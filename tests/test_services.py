"""The model services (services/*) with stand-in models, and the app's client talking to them.

The real models (faster-whisper-turbo, IndexTTS 2.5, MMS aligner) only run on the GPU PC;
here each service's web layer, file checks, options and the client/server protocol are tested.
"""

import argparse
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mywebtools.errors import ToolError
from mywebtools.nodes import Node

SERVICES = Path(__file__).resolve().parent.parent / "services"
sys.path.append(str(SERVICES))
sys.path.append(str(SERVICES / "aligner"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"svc_{name}", SERVICES / name / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


whisper_svc, indextts_svc, aligner_svc = _load("whisper"), _load("indextts"), _load("aligner")


def _wav(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


def _folder(root: Path, name: str, files) -> Path:
    folder = root / name
    for entry in files:
        rel = entry[0] if isinstance(entry, tuple) else entry
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"x")
    return folder


def _args(root: Path, folder: Path, **kw):
    return argparse.Namespace(model_dir=str(folder), models_root=str(root), device="cpu", idle_unload=0, token="",
                              host="127.0.0.1", port=0, test=None, **kw)


# ---------------------------------------------------------------- whisper
class FakeWhisper:
    def transcribe(self, path, **kw):
        assert Path(path).is_file()
        self.kw = kw
        segs = [SimpleNamespace(start=0.0, end=1.2, text=" Hello there", words=None),
                SimpleNamespace(start=1.5, end=2.5, text=" 大家好", words=None)]
        return iter(segs), SimpleNamespace(language="en", language_probability=0.98, duration=3.0)


@pytest.fixture
def whisper(tmp_path, monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(whisper_svc, "load_model", lambda key: fake)
    folder = _folder(tmp_path, "faster-whisper-turbo", whisper_svc.FILES + ["preprocessor_config.json"])
    app, slot = whisper_svc.build(_args(tmp_path, folder))
    return app, slot, fake, folder


def test_whisper_health_and_stream(whisper, tmp_path):
    app, slot, fake, folder = whisper
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["service"] == "whisper" and health["model"]["ok"]
    assert {f["name"] for f in health["model"]["files"]} >= {"model.bin", "vocabulary.json | vocabulary.txt"}
    missing = client.get("/health", params={"model_dir": str(tmp_path / "nope")}).json()
    assert missing["model"]["exists"] is False and missing["ok"] is False
    resp = client.post("/transcribe", files={"file": ("a.wav", _wav(), "audio/wav")},
                       data={"options": json.dumps({"language": "en", "beam_size": 3, "hotwords": "Duix"})})
    lines = [json.loads(x) for x in resp.text.splitlines()]
    assert [x["type"] for x in lines] == ["info", "segment", "segment", "done"]
    assert lines[1]["text"] == "Hello there" and fake.kw["beam_size"] == 3 and fake.kw["hotwords"] == "Duix"
    assert slot.status()["loaded"] and client.post("/unload").json()["unloaded"] is True


def test_whisper_rejects_folders(whisper, tmp_path):
    app, _, _, folder = whisper
    client = TestClient(app)
    outside = client.post("/transcribe", files={"file": ("a.wav", _wav(), "audio/wav")},
                          data={"options": json.dumps({"model_dir": str(tmp_path.parent)})})
    assert outside.status_code == 403 and outside.json()["error"]["code"] == "model_outside_root"
    (folder / "model.bin").unlink()
    broken = client.post("/transcribe", files={"file": ("a.wav", _wav(), "audio/wav")}, data={"options": "{}"})
    assert broken.status_code == 404 and broken.json()["error"]["code"] == "files_missing"


def test_service_token(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_svc, "load_model", lambda key: FakeWhisper())
    folder = _folder(tmp_path, "w", whisper_svc.FILES)
    args = _args(tmp_path, folder)
    args.token = "s3cret"
    client = TestClient(whisper_svc.build(args)[0])
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200


async def test_client_transcribe_through_service(whisper, monkeypatch):
    app, _, _, folder = whisper
    seen = []
    node = Node("whisper", url="http://gpu", model_dir=str(folder), transport=httpx.ASGITransport(app=app))
    result = await node.transcribe(Path(__file__).parent / "conftest.py", {"language": "en"},
                                   on_segment=lambda seg, total: seen.append((seg["end"], total)))
    assert result["language"] == "en" and len(result["segments"]) == 2 and seen[-1] == (2.5, 3.0)
    health = await node.health()
    assert health["reachable"] and health["model"]["ok"]
    bad = Node("whisper", url="http://gpu", model_dir=str(folder / "missing"), transport=httpx.ASGITransport(app=app))
    with pytest.raises(ToolError) as err:
        await bad.transcribe(Path(__file__), {})
    assert err.value.code == "node.model_missing"


async def test_client_offline():
    def refuse(request):
        raise httpx.ConnectError("refused")

    node = Node("tts", url="http://192.0.2.1:8392", transport=httpx.MockTransport(refuse))
    health = await node.health()
    assert health["reachable"] is False and health["error"]["code"] == "node.offline"
    with pytest.raises(ToolError) as err:
        await node.unload()
    assert err.value.code == "node.offline"


# ---------------------------------------------------------------- IndexTTS
class FakeIndexTTS:
    instances = 0

    def __init__(self, cfg_path, model_dir, use_bf16, device, use_cuda_kernel, use_qwen_emo):
        FakeIndexTTS.instances += 1
        self.qwen_emo = object() if use_qwen_emo else None
        self.calls = []
        FakeIndexTTS.last = self

    def normalize_emo_vec(self, vec, apply_bias=True):
        return [v / 2 for v in vec]

    def infer(self, spk_audio_prompt, text, output_path, lang, **kw):
        self.calls.append({"voice": spk_audio_prompt, "text": text, "lang": lang, **kw})
        Path(output_path).write_bytes(_wav(1.5, 22050))
        return output_path


@pytest.fixture
def indextts(tmp_path, monkeypatch):
    FakeIndexTTS.instances = 0
    monkeypatch.setattr(indextts_svc, "import_engine", lambda repo: FakeIndexTTS)
    folder = _folder(tmp_path, "IndexTTS-2.5", indextts_svc.FILES)
    (folder / "qwen0.6bemo4-merge").mkdir()  # empty, like on the GPU PC today
    args = _args(tmp_path, folder, repo="", bf16=True, emotion_model="auto", cache_dir=str(tmp_path / "refs"))
    app, slot = indextts_svc.build(args)
    return TestClient(app), folder


def _tts(client, text, options, emo=False):
    files = {"voice": ("me.wav", _wav(2), "audio/wav")}
    if emo:
        files["emo_audio"] = ("angry.wav", _wav(1), "audio/wav")
    return client.post("/tts", data={"text": text, "options": json.dumps(options)}, files=files)


def test_indextts_synthesis_and_emotions(indextts):
    client, folder = indextts
    health = client.get("/health").json()
    assert health["model"]["ok"] and health["model"]["optional"]["emotion_text"]["ok"] is False
    resp = _tts(client, "大家好，今天去海边。", {"duration_factor": 1.2})
    assert resp.status_code == 200 and resp.headers["x-lang"] == "ZH" and float(resp.headers["x-seconds"]) == 1.5
    call = FakeIndexTTS.last.calls[-1]
    assert call["duration_factor"] == 1.2 and "emo_vector" not in call
    resp = _tts(client, "Hello there", {"emotion": "vector", "emo_vector": [0.8, 0, 0, 0, 0, 0, 0, 0.2]})
    assert FakeIndexTTS.last.calls[-1]["emo_vector"] == [0.4, 0, 0, 0, 0, 0, 0, 0.1]
    assert FakeIndexTTS.last.calls[-1]["lang"] == "EN"
    _tts(client, "Hello", {"emotion": "audio", "emo_alpha": 0.5}, emo=True)
    assert FakeIndexTTS.last.calls[-1]["emo_audio_prompt"].endswith(".wav")
    voices = {c["voice"] for c in FakeIndexTTS.last.calls}
    assert len(voices) == 1, "same reference audio -> same cached path (IndexTTS reuses its speaker features)"
    no_emo = _tts(client, "Hello", {"emotion": "text"})
    assert no_emo.status_code == 400 and no_emo.json()["error"]["code"] == "emotion_model_missing"
    bad = _tts(client, "Hello", {"lang": "FR"})
    assert bad.status_code == 400


def test_indextts_emotion_text_when_downloaded(indextts):
    client, folder = indextts
    for name in indextts_svc.EMOTION_FILES:
        (folder / name).write_bytes(b"x")
    assert client.get("/health").json()["model"]["optional"]["emotion_text"]["ok"]
    resp = _tts(client, "太好了！", {"emotion": "text", "emo_text": "excited", "emo_alpha": 0.6})
    assert resp.status_code == 200
    call = FakeIndexTTS.last.calls[-1]
    assert call["use_emo_text"] is True and call["emo_text"] == "excited" and call["emo_alpha"] == 0.6


async def test_client_tts_through_service(indextts, tmp_path):
    client, folder = indextts
    voice = tmp_path / "voice.wav"
    voice.write_bytes(_wav(3))
    node = Node("tts", url="http://gpu", model_dir=str(folder), transport=httpx.ASGITransport(app=client.app))
    out = await node.tts("Hello world", voice, tmp_path / "out.wav", {"lang": "EN"})
    with wave.open(str(out)) as w:
        assert w.getframerate() == 22050
    with pytest.raises(ToolError) as err:
        await node.tts("Hello", voice, tmp_path / "x.wav", {"emotion": "text"})
    assert err.value.code == "node.emotion_model_missing"


# ---------------------------------------------------------------- aligner
VOCAB = {"<blank>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{c: i + 4 for i, c in enumerate("aienoutsrmkldghybpwcvjzf'qx")}}


def _emissions(frames, placements):
    lp = np.full((frames, len(VOCAB)), -8.0, dtype=np.float32)
    lp[:, 0] = -0.05
    for start, word in placements:
        for i, ch in enumerate(word):
            lp[start + 2 * i] = -8.0
            lp[start + 2 * i, VOCAB[ch]] = -0.05
    return lp


def test_ctc_alignment_places_words():
    import ctc

    lp, star = ctc.emissions_with_star(_emissions(250, [(50, "hello"), (62, "there"), (100, "um"), (150, "good"), (160, "morning")]))
    result = ctc.align(lp, 0.02, ["Hello there!", "2026", "Good morning."], VOCAB, None, star=star)
    first, digits, last = result["cues"]
    assert first["start"] == pytest.approx(1.0) and first["aligned"]
    assert last["start"] == pytest.approx(3.0) and last["end"] == pytest.approx(3.46)
    assert digits["aligned"] is False and first["end"] <= digits["start"] <= digits["end"] <= last["start"]
    assert [w["text"] for w in result["words"]] == ["Hello", "there!", "Good", "morning."]


def test_ctc_units_cjk_and_errors():
    import ctc

    pieces = ctc.units(["你好 OK"], VOCAB, lambda s: {"你": "ni", "好": "hao"}.get(s, s))
    assert [(p["text"], "".join(p["tokens"])) for p in pieces] == [("你", "ni"), ("好", "hao"), ("OK", "ok")]
    with pytest.raises(ctc.AlignError) as err:
        ctc.viterbi(np.zeros((3, 5), dtype=np.float32), np.array([4, 4, 4, 4]))
    assert err.value.code == "audio_too_short"
    assert ctc.iso3("zh-CN") == "zho" and ctc.iso3("eng") == "eng"


def test_ctc_real_uroman_chinese():
    import ctc

    uroman = pytest.importorskip("uroman")
    u = uroman.Uroman()
    pieces = ctc.units(["大家好"], VOCAB, lambda s: u.romanize_string(s, lcode="zho"))
    assert "".join("".join(p["tokens"]) for p in pieces) == "dajiahao"


@pytest.fixture
def aligner(tmp_path, monkeypatch):
    monkeypatch.setattr(aligner_svc, "load_model", lambda key: {"vocab": VOCAB})
    monkeypatch.setattr(aligner_svc, "emissions",
                        lambda bundle, audio: (_emissions(250, [(50, "hello"), (150, "world")]), 0.02))
    monkeypatch.setattr(aligner_svc, "romanizer", lambda lang: None)
    folder = _folder(tmp_path, "mms-300m-1130-forced-aligner", aligner_svc.FILES)
    app, _ = aligner_svc.build(_args(tmp_path, folder))
    return app, folder


async def test_client_align_through_service(aligner, tmp_path):
    app, folder = aligner
    audio = tmp_path / "v.wav"
    audio.write_bytes(_wav(5))
    node = Node("align", url="http://gpu", model_dir=str(folder), transport=httpx.ASGITransport(app=app))
    result = await node.align(audio, ["Hello,", "world."], "en")
    assert [c["start"] for c in result["cues"]] == [pytest.approx(1.0), pytest.approx(3.0)]
    assert result["language"] == "eng"
    with pytest.raises(ToolError) as err:
        await node.align(audio, ["123"], "en")
    assert err.value.code == "node.no_letters"


def test_aligner_reads_wav(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(_wav(0.5))
    assert aligner_svc.read_audio(path).shape == (8000,)
