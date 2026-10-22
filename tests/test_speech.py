"""Engine choice and fallbacks of speech.py (services faked)."""

import pytest

from mywebtools import nodes, speech
from mywebtools.config import settings
from mywebtools.errors import ToolError
from mywebtools.llm import LLM


def _offline(*args, **kwargs):
    raise ToolError("node.offline", service="x", url="http://pc")


async def test_auto_recognition_falls_back_when_the_pc_is_off(monkeypatch, tmp_path):
    async def offline(self, *a, **kw):
        _offline()

    async def api(self, audio, language=None, prompt=""):
        return [{"start": 0.0, "end": 1.0, "text": "hi"}]

    monkeypatch.setattr(nodes.Node, "transcribe", offline)
    monkeypatch.setattr(LLM, "transcribe", api)
    monkeypatch.setattr(LLM, "configured", property(lambda self: True))
    monkeypatch.setattr(speech, "local_whisper", lambda: False)
    logs = []
    result = await speech.transcribe(tmp_path / "a.wav", engine="auto", log=logs.append)
    assert result["engine"] == "api" and result["segments"][0]["text"] == "hi"
    assert "node unavailable (node.offline)" in logs[0]
    with pytest.raises(ToolError) as err:
        await speech.transcribe(tmp_path / "a.wav", engine="node")
    assert err.value.code == "node.offline"


def test_asr_order(monkeypatch):
    monkeypatch.setattr(speech, "local_whisper", lambda: True)
    monkeypatch.setattr(LLM, "configured", property(lambda self: False))
    assert speech.asr_order("auto") == ["node", "local"]
    monkeypatch.setattr(settings.runtime, "node_whisper_url", "")
    assert speech.asr_order("auto") == ["local"]
    monkeypatch.setattr(speech, "local_whisper", lambda: False)
    with pytest.raises(ToolError):
        speech.asr_order("auto")


async def test_align_auto_falls_back_to_recognition(monkeypatch, tmp_path):
    async def offline(self, *a, **kw):
        _offline()

    async def asr(audio, language="", prompt="", log=None):
        return [{"start": 1.0, "end": 2.0, "text": "AAAA"}, {"start": 3.0, "end": 4.0, "text": "BBBB"}]

    monkeypatch.setattr(nodes.Node, "align", offline)
    monkeypatch.setattr(speech, "recognize", asr)
    notes = []
    cues, words = await speech.align_script(["AAAA", "BBBB"], tmp_path / "v.wav", 5.0, engine="auto", notes=notes)
    assert notes == ["subs.align_fallback"] and words == []
    assert cues[1].start == pytest.approx(3.0)
    with pytest.raises(ToolError):
        await speech.align_script(["AAAA"], tmp_path / "v.wav", 5.0, engine="mms")


async def test_align_mms_keeps_text_and_holds_lines(monkeypatch, tmp_path):
    async def aligned(self, audio, cues, language):
        assert language == "zh"  # guessed from the script
        return {"cues": [{"start": 0.5, "end": 1.0, "aligned": True}, {"start": 1.1, "end": 2.0, "aligned": True}],
                "words": [{"cue": 0, "text": "大", "start": 0.5, "end": 0.7}]}

    monkeypatch.setattr(nodes.Node, "align", aligned)
    cues, words = await speech.align_script(["大家好。", "今天去海边！"], tmp_path / "v.wav", 3.0, engine="auto")
    assert [c.text for c in cues] == ["大家好。", "今天去海边！"]
    assert cues[0].end == pytest.approx(1.1)  # held into the pause, but never over the next line
    assert cues[1].end == pytest.approx(2.25) and words


async def test_indextts_needs_a_voice_sample(tmp_path):
    with pytest.raises(ToolError) as err:
        await speech.synthesize("hello", tmp_path / "o.wav", tmp_path, engine="indextts")
    assert err.value.code == "tts.no_reference"


async def test_long_scripts_are_spoken_in_parts(monkeypatch, tmp_path):
    spoken = []

    async def fake_tts(self, text, voice, dest, options, emo_audio=None):
        spoken.append(text)
        dest.write_bytes(b"RIFF")
        return dest

    async def fake_concat(args, **kw):
        (tmp_path / "o.wav").write_bytes(b"joined")

    monkeypatch.setattr(nodes.Node, "tts", fake_tts)
    monkeypatch.setattr(speech, "run_cmd", fake_concat)
    script = "。".join(["这是一个很长的句子" * 10] * 10) + "。"
    await speech.synthesize(script, tmp_path / "o.wav", tmp_path, engine="indextts", reference=tmp_path / "r.wav")
    assert len(spoken) > 1 and all(len(s) <= 600 for s in spoken) and "".join(spoken) == script


def test_guess_language():
    assert speech.guess_language("今日はいい天気") == "ja"
    assert speech.guess_language("今天天气很好") == "zh"
    assert speech.guess_language("안녕하세요") == "ko"
    assert speech.guess_language("Hello there") == "en"
