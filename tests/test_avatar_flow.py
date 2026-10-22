"""End-to-end avatar job with fake Duix / TTS / ASR and real ffmpeg (skipped without ffmpeg)."""

import asyncio
import shutil
import subprocess

import pytest

from mywebtools import duix, llm, nodes, speech
from mywebtools.config import settings
from mywebtools.jobs import JobManager
from mywebtools.storage import Library, UploadStore
from mywebtools.tools import REGISTRY

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _video(path, size, seconds, color, audio):
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=25:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", "-shortest"]
    subprocess.run(args + ["-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)], check=True)


class FakeEngine:
    name = "duix_local"

    async def account(self):
        return {"valid": True, "plan": "local", "credits_left": None, "max_minutes": 0}

    def check(self, account, seconds):
        return {"credits_needed": 0, "credits_left": None}

    async def generate(self, video, audio, dest, progress, save_task, task_id="", model_name=""):
        save_task("mwt_fake")
        progress(50, "duix.running", {"task": "mwt_fake"})
        await asyncio.sleep(0.2)  # "rendering" while the subtitle draft waits for review
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-map", "0:v",
                        "-map", "1:a", "-shortest", "-c:v", "copy", "-c:a", "aac", str(dest)], check=True)
        return dest


async def _wait(job, states):
    for _ in range(600):
        if job.state in states:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"stuck in {job.state}: {job.error}")


def _sine(dest, seconds=3, rate=None):
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=500:duration={seconds}"]
    subprocess.run(args + (["-ar", str(rate)] if rate else []) + [str(dest)], check=True)
    return dest


async def _avatar_job(tmp_path, params_extra):
    main, model = tmp_path / "vlog.mp4", tmp_path / "model.mp4"
    _video(main, "1280x720", 4, "blue", audio=True)  # landscape iPhone-style clip
    _video(model, "720x1280", 3, "white", audio=False)
    uploads, library = UploadStore(settings.uploads), Library(settings.library)

    async def chunks(path):
        yield path.read_bytes()

    up = await uploads.save("vlog.mp4", chunks(main), 10**9)
    item = library.add("avatar_models", model, "presenter")
    tool = REGISTRY["avatar_pip"]
    params = tool.coerce({
        "avatar_source": "duix", "model": item.id, "audio_source": "tts", "script": "大家好。今天去海边！",
        "subtitles": True, "sub_source": "auto", "sub_ai": "none", "sub_review": True, "sub_burn": True,
        "canvas": "shorts_720", "size": 200, "crf": 30, "preset": "ultrafast", "fps": "25", "avatar_delay": 0.5,
        **params_extra,
    })
    mgr = JobManager(settings, REGISTRY, uploads, library)
    await mgr.start()
    return mgr, mgr.submit("avatar_pip", params, {"main_video": [up.id]}), library


async def _approve_and_finish(mgr, job, library, first_start_srt):
    await _wait(job, ("review", "failed"))
    assert job.state == "review", job.error
    draft = job.meta["review_draft"]["cues"]
    assert [c["text"] for c in draft] == ["大家好。", "今天去海边！"]  # the typed script, timed to the voice
    edited = [{**draft[0], "text": "大家好！"}, draft[1]]
    mgr.submit_review(job.id, {"cues": edited})
    await _wait(job, ("done", "failed"))
    assert job.state == "done", job.error
    names = {o["name"] for o in job.outputs}
    assert {"vlog_avatar.mp4", "vlog_avatar.srt"} <= names
    srt = (settings.outputs / job.id / "vlog_avatar.srt").read_text(encoding="utf-8")
    assert "大家好！" in srt and f"{first_start_srt} -->" in srt
    assert job.result["subtitles"] == 2
    assert library.list("avatar_clips"), "generated clip saved for re-compositing"
    out = await __import__("mywebtools.media", fromlist=["probe"]).probe(settings.outputs / job.id / "vlog_avatar.mp4")
    assert (out.width, out.height) == (720, 1280) and out.has_audio
    return draft


async def test_indextts_voice_clone_mms_subtitles_review_and_burn(tmp_path, monkeypatch):
    """Typed script -> IndexTTS (voice cloned from the main video) -> Duix -> MMS-timed subtitles."""
    calls = []

    async def fake_tts(self, text, voice, dest, options, emo_audio=None):
        assert voice.is_file() and voice.stat().st_size > 1000, "voice sample cut from the main video"
        calls.append(("tts", text, options))
        return _sine(dest, rate=22050)

    async def fake_align(self, audio, cues, language):
        calls.append(("align", cues, language))
        return {"cues": [{"start": 0.3, "end": 1.2, "aligned": True}, {"start": 1.5, "end": 2.8, "aligned": True}],
                "words": []}

    async def fake_unload(self):
        calls.append(("unload", self.kind))
        return {"ok": True}

    monkeypatch.setattr(duix, "engine", lambda kind=None: FakeEngine())
    monkeypatch.setattr(nodes.Node, "tts", fake_tts)
    monkeypatch.setattr(nodes.Node, "align", fake_align)
    monkeypatch.setattr(nodes.Node, "unload", fake_unload)
    mgr, job, library = await _avatar_job(tmp_path, {
        "tts_engine": "indextts", "tts_ref": "main", "tts_lang": "ZH", "tts_emotion": "vector", "emo_happy": 0.6,
        "tts_duration": 1.1})
    try:
        draft = await _approve_and_finish(mgr, job, library, "00:00:00,800")  # 0.3 s + 0.5 s avatar delay
        assert draft[0]["end"] == pytest.approx(1.45)  # held 0.25 s into the pause
        tts = next(c for c in calls if c[0] == "tts")
        assert tts[1] == "大家好。今天去海边！" and tts[2]["lang"] == "ZH" and tts[2]["duration_factor"] == 1.1
        assert tts[2]["emotion"] == "vector" and tts[2]["emo_vector"][0] == 0.6
        assert ("align", ["大家好。", "今天去海边！"], "zh") in calls
        assert ("unload", "tts") in calls, "GPU memory freed for Duix"
    finally:
        await mgr.stop()


async def test_openai_voice_basic_timing_review_and_burn(tmp_path, monkeypatch):
    """Typed script -> OpenAI voice -> Duix -> script timed by speech recognition."""

    async def fake_speech(self, text, dest, **kw):
        return _sine(dest)

    async def fake_asr(audio, language="", prompt="", log=None):
        return [{"start": 0.2, "end": 1.4, "text": "大家好"}, {"start": 1.6, "end": 2.9, "text": "今天去海边"}]

    monkeypatch.setattr(duix, "engine", lambda kind=None: FakeEngine())
    monkeypatch.setattr(llm.LLM, "speech", fake_speech)
    monkeypatch.setattr(speech, "recognize", fake_asr)
    mgr, job, library = await _avatar_job(tmp_path, {"tts_engine": "openai", "sub_align": "basic"})
    try:
        draft = await _approve_and_finish(mgr, job, library, "00:00:00,700")  # 0.2 s + 0.5 s avatar delay
        assert draft[1]["start"] == pytest.approx(1.6, abs=0.05)
    finally:
        await mgr.stop()


async def test_tts_tool_with_subtitles(tmp_path, monkeypatch):
    """The stand-alone voice tool: uploaded sample -> IndexTTS -> MP3 + aligned SRT; sample saved to the library."""
    async def fake_tts(self, text, voice, dest, options, emo_audio=None):
        return _sine(dest, rate=22050)

    async def fake_align(self, audio, cues, language):
        assert language == "en"
        return {"cues": [{"start": 0.1 + i, "end": 0.8 + i, "aligned": True} for i in range(len(cues))], "words": []}

    monkeypatch.setattr(nodes.Node, "tts", fake_tts)
    monkeypatch.setattr(nodes.Node, "align", fake_align)
    uploads, library = UploadStore(settings.uploads), Library(settings.library)
    sample = _sine(tmp_path / "me.m4a", 5)

    async def chunks(path):
        yield path.read_bytes()

    up = await uploads.save("me.m4a", chunks(sample), 10**9)
    tool = REGISTRY["tts"]
    params = tool.coerce({"text": "Hello there. This is my cloned voice.", "tts_ref": "upload", "save_ref": True,
                          "format": "mp3", "subtitles": True})
    mgr = JobManager(settings, REGISTRY, uploads, library)
    await mgr.start()
    try:
        job = mgr.submit("tts", params, {"ref_file": [up.id]})
        await _wait(job, ("done", "failed"))
        assert job.state == "done", job.error
        assert {o["name"] for o in job.outputs} == {"speech.mp3", "speech.srt"}
        assert job.result["subtitles"] == 2 and library.list("voices")
    finally:
        await mgr.stop()


async def test_align_tool_retimes_an_srt(tmp_path, monkeypatch):
    async def fake_align(self, audio, cues, language):
        return {"cues": [{"start": 1.0, "end": 1.5, "aligned": True}, {"start": 2.0, "end": 2.6, "aligned": True}],
                "words": [{"cue": 0, "text": "one", "start": 1.0, "end": 1.5}]}

    monkeypatch.setattr(nodes.Node, "align", fake_align)
    uploads, library = UploadStore(settings.uploads), Library(settings.library)
    voice = _sine(tmp_path / "voice.wav", 4)
    srt = tmp_path / "script.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nLine one\n\n2\n00:00:01,000 --> 00:00:02,000\nLine two\n",
                   encoding="utf-8")

    async def chunks(path):
        yield path.read_bytes()

    media, script = await uploads.save("voice.wav", chunks(voice), 10**9), await uploads.save("script.srt", chunks(srt), 10**9)
    tool = REGISTRY["align_subs"]
    mgr = JobManager(settings, REGISTRY, uploads, library)
    await mgr.start()
    try:
        job = mgr.submit("align_subs", tool.coerce({"formats": ["srt", "json"]}),
                         {"media": [media.id], "script_file": [script.id]})
        await _wait(job, ("done", "failed"))
        assert job.state == "done", job.error
        out = (settings.outputs / job.id / "voice.srt").read_text(encoding="utf-8")
        assert "00:00:01,000 --> 00:00:01,750\nLine one" in out and job.result["engine"] == "mms"
    finally:
        await mgr.stop()
