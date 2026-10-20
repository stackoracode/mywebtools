"""End-to-end avatar job with fake Duix / TTS / ASR and real ffmpeg (skipped without ffmpeg)."""

import asyncio
import shutil
import subprocess

import pytest

from mywebtools import duix, subtitles
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


async def test_typed_script_tts_subtitles_review_and_burn(tmp_path, monkeypatch):
    main, model = tmp_path / "vlog.mp4", tmp_path / "model.mp4"
    _video(main, "1280x720", 4, "blue", audio=True)  # landscape iPhone-style clip
    _video(model, "720x1280", 3, "white", audio=False)

    async def fake_tts(script, dest, workdir, **kw):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=500:duration=3",
                        str(dest)], check=True)
        return dest

    async def fake_asr(audio, language="", prompt=""):
        return [{"start": 0.2, "end": 1.4, "text": "大家好"}, {"start": 1.6, "end": 2.9, "text": "今天去海边"}]

    monkeypatch.setattr(duix, "engine", lambda kind=None: FakeEngine())
    monkeypatch.setattr(subtitles, "synthesize", fake_tts)
    monkeypatch.setattr(subtitles, "recognize", fake_asr)

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
    })
    mgr = JobManager(settings, REGISTRY, uploads, library)
    await mgr.start()
    try:
        job = mgr.submit("avatar_pip", params, {"main_video": [up.id]})
        await _wait(job, ("review", "failed"))
        assert job.state == "review", job.error
        draft = job.meta["review_draft"]["cues"]
        assert [c["text"] for c in draft] == ["大家好。", "今天去海边！"]  # typed script, timed by speech
        assert draft[1]["start"] == pytest.approx(1.6, abs=0.05)
        edited = [{**draft[0], "text": "大家好！"}, draft[1]]
        mgr.submit_review(job.id, {"cues": edited})
        await _wait(job, ("done", "failed"))
        assert job.state == "done", job.error
        names = {o["name"] for o in job.outputs}
        assert {"vlog_avatar.mp4", "vlog_avatar.srt"} <= names
        srt = (settings.outputs / job.id / "vlog_avatar.srt").read_text(encoding="utf-8")
        assert "大家好！" in srt and "00:00:00,700 -->" in srt  # 0.2 s + 0.5 s avatar delay
        assert job.result["subtitles"] == 2
        assert library.list("avatar_clips"), "generated clip saved for re-compositing"
        out = await __import__("mywebtools.media", fromlist=["probe"]).probe(settings.outputs / job.id / "vlog_avatar.mp4")
        assert (out.width, out.height) == (720, 1280) and out.has_audio
    finally:
        await mgr.stop()
