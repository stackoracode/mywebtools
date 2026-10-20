import asyncio

import pytest

from mywebtools.config import settings
from mywebtools.errors import ToolError
from mywebtools.jobs import Job, JobManager
from mywebtools.storage import UploadStore
from mywebtools.tools.base import Param, Tool


class Echo(Tool):
    id = "echo"
    lane = "cpu"

    def params(self):
        return [Param("text", "text", "Text", default="hi"), Param("sleep", "number", "Sleep", default=0)]

    async def run(self, ctx, params):
        ctx.progress(10, "test.stage")
        await asyncio.sleep(params["sleep"])
        if params["text"] == "boom":
            raise ToolError("test.boom", "detail", n=1)
        path = ctx.out_path("out.txt")
        path.write_text(params["text"])
        ctx.add_output(path)
        ctx.set_result({"len": len(params["text"])})


async def _wait(job, states=("done", "failed", "cancelled")):
    for _ in range(200):
        if job.state in states:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(job.state)


@pytest.fixture
async def manager():
    tool = Echo()
    mgr = JobManager(settings, {"echo": tool}, UploadStore(settings.uploads))
    await mgr.start()
    yield mgr
    await mgr.stop()


async def test_job_runs_and_produces_output(manager):
    job = manager.submit("echo", Echo().coerce({"text": "hello"}), {})
    await _wait(job)
    assert job.state == "done" and job.progress == 100
    assert job.outputs[0]["name"] == "out.txt"
    assert manager.output_path(job.id, "out.txt").read_text() == "hello"
    assert job.result == {"len": 5}


async def test_tool_error_is_reported(manager):
    job = manager.submit("echo", {"text": "boom", "sleep": 0}, {})
    await _wait(job)
    assert job.state == "failed" and job.error["code"] == "test.boom" and job.error["params"] == {"n": 1}


async def test_cancel_running_and_queued(manager):
    first = manager.submit("echo", {"text": "a", "sleep": 5}, {})
    second = manager.submit("echo", {"text": "b", "sleep": 0}, {})  # waits: cpu lane has one worker
    await _wait(first, ("running",))
    manager.cancel(second.id)
    manager.cancel(first.id)
    await _wait(first)
    assert first.state == "cancelled" and second.state == "cancelled"


async def test_events_are_broadcast(manager):
    queue = manager.open_subscription()
    job = manager.submit("echo", {"text": "x", "sleep": 0}, {})
    await _wait(job)
    states = set()
    while not queue.empty():
        event = queue.get_nowait()
        if event["type"] == "job" and event["job"]["id"] == job.id:
            states.add(event["job"]["state"])
    manager.close_subscription(queue)
    assert {"queued", "done"} <= states


def test_restore_marks_interrupted_and_keeps_resume_meta():
    job = Job(tool="avatar_pip", lane="net", params={}, files={}, state="running",
              meta={"resume_duix_task": "mwt_1", "resume_engine": "duix_local"})
    restored = Job.restore(job.snapshot())
    assert restored.state == "interrupted"
    assert restored.meta["resume_duix_task"] == "mwt_1"


def test_coerce_clamps_and_drops_unknown():
    tool = Echo()
    clean = tool.coerce({"text": 5, "sleep": "-3", "evil": 1})
    assert clean == {"text": "5", "sleep": -3.0}


class Reviewer(Tool):
    id = "reviewer"
    lane = "net"

    async def run(self, ctx, params):
        result = await ctx.request_review("subtitles", {"cues": [{"start": 0, "end": 1, "text": "draft"}]})
        ctx.set_result({"text": result["cues"][0]["text"]})


async def test_review_pauses_until_approved():
    mgr = JobManager(settings, {"reviewer": Reviewer()}, UploadStore(settings.uploads))
    await mgr.start()
    try:
        job = mgr.submit("reviewer", {}, {})
        await _wait(job, ("review",))
        assert job.meta["review_draft"]["cues"][0]["text"] == "draft"
        with pytest.raises(ToolError):
            mgr.submit_review("0" * 32, {})
        mgr.submit_review(job.id, {"cues": [{"start": 0, "end": 1, "text": "edited"}]})
        await _wait(job)
        assert job.state == "done" and job.result == {"text": "edited"}
        again = mgr.retry(job.id)  # a re-run offers the edited draft again
        assert again.meta["resume_review_draft"]["cues"][0]["text"] == "edited"
        await _wait(again, ("review",))
        mgr.cancel(again.id)
        await _wait(again)
        assert again.state == "cancelled"
    finally:
        await mgr.stop()
