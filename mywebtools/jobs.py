"""Single-process, in-memory job queue built on asyncio.

Jobs are routed to *lanes*; each lane is an ``asyncio.Queue`` drained by a
fixed number of worker coroutines on the one event loop:

* ``cpu``   - ffmpeg / OCR / Whisper. Default 1 worker so heavy jobs never
              fight over the board's CPU.
* ``net``   - cloud work that mostly waits on the network (DUIX upload/poll).
* ``light`` - quick jobs (probe, small PDFs).

Heavy work never blocks the loop: external programs run through
``asyncio.create_subprocess_exec`` and Python CPU work through
``asyncio.to_thread``. Finished jobs are snapshotted to ``data/jobs.json`` so
history survives a restart; queued/running jobs come back as *interrupted* and
can be retried (an avatar job resumes its cloud task instead of paying again).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Settings
from .errors import ToolError
from .storage import Library, UploadStore, new_id, safe_name

if TYPE_CHECKING:
    from .tools.base import Tool

log = logging.getLogger("mywebtools.jobs")

ACTIVE = ("queued", "running")
FINAL = ("done", "failed", "cancelled", "interrupted")
HISTORY_LIMIT = 300


@dataclass
class Job:
    tool: str
    lane: str
    params: dict[str, Any]
    files: dict[str, list[str]]
    title: str = ""
    batch_id: str = ""
    id: str = field(default_factory=new_id)
    state: str = "queued"
    progress: float | None = None
    stage: str = ""
    stage_params: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    logs: deque = field(default_factory=lambda: deque(maxlen=200))
    cancel_requested: bool = False

    def public(self, with_logs: bool = False) -> dict[str, Any]:
        data = {
            k: getattr(self, k)
            for k in ("id", "tool", "lane", "title", "batch_id", "state", "progress", "stage", "stage_params",
                      "created", "started", "finished", "outputs", "result", "error", "params", "files")
        }
        data["meta"] = {k: v for k, v in self.meta.items() if not k.startswith("_")}
        if with_logs:
            data["logs"] = list(self.logs)
        return data

    def snapshot(self) -> dict[str, Any]:
        data = self.public(with_logs=True)
        data["meta"] = self.meta
        return data

    @classmethod
    def restore(cls, data: dict[str, Any]) -> Job:
        job = cls(tool=data["tool"], lane=data["lane"], params=data.get("params", {}), files=data.get("files", {}))
        for key in ("id", "title", "batch_id", "state", "progress", "stage", "stage_params", "created", "started",
                    "finished", "outputs", "result", "error", "meta"):
            if key in data:
                setattr(job, key, data[key])
        job.logs.extend(data.get("logs", []))
        if job.state in ACTIVE:
            job.state = "interrupted"
            job.error = {"code": "job.interrupted", "params": {}, "detail": ""}
        return job


class JobContext:
    """What a tool sees while it runs."""

    def __init__(self, manager: JobManager, job: Job) -> None:
        self.manager = manager
        self.job = job
        self.settings: Settings = manager.settings
        self.workdir = manager.settings.work / job.id
        self.outdir = manager.settings.outputs / job.id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(parents=True, exist_ok=True)

    @property
    def meta(self) -> dict[str, Any]:
        return self.job.meta

    def inputs(self, name: str) -> list[Path]:
        return [self.manager.uploads.get(uid).path for uid in self.job.files.get(name, [])]

    def input(self, name: str, required: bool = True) -> Path | None:
        paths = self.inputs(name)
        if not paths and required:
            raise ToolError("input.missing", field=name)
        return paths[0] if paths else None

    def progress(self, value: float | None = None, stage: str | None = None, **params: Any) -> None:
        if value is not None:
            self.job.progress = max(0.0, min(100.0, round(float(value), 1)))
        if stage is not None:
            self.job.stage = stage
            self.job.stage_params = params
        self.manager.notify(self.job)

    def log(self, message: str) -> None:
        self.job.logs.append(f"{time.strftime('%H:%M:%S')} {message}")
        log.info("[%s] %s", self.job.id[:8], message)

    def out_path(self, name: str) -> Path:
        name = safe_name(name)
        path = self.outdir / name
        stem, suffix, n = path.stem, path.suffix, 1
        while path.exists():
            path = self.outdir / f"{stem}_{n}{suffix}"
            n += 1
        return path

    def add_output(self, path: Path, label: str = "") -> None:
        if path.parent != self.outdir:
            target = self.out_path(path.name)
            shutil.move(str(path), str(target))
            path = target
        self.job.outputs.append({"name": path.name, "size": path.stat().st_size, "label": label})
        self.manager.notify(self.job, force=True)

    def set_result(self, result: dict[str, Any]) -> None:
        self.job.result = result

    def save_meta(self, **values: Any) -> None:
        """Persist resumable state (e.g. remote task ids) immediately."""
        self.job.meta.update(values)
        self.manager.persist()

    def check_cancel(self) -> None:
        if self.job.cancel_requested:
            raise asyncio.CancelledError()


class JobManager:
    def __init__(self, settings: Settings, tools: dict[str, Tool], uploads: UploadStore, library: Library | None = None) -> None:
        self.settings = settings
        self.tools = tools
        self.uploads = uploads
        self.library = library
        # Shared CPU budget: cpu-lane jobs hold it for their whole run, other lanes
        # take it only around heavy steps (e.g. the avatar job's ffmpeg encode).
        self.cpu_slot = asyncio.Semaphore(max(1, settings.lane_cpu))
        self.jobs: dict[str, Job] = {}
        self.queues: dict[str, asyncio.Queue[str]] = {}
        self._workers: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._last_notify: dict[str, float] = {}
        self._stopping = False
        self._history = settings.data_dir / "jobs.json"
        self._hooks: list[Callable[[Job], None]] = []

    # ---- lifecycle -----------------------------------------------------
    def lane_sizes(self) -> dict[str, int]:
        s = self.settings
        return {"cpu": max(1, s.lane_cpu), "net": max(1, s.lane_net), "light": max(1, s.lane_light)}

    async def start(self) -> None:
        self._load()
        for lane, size in self.lane_sizes().items():
            self.queues[lane] = asyncio.Queue()
            for n in range(size):
                self._workers.append(asyncio.create_task(self._worker(lane), name=f"worker-{lane}-{n}"))
        self._workers.append(asyncio.create_task(self._sweeper(), name="sweeper"))

    async def stop(self) -> None:
        self._stopping = True
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.persist()

    def _load(self) -> None:
        try:
            data = json.loads(self._history.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in data:
            try:
                job = Job.restore(item)
            except (KeyError, TypeError):
                continue
            self.jobs[job.id] = job

    def persist(self) -> None:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created)[-HISTORY_LIMIT:]
        tmp = self._history.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps([j.snapshot() for j in jobs], ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._history)
        except OSError as exc:  # disk full must not kill the worker
            log.warning("could not persist jobs: %s", exc)

    # ---- public API ----------------------------------------------------
    def submit(self, tool_id: str, params: dict[str, Any], files: dict[str, list[str]], *, title: str = "",
               batch_id: str = "", meta: dict[str, Any] | None = None) -> Job:
        tool = self.tools.get(tool_id)
        if tool is None:
            raise ToolError("tool.unknown", tool=tool_id)
        job = Job(tool=tool_id, lane=tool.lane, params=params, files=files, title=title or tool_id,
                  batch_id=batch_id, meta=dict(meta or {}))
        self.jobs[job.id] = job
        self.queues[tool.lane].put_nowait(job.id)
        self.notify(job, force=True)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError("job.missing")
        return job

    def list(self, limit: int = 100) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state in ACTIVE:
            job.cancel_requested = True
            task = self._running.get(job_id)
            if task:
                task.cancel()
            else:
                self._finish(job, "cancelled")
        return job

    def retry(self, job_id: str) -> Job:
        old = self.get(job_id)
        if old.state in ACTIVE:
            raise ToolError("job.active")
        missing = [uid for ids in old.files.values() for uid in ids if not self.uploads.exists(uid)]
        if missing:
            raise ToolError("job.inputs_gone")
        resume = {k: v for k, v in old.meta.items() if k.startswith("resume_")}
        return self.submit(old.tool, old.params, old.files, title=old.title, batch_id=old.batch_id, meta=resume)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in ACTIVE:
            raise ToolError("job.active")
        shutil.rmtree(self.settings.outputs / job_id, ignore_errors=True)
        shutil.rmtree(self.settings.work / job_id, ignore_errors=True)
        del self.jobs[job_id]
        self.persist()
        self._broadcast({"type": "deleted", "id": job_id})

    def output_path(self, job_id: str, name: str) -> Path:
        job = self.get(job_id)
        if name not in {o["name"] for o in job.outputs}:
            raise ToolError("file.missing")
        path = self.settings.outputs / job_id / name
        if not path.is_file():
            raise ToolError("file.missing")
        return path

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job.state] = counts.get(job.state, 0) + 1
        return {"counts": counts, "lanes": {k: {"workers": v, "waiting": self.queues[k].qsize() if k in self.queues else 0}
                                            for k, v in self.lane_sizes().items()}}

    def on_finish(self, hook: Callable[[Job], None]) -> None:
        self._hooks.append(hook)

    # ---- events (Server-Sent Events) -----------------------------------
    def open_subscription(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def close_subscription(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # slow client: drop it, it will resync on reconnect
                self._subscribers.discard(queue)

    def notify(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_notify.get(job.id, 0) < 0.5:
            return
        self._last_notify[job.id] = now
        self._broadcast({"type": "job", "job": job.public()})

    # ---- execution -----------------------------------------------------
    async def _worker(self, lane: str) -> None:
        queue = self.queues[lane]
        while True:
            job_id = await queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.state != "queued":
                continue
            task = asyncio.create_task(self._execute(job))
            self._running[job_id] = task
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:  # server shutdown
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            finally:
                self._running.pop(job_id, None)

    async def _execute(self, job: Job) -> None:
        tool = self.tools[job.tool]
        job.state, job.started, job.progress = "running", time.time(), 0.0
        job.error = None
        self.notify(job, force=True)
        ctx = JobContext(self, job)
        try:
            if job.lane == "cpu":
                async with self.cpu_slot:
                    await tool.run(ctx, job.params)
            else:
                await tool.run(ctx, job.params)
        except asyncio.CancelledError:
            if self._stopping and not job.cancel_requested:
                self._finish(job, "interrupted", {"code": "job.interrupted", "params": {}, "detail": ""})
            else:
                self._finish(job, "cancelled")
            return
        except ToolError as exc:
            ctx.log(f"error {exc.code} {exc.detail[-400:]}")
            self._finish(job, "failed", exc.public())
            return
        except Exception as exc:  # noqa: BLE001 - report anything to the user
            ctx.log(traceback.format_exc()[-1500:])
            self._finish(job, "failed", {"code": "internal", "params": {}, "detail": f"{type(exc).__name__}: {exc}"[:500]})
            return
        finally:
            shutil.rmtree(ctx.workdir, ignore_errors=True)
        job.progress = 100.0
        self._finish(job, "done")

    def _finish(self, job: Job, state: str, error: dict[str, Any] | None = None) -> None:
        job.state, job.finished = state, time.time()
        if error is not None:
            job.error = error
        if state == "done" and self.settings.runtime.delete_inputs_on_success:
            active = {uid for j in self.jobs.values() if j.state in ACTIVE for ids in j.files.values() for uid in ids}
            pinned = {uid for j in self.jobs.values() for uid in j.meta.get("_keep_uploads", [])}
            for ids in job.files.values():
                for uid in ids:
                    if uid not in active and uid not in pinned:
                        self.uploads.delete(uid)
        for hook in self._hooks:
            try:
                hook(job)
            except Exception:  # noqa: BLE001
                log.exception("finish hook failed")
        self.persist()
        self.notify(job, force=True)

    async def _sweeper(self) -> None:
        """Hourly: expire old uploads and outputs according to retention settings."""
        while True:
            try:
                rt = self.settings.runtime
                keep = {uid for j in self.jobs.values() if j.state in ACTIVE for ids in j.files.values() for uid in ids}
                self.uploads.sweep(rt.upload_ttl_hours, keep)
                cutoff = time.time() - rt.retention_days * 86400
                for job in list(self.jobs.values()):
                    if job.state in FINAL and (job.finished or job.created) < cutoff:
                        self.delete(job.id)
                for folder in self.settings.work.iterdir():
                    if folder.name not in self._running:
                        shutil.rmtree(folder, ignore_errors=True)
                for path in self.settings.tmp.iterdir():
                    if path.stat().st_mtime < time.time() - 6 * 3600:
                        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                log.exception("sweeper failed")
            await asyncio.sleep(3600)
