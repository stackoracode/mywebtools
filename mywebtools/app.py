"""FastAPI application: JSON API + the static single-page UI."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, duix, nodes, ocr
from .config import SECRET_KEYS, Runtime, settings, write_secrets
from .errors import ToolError
from .i18n import TESSERACT_TO_BCP47, UI_LANGUAGES
from .jobs import JobManager
from .llm import EFFORTS, LLM
from .media import ffmpeg_caps, probe, thumbnail
from .storage import LIBRARIES, Library, Prefs, UploadStore, new_id
from .tools import REGISTRY

STATIC = Path(__file__).parent / "static"
COOKIE = "mwt_session"
SESSION_DAYS = 30


class Services:
    """Shared singletons (also handed to tool actions)."""

    def __init__(self) -> None:
        settings.ensure_dirs()
        settings.load_runtime()
        self.uploads = UploadStore(settings.uploads)
        self.library = Library(settings.library)
        self.prefs = Prefs(settings.data_dir / "prefs.json")
        self.jobs = JobManager(settings, REGISTRY, self.uploads, self.library)


def create_app(services: Services | None = None) -> FastAPI:
    svc = services or Services()
    # multipart/tempfile spill-over must not land on a small tmpfs /tmp
    os.environ.setdefault("TMPDIR", str(settings.tmp))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await svc.jobs.start()
        yield
        await svc.jobs.stop()

    app = FastAPI(title="mywebtools", version=__version__, lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.state.svc = svc

    # ---- errors & auth -------------------------------------------------
    @app.exception_handler(ToolError)
    async def tool_error(_: Request, exc: ToolError):
        status = 404 if exc.code in ("job.missing", "file.missing", "library.missing", "upload.missing") else 400
        return JSONResponse({"error": exc.public()}, status_code=status)

    def _sign(expiry: int) -> str:
        msg = f"{expiry}".encode()
        return hmac.new(settings.session_key() + settings.secret("APP_PASSWORD").encode(), msg, hashlib.sha256).hexdigest()

    def _authed(request: Request) -> bool:
        if not settings.secret("APP_PASSWORD"):
            return True
        token = request.cookies.get(COOKIE, "")
        expiry, _, sig = token.partition(".")
        return expiry.isdigit() and int(expiry) > time.time() and hmac.compare_digest(sig, _sign(int(expiry)))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in ("/api/auth", "/api/login") and not _authed(request):
            return JSONResponse({"error": {"code": "auth.required", "params": {}, "detail": ""}}, status_code=401)
        return await call_next(request)

    @app.get("/api/auth")
    async def auth_state(request: Request):
        return {"required": bool(settings.secret("APP_PASSWORD")), "ok": _authed(request)}

    @app.post("/api/login")
    async def login(request: Request):
        body = await _json(request)
        expected = settings.secret("APP_PASSWORD")
        if expected and not hmac.compare_digest(str(body.get("password", "")).encode(), expected.encode()):
            await asyncio.sleep(1)
            raise ToolError("auth.wrong_password")
        expiry = int(time.time()) + SESSION_DAYS * 86400
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, f"{expiry}.{_sign(expiry)}", max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax")
        return resp

    @app.post("/api/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    # ---- meta & tools --------------------------------------------------
    @app.get("/api/meta")
    async def meta():
        return {
            "version": __version__,
            "ui_languages": UI_LANGUAGES,
            "default_language": settings.runtime.default_language,
            "tesseract_bcp47": TESSERACT_TO_BCP47,
            "libraries": LIBRARIES,
            "engine": settings.runtime.avatar_engine,
            "os": os.name,
            "llm": {"configured": LLM().configured, "model": settings.runtime.llm_model, "efforts": EFFORTS},
            "nodes": list(nodes.KINDS),
        }

    @app.get("/api/tools")
    async def tools():
        return [tool.schema() for tool in REGISTRY.values()]

    @app.post("/api/tools/{tool_id}/actions/{action}")
    async def tool_action(tool_id: str, action: str, request: Request):
        tool = _tool(tool_id)
        if action not in tool.actions:
            raise ToolError("tool.no_action", action=action)
        body = await _json(request)
        params = tool.coerce(body.get("params") or {})
        return await tool.action(action, params, _files(body.get("files")), svc)

    # ---- uploads -------------------------------------------------------
    @app.put("/api/uploads")
    async def upload(request: Request, name: str = "file"):
        max_bytes = settings.runtime.max_upload_mb * 1024 * 1024
        declared = int(request.headers.get("content-length") or 0)
        if declared > max_bytes:
            raise ToolError("upload.too_large", max_mb=settings.runtime.max_upload_mb)
        item = await svc.uploads.save(name, request.stream(), max_bytes)
        return item.public()

    @app.get("/api/uploads/{upload_id}")
    async def upload_info(upload_id: str):
        return svc.uploads.get(upload_id).public()

    @app.post("/api/uploads/{upload_id}/probe")
    async def upload_probe(upload_id: str):
        return (await probe(svc.uploads.get(upload_id).path)).public()

    # ---- jobs ----------------------------------------------------------
    @app.post("/api/jobs")
    async def submit(request: Request):
        body = await _json(request)
        tool = _tool(str(body.get("tool", "")))
        ok, reason = tool.availability()
        if not ok:
            raise ToolError("tool.unavailable", reason=reason)
        raw_params = body.get("params") or {}
        params = tool.coerce(raw_params)
        files = _files(body.get("files"))
        file_params = [p for p in tool.params() if p.type == "file"]
        for p in file_params:
            visible = _visible(p.show_if, params)
            if p.required and visible and not files.get(p.name):
                raise ToolError("input.missing", field=p.name)
            for uid in files.get(p.name, []):
                svc.uploads.get(uid)  # validates existence
        svc.prefs.set_last(tool.id, {p.name: params[p.name] for p in tool.params() if p.remember and p.name in params})

        batch = next((p for p in file_params if p.batch and len(files.get(p.name, [])) > 1), None)
        batch_id = new_id() if batch else ""
        runs = [{**files, batch.name: [uid]} for uid in files[batch.name]] if batch else [files]
        jobs = []
        for run_files in runs:
            names = [svc.uploads.get(uid).name for ids in run_files.values() for uid in ids]
            title = tool.title_for(params, names)
            jobs.append(svc.jobs.submit(tool.id, params, run_files, title=title, batch_id=batch_id).public())
        return {"jobs": jobs}

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 100):
        return {"jobs": [j.public() for j in svc.jobs.list(min(limit, 500))], "stats": svc.jobs.stats()}

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str):
        return svc.jobs.get(job_id).public(with_logs=True)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str):
        return svc.jobs.cancel(job_id).public()

    @app.post("/api/jobs/{job_id}/retry")
    async def retry(job_id: str):
        return svc.jobs.retry(job_id).public()

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str):
        svc.jobs.delete(job_id)
        return {"ok": True}

    @app.post("/api/jobs/clear")
    async def clear_jobs():
        removed = 0
        for job in list(svc.jobs.jobs.values()):
            if job.state not in ("queued", "running", "review"):
                svc.jobs.delete(job.id)
                removed += 1
        return {"removed": removed}

    @app.get("/api/jobs/{job_id}/review")
    async def review_draft(job_id: str):
        job = svc.jobs.get(job_id)
        return {"state": job.state, "kind": job.meta.get("review_kind", ""), "draft": job.meta.get("review_draft") or {}}

    @app.post("/api/jobs/{job_id}/review")
    async def review_submit(job_id: str, request: Request):
        """Approve the (edited) subtitle draft; the paused job continues."""
        from .subtitles import cues_from

        body = await _json(request)
        cues = cues_from(body.get("cues") or [])
        if not cues:
            raise ToolError("subs.empty")
        return svc.jobs.submit_review(job_id, {"cues": [c.public() for c in cues]}).public()

    @app.post("/api/jobs/{job_id}/review/revise")
    async def review_revise(job_id: str, request: Request):
        """Ask the AI to rework the current draft (fix / translate / follow an instruction)."""
        from .subtitles import ai_edit, cues_from

        svc.jobs.get(job_id)
        body = await _json(request)
        cues = cues_from(body.get("cues") or [])
        mode = str(body.get("mode") or "revise")
        if mode not in ("fix", "translate", "bilingual", "revise"):
            raise ToolError("tool.no_action", action=mode)
        edited = await ai_edit(cues, mode, target=str(body.get("target") or "en"),
                               instruction=str(body.get("instruction") or ""), effort=body.get("effort") or None)
        return {"cues": [c.public() for c in edited]}

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def job_file(job_id: str, name: str, inline: int = 0):
        path = svc.jobs.output_path(job_id, name)
        return FileResponse(path, filename=path.name, content_disposition_type="inline" if inline else "attachment")

    @app.get("/api/events")
    async def events(request: Request):
        queue = svc.jobs.open_subscription()

        async def stream():
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), 15)
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                svc.jobs.close_subscription(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/previews/{name}")
    async def preview_file(name: str):
        if not re.fullmatch(r"preview_[a-f0-9]{32}\.jpg", name) or not (settings.tmp / name).is_file():
            raise ToolError("file.missing")
        return FileResponse(settings.tmp / name, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- prefs (form memory) -------------------------------------------
    @app.get("/api/prefs/{tool_id}")
    async def prefs(tool_id: str):
        _tool(tool_id)
        return {"last": svc.prefs.last(tool_id), "presets": svc.prefs.presets(tool_id)}

    @app.put("/api/prefs/{tool_id}/presets/{name}")
    async def save_preset(tool_id: str, name: str, request: Request):
        tool = _tool(tool_id)
        body = await _json(request)
        values = tool.coerce(body.get("values") or {})
        svc.prefs.save_preset(tool_id, name, {p.name: values[p.name] for p in tool.params() if p.remember and p.name in values})
        return {"presets": svc.prefs.presets(tool_id)}

    @app.delete("/api/prefs/{tool_id}/presets/{name}")
    async def delete_preset(tool_id: str, name: str):
        svc.prefs.delete_preset(tool_id, name)
        return {"presets": svc.prefs.presets(tool_id)}

    # ---- libraries -----------------------------------------------------
    @app.get("/api/library/{lib}")
    async def library_list(lib: str):
        return {"items": [i.public() for i in svc.library.list(lib)]}

    @app.post("/api/library/{lib}")
    async def library_add(lib: str, request: Request):
        body = await _json(request)
        upload = svc.uploads.get(str(body.get("upload_id", "")))
        info = await probe(upload.path)
        if LIBRARIES[lib]["kind"] == "video" and not info.has_video:
            raise ToolError("media.no_video")
        if LIBRARIES[lib]["kind"] == "audio" and not info.has_audio:
            raise ToolError("media.no_audio")
        item = svc.library.add(lib, upload.path, str(body.get("name") or Path(upload.name).stem), move=True,
                               info={k: v for k, v in info.public().items()
                                     if k in ("duration", "width", "height", "fps", "orientation", "is_hdr", "vcodec")})
        svc.uploads.delete(upload.id)
        if await thumbnail(svc.library.path(item), svc.library.thumb_path(item), at=min(1.0, info.duration / 2)):
            svc.library.update(item, has_thumb=True)
        return item.public()

    @app.patch("/api/library/{lib}/{item_id}")
    async def library_rename(lib: str, item_id: str, request: Request):
        body = await _json(request)
        item = svc.library.get(lib, item_id)
        return svc.library.update(item, name=str(body.get("name", item.name)).strip()[:80] or item.name).public()

    @app.delete("/api/library/{lib}/{item_id}")
    async def library_delete(lib: str, item_id: str):
        svc.library.delete(lib, item_id)
        return {"ok": True}

    @app.get("/api/library/{lib}/{item_id}/file")
    async def library_file(lib: str, item_id: str):
        item = svc.library.get(lib, item_id)
        return FileResponse(svc.library.path(item), filename=item.file, content_disposition_type="inline")

    @app.get("/api/library/{lib}/{item_id}/thumb")
    async def library_thumb(lib: str, item_id: str):
        item = svc.library.get(lib, item_id)
        path = svc.library.thumb_path(item)
        if not path.is_file():
            raise ToolError("file.missing")
        return FileResponse(path, media_type="image/jpeg")

    # ---- settings & admin ----------------------------------------------
    @app.get("/api/settings")
    async def get_settings():
        secrets = {}
        for key in SECRET_KEYS:
            value = settings.secret(key)
            secrets[key] = {"configured": bool(value), "hint": f"…{value[-4:]}" if len(value) > 8 else ""}
        return {"values": asdict(settings.runtime), "fields": Runtime.editable(), "secrets": secrets}

    @app.put("/api/settings")
    async def put_settings(request: Request):
        body = await _json(request)
        values = {k: v for k, v in (body.get("values") or {}).items() if k in Runtime.__dataclass_fields__}
        ffmpeg_caps.cache_clear()
        return {"values": asdict(settings.save_runtime(values))}

    @app.put("/api/secrets")
    async def put_secrets(request: Request):
        body = await _json(request)
        values = {k: str(v).strip() for k, v in body.items() if k in SECRET_KEYS and v is not None}
        try:
            write_secrets(values)
        except ValueError as exc:
            raise ToolError("settings.bad_secret", str(exc)) from exc
        resp = JSONResponse({"ok": True})
        if "APP_PASSWORD" in values and values["APP_PASSWORD"]:  # keep the current browser logged in
            expiry = int(time.time()) + SESSION_DAYS * 86400
            resp.set_cookie(COOKIE, f"{expiry}.{_sign(expiry)}", max_age=SESSION_DAYS * 86400, httponly=True,
                            samesite="lax")
        return resp

    @app.get("/api/duix/health")
    async def duix_health():
        """Is the Duix engine usable right now? Drives the 'start Docker' banner."""
        rt = settings.runtime
        if rt.avatar_engine == "duix_cloud":
            key = settings.secret("DUIX_API_KEY")
            return {"engine": "duix_cloud", "ok": bool(key), "reachable": bool(key), "wol": False}
        health = await duix.DuixLocal().health()
        health["wol"] = bool(rt.duix_wol_mac)
        return health

    @app.post("/api/duix/wake")
    async def duix_wake():
        if not settings.runtime.duix_wol_mac:
            raise ToolError("duix.bad_mac")
        duix.wake_on_lan(settings.runtime.duix_wol_mac)
        return {"ok": True}

    @app.post("/api/duix/verify")
    async def duix_verify(request: Request):
        """Test the engine with the given (unsaved) values: local URL/share or a cloud key."""
        body = await _json(request)
        rt = settings.runtime
        engine_kind = str(body.get("engine") or rt.avatar_engine)
        if engine_kind == "duix_local":
            share = body.get("share_path") or rt.duix_share()
            return await duix.DuixLocal(base_url=str(body.get("url") or rt.duix_local_url), share=Path(share)).health()
        key = str(body.get("key") or "").strip()
        try:
            return await duix.DuixCloud(key or settings.secret("DUIX_API_KEY")).account()
        except ToolError as exc:
            return {"valid": False, "ok": False, "error": exc.public()}

    @app.post("/api/llm/models")
    async def llm_models(request: Request):
        """Load the model list from the configured (or the given, unsaved) API."""
        body = await _json(request)
        client = LLM(base_url=body.get("base_url") or None, api_key=body.get("api_key") or None)
        return await client.models()

    @app.post("/api/llm/test")
    async def llm_test(request: Request):
        body = await _json(request)
        client = LLM(base_url=body.get("base_url") or None, api_key=body.get("api_key") or None,
                     provider=body.get("provider") or None, model=body.get("model") or None,
                     effort=body.get("effort") if body.get("effort") is not None else None)
        try:
            return await client.test()
        except ToolError as exc:
            return {"ok": False, "error": exc.public()}

    # ---- model services on the GPU PC -----------------------------------
    @app.get("/api/nodes")
    async def nodes_health():
        kinds = list(nodes.KINDS)
        results = await asyncio.gather(*(nodes.Node(k).health() for k in kinds))
        return dict(zip(kinds, results, strict=True))

    @app.post("/api/nodes/{kind}/check")
    async def node_check(kind: str, request: Request):
        """Health of one service with the given (unsaved) URL / model folder."""
        body = await _json(request)
        return await nodes.Node(kind, url=body.get("url") or None, model_dir=body.get("model_dir") or None).health()

    @app.post("/api/nodes/{kind}/{action}")
    async def node_action(kind: str, action: str):
        if action not in ("load", "unload"):
            raise ToolError("tool.no_action", action=action)
        node = nodes.Node(kind)
        return await (node.load() if action == "load" else node.unload())

    @app.get("/api/system")
    async def system():
        caps = ffmpeg_caps()
        disk = shutil.disk_usage(settings.data_dir)
        return {
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "cpu_count": os.cpu_count(),
            "load": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "memory": _meminfo(),
            "temperature_c": _temperature(),
            "disk": {"total": disk.total, "free": disk.free, "used": disk.used},
            "data": {"uploads": svc.uploads.usage(), "outputs": _du(settings.outputs), "library": svc.library.usage(),
                     "models": _du(settings.models)},
            "ffmpeg": {"available": caps["available"], "version": caps["version"],
                       "zscale": "zscale" in caps["filters"],
                       "hw_encoders": sorted(e for e in caps["encoders"] if e.startswith("h264_") or e.startswith("hevc_"))},
            "tesseract": {"available": ocr.available(), "languages": ocr.languages()},
            "llm": {"configured": LLM().configured, "base_url": settings.runtime.llm_base_url,
                    "model": settings.runtime.llm_model},
            "tools": {t.id: dict(zip(("available", "reason"), t.availability(), strict=True)) for t in REGISTRY.values()},
            "jobs": svc.jobs.stats(),
        }

    @app.post("/api/system/cleanup")
    async def cleanup(request: Request):
        body = await _json(request)
        removed = 0
        if body.get("uploads"):
            removed += svc.uploads.sweep(0, keep={uid for j in svc.jobs.jobs.values() if j.state in ("queued", "running", "review")
                                                  for ids in j.files.values() for uid in ids})
        if body.get("tmp"):
            for path in settings.tmp.iterdir():
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        return {"removed": removed}

    # ---- static UI -----------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/manifest.webmanifest")
    async def manifest():
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    return app


def _tool(tool_id: str):
    tool = REGISTRY.get(tool_id)
    if tool is None:
        raise ToolError("tool.unknown", tool=tool_id)
    return tool


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _files(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): [str(v) for v in (vals if isinstance(vals, list) else [vals]) if v] for k, vals in raw.items()}


def _visible(show_if: dict[str, Any] | None, params: dict[str, Any]) -> bool:
    if not show_if:
        return True
    for key, expected in show_if.items():
        allowed = expected if isinstance(expected, list) else [expected]
        if params.get(key) not in allowed:
            return False
    return True


def _du(path: Path) -> int:
    from .storage import dir_size

    return dir_size(path)


def _meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return out


def _temperature() -> float | None:
    temps = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(zone.read_text().strip()) / 1000)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


app = create_app() if os.getenv("MWT_NO_APP") != "1" else None  # uvicorn entry: mywebtools.app:app
