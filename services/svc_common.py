"""Shared plumbing for the mywebtools model services.

Each model runs as its own small web service, in its own Python environment
(their dependencies conflict), on the PC with the graphics card:

    whisper   :8391  faster-whisper (CTranslate2)   speech -> timed text
    indextts  :8392  IndexTTS 2.5                    text + voice sample -> speech
    aligner   :8393  MMS-300m CTC forced aligner     speech + known text -> timings

Every service has ``GET /health`` (model folder + file check, GPU, loaded state),
``POST /load``, ``POST /unload`` and its own work endpoint, and can also be run
once from the command line with ``--test`` (no web server).

This module needs only fastapi, uvicorn and python-multipart. It never imports
torch itself; it only uses it when the service already has.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hmac
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

PROTOCOL = 1


class ServiceError(Exception):
    def __init__(self, code: str, detail: str = "", status: int = 400):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.status = code, detail, status


def error_payload(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, ServiceError):
        return {"code": exc.code, "detail": exc.detail}
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower():
        return {"code": "out_of_memory", "detail": text[:500]}
    return {"code": "internal", "detail": text[:500]}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def default_models_root() -> str:
    return os.environ.get("MWT_MODELS_ROOT") or str(Path.home() / "Models")


def base_parser(name: str, port: int, model_folder: str, description: str) -> argparse.ArgumentParser:
    env = name.upper()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default=os.environ.get("MWT_SERVICE_HOST", "0.0.0.0"),
                        help="0.0.0.0 = reachable from the LAN (the board), 127.0.0.1 = this PC only")
    parser.add_argument("--port", type=int, default=int(os.environ.get(f"MWT_{env}_PORT", port)))
    parser.add_argument("--model-dir", default=os.environ.get(f"MWT_{env}_MODEL") or
                        str(Path(default_models_root()) / model_folder), help="default model folder")
    parser.add_argument("--models-root", default=default_models_root(),
                        help="model folders sent by mywebtools must be inside this folder ('*' = anywhere)")
    parser.add_argument("--device", default=os.environ.get(f"MWT_{env}_DEVICE", "auto"), help="auto | cuda | cpu")
    parser.add_argument("--idle-unload", type=int, default=int(os.environ.get("MWT_IDLE_UNLOAD", 900)),
                        help="free the model (and GPU memory) after this many idle seconds; 0 = never")
    parser.add_argument("--token", default=os.environ.get("MWT_SERVICE_TOKEN", ""),
                        help="require 'Authorization: Bearer <token>' (set NODE_TOKEN to the same value in mywebtools)")
    parser.add_argument("--test", nargs="+", metavar="ARG", help="run once without the web server and exit")
    return parser


# ---------------------------------------------------------------------------
# model folders
# ---------------------------------------------------------------------------
class Folders:
    def __init__(self, default: str, root: str):
        self.default = default
        self.root = root

    def resolve(self, value: str | None) -> Path:
        """Model folder from a request (or the default); must be inside the models root."""
        path = Path(value or self.default).expanduser()
        if self.root != "*":
            root = os.path.normcase(os.path.abspath(Path(self.root).expanduser()))
            full = os.path.normcase(os.path.abspath(path))
            if full != root and not full.startswith(root.rstrip("\\/") + os.sep):
                raise ServiceError("model_outside_root", f"{path} is not inside {self.root}", 403)
        return path

    def require(self, value: str | None, files: list[Any]) -> Path:
        path = self.resolve(value)
        if not path.is_dir():
            raise ServiceError("model_missing", str(path), 404)
        missing = [f["name"] for f in check_files(path, files) if not f["ok"]]
        if missing:
            raise ServiceError("files_missing", f"{path}: {', '.join(missing)}", 404)
        return path


def check_files(folder: Path, files: list[Any]) -> list[dict[str, Any]]:
    """files: names relative to the folder; a tuple means "any one of these"."""
    out = []
    for entry in files:
        names = entry if isinstance(entry, tuple) else (entry,)
        found = next((folder / n for n in names if (folder / n).is_file() and (folder / n).stat().st_size > 0), None)
        out.append({"name": " | ".join(names), "ok": found is not None,
                    "size": found.stat().st_size if found else 0})
    return out


def folder_report(folders: Folders, value: str | None, files: list[Any],
                  optional: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    try:
        path = folders.resolve(value)
    except ServiceError as exc:
        return {"dir": value or folders.default, "exists": False, "ok": False, "files": [], "error": exc.code}
    exists = path.is_dir()
    report: dict[str, Any] = {"dir": str(path), "exists": exists, "files": check_files(path, files) if exists else []}
    report["ok"] = exists and all(f["ok"] for f in report["files"])
    for feature, names in (optional or {}).items():
        checked = check_files(path, names) if exists else []
        report.setdefault("optional", {})[feature] = {"ok": bool(checked) and all(f["ok"] for f in checked),
                                                       "files": checked}
    return report


# ---------------------------------------------------------------------------
# GPU / CUDA helpers
# ---------------------------------------------------------------------------
def add_cuda_dll_dirs() -> None:
    """Windows: make the CUDA DLLs from pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable."""
    if os.name != "nt":
        return
    for base in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for bin_dir in (base / "nvidia").glob("*/bin"):
            with contextlib.suppress(OSError):
                os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_gpu_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])


def gpu_info() -> list[dict[str, Any]]:
    """Graphics cards with free/total memory (MB); empty without NVIDIA tools."""
    global _gpu_cache
    if time.time() - _gpu_cache[0] < 5:
        return _gpu_cache[1]
    gpus: list[dict[str, Any]] = []
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    gpus.append({"name": torch.cuda.get_device_name(i), "free_mb": free // 2**20, "total_mb": total // 2**20})
    if not gpus and shutil.which("nvidia-smi"):
        with contextlib.suppress(Exception):
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.strip().splitlines():
                name, free, total = (x.strip() for x in line.rsplit(",", 2))
                gpus.append({"name": name, "free_mb": int(free), "total_mb": int(total)})
    _gpu_cache = (time.time(), gpus)
    return gpus


def free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# the loaded model
# ---------------------------------------------------------------------------
class ModelSlot:
    """Holds one loaded model. One job at a time; reloads when the key (folder, device, ...)
    changes; unloads after ``idle_seconds`` without use to give the GPU back (e.g. to Duix)."""

    def __init__(self, loader: Callable[[tuple], Any], idle_seconds: int = 0):
        self._loader = loader
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self.model: Any = None
        self.key: tuple | None = None
        self.busy = False
        self.last_used = 0.0
        self.load_seconds: float | None = None
        if idle_seconds > 0:
            threading.Thread(target=self._watch, daemon=True, name="idle-unload").start()

    @contextlib.contextmanager
    def use(self, key: tuple) -> Iterator[Any]:
        with self._lock:
            if self.model is None or self.key != key:
                self._drop()
                started = time.monotonic()
                self.model = self._loader(key)
                self.key = key
                self.load_seconds = round(time.monotonic() - started, 1)
            self.busy = True
            try:
                yield self.model
            finally:
                self.busy = False
                self.last_used = time.time()

    def unload(self) -> bool:
        with self._lock:
            had = self.model is not None
            self._drop()
            return had

    def _drop(self) -> None:
        if self.model is not None:
            self.model = None
            self.key = None
            free_memory()

    def _watch(self) -> None:
        while True:
            time.sleep(min(30, max(1, self.idle_seconds)))
            if self.model is not None and not self.busy and time.time() - self.last_used > self.idle_seconds:
                if self._lock.acquire(blocking=False):
                    try:
                        if not self.busy:
                            self._drop()
                    finally:
                        self._lock.release()

    def status(self) -> dict[str, Any]:
        return {"loaded": self.model is not None, "busy": self.busy, "key": [str(k) for k in self.key or ()],
                "load_seconds": self.load_seconds, "idle_unload_seconds": self.idle_seconds}


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def parse_options(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ServiceError("bad_request", f"options is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ServiceError("bad_request", "options must be a JSON object")
    return value


def work_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"mwt-{name}-"))


async def save_upload(upload: Any, dest: Path) -> Path:
    with dest.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            out.write(chunk)
    if dest.stat().st_size == 0:
        raise ServiceError("bad_request", f"empty file {upload.filename}")
    return dest


def wav_seconds(path: Path) -> float:
    import wave

    try:
        with wave.open(str(path), "rb") as wav:
            return round(wav.getnframes() / float(wav.getframerate() or 1), 3)
    except (wave.Error, OSError, EOFError):
        return 0.0


def make_app(name: str, version: str, args: argparse.Namespace, slot: ModelSlot,
             health: Callable[[str | None], dict[str, Any]], preload: Callable[[dict[str, Any]], None]):
    """The common part of every service: auth, errors, /health, /load, /unload."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title=f"mywebtools {name} service", version=version)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if args.token:
            sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(sent.encode(), args.token.encode()):
                return JSONResponse({"error": {"code": "unauthorized", "detail": "wrong or missing token"}}, status_code=401)
        return await call_next(request)

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError):
        return JSONResponse({"error": error_payload(exc)}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def crash(_: Request, exc: Exception):
        return JSONResponse({"error": error_payload(exc)}, status_code=500)

    @app.get("/health")
    def get_health(model_dir: str | None = None):
        body = {"service": name, "version": version, "protocol": PROTOCOL, "python": sys.version.split()[0],
                "platform": f"{platform.system()} {platform.release()}", "host": platform.node(),
                "gpu": gpu_info(), "slot": slot.status(), "token": bool(args.token)}
        body.update(health(model_dir))
        body["ok"] = bool(body.get("model", {}).get("ok")) and not body.get("import_error")
        return body

    @app.post("/load")
    async def load(request: Request):
        import asyncio

        try:
            options = await request.json()
        except ValueError:
            options = {}
        started = time.monotonic()
        await asyncio.to_thread(preload, options if isinstance(options, dict) else {})
        return {"ok": True, "seconds": round(time.monotonic() - started, 1), "slot": slot.status()}

    @app.post("/unload")
    def unload():
        return {"ok": True, "unloaded": slot.unload()}

    return app


def serve(app: Any, args: argparse.Namespace, name: str) -> None:
    import uvicorn

    print(f"mywebtools {name} service on http://{args.host}:{args.port}  model: {args.model_dir}")
    print(f"  models root: {args.models_root}   device: {args.device}   idle unload: {args.idle_unload}s")
    if not args.token and args.host not in ("127.0.0.1", "localhost"):
        print("  note: no --token set; anyone on the LAN can use this service")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
