"""Digital-avatar (lip-sync) engines.

``DuixLocal`` (default) drives the self-hosted Duix.Avatar Docker container on a
Windows PC in the LAN; see its docstring for the protocol.

``DuixCloud`` (optional) talks to the hosted DUIX API with an API key — the same HTTP
endpoints the official ``duix-cli`` (npm, v0.7.x) uses, re-implemented with
httpx so no Node.js is needed on the board:

    Key: <DUIX_API_KEY>                         (request header)
    GET  /skills-api/skill/credit/queryAccount  plan (packageItem JSON string)
    GET  /skills-api/skill/credit/queryCreditsLeft
    POST /skills-api/skill/file/upload-url      -> presigned PUT url
    POST /skills-api/skill/file/download-url    -> readable url for the upload
    POST /skills-api/skill/video/synthesis      {videoUrl, audioUrl, needTranscode}
    GET  /skills-api/skill/task/status?taskId=  status 0..4, progress, outputUrl
    POST /skills-api/skill/file/deleteSynthResult {taskId}

Rules mirrored from ``duix-cli compose check``: video mp4/mov/webm, 9:16 or
16:9 (±2 %), short side 720..2160 px, <=1 GB; audio mp3/wav/m4a/aac/flac;
audio length <= plan ``durationMinutes``; cost = ceil(seconds / 5) credits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .errors import ToolError

SECONDS_PER_CREDIT = 5
VIDEO_EXTS = {".mp4", ".mov", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
MAX_BYTES = 1024 ** 3
STATUS_NAMES = {0: "pending", 1: "queued", 2: "running", 3: "succeeded", 4: "failed"}

Progress = Callable[[float, str, dict[str, Any]], None]


def credits_for(seconds: float) -> int:
    return max(1, math.ceil(seconds / SECONDS_PER_CREDIT))


def ratio_ok(width: int, height: int) -> bool:
    if not width or not height:
        return False
    ratio = width / height
    return any(abs(ratio - target) < target * 0.02 for target in (9 / 16, 16 / 9))


def resolution_ok(width: int, height: int) -> bool:
    short, long = min(width, height), max(width, height)
    return 720 <= short <= 2160 and long <= 3840


def _status_from_desc(desc: str) -> str | None:
    text = (desc or "").replace(" ", "")
    for needle, status in (("成功", "succeeded"), ("失败", "failed"), ("排队", "queued"), ("处理中", "running"),
                           ("合成中", "running"), ("待处理", "pending")):
        if needle in text:
            return status
    return None


def normalize_status(data: dict[str, Any]) -> dict[str, Any]:
    status = _status_from_desc(str(data.get("statusDesc") or "")) or STATUS_NAMES.get(data.get("status"), "pending")
    try:
        progress = max(0, min(100, int(float(data.get("progress")))))
    except (TypeError, ValueError):
        progress = 100 if status == "succeeded" else 0
    return {"status": status, "progress": progress, "output_url": data.get("outputUrl") or "",
            "error": data.get("errorMsg") or "", "desc": data.get("statusDesc") or ""}


def remote_error(msg: str) -> ToolError:
    """Map the API's (Chinese) messages to translatable codes, keep the original as detail."""
    text = msg or ""
    if "认证失败" in text or "auth" in text.lower():
        return ToolError("duix.auth_failed", text)
    if "积分" in text and ("不足" in text or "余额" in text):
        return ToolError("duix.no_credits", text)
    return ToolError("duix.remote_error", text)


class DuixCloud:
    name = "duix_cloud"

    def __init__(self, api_key: str, base_url: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise ToolError("duix.no_key")
        self.api_key = api_key
        self.base_url = (base_url or settings.runtime.duix_base_url).rstrip("/")
        self.transport = transport

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=self.transport,
                                 headers={"Key": self.api_key, "Content-Type": "application/json"})

    async def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            async with self._client() as client:
                resp = await client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise ToolError("duix.network", str(exc)) from exc
        if resp.status_code >= 400:
            raise ToolError("duix.http", f"HTTP {resp.status_code} {resp.text[:300]}", status=resp.status_code)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ToolError("duix.bad_response", resp.text[:300]) from exc
        if data.get("code") != 0:
            raise remote_error(str(data.get("msg")))
        return data.get("data")

    # ---- account -------------------------------------------------------
    async def account(self) -> dict[str, Any]:
        account, credits = await asyncio.gather(
            self._call("GET", "/skills-api/skill/credit/queryAccount"),
            self._call("GET", "/skills-api/skill/credit/queryCreditsLeft"),
        )
        account = account or {}
        package: dict[str, Any] = {}
        if isinstance(account.get("packageItem"), str):
            try:
                package = json.loads(account["packageItem"])
            except ValueError:
                package = {}
        try:
            credits_left = int(float(credits))
        except (TypeError, ValueError):
            credits_left = 0
        minutes = package.get("durationMinutes") or account.get("maxDuration") or 0
        return {
            "valid": True,
            "engine": self.name,
            "plan": package.get("gradeName", ""),
            "summary": package.get("equitySummary", ""),
            "max_minutes": float(minutes or 0),
            "max_resolution": account.get("maxResolution") or package.get("resolution", ""),
            "credits_left": credits_left,
            "credits_total": package.get("credits"),
            "seconds_left": credits_left * SECONDS_PER_CREDIT,
            "cycle": f"{package.get('intervalQuantity', '')} {str(package.get('intervalUnit', '')).lower()}".strip(),
            "rights": package.get("rightsDesc", []),
        }

    def check(self, account: dict[str, Any], audio_seconds: float) -> dict[str, Any]:
        limit = account.get("max_minutes", 0) * 60
        needed = credits_for(audio_seconds)
        if limit and audio_seconds > limit:
            raise ToolError("duix.too_long", seconds=round(audio_seconds, 1), limit=int(limit))
        if needed > account.get("credits_left", 0):
            raise ToolError("duix.no_credits", needed=needed, left=account.get("credits_left", 0))
        return {"credits_needed": needed, "credits_left": account.get("credits_left", 0)}

    # ---- files ---------------------------------------------------------
    async def upload(self, path: Path, kind: str, on_progress: Callable[[float], None] | None = None) -> str:
        size = path.stat().st_size
        if size > MAX_BYTES:
            raise ToolError("duix.file_too_large", kind=kind)
        body = {"remoteFileAbsPath": f"duix-cli/upload/{self.api_key}/{uuid.uuid4().hex}{path.suffix.lower()}",
                "bucketLevel": 3, "expireSeconds": 7200}
        put_url = await self._call("POST", "/skills-api/skill/file/upload-url", body)
        if not put_url:
            raise ToolError("duix.bad_response", "no upload url")

        async def chunks():
            sent = 0
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 512):
                    sent += len(chunk)
                    if on_progress:
                        on_progress(sent / size if size else 1.0)
                    yield chunk

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, write=300.0), transport=self.transport) as client:
                resp = await client.put(put_url, content=chunks(), headers={"Content-Length": str(size)})
        except httpx.HTTPError as exc:
            raise ToolError("duix.upload_failed", str(exc), kind=kind) from exc
        if resp.status_code >= 300:
            raise ToolError("duix.upload_failed", f"HTTP {resp.status_code} {resp.text[:300]}", kind=kind)
        url = await self._call("POST", "/skills-api/skill/file/download-url", body)
        if not url:
            raise ToolError("duix.bad_response", "no download url")
        return url

    async def download(self, url: str, dest: Path, on_progress: Callable[[float], None] | None = None) -> Path:
        tmp = dest.with_suffix(".part")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=True,
                                         transport=self.transport) as client, client.stream("GET", url) as resp:
                if resp.status_code >= 300:
                    raise ToolError("duix.download_failed", f"HTTP {resp.status_code}")
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                with tmp.open("wb") as out:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        out.write(chunk)
                        done += len(chunk)
                        if on_progress and total:
                            on_progress(done / total)
        except httpx.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise ToolError("duix.download_failed", str(exc)) from exc
        tmp.replace(dest)
        return dest

    # ---- tasks ---------------------------------------------------------
    async def create(self, video_url: str, audio_url: str) -> str:
        data = await self._call("POST", "/skills-api/skill/video/synthesis",
                                {"videoUrl": video_url, "audioUrl": audio_url, "needTranscode": 1})
        if not data or data.get("taskId") in (None, ""):
            raise ToolError("duix.bad_response", "no taskId")
        return str(data["taskId"])

    async def status(self, task_id: str) -> dict[str, Any]:
        data = await self._call("GET", f"/skills-api/skill/task/status?taskId={task_id}")
        return normalize_status(data or {})

    async def delete(self, task_id: str) -> None:
        body = {"taskId": int(task_id) if task_id.isdigit() else task_id}
        await self._call("POST", "/skills-api/skill/file/deleteSynthResult", body)

    async def generate(self, video: Path, audio: Path, dest: Path, progress: Progress,
                       save_task: Callable[[str], None], task_id: str = "", model_name: str = "") -> Path:
        rt = settings.runtime
        if not task_id:
            progress(2, "duix.uploading_video", {})
            video_url = await self.upload(video, "video", lambda f: progress(2 + 18 * f, "duix.uploading_video", {}))
            progress(20, "duix.uploading_audio", {})
            audio_url = await self.upload(audio, "audio", lambda f: progress(20 + 5 * f, "duix.uploading_audio", {}))
            task_id = await self.create(video_url, audio_url)
            save_task(task_id)
        progress(25, "duix.queued", {"task": task_id})
        deadline = time.monotonic() + rt.duix_timeout_seconds
        errors = 0
        while True:
            try:
                state = await self.status(task_id)
                errors = 0
            except ToolError as exc:
                if exc.code in ("duix.auth_failed",) or errors >= 5:
                    raise
                errors += 1
                await asyncio.sleep(rt.duix_poll_seconds * 2)
                continue
            if state["status"] == "succeeded" and state["output_url"]:
                break
            if state["status"] == "failed":
                raise ToolError("duix.task_failed", state["error"] or state["desc"], task=task_id)
            progress(25 + 0.65 * state["progress"], f"duix.{state['status']}", {"task": task_id})
            if time.monotonic() > deadline:
                raise ToolError("duix.timeout", task=task_id)
            await asyncio.sleep(max(3, rt.duix_poll_seconds))
        progress(90, "duix.downloading", {"task": task_id})
        await self.download(state["output_url"], dest, lambda f: progress(90 + 9 * f, "duix.downloading", {}))
        if rt.duix_delete_remote:
            try:
                await self.delete(task_id)
            except ToolError:
                pass
        return dest


class DuixLocal:
    """Self-hosted Duix.Avatar container (``guiji2025/duix.avatar``, ``app_local.py``).

    Protocol (from the Duix.Avatar desktop client):
      POST /easy/submit {audio_url, video_url, code, chaofen:0, watermark_switch:0, pn:1}
           -> {"code": 10000, ...} on success
      GET  /easy/query?code=<code>
           -> {"code": 10000, "data": {"status": 1 running | 2 done | 3 failed,
                                       "progress": 0..100, "result": "/<code>-r.mp4", "msg"}}
           -> code 9999 / 10002 / 10003 = failed
    ``audio_url`` / ``video_url`` are file names inside the container's
    ``/code/data/temp`` = the Windows host folder ``duix_windows_dir``. This server
    reaches that folder through ``duix_share()`` (an SMB mount on the board, or
    the folder itself when running on Windows).
    """

    name = "duix_local"
    OK = 10000
    FAILED = {9999, 10002, 10003}

    def __init__(self, base_url: str | None = None, share: Path | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = (base_url or settings.runtime.duix_local_url).rstrip("/")
        self.share = share or settings.runtime.duix_share()
        self.transport = transport

    async def _request(self, method: str, path: str, timeout: float = 30, **kw: Any) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=self.transport) as client:
                resp = await client.request(method, path, **kw)
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError("duix.local_unreachable", str(exc), url=self.base_url) from exc

    def share_status(self) -> dict[str, Any]:
        folder = self.share
        status = {"path": str(folder), "exists": folder.is_dir(), "writable": False}
        if status["exists"]:
            probe_file = folder / f".mwt_probe_{uuid.uuid4().hex[:8]}"
            try:
                probe_file.write_bytes(b"ok")
                probe_file.unlink()
                status["writable"] = True
            except OSError as exc:
                status["error"] = str(exc)
        return status

    async def health(self) -> dict[str, Any]:
        started = time.monotonic()
        out: dict[str, Any] = {"engine": self.name, "url": self.base_url,
                               "windows_dir": settings.runtime.duix_windows_dir}
        try:
            await self._request("GET", "/easy/query", timeout=5, params={"code": "mwt_healthcheck"})
            out["reachable"] = True
            out["latency_ms"] = int((time.monotonic() - started) * 1000)
        except ToolError as exc:
            out["reachable"] = False
            out["error"] = exc.public()
        out["share"] = await asyncio.to_thread(self.share_status)
        out["ok"] = bool(out["reachable"] and out["share"]["writable"])
        return out

    async def account(self) -> dict[str, Any]:
        health = await self.health()
        if not health["reachable"]:
            raise ToolError("duix.local_unreachable", url=self.base_url)
        return {"valid": True, "plan": "local", "max_minutes": 0, "credits_left": None, **health}

    def check(self, account: dict[str, Any], audio_seconds: float) -> dict[str, Any]:
        return {"credits_needed": 0, "credits_left": None}

    async def wait_until_online(self, progress: Progress, max_wait: float = 600) -> None:
        """The Windows PC may still be booting / starting Docker: wait instead of failing."""
        deadline = time.monotonic() + max_wait
        while True:
            try:
                await self._request("GET", "/easy/query", timeout=5, params={"code": "mwt_healthcheck"})
                return
            except ToolError:
                if time.monotonic() > deadline:
                    raise
                progress(None, "duix.waiting_engine", {"url": self.base_url})
                await asyncio.sleep(15)

    async def _require_share(self) -> None:
        # a stale SMB mount can block for seconds: keep it off the event loop
        status = await asyncio.to_thread(self.share_status)
        if not status["writable"]:
            raise ToolError("duix.share_unavailable", status.get("error", ""), path=status["path"],
                            windows_dir=settings.runtime.duix_windows_dir)

    async def put_model(self, video: Path, cache_name: str) -> str:
        """Copy the base model video into the share once; reuse it while unchanged."""
        target = self.share / cache_name
        if not (target.is_file() and target.stat().st_size == video.stat().st_size):
            tmp = self.share / (cache_name + ".part")
            await asyncio.to_thread(shutil.copyfile, video, tmp)
            await asyncio.to_thread(tmp.replace, target)
        return target.name

    async def generate(self, video: Path, audio: Path, dest: Path, progress: Progress,
                       save_task: Callable[[str], None], task_id: str = "", model_name: str = "") -> Path:
        rt = settings.runtime
        await self.wait_until_online(progress)
        await self._require_share()
        code = task_id or "mwt_" + uuid.uuid4().hex
        audio_name = f"{code}_voice{audio.suffix}"
        if not task_id:
            progress(2, "duix.uploading_video", {})
            video_name = await self.put_model(video, model_name or f"{code}_model{video.suffix}")
            progress(15, "duix.uploading_audio", {})
            await asyncio.to_thread(shutil.copyfile, audio, self.share / audio_name)
            body = {"audio_url": audio_name, "video_url": video_name, "code": code,
                    "chaofen": 0, "watermark_switch": 0, "pn": 1}
            result = await self._request("POST", "/easy/submit", timeout=60, json=body)
            if result.get("code") != self.OK:
                raise ToolError("duix.task_failed", str(result.get("msg", result))[:300], task=code)
            save_task(code)
        progress(25, "duix.queued", {"task": code})
        deadline = time.monotonic() + rt.duix_timeout_seconds
        errors = 0
        while True:
            try:
                answer = await self._request("GET", "/easy/query", params={"code": code})
                errors = 0
            except ToolError:
                errors += 1
                if errors > 6:  # ~1 min of the PC being unreachable
                    raise
                progress(None, "duix.reconnecting", {"task": code})
                await asyncio.sleep(10)
                continue
            if answer.get("code") in self.FAILED:
                raise ToolError("duix.task_failed", str(answer.get("msg", ""))[:300], task=code)
            data = answer.get("data") or {}
            status = data.get("status")
            if status == 2:
                break
            if status == 3:
                raise ToolError("duix.task_failed", str(data.get("msg", ""))[:300], task=code)
            try:
                pct = float(data.get("progress") or 0)
            except (TypeError, ValueError):
                pct = 0.0
            progress(25 + 0.65 * pct, "duix.running", {"task": code})
            if time.monotonic() > deadline:
                raise ToolError("duix.timeout", task=code, seconds=rt.duix_timeout_seconds)
            await asyncio.sleep(max(2, rt.duix_poll_seconds))
        result_name = Path(str(data.get("result") or f"{code}-r.mp4").replace("\\", "/")).name
        source = self.share / result_name
        progress(90, "duix.downloading", {"task": code})
        for _ in range(10):  # SMB caches can lag a moment behind the container
            if source.is_file() and source.stat().st_size > 0:
                break
            await asyncio.sleep(2)
        else:
            raise ToolError("duix.download_failed", str(source))
        await asyncio.to_thread(shutil.copyfile, source, dest)
        if rt.duix_cleanup_share:
            for name in (audio_name, result_name):
                with contextlib.suppress(OSError):
                    (self.share / name).unlink()
        return dest


def wake_on_lan(mac: str, broadcast: str = "255.255.255.255") -> None:
    import socket

    digits = re.sub(r"[^0-9A-Fa-f]", "", mac)
    if len(digits) != 12:
        raise ToolError("duix.bad_mac")
    packet = b"\xff" * 6 + bytes.fromhex(digits) * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for port in (9, 7):
            sock.sendto(packet, (broadcast, port))


def engine(kind: str | None = None) -> DuixCloud | DuixLocal:
    kind = kind or settings.runtime.avatar_engine
    if kind == "duix_cloud":
        return DuixCloud(settings.secret("DUIX_API_KEY"))
    return DuixLocal()
