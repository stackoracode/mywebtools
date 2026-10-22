"""Client for the model services on the GPU PC (see services/README.md).

    whisper  faster-whisper    POST /transcribe  (NDJSON stream of segments)
    tts      IndexTTS 2.5      POST /tts         (WAV)
    align    MMS CTC aligner   POST /align       (JSON timings)

Model folders are configured here (Admin → Model services) and sent with every
request, so the services need no configuration of their own. Each service also
has /health, /load and /unload.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .errors import ToolError

KINDS: dict[str, dict[str, str]] = {
    "whisper": {"url": "node_whisper_url", "model_dir": "whisper_model_dir", "device": "whisper_device", "port": "8391"},
    "tts": {"url": "node_tts_url", "model_dir": "indextts_model_dir", "device": "indextts_device", "port": "8392"},
    "align": {"url": "node_align_url", "model_dir": "mms_model_dir", "device": "mms_device", "port": "8393"},
}
# service error codes with their own message (err.node.<code>); anything else is node.failed
KNOWN = {"model_missing", "model_outside_root", "files_missing", "out_of_memory", "emotion_model_missing",
         "import_failed", "audio_too_short", "too_long", "no_letters", "unauthorized"}


class Node:
    def __init__(self, kind: str, *, url: str | None = None, model_dir: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        if kind not in KINDS:
            raise ToolError("node.unknown", kind=kind)
        rt = settings.runtime
        self.kind = kind
        self.url = (url if url is not None else getattr(rt, KINDS[kind]["url"])).strip().rstrip("/")
        self.model_dir = model_dir if model_dir is not None else getattr(rt, KINDS[kind]["model_dir"])
        self.device = getattr(rt, KINDS[kind]["device"])
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def _client(self, timeout: float, connect: float = 10) -> httpx.AsyncClient:
        token = settings.secret("NODE_TOKEN")
        return httpx.AsyncClient(base_url=self.url, transport=self.transport,
                                 headers={"Authorization": f"Bearer {token}"} if token else {},
                                 timeout=httpx.Timeout(timeout, connect=connect))

    def _fail(self, resp: httpx.Response) -> ToolError:
        try:
            err = resp.json().get("error") or {}
        except ValueError:
            err = {}
        return self._error(err.get("code") or f"http_{resp.status_code}", err.get("detail") or resp.text[:300])

    def _error(self, code: str, detail: str) -> ToolError:
        if code in KNOWN:
            return ToolError(f"node.{code}", detail[:500], service=self.kind)
        return ToolError("node.failed", detail[:500], service=self.kind, remote_code=code)

    async def _post(self, path: str, *, timeout: float | None = None, **kw: Any) -> httpx.Response:
        if not self.configured:
            raise ToolError("node.not_configured", service=self.kind)
        try:
            async with self._client(timeout or settings.runtime.node_timeout_seconds) as client:
                resp = await client.post(path, **kw)
        except httpx.ConnectTimeout as exc:  # PC off / unreachable: not a slow job
            raise ToolError("node.offline", str(exc), service=self.kind, url=self.url) from exc
        except httpx.TimeoutException as exc:
            raise ToolError("node.timeout", str(exc), service=self.kind, url=self.url) from exc
        except httpx.HTTPError as exc:
            raise ToolError("node.offline", str(exc), service=self.kind, url=self.url) from exc
        if resp.status_code >= 400:
            raise self._fail(resp)
        return resp

    def _options(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {"model_dir": self.model_dir, "device": self.device, **extra}

    # ---- management ------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        """Never raises: {"reachable": False, "error": ...} when the service cannot be reached."""
        base = {"kind": self.kind, "url": self.url, "model_dir": self.model_dir}
        if not self.configured:
            return {**base, "reachable": False, "ok": False, "error": ToolError("node.not_configured", service=self.kind).public()}
        try:
            async with self._client(20, connect=4) as client:
                resp = await client.get("/health", params={"model_dir": self.model_dir} if self.model_dir else None)
        except httpx.HTTPError as exc:
            return {**base, "reachable": False, "ok": False,
                    "error": ToolError("node.offline", str(exc), service=self.kind, url=self.url).public()}
        if resp.status_code >= 400:
            return {**base, "reachable": True, "ok": False, "error": self._fail(resp).public()}
        return {**base, "reachable": True, **resp.json()}

    async def load(self) -> dict[str, Any]:
        return (await self._post("/load", json=self._options(self._load_extra()), timeout=900)).json()

    async def unload(self) -> dict[str, Any]:
        return (await self._post("/unload", timeout=60)).json()

    async def unload_quietly(self) -> None:
        try:
            await self.unload()
        except ToolError:
            pass

    def _load_extra(self) -> dict[str, Any]:
        rt = settings.runtime
        if self.kind == "whisper":
            return {"compute_type": rt.whisper_compute_type}
        if self.kind == "tts":
            return {"bf16": rt.indextts_bf16}
        return {}

    # ---- work ------------------------------------------------------------
    async def transcribe(self, audio: Path, options: dict[str, Any],
                         on_segment: Callable[[dict[str, Any], float], None] | None = None) -> dict[str, Any]:
        """{"language", "duration", "segments": [{start, end, text[, words]}]} (streamed by the service)."""
        if not self.configured:
            raise ToolError("node.not_configured", service=self.kind)
        rt = settings.runtime
        opts = self._options({"compute_type": rt.whisper_compute_type, "beam_size": rt.whisper_beam_size, **options})
        info: dict[str, Any] = {"language": "", "duration": 0.0, "segments": []}
        try:
            with audio.open("rb") as handle:
                async with self._client(rt.node_timeout_seconds) as client, \
                        client.stream("POST", "/transcribe", data={"options": json.dumps(opts, ensure_ascii=False)},
                                      files={"file": (audio.name, handle, "audio/wav")}) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        raise self._fail(resp)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        kind = item.pop("type", "")
                        if kind == "info":
                            info.update(language=item.get("language", ""), duration=float(item.get("duration") or 0))
                        elif kind == "segment":
                            info["segments"].append(item)
                            if on_segment:
                                on_segment(item, info["duration"])
                        elif kind == "error":
                            raise self._error(item.get("code", "internal"), item.get("detail", ""))
        except httpx.ConnectTimeout as exc:
            raise ToolError("node.offline", str(exc), service=self.kind, url=self.url) from exc
        except httpx.TimeoutException as exc:
            raise ToolError("node.timeout", str(exc), service=self.kind, url=self.url) from exc
        except httpx.HTTPError as exc:
            raise ToolError("node.offline", str(exc), service=self.kind, url=self.url) from exc
        return info

    async def tts(self, text: str, voice: Path, dest: Path, options: dict[str, Any],
                  emo_audio: Path | None = None) -> Path:
        opts = self._options({"bf16": settings.runtime.indextts_bf16, **options})
        handles = {"voice": (voice.name, voice.open("rb"), "audio/wav")}
        if emo_audio is not None:
            handles["emo_audio"] = (emo_audio.name, emo_audio.open("rb"), "audio/wav")
        try:
            resp = await self._post("/tts", data={"text": text, "options": json.dumps(opts, ensure_ascii=False)},
                                    files=handles)
        finally:
            for _, handle, _ in handles.values():
                handle.close()
        dest.write_bytes(resp.content)
        return dest

    async def align(self, audio: Path, cues: list[str], language: str | None) -> dict[str, Any]:
        opts = self._options({"cues": cues, "language": language or ""})
        with audio.open("rb") as handle:
            resp = await self._post("/align", data={"options": json.dumps(opts, ensure_ascii=False)},
                                    files={"file": (audio.name, handle, "audio/wav")})
        return resp.json()
