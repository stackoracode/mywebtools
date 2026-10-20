"""Large-language-model / speech API client (OpenAI by default, any compatible server).

Two request styles:
* ``openai``            - OpenAI Responses API  (POST {base}/responses, ``reasoning.effort``)
* ``openai_compatible`` - Chat Completions      (POST {base}/chat/completions, ``reasoning_effort``)
  for DeepSeek, Qwen/DashScope, Ollama, vLLM, LM Studio, OpenRouter, ...

Also: model listing (GET {base}/models), speech-to-text (``/audio/transcriptions``)
and text-to-speech (``/audio/speech``). Model ids are never hard-coded: the
admin page loads them from the server or accepts any typed id.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .errors import ToolError

EFFORTS = ["", "none", "minimal", "low", "medium", "high", "xhigh"]
_NON_CHAT = re.compile(r"(embed|whisper|tts|transcribe|dall-e|image|moderation|audio|realtime|search|computer|davinci|babbage)", re.I)
_SPEECH = re.compile(r"(tts|speech)", re.I)
_ASR = re.compile(r"(whisper|transcribe|asr|stt)", re.I)


class LLM:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None, provider: str | None = None,
                 model: str | None = None, effort: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        rt = settings.runtime
        self.base_url = (base_url or rt.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.secret("LLM_API_KEY")
        self.provider = provider or rt.llm_provider
        self.model = model or rt.llm_model
        self.effort = rt.llm_reasoning_effort if effort is None else effort
        self.timeout = rt.llm_timeout_seconds
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.base_url and (self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url))

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.AsyncClient(base_url=self.base_url, headers=headers, transport=self.transport,
                                 timeout=httpx.Timeout(timeout or self.timeout, connect=15))

    async def _post(self, path: str, *, timeout: float | None = None, **kw: Any) -> httpx.Response:
        if not self.configured:
            raise ToolError("llm.not_configured")
        try:
            async with self._client(timeout) as client:
                resp = await client.post(path, **kw)
        except httpx.TimeoutException as exc:
            raise ToolError("llm.timeout", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ToolError("llm.network", str(exc), url=self.base_url) from exc
        if resp.status_code in (401, 403):
            raise ToolError("llm.auth_failed", resp.text[:300])
        return resp

    @staticmethod
    def _error(resp: httpx.Response) -> ToolError:
        try:
            message = resp.json().get("error", {}).get("message") or resp.text
        except ValueError:
            message = resp.text
        return ToolError("llm.http", str(message)[:400], status=resp.status_code)

    # ---- models --------------------------------------------------------
    async def models(self) -> dict[str, list[str]]:
        if not self.configured:
            raise ToolError("llm.not_configured")
        try:
            async with self._client(30) as client:
                resp = await client.get("/models")
        except httpx.HTTPError as exc:
            raise ToolError("llm.network", str(exc), url=self.base_url) from exc
        if resp.status_code in (401, 403):
            raise ToolError("llm.auth_failed", resp.text[:300])
        if resp.status_code >= 400:
            raise self._error(resp)
        ids = sorted({str(m.get("id")) for m in resp.json().get("data", []) if m.get("id")})
        return {
            "chat": [i for i in ids if not _NON_CHAT.search(i)] or ids,
            "tts": [i for i in ids if _SPEECH.search(i)],
            "transcribe": [i for i in ids if _ASR.search(i)],
            "all": ids,
        }

    # ---- text generation -----------------------------------------------
    async def complete(self, system: str, user: str, *, json_mode: bool = False, model: str | None = None,
                       effort: str | None = None) -> str:
        if not self.configured:
            raise ToolError("llm.not_configured")
        model = model or self.model
        if not model:
            raise ToolError("llm.no_model")
        effort = self.effort if effort in (None, "") else effort
        if self.provider == "openai_compatible":
            return await self._chat(system, user, model, effort, json_mode)
        return await self._responses(system, user, model, effort, json_mode)

    async def _responses(self, system: str, user: str, model: str, effort: str, json_mode: bool) -> str:
        body: dict[str, Any] = {"model": model, "instructions": system, "input": user}
        if effort:
            body["reasoning"] = {"effort": effort}
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        resp = await self._post("/responses", json=body)
        if resp.status_code == 400 and effort and "reasoning" in resp.text.lower():
            body.pop("reasoning")  # non-reasoning model: retry without the effort setting
            resp = await self._post("/responses", json=body)
        if resp.status_code >= 400:
            raise self._error(resp)
        data = resp.json()
        if data.get("output_text"):
            return data["output_text"]
        parts = [c.get("text", "") for item in data.get("output", []) for c in item.get("content", []) or []
                 if c.get("type") in ("output_text", "text")]
        if not parts:
            raise ToolError("llm.empty", json.dumps(data)[:300])
        return "".join(parts)

    async def _chat(self, system: str, user: str, model: str, effort: str, json_mode: bool) -> str:
        body: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": system},
                                                             {"role": "user", "content": user}]}
        if effort:
            body["reasoning_effort"] = effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await self._post("/chat/completions", json=body)
        if resp.status_code == 400 and (effort or json_mode):
            body.pop("reasoning_effort", None)  # many compatible servers reject unknown options
            body.pop("response_format", None)
            resp = await self._post("/chat/completions", json=body)
        if resp.status_code >= 400:
            raise self._error(resp)
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ToolError("llm.empty", resp.text[:300]) from exc
        if not content:
            raise ToolError("llm.empty")
        return re.sub(r"^<think>.*?</think>\s*", "", content, flags=re.S)  # reasoning models served via chat

    async def complete_json(self, system: str, user: str, **kw: Any) -> Any:
        text = await self.complete(system, user, json_mode=True, **kw)
        try:
            return json.loads(text)
        except ValueError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except ValueError:
                    pass
        raise ToolError("llm.bad_json", text[:300])

    async def test(self) -> dict[str, Any]:
        started = time.monotonic()
        reply = await self.complete("You are a health check.", "Reply with the single word: OK")
        return {"ok": True, "reply": reply.strip()[:200], "model": self.model, "effort": self.effort,
                "seconds": round(time.monotonic() - started, 2)}

    # ---- speech ----------------------------------------------------------
    async def transcribe(self, audio: Path, language: str | None = None, prompt: str = "") -> list[dict[str, Any]]:
        """Timed segments [{start, end, text}] via /audio/transcriptions (verbose_json)."""
        model = settings.runtime.llm_transcribe_model or "whisper-1"
        data: dict[str, Any] = {"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        with audio.open("rb") as handle:
            resp = await self._post("/audio/transcriptions", data=data,
                                    files={"file": (audio.name, handle, "audio/wav")}, timeout=max(self.timeout, 300))
        if resp.status_code >= 400:
            raise self._error(resp)
        body = resp.json()
        segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")).strip()}
                    for s in body.get("segments") or [] if str(s.get("text", "")).strip()]
        if not segments and body.get("text"):  # model without timestamps: one segment, aligned later
            segments = [{"start": 0.0, "end": float(body.get("duration") or 0), "text": body["text"].strip()}]
        return segments

    async def speech(self, text: str, dest: Path, *, voice: str | None = None, instructions: str = "",
                     speed: float = 1.0) -> Path:
        rt = settings.runtime
        body: dict[str, Any] = {"model": rt.llm_tts_model or "gpt-4o-mini-tts", "voice": voice or rt.llm_tts_voice or "alloy",
                                "input": text, "response_format": "wav"}
        if instructions:
            body["instructions"] = instructions
        if abs(speed - 1.0) > 0.01:
            body["speed"] = speed
        resp = await self._post("/audio/speech", json=body, timeout=max(self.timeout, 300))
        if resp.status_code == 400 and "instructions" in body:
            body.pop("instructions")  # older TTS models do not take style instructions
            resp = await self._post("/audio/speech", json=body, timeout=max(self.timeout, 300))
        if resp.status_code >= 400:
            raise self._error(resp)
        dest.write_bytes(resp.content)
        return dest
