import asyncio
import json

import httpx
import pytest

from mywebtools.errors import ToolError
from mywebtools.llm import LLM


def make(handler, **kw):
    kw.setdefault("provider", "openai")
    return LLM(base_url="https://api.test/v1", api_key="sk-test", model="gpt-test", effort="xhigh",
               transport=httpx.MockTransport(handler), **kw)


def test_responses_api_with_reasoning_effort():
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen.update(body)
        assert request.headers["Authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"output": [{"type": "reasoning", "content": []},
                                                    {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]})

    assert asyncio.run(make(handler).complete("sys", "hi")) == "OK"
    assert seen["reasoning"] == {"effort": "xhigh"} and seen["model"] == "gpt-test" and seen["instructions"] == "sys"


def test_non_reasoning_model_retries_without_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "reasoning" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'reasoning.effort'"}})
        return httpx.Response(200, json={"output_text": "fine"})

    assert asyncio.run(make(handler).complete("s", "u")) == "fine"
    assert len(calls) == 2 and "reasoning" not in calls[1]


def test_chat_completions_compatible_server_and_think_tags():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "xhigh" and body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\n{\"a\": 1}"}}]})

    client = make(handler, provider="openai_compatible")
    assert asyncio.run(client.complete_json("s", "u")) == {"a": 1}


def test_models_are_grouped():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": i} for i in
                                                  ["gpt-6-luna", "gpt-4o-mini-tts", "whisper-1", "text-embedding-3-small"]]})

    models = asyncio.run(make(handler).models())
    assert models["chat"] == ["gpt-6-luna"]
    assert models["tts"] == ["gpt-4o-mini-tts"] and models["transcribe"] == ["whisper-1"]


def test_auth_error_and_missing_config():
    with pytest.raises(ToolError) as exc:
        asyncio.run(make(lambda r: httpx.Response(401, text="bad key")).complete("s", "u"))
    assert exc.value.code == "llm.auth_failed"
    with pytest.raises(ToolError) as exc:
        asyncio.run(LLM(base_url="https://api.test/v1", api_key="").complete("s", "u"))
    assert exc.value.code == "llm.not_configured"


def test_transcribe_and_speech(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")

    def handler(request):
        if request.url.path.endswith("/audio/transcriptions"):
            assert b"verbose_json" in request.content and b'name="file"' in request.content
            return httpx.Response(200, json={"segments": [{"start": 0, "end": 1.5, "text": " Hello "}]})
        if request.url.path.endswith("/audio/speech"):
            assert json.loads(request.content)["response_format"] == "wav"
            return httpx.Response(200, content=b"WAVDATA")
        return httpx.Response(404)

    client = make(handler)
    assert asyncio.run(client.transcribe(wav, "en")) == [{"start": 0.0, "end": 1.5, "text": "Hello"}]
    out = asyncio.run(client.speech("hi", tmp_path / "s.wav", instructions="cheerful"))
    assert out.read_bytes() == b"WAVDATA"
