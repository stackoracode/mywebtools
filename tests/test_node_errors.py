"""Regression coverage for errors returned by the GPU model services."""

import json
from pathlib import Path

import httpx
import pytest

from mywebtools.errors import ToolError
from mywebtools.nodes import Node


@pytest.mark.parametrize("kind", ["tts", "align", "whisper"])
@pytest.mark.parametrize("remote_code", ["internal", "bad_request"])
async def test_service_failure_preserves_remote_error(kind, remote_code):
    transport = httpx.MockTransport(lambda request: httpx.Response(
        500, json={"error": {"code": remote_code, "detail": "original model failure"}}))
    node = Node(kind, url="http://gpu", transport=transport)
    with pytest.raises(ToolError) as caught:
        await node._post("/load")
    error = caught.value.public()
    assert error == {"code": "node.failed", "detail": "original model failure",
                     "params": {"service": kind, "remote_code": remote_code}}
    catalogs = Path(__file__).resolve().parents[1] / "mywebtools" / "static" / "i18n"
    for language in ("en", "zh-CN"):
        message = json.loads((catalogs / f"{language}.json").read_text(encoding="utf-8"))["err.node.failed"]
        assert remote_code in message.format(**error["params"])


async def test_non_json_service_failure():
    node = Node("tts", url="http://gpu", transport=httpx.MockTransport(
        lambda request: httpx.Response(502, text="Bad gateway")))
    with pytest.raises(ToolError) as caught:
        await node._post("/load")
    assert caught.value.code == "node.failed"
    assert caught.value.params["remote_code"] == "http_502"
    assert caught.value.detail == "Bad gateway"


async def test_known_service_error_keeps_specific_message():
    node = Node("tts", url="http://gpu", transport=httpx.MockTransport(lambda request: httpx.Response(
        404, json={"error": {"code": "files_missing", "detail": "missing weights"}})))
    with pytest.raises(ToolError) as caught:
        await node._post("/load")
    assert caught.value.public() == {"code": "node.files_missing", "detail": "missing weights",
                                    "params": {"service": "tts"}}


async def test_streamed_service_error(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    node = Node("whisper", url="http://gpu", transport=httpx.MockTransport(lambda request: httpx.Response(
        200, text=json.dumps({"type": "error", "code": "internal", "detail": "stream failed"}) + "\n")))
    with pytest.raises(ToolError) as caught:
        await node.transcribe(audio, {})
    assert caught.value.public() == {"code": "node.failed", "detail": "stream failed",
                                    "params": {"service": "whisper", "remote_code": "internal"}}
