import asyncio
import json
from pathlib import Path

import httpx
import pytest

from mywebtools import duix
from mywebtools.errors import ToolError

ACCOUNT = {
    "userId": 1, "maxDuration": 3, "maxResolution": "720P",
    "packageItem": json.dumps({"gradeName": "Free", "durationMinutes": 3, "credits": 36,
                               "equitySummary": "36 credits + 3 minutes + 720p", "intervalQuantity": 30, "intervalUnit": "DAYS"}),
}


def ok(data):
    return httpx.Response(200, json={"success": True, "code": 0, "msg": "成功", "data": data})


def test_credit_rules():
    assert duix.credits_for(0.5) == 1
    assert duix.credits_for(5) == 1
    assert duix.credits_for(5.1) == 2
    assert duix.credits_for(180) == 36
    assert duix.ratio_ok(720, 1280) and duix.ratio_ok(1920, 1080) and not duix.ratio_ok(1080, 1350)
    assert duix.resolution_ok(720, 1280) and not duix.resolution_ok(480, 854)


def test_status_mapping():
    assert duix.normalize_status({"status": 3, "outputUrl": "u"})["status"] == "succeeded"
    assert duix.normalize_status({"status": 2, "statusDesc": "处理中", "progress": "42"}) == {
        "status": "running", "progress": 42, "output_url": "", "error": "", "desc": "处理中"}
    assert duix.normalize_status({"status": 1, "statusDesc": "失败"})["status"] == "failed"


def test_auth_error_mapping():
    assert duix.remote_error("查询剩余积分失败: 认证失败，请检查key是否正确").code == "duix.auth_failed"


def cloud_transport(tmp_path: Path, calls: list):
    statuses = iter([{"status": 1, "progress": 0}, {"status": 2, "progress": 50}, {"status": 3, "progress": 100, "outputUrl": "https://cdn/out.mp4"}])

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/credit/queryAccount"):
            assert request.headers["Key"] == "k"
            return ok(ACCOUNT)
        if path.endswith("/credit/queryCreditsLeft"):
            return ok(28)
        if path.endswith("/file/upload-url"):
            body = json.loads(request.content)
            assert body["remoteFileAbsPath"].startswith("duix-cli/upload/k/") and body["bucketLevel"] == 3
            return ok("https://oss/put")
        if path == "/put":
            assert int(request.headers["Content-Length"]) > 0
            return httpx.Response(200)
        if path.endswith("/file/download-url"):
            return ok("https://oss/get")
        if path.endswith("/video/synthesis"):
            assert json.loads(request.content) == {"videoUrl": "https://oss/get", "audioUrl": "https://oss/get", "needTranscode": 1}
            return ok({"taskId": 987, "status": 0})
        if path.endswith("/task/status"):
            return ok(next(statuses))
        if path == "/out.mp4":
            return httpx.Response(200, content=b"mp4data", headers={"content-length": "7"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_cloud_account_and_check(tmp_path):
    client = duix.DuixCloud("k", "https://api", transport=cloud_transport(tmp_path, []))
    account = asyncio.run(client.account())
    assert account["plan"] == "Free" and account["credits_left"] == 28 and account["max_minutes"] == 3
    assert client.check(account, 60)["credits_needed"] == 12
    with pytest.raises(ToolError) as exc:
        client.check(account, 200)
    assert exc.value.code == "duix.too_long"
    with pytest.raises(ToolError) as exc:
        client.check(account, 179)  # 36 credits needed, 28 left
    assert exc.value.code == "duix.no_credits"


def test_cloud_generate(tmp_path, monkeypatch):
    from mywebtools.config import settings

    monkeypatch.setattr(settings.runtime, "duix_poll_seconds", 0)
    monkeypatch.setattr(duix.asyncio, "sleep", _no_sleep)
    calls: list = []
    client = duix.DuixCloud("k", "https://api", transport=cloud_transport(tmp_path, calls))
    video, audio = tmp_path / "v.mp4", tmp_path / "a.wav"
    video.write_bytes(b"v" * 1000)
    audio.write_bytes(b"a" * 100)
    saved = []
    out = asyncio.run(client.generate(video, audio, tmp_path / "out.mp4", lambda *a: None, saved.append))
    assert out.read_bytes() == b"mp4data"
    assert saved == ["987"]


async def _no_sleep(*_):
    return None


def local_transport(share: Path, submitted: list, fail=False):
    polls = {"n": 0}

    def handler(request: httpx.Request):
        if request.url.path == "/easy/submit":
            body = json.loads(request.content)
            assert (share / body["audio_url"]).is_file() and (share / body["video_url"]).is_file()
            assert body["chaofen"] == 0 and body["pn"] == 1
            submitted.append(body)
            return httpx.Response(200, json={"code": 10000, "success": True})
        if request.url.path == "/easy/query":
            code = request.url.params["code"]
            if code == "mwt_healthcheck":
                return httpx.Response(200, json={"code": 10004, "msg": "not found"})
            if fail:
                return httpx.Response(200, json={"code": 10002, "msg": "face not detected"})
            polls["n"] += 1
            if polls["n"] < 2:
                return httpx.Response(200, json={"code": 10000, "data": {"status": 1, "progress": 40}})
            (share / f"{code}-r.mp4").write_bytes(b"result")
            return httpx.Response(200, json={"code": 10000, "data": {"status": 2, "progress": 100, "result": f"/{code}-r.mp4"}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_local_generate_via_share(tmp_path, monkeypatch):
    from mywebtools.config import settings

    monkeypatch.setattr(duix.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(settings.runtime, "duix_cleanup_share", True)
    share = tmp_path / "share"
    share.mkdir()
    submitted: list = []
    engine = duix.DuixLocal("http://pc:8383", share, transport=local_transport(share, submitted))
    health = asyncio.run(engine.health())
    assert health["reachable"] and health["share"]["writable"] and health["ok"]
    video, audio = tmp_path / "model.mp4", tmp_path / "voice.wav"
    video.write_bytes(b"m" * 50)
    audio.write_bytes(b"a" * 10)
    codes = []
    out = asyncio.run(engine.generate(video, audio, tmp_path / "clip.mp4", lambda *a: None, codes.append,
                                      model_name="mwt_model_x.mp4"))
    assert out.read_bytes() == b"result"
    assert submitted[0]["video_url"] == "mwt_model_x.mp4"
    remaining = sorted(p.name for p in share.iterdir())
    assert remaining == ["mwt_model_x.mp4"]  # model cached, voice + result cleaned up
    assert codes and codes[0].startswith("mwt_")


def test_local_failure_and_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(duix.asyncio, "sleep", _no_sleep)
    share = tmp_path / "share"
    share.mkdir()
    engine = duix.DuixLocal("http://pc:8383", share, transport=local_transport(share, [], fail=True))
    video, audio = tmp_path / "m.mp4", tmp_path / "v.wav"
    video.write_bytes(b"m")
    audio.write_bytes(b"a")
    with pytest.raises(ToolError) as exc:
        asyncio.run(engine.generate(video, audio, tmp_path / "o.mp4", lambda *a: None, lambda c: None))
    assert exc.value.code == "duix.task_failed"

    def down(request):
        raise httpx.ConnectError("refused")

    offline = duix.DuixLocal("http://pc:8383", tmp_path / "missing", transport=httpx.MockTransport(down))
    health = asyncio.run(offline.health())
    assert not health["reachable"] and not health["share"]["exists"] and not health["ok"]


def test_wake_on_lan_validates_mac():
    with pytest.raises(ToolError):
        duix.wake_on_lan("nope")
