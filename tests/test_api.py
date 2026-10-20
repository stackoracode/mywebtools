import json
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from mywebtools.app import Services, create_app
from mywebtools.tools import REGISTRY

STATIC = Path(__file__).resolve().parent.parent / "mywebtools" / "static"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Services())) as c:
        yield c


def _pdf(text: str, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page().insert_text((72, 72), f"{text} page {n + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client, name: str, data: bytes) -> str:
    resp = client.put(f"/api/uploads?name={name}", content=data)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _wait(client, job_id: str) -> dict:
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_meta_and_tools(client):
    meta = client.get("/api/meta").json()
    assert "zh-CN" in meta["ui_languages"]
    tools = {t["id"]: t for t in client.get("/api/tools").json()}
    assert {"avatar_pip", "pdf_merge", "video_ocr", "transcribe"} <= set(tools)
    avatar = {p["name"]: p for p in tools["avatar_pip"]["params"]}
    assert avatar["shape"]["options"][0][0] == "circle"
    assert avatar["model"]["type"] == "library"


def test_index_and_static(client):
    assert "My Web Tools" in client.get("/").text
    assert client.get("/static/js/app.js").status_code == 200


def test_pdf_merge_split_extract_flow(client):
    a = _upload(client, "a.pdf", _pdf("Alpha", 2))
    b = _upload(client, "b.pdf", _pdf("Beta", 1))
    resp = client.post("/api/jobs", json={"tool": "pdf_merge", "params": {"output_name": "both"}, "files": {"pdfs": [a, b]}})
    job = _wait(client, resp.json()["jobs"][0]["id"])
    assert job["state"] == "done", job
    assert job["result"]["pages"] == 3
    merged = client.get(f"/api/jobs/{job['id']}/files/both.pdf")
    assert merged.status_code == 200 and merged.content.startswith(b"%PDF")

    c = _upload(client, "c.pdf", _pdf("Gamma", 3))
    resp = client.post("/api/jobs", json={"tool": "pdf_split", "params": {"mode": "ranges", "ranges": "1,2-"}, "files": {"pdf": [c]}})
    job = _wait(client, resp.json()["jobs"][0]["id"])
    assert job["state"] == "done" and job["result"]["parts"] == 2

    d = _upload(client, "d.pdf", _pdf("Delta"))
    e = _upload(client, "e.pdf", _pdf("Epsilon"))
    resp = client.post("/api/jobs", json={"tool": "pdf_extract", "params": {"ocr_mode": "never"}, "files": {"pdf": [d, e]}})
    jobs = resp.json()["jobs"]
    assert len(jobs) == 2 and jobs[0]["batch_id"] == jobs[1]["batch_id"] != ""  # batch fan-out
    done = [_wait(client, j["id"]) for j in jobs]
    text = client.get(f"/api/jobs/{done[0]['id']}/files/d.txt").text
    assert "Delta page 1" in text


def test_missing_file_is_rejected(client):
    resp = client.post("/api/jobs", json={"tool": "pdf_merge", "params": {}, "files": {}})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "input.missing"


def test_form_memory_and_presets(client):
    a = _upload(client, "x.pdf", _pdf("X"))
    client.post("/api/jobs", json={"tool": "pdf_to_images", "params": {"dpi": 300, "format": "jpg"}, "files": {"pdf": [a]}})
    last = client.get("/api/prefs/pdf_to_images").json()["last"]
    assert last["dpi"] == 300 and last["format"] == "jpg"
    resp = client.put("/api/prefs/avatar_pip/presets/Shorts%20left", json={"values": {"shape": "heart", "size": 420}})
    assert resp.json()["presets"]["Shorts left"]["shape"] == "heart"
    client.delete("/api/prefs/avatar_pip/presets/Shorts%20left")
    assert "Shorts left" not in client.get("/api/prefs/avatar_pip").json()["presets"]


def test_settings_defaults_point_at_windows_duix(client):
    values = client.get("/api/settings").json()["values"]
    assert values["avatar_engine"] == "duix_local"
    assert values["duix_local_url"] == "http://192.168.8.106:8383"
    assert values["duix_windows_dir"] == r"C:\AI\Duix\data\face2face\temp"
    assert values["duix_timeout_seconds"] == 3600
    resp = client.put("/api/settings", json={"values": {"duix_timeout_seconds": "1800", "bogus": 1}})
    assert resp.json()["values"]["duix_timeout_seconds"] == 1800
    client.put("/api/settings", json={"values": {"duix_timeout_seconds": 3600}})


def test_duix_health_reports_offline(client):
    client.put("/api/settings", json={"values": {"duix_local_url": "http://127.0.0.1:9", "duix_share_path": "/nonexistent/share"}})
    health = client.get("/api/duix/health").json()
    assert health["engine"] == "duix_local" and health["ok"] is False and health["reachable"] is False
    client.put("/api/settings", json={"values": {"duix_local_url": "http://192.168.8.106:8383", "duix_share_path": ""}})


def test_password_protection(client):
    assert client.put("/api/secrets", json={"APP_PASSWORD": "s3cret"}).status_code == 200
    anon = TestClient(client.app)
    assert anon.get("/api/tools").status_code == 401
    assert anon.post("/api/login", json={"password": "nope"}).status_code == 400
    assert anon.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert anon.get("/api/tools").status_code == 200
    client.put("/api/secrets", json={"APP_PASSWORD": ""})


def test_zh_cn_catalog_covers_ui_and_tools():
    en = json.loads((STATIC / "i18n" / "en.json").read_text(encoding="utf-8"))
    zh = json.loads((STATIC / "i18n" / "zh-CN.json").read_text(encoding="utf-8"))
    missing = [k for k in en if k not in zh]
    for tool in REGISTRY.values():
        schema = tool.schema()
        keys = [f"tool.{tool.id}.name", f"tool.{tool.id}.desc"]
        keys += [f"tool.{tool.id}.g.{g['name']}" for g in schema["groups"] if g["label"]]
        for p in schema["params"]:
            if p.get("label"):
                keys.append(f"tool.{tool.id}.p.{p['name']}")
            if p.get("help"):
                keys.append(f"tool.{tool.id}.h.{p['name']}")
            if p["name"] not in ("ocr_langs", "language", "sub_language"):
                keys += [f"tool.{tool.id}.o.{p['name']}.{v}" for v, label in p.get("options", []) if label != v]
        missing += [k for k in keys if k not in zh]
    assert not missing, missing
