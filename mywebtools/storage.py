"""Uploads, libraries and user preferences on disk.

Everything lives under ``data/`` so the whole state of the app can be backed up
or wiped by copying/removing one directory.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ToolError

ID_RE = re.compile(r"^[a-f0-9]{32}$")


def new_id() -> str:
    return uuid.uuid4().hex


def safe_name(name: str, fallback: str = "file") -> str:
    """Keep a readable, unicode-friendly file name without any path parts."""
    name = unicodedata.normalize("NFC", Path(str(name).replace("\\", "/")).name)
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip(" ._") or fallback
    stem = re.sub(r"\s+", "_", stem)[:100]
    ext = re.sub(r"[^A-Za-z0-9]", "", ext).lower()[:10]
    return f"{stem}.{ext}" if ext else stem


def check_id(value: str) -> str:
    if not ID_RE.match(value or ""):
        raise ToolError("bad_id")
    return value


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


async def write_stream(stream: AsyncIterator[bytes], dest: Path, max_bytes: int) -> int:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            async for chunk in stream:
                size += len(chunk)
                if size > max_bytes:
                    raise ToolError("upload.too_large", max_mb=max_bytes // (1024 * 1024))
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise ToolError("upload.empty")
    return size


# ---------------------------------------------------------------------------
# Uploads: raw files sent by the browser before a job is submitted. Uploading
# first lets the phone start sending a big video while the form is filled in,
# and lets "preview" and the real job share the same file.
# ---------------------------------------------------------------------------
@dataclass
class Upload:
    id: str
    name: str
    size: int
    created: float
    path: Path

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "size": self.size, "created": self.created}


class UploadStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def save(self, name: str, stream: AsyncIterator[bytes], max_bytes: int) -> Upload:
        upload_id = new_id()
        filename = safe_name(name)
        folder = self.root / upload_id
        try:
            size = await write_stream(stream, folder / filename, max_bytes)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        upload = Upload(upload_id, filename, size, time.time(), folder / filename)
        _write_json(folder / "meta.json", {"name": filename, "size": size, "created": upload.created})
        return upload

    def get(self, upload_id: str) -> Upload:
        folder = self.root / check_id(upload_id)
        meta = _read_json(folder / "meta.json", None)
        if not meta or not (folder / meta["name"]).is_file():
            raise ToolError("upload.missing")
        return Upload(upload_id, meta["name"], meta["size"], meta["created"], folder / meta["name"])

    def exists(self, upload_id: str) -> bool:
        try:
            self.get(upload_id)
            return True
        except ToolError:
            return False

    def delete(self, upload_id: str) -> None:
        shutil.rmtree(self.root / check_id(upload_id), ignore_errors=True)

    def sweep(self, ttl_hours: int, keep: set[str]) -> int:
        cutoff = time.time() - ttl_hours * 3600
        removed = 0
        for folder in self.root.iterdir() if self.root.is_dir() else []:
            if folder.name in keep or not ID_RE.match(folder.name):
                continue
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        return removed

    def usage(self) -> int:
        return dir_size(self.root)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Libraries: reusable assets such as avatar base-model videos, generated
# lip-sync clips (re-compositing a clip costs no cloud credits) and voice
# samples for voice cloning.
# ---------------------------------------------------------------------------
LIBRARIES: dict[str, dict[str, Any]] = {
    "avatar_models": {"accept": "video/*,.mp4,.mov,.webm,.mkv", "kind": "video"},
    "avatar_clips": {"accept": "video/*,.mp4,.mov,.webm", "kind": "video"},
    "voices": {"accept": "audio/*,video/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.mp4,.mov", "kind": "audio"},
}


@dataclass
class LibraryItem:
    id: str
    library: str
    name: str
    file: str
    size: int
    created: float
    info: dict[str, Any] = field(default_factory=dict)
    has_thumb: bool = False

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("file")
        return data


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _folder(self, library: str, item_id: str = "") -> Path:
        if library not in LIBRARIES:
            raise ToolError("library.unknown")
        return self.root / library / check_id(item_id) if item_id else self.root / library

    def add(self, library: str, source: Path, name: str, info: dict[str, Any] | None = None, move: bool = False) -> LibraryItem:
        item_id = new_id()
        folder = self._folder(library, item_id)
        folder.mkdir(parents=True, exist_ok=True)
        filename = safe_name(source.name)
        (shutil.move if move else shutil.copy2)(str(source), str(folder / filename))
        item = LibraryItem(item_id, library, name.strip()[:80] or Path(filename).stem, filename,
                           (folder / filename).stat().st_size, time.time(), info or {})
        self._save(item)
        return item

    def _save(self, item: LibraryItem) -> None:
        _write_json(self._folder(item.library, item.id) / "meta.json", asdict(item))

    def list(self, library: str) -> list[LibraryItem]:
        folder = self._folder(library)
        items = []
        for meta in folder.glob("*/meta.json") if folder.is_dir() else []:
            data = _read_json(meta, None)
            if data:
                items.append(LibraryItem(**data))
        return sorted(items, key=lambda i: i.created, reverse=True)

    def get(self, library: str, item_id: str) -> LibraryItem:
        data = _read_json(self._folder(library, item_id) / "meta.json", None)
        if not data:
            raise ToolError("library.missing")
        return LibraryItem(**data)

    def path(self, item: LibraryItem) -> Path:
        return self._folder(item.library, item.id) / item.file

    def thumb_path(self, item: LibraryItem) -> Path:
        return self._folder(item.library, item.id) / "thumb.jpg"

    def cache_path(self, item: LibraryItem, key: str) -> Path:
        return self._folder(item.library, item.id) / f"cache_{safe_name(key)}"

    def update(self, item: LibraryItem, **changes: Any) -> LibraryItem:
        for key, value in changes.items():
            setattr(item, key, value)
        self._save(item)
        return item

    def delete(self, library: str, item_id: str) -> None:
        shutil.rmtree(self._folder(library, item_id), ignore_errors=True)

    def usage(self) -> int:
        return dir_size(self.root)


# ---------------------------------------------------------------------------
# Preferences: the "form memory". Last-used values per tool + named presets.
# ---------------------------------------------------------------------------
class Prefs:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = _read_json(path, {}) or {}
        self.data.setdefault("last", {})
        self.data.setdefault("presets", {})

    def _flush(self) -> None:
        _write_json(self.path, self.data)

    def last(self, tool: str) -> dict[str, Any]:
        return dict(self.data["last"].get(tool, {}))

    def set_last(self, tool: str, values: dict[str, Any]) -> None:
        self.data["last"][tool] = values
        self._flush()

    def presets(self, tool: str) -> dict[str, dict[str, Any]]:
        return dict(self.data["presets"].get(tool, {}))

    def save_preset(self, tool: str, name: str, values: dict[str, Any]) -> None:
        name = name.strip()[:60]
        if not name:
            raise ToolError("preset.name")
        self.data["presets"].setdefault(tool, {})[name] = values
        self._flush()

    def delete_preset(self, tool: str, name: str) -> None:
        self.data["presets"].get(tool, {}).pop(name, None)
        self._flush()
