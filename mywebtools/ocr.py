"""Tesseract OCR through its CLI (async subprocess, TSV output)."""

from __future__ import annotations

import asyncio
import csv
import io
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .i18n import join_words
from .media import run_cmd

PSM_OPTIONS = [["3", "Auto page"], ["6", "Single block"], ["7", "Single line"], ["11", "Sparse text"]]

_lang_cache: tuple[float, list[str]] = (0.0, [])


def available() -> bool:
    return shutil.which(settings.tesseract) is not None


def languages() -> list[str]:
    """Installed traineddata (``tesseract --list-langs``), cached for 5 minutes."""
    global _lang_cache
    if time.time() - _lang_cache[0] < 300 and _lang_cache[1]:
        return _lang_cache[1]
    try:
        out = subprocess.run([settings.tesseract, "--list-langs"], capture_output=True, text=True, timeout=15)
        langs = [line.strip() for line in (out.stdout or out.stderr).splitlines()[1:] if line.strip() and line.strip() != "osd"]
    except (OSError, subprocess.SubprocessError):
        langs = []
    _lang_cache = (time.time(), sorted(langs))
    return _lang_cache[1]


def lang_arg(langs: list[str]) -> str:
    installed = set(languages())
    chosen = [lang for lang in langs if not installed or lang in installed]
    return "+".join(chosen or ["eng"])


@dataclass
class OcrLine:
    text: str
    conf: float
    top: int
    left: int


def parse_tsv(tsv: str, min_conf: float) -> list[OcrLine]:
    lines: dict[tuple[int, int, int, int], list[tuple[str, float, int, int]]] = {}
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf") or -1)
            key = (int(row["page_num"]), int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
            top, left = int(row["top"]), int(row["left"])
        except (KeyError, ValueError):
            continue
        if conf < min_conf:
            continue
        lines.setdefault(key, []).append((text, conf, top, left))
    out = []
    for words in lines.values():
        words.sort(key=lambda w: w[3])
        text = join_words([w[0] for w in words])
        conf = sum(w[1] for w in words) / len(words)
        out.append(OcrLine(text, conf, min(w[2] for w in words), min(w[3] for w in words)))
    out.sort(key=lambda line: (line.top, line.left))
    return out


async def image_lines(image: Path, langs: list[str], psm: str = "6", min_conf: float = 60.0) -> list[OcrLine]:
    tsv = await run_cmd(
        [settings.tesseract, str(image), "stdout", "-l", lang_arg(langs), "--psm", str(psm), "tsv"],
        env={"OMP_THREAD_LIMIT": "1"}, error_code="ocr.failed",
    )
    return parse_tsv(tsv, min_conf)


async def image_text(image: Path, langs: list[str], psm: str = "3") -> str:
    return await run_cmd([settings.tesseract, str(image), "stdout", "-l", lang_arg(langs), "--psm", str(psm)],
                         env={"OMP_THREAD_LIMIT": "1"}, error_code="ocr.failed")


async def gather_limited(coros, limit: int):
    sem = asyncio.Semaphore(max(1, limit))

    async def guarded(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(guarded(c) for c in coros))
