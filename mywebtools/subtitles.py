"""Subtitles: drafts from speech or a typed script, AI polish/translation, SRT/ASS.

Pipeline used by the avatar tool (and partly by speech-to-text):

  voice wav ──ASR──▶ timed segments ─┐
  typed script ──split──▶ sentences ─┴─▶ align ─▶ cues ─▶ (AI fix / translate) ─▶ review ─▶ SRT + burned ASS
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .errors import ToolError
from .i18n import is_cjk_char
from .llm import LLM
from .media import ffmpeg_base, run_cmd

TARGET_LANGUAGES = {
    "en": "English", "zh-Hans": "Simplified Chinese", "zh-Hant": "Traditional Chinese", "ja": "Japanese",
    "ko": "Korean", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
}
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…])|(?<=[.!?])\s+|\n+")


@dataclass
class Cue:
    start: float
    end: float
    text: str

    def public(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


def cues_from(data: list[dict[str, Any]]) -> list[Cue]:
    cues = []
    for item in data or []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if text and end > start:
            cues.append(Cue(max(0.0, start), end, text[:500]))
    return sorted(cues, key=lambda c: c.start)


def cjk_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    return sum(is_cjk_char(c) for c in letters) / len(letters) if letters else 0.0


def auto_max_chars(text: str, canvas_w: int = 1080, font_px: int = 64) -> int:
    """Characters per line that fit the frame (CJK glyphs are ~1 em, Latin ~0.5 em)."""
    usable = canvas_w * 0.86
    return max(8, int(usable / (font_px * (0.98 if cjk_ratio(text) > 0.3 else 0.52))))


def wrap(text: str, max_chars: int) -> list[str]:
    """Break one cue into lines of at most ``max_chars`` (CJK by char, others by word)."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    if cjk_ratio(text) > 0.3:
        lines, line = [], ""
        for ch in text:
            line += ch
            if len(line) >= max_chars:
                lines.append(line.strip())
                line = ""
        return [ln for ln in lines + [line.strip()] if ln]
    lines, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > max_chars:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def split_script(script: str, max_chars: int, max_lines: int = 2) -> list[str]:
    """Typed script -> subtitle-sized chunks (sentence boundaries first)."""
    chunks = []
    for sentence in _SENTENCE_END.split(script or ""):
        sentence = (sentence or "").strip()
        if not sentence:
            continue
        lines = wrap(sentence, max_chars)
        for i in range(0, len(lines), max_lines):
            chunks.append(" ".join(lines[i:i + max_lines]) if cjk_ratio(sentence) <= 0.3 else "".join(lines[i:i + max_lines]))
    return chunks


def fit_cues(cues: list[Cue], max_chars: int, max_lines: int = 2) -> list[Cue]:
    """Split cues that would need more than ``max_lines`` lines; time is shared by length."""
    out = []
    for cue in cues:
        lines = wrap(cue.text, max_chars)
        if len(lines) <= max_lines:
            out.append(Cue(cue.start, cue.end, "\n".join(lines)))
            continue
        groups = ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]
        total = sum(len(g) for g in groups) or 1
        t = cue.start
        for group in groups:
            span = (cue.end - cue.start) * len(group) / total
            out.append(Cue(t, t + span, group))
            t += span
    return out


def _weight(text: str) -> int:
    return len(re.sub(r"\s+", "", text)) or 1


def align_script(chunks: list[str], segments: list[dict[str, Any]], duration: float) -> list[Cue]:
    """Time typed text against recognised speech by relative text position.

    Speech segments give anchors (char offset -> time); script chunks are placed
    by their own char offsets scaled onto the recognised text. Without segments
    the chunks are spread over the whole voice proportionally.
    """
    if not chunks:
        return []
    anchors = []  # (char offset, time)
    pos = 0
    for seg in segments:
        anchors.append((pos, float(seg["start"])))
        pos += _weight(seg["text"])
        anchors.append((pos, float(seg["end"])))
    if not anchors:
        anchors = [(0, 0.0), (1, max(duration, 0.5))]
        pos = 1
    total_script = sum(_weight(c) for c in chunks)
    scale = pos / total_script

    def time_at(offset: float) -> float:
        q = offset * scale
        for (c0, t0), (c1, t1) in zip(anchors, anchors[1:], strict=False):
            if c0 <= q <= c1:
                return t0 if c1 == c0 else t0 + (q - c0) / (c1 - c0) * (t1 - t0)
        return anchors[-1][1]

    # nudge by a fraction of a character so a chunk that starts exactly at a pause
    # begins with the next speech segment and one that ends there stops with the previous
    cues, offset = [], 0
    for chunk in chunks:
        start = time_at(offset + 1e-6)
        offset += _weight(chunk)
        end = time_at(offset - 1e-6)
        cues.append(Cue(start, max(end, start + 0.4), chunk))
    for a, b in zip(cues, cues[1:], strict=False):  # no overlaps
        a.end = min(a.end, b.start) if b.start > a.start else a.end
    return cues


# ---------------------------------------------------------------------------
# speech recognition
# ---------------------------------------------------------------------------
def asr_engine() -> str:
    choice = settings.runtime.asr_engine
    local_ok = importlib.util.find_spec("faster_whisper") is not None
    api_ok = LLM().configured
    if choice == "local" or (choice == "auto" and local_ok):
        if not local_ok:
            raise ToolError("dependency.faster_whisper")
        return "local"
    if not api_ok:
        raise ToolError("subs.no_asr")
    return "api"


async def recognize(audio: Path, language: str = "", prompt: str = "") -> list[dict[str, Any]]:
    lang = None if language in ("", "auto") else language.split("-")[0]
    if asr_engine() == "api":
        return await LLM().transcribe(audio, lang, prompt)
    from .tools.transcribe import _load_model

    model = await asyncio.to_thread(_load_model, settings.runtime.whisper_model)

    def work():
        segments, _ = model.transcribe(str(audio), language=lang, vad_filter=True, beam_size=2,
                                       initial_prompt=prompt or None)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments if s.text.strip()]

    return await asyncio.to_thread(work)


# ---------------------------------------------------------------------------
# AI editing
# ---------------------------------------------------------------------------
_EDITOR = ("You are a professional subtitle editor. The user sends JSON {\"cues\": [strings]}. Reply with JSON "
           "{\"cues\": [strings]} containing EXACTLY the same number of items in the same order: item i is the edited "
           "version of input item i. Never merge, split, reorder, drop or add items. Each item is one on-screen "
           "subtitle, keep it short. Do not add quotes, numbering or timestamps.")
_TASKS = {
    "fix": "Correct speech-recognition mistakes (homophones, names, technical terms), punctuation and casing. "
           "Keep the original language, wording and meaning.",
    "translate": "Translate every item into {target}. Natural, concise, spoken style.",
    "revise": "Apply the user's instruction to every item where it is relevant.",
}


async def ai_edit(cues: list[Cue], mode: str, *, target: str = "", instruction: str = "", context: str = "",
                  effort: str | None = None, llm: LLM | None = None, batch: int = 60) -> list[Cue]:
    """Edit cue texts with the LLM; timing is kept. ``mode``: fix | translate | bilingual | revise."""
    if not cues:
        return cues
    llm = llm or LLM()
    task_mode = "translate" if mode == "bilingual" else mode
    target_name = TARGET_LANGUAGES.get(target, target or "English")
    system = _EDITOR + " " + _TASKS[task_mode].format(target=target_name)
    if target.startswith("zh-Hans"):
        system += " Use Simplified Chinese characters and Chinese punctuation."
    if context:
        system += f" Vocabulary and context: {context[:1500]}"
    if instruction:
        system += f" Instruction from the user: {instruction[:1500]}"
    out: list[Cue] = []
    for i in range(0, len(cues), batch):
        part = cues[i:i + batch]
        data = await llm.complete_json(system, json.dumps({"cues": [c.text for c in part]}, ensure_ascii=False),
                                       effort=effort)
        texts = data.get("cues") if isinstance(data, dict) else None
        if not isinstance(texts, list) or len(texts) != len(part):
            raise ToolError("llm.count_mismatch", expected=len(part), got=len(texts) if isinstance(texts, list) else 0)
        for cue, text in zip(part, texts, strict=True):
            text = str(text).strip() or cue.text
            out.append(Cue(cue.start, cue.end, f"{cue.text}\n{text}" if mode == "bilingual" else text))
    return out


# ---------------------------------------------------------------------------
# text to speech (long scripts are spoken in chunks and concatenated)
# ---------------------------------------------------------------------------
async def synthesize(script: str, dest: Path, workdir: Path, *, voice: str = "", instructions: str = "",
                     speed: float = 1.0, max_chars: int = 3000) -> Path:
    script = script.strip()
    if not script:
        raise ToolError("subs.no_script")
    chunks, current = [], ""
    for sentence in _SENTENCE_END.split(script):
        sentence = (sentence or "").strip()
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip() if cjk_ratio(sentence) <= 0.3 else current + sentence
    if current:
        chunks.append(current)
    llm = LLM()
    parts = []
    for n, chunk in enumerate(chunks):
        parts.append(await llm.speech(chunk, workdir / f"tts_{n:03d}.wav", voice=voice, instructions=instructions,
                                      speed=speed))
    if len(parts) == 1:
        parts[0].replace(dest)
        return dest
    inputs = [arg for p in parts for arg in ("-i", str(p))]
    graph = "".join(f"[{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[a]"
    await run_cmd(ffmpeg_base() + inputs + ["-filter_complex", graph, "-map", "[a]", str(dest)])
    return dest


# ---------------------------------------------------------------------------
# formats
# ---------------------------------------------------------------------------
def _ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(max(seconds, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue], offset: float = 0.0) -> str:
    return "\n".join(f"{n}\n{_ts(c.start + offset)} --> {_ts(c.end + offset)}\n{c.text}\n" for n, c in enumerate(cues, 1))


def parse_srt(text: str) -> list[Cue]:
    cues = []
    pattern = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + int(m.group(4).ljust(3, "0")) / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + int(m.group(8).ljust(3, "0")) / 1000
                body = "\n".join(lines[i + 1:]).strip()
                if body:
                    cues.append(Cue(start, end, body))
                break
    return cues


@dataclass
class SubStyle:
    font: str = "Noto Sans CJK SC"
    size: int = 64
    color: str = "#ffffff"
    outline_color: str = "#000000"
    outline: int = 4
    bold: bool = True
    box: bool = False
    box_color: str = "#000000"
    box_opacity: int = 55
    position: str = "bottom"  # bottom | middle | top
    margin_v: int = 0  # 0 = automatic (above the Shorts caption area on 9:16)
    max_chars: int = 0  # 0 = automatic from font size and frame width


def _ass_color(hex_color: str, alpha_pct: int = 0) -> str:
    value = (hex_color or "#ffffff").lstrip("#").ljust(6, "0")[:6]
    r, g, b = value[0:2], value[2:4], value[4:6]
    alpha = max(0, min(255, round(255 * alpha_pct / 100)))
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def margin_for(style: SubStyle, width: int, height: int) -> int:
    if style.margin_v > 0:
        return style.margin_v
    if style.position == "middle":
        return 0
    if height > width:  # vertical video: stay clear of the Shorts title/caption area or the top bar
        return int(height * (0.24 if style.position == "bottom" else 0.10))
    return int(height * 0.06)


def to_ass(cues: list[Cue], style: SubStyle, width: int, height: int, offset: float = 0.0) -> str:
    align = {"bottom": 2, "middle": 5, "top": 8}.get(style.position, 2)
    border_style = 3 if style.box else 1
    outline_col = _ass_color(style.box_color if style.box else style.outline_color, 100 - style.box_opacity if style.box else 0)
    back_col = _ass_color(style.box_color, 100 - style.box_opacity if style.box else 100)
    outline = max(4, style.size // 6) if style.box else style.outline
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font.replace(',', ' ')},{style.size},{_ass_color(style.color)},&H000000FF,{outline_col},"
        f"{back_col},{-1 if style.bold else 0},0,0,0,100,100,0,0,{border_style},{outline},0,{align},"
        f"{int(width * 0.06)},{int(width * 0.06)},{margin_for(style, width, height)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def ass_time(seconds: float) -> str:
        cs = int(round(max(seconds, 0) * 100))
        h, cs = divmod(cs, 360000)
        m, cs = divmod(cs, 6000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def escape(text: str) -> str:
        return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", "\\N")

    lines = [f"Dialogue: 0,{ass_time(c.start + offset)},{ass_time(c.end + offset)},Default,,0,0,0,,{escape(c.text)}"
             for c in cues]
    return head + "\n".join(lines) + "\n"


def style_from(params: dict[str, Any]) -> SubStyle:
    return SubStyle(
        font=params.get("sub_font") or settings.runtime.subtitle_font, size=int(params.get("sub_size", 64)),
        color=params.get("sub_color", "#ffffff"), outline_color=params.get("sub_outline_color", "#000000"),
        outline=int(params.get("sub_outline", 4)), bold=bool(params.get("sub_bold", True)),
        box=bool(params.get("sub_box", False)), box_color=params.get("sub_box_color", "#000000"),
        box_opacity=int(params.get("sub_box_opacity", 55)), position=params.get("sub_position", "bottom"),
        margin_v=int(params.get("sub_margin", 0)), max_chars=int(params.get("sub_max_chars", 0)),
    )


def draft_payload(cues: list[Cue], **extra: Any) -> dict[str, Any]:
    return {"cues": [c.public() for c in cues], **extra}

