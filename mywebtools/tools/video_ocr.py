"""On-screen text from video (hard subtitles, captions, slides) via frame OCR."""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from .. import ocr
from ..errors import ToolError
from ..media import ffmpeg_base, probe, progress_args, run_cmd
from ..textclean import clean_text
from .base import Group, Param, Tool, file_param, register
from .pdf_tools import ascii_flag, ocr_params

VIDEO_ACCEPT = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v"

REGIONS = {
    "full": (0.0, 1.0),
    "bottom_third": (0.66, 1.0),
    "bottom_quarter": (0.75, 1.0),
    "middle": (0.33, 0.66),
    "top_third": (0.0, 0.34),
}


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def similar(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return a == b
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def merge_cues(frames: list[tuple[float, str]], interval: float, threshold: float) -> list[tuple[float, float, str]]:
    """Consecutive frames with (nearly) the same text become one subtitle cue."""
    cues: list[list] = []
    for t, text in frames:
        if not text:
            continue
        if cues and similar(cues[-1][2], text, threshold) and t - cues[-1][1] <= interval * 1.5 + 0.01:
            cues[-1][1] = t + interval
            if len(text) > len(cues[-1][2]):
                cues[-1][2] = text  # keep the most complete reading
        else:
            cues.append([t, t + interval, text])
    return [(a, b, c) for a, b, c in cues]


def unique_lines(cues: list[tuple[float, float, str]], threshold: float) -> list[str]:
    seen: list[str] = []
    for _, _, text in cues:
        for line in text.splitlines():
            if line and not any(similar(line, s, threshold) for s in seen):
                seen.append(line)
    return seen


@register
class VideoOcr(Tool):
    id = "video_ocr"
    category = "video"
    icon = "scan-text"
    name = "Video on-screen text (OCR)"
    description = "Read burned-in subtitles, captions or slides from a video into TXT / SRT."
    groups = [Group("main", ""), Group("ocr", "OCR"), Group("frames", "Frames", collapsed=True)]

    def params(self):
        return [
            file_param("video", "Videos", VIDEO_ACCEPT, multiple=True, batch=True),
            Param("region", "select", "Text area", default="full",
                  options=[["full", "Whole frame"], ["bottom_third", "Bottom third (subtitles)"],
                           ["bottom_quarter", "Bottom quarter"], ["middle", "Middle band"],
                           ["top_third", "Top third"], ["custom", "Custom"]]),
            Param("region_top", "int", "Area top", default=70, min=0, max=99, unit="%", show_if={"region": "custom"}),
            Param("region_bottom", "int", "Area bottom", default=100, min=1, max=100, unit="%",
                  show_if={"region": "custom"}),
            Param("output", "multiselect", "Outputs", default=["txt", "srt"],
                  options=[["txt", "Unique lines (TXT)"], ["srt", "Timed subtitles (SRT)"]]),
            Param("interval", "number", "Sample every", default=1.0, min=0.2, max=10, step=0.1, unit="s", group="frames"),
            Param("min_conf", "int", "Minimum confidence", default=60, min=0, max=100, group="ocr"),
            Param("psm", "select", "Layout mode", default="6", options=ocr.PSM_OPTIONS, group="ocr", advanced=True),
            Param("similarity", "int", "Merge similar text", default=85, min=50, max=100, unit="%", group="frames",
                  help="Readings this similar count as the same line / cue."),
            Param("upscale", "bool", "Upscale small text 2x", default=False, group="frames"),
            Param("invert", "bool", "Light text on dark background", default=False, group="frames", advanced=True),
            *ocr_params(default_dates=True),
        ]

    def availability(self):
        return (True, "") if ocr.available() else (False, "dependency.tesseract")

    async def run(self, ctx, params):
        source = ctx.input("video")
        info = await probe(source)
        if not info.has_video:
            raise ToolError("media.no_video")
        interval = params["interval"]
        top, bottom = REGIONS.get(params["region"], (params["region_top"] / 100, params["region_bottom"] / 100))
        if bottom <= top:
            raise ToolError("ocr.bad_region")
        frames_dir = ctx.workdir / "frames"
        frames_dir.mkdir()
        vf = [f"fps=1/{interval:g}"]
        if (top, bottom) != (0.0, 1.0):
            vf.append(f"crop=iw:trunc(ih*{bottom - top:.4f}/2)*2:0:trunc(ih*{top:.4f}/2)*2")
        if params["upscale"]:
            vf.append("scale=iw*2:-2:flags=lanczos")
        vf.append("format=gray")
        ctx.progress(0, "ocr.sampling")
        await run_cmd(ffmpeg_base() + ["-i", str(source), "-vf", ",".join(vf), "-an", "-sn"] + progress_args()
                      + [str(frames_dir / "f_%06d.png")],
                      duration=info.duration, on_progress=lambda f: ctx.progress(25 * f, "ocr.sampling"))
        frames = sorted(frames_dir.glob("f_*.png"))
        if not frames:
            raise ToolError("ocr.no_frames")

        # Skip OCR for frames identical to the previous one (static captions are common).
        prev = None
        unique: list[tuple[int, Path]] = []
        mapping: list[int] = []
        for index, frame in enumerate(frames):
            small = await asyncio.to_thread(lambda p=frame: Image.open(p).convert("L").resize((96, 54)))
            if params["invert"]:
                await asyncio.to_thread(lambda p=frame: ImageOps.invert(Image.open(p).convert("L")).save(p))
            if prev is not None and _diff(small, prev) < 2.5:
                mapping.append(len(unique) - 1)
            else:
                unique.append((index, frame))
                mapping.append(len(unique) - 1)
            prev = small

        done = 0
        langs = params["ocr_langs"] or ["eng"]

        async def read(frame: Path) -> str:
            nonlocal done
            ctx.check_cancel()
            lines = await ocr.image_lines(frame, langs, params["psm"], params["min_conf"])
            done += 1
            ctx.progress(25 + 70 * done / len(unique), "ocr.reading", frame=done, frames=len(unique))
            text = "\n".join(line.text for line in lines)
            return clean_text(text, ascii_only=ascii_flag(params), remove_dates=params["remove_dates"])

        texts = await ocr.gather_limited([read(f) for _, f in unique], ctx.settings.runtime.ocr_parallel)
        timeline = [(i * interval, texts[mapping[i]]) for i in range(len(frames))]
        threshold = params["similarity"] / 100
        cues = merge_cues(timeline, interval, threshold)
        outputs = params["output"] or ["txt"]
        if "txt" in outputs:
            path = ctx.out_path(f"{source.stem}_ocr.txt")
            path.write_text("\n".join(unique_lines(cues, threshold)) + "\n", encoding="utf-8")
            ctx.add_output(path, "text")
        if "srt" in outputs:
            path = ctx.out_path(f"{source.stem}_ocr.srt")
            body = "\n".join(f"{n}\n{srt_time(a)} --> {srt_time(min(b, info.duration or b))}\n{text}\n"
                             for n, (a, b, text) in enumerate(cues, 1))
            path.write_text(body, encoding="utf-8")
            ctx.add_output(path, "subtitles")
        ctx.set_result({"frames": len(frames), "ocr_frames": len(unique), "cues": len(cues)})


def _diff(a: Image.Image, b: Image.Image) -> float:
    hist = ImageChops.difference(a, b).histogram()
    total = sum(hist) or 1
    return sum(i * n for i, n in enumerate(hist)) / total
