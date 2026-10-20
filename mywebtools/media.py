"""Async wrappers around ffmpeg / ffprobe / any external command."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import settings
from .errors import ToolError

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG (iPhone default) and PQ
_KV = re.compile(r"^\w+=\S*\s*$")


async def run_cmd(
    args: list[str],
    *,
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    error_code: str = "media.failed",
) -> str:
    """Run a command without blocking the loop; kill it if the job is cancelled.

    For ffmpeg, pass ``duration`` + ``on_progress`` and the command must contain
    ``-progress pipe:1``; progress lines are parsed into a 0..1 fraction.
    """
    proc_env = {**os.environ, **(env or {})}
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=proc_env
        )
    except FileNotFoundError as exc:
        raise ToolError("dependency.missing", name=Path(args[0]).name) from exc

    stdout_lines: list[str] = []
    stderr_tail: list[str] = []

    async def read_stdout() -> None:
        assert proc.stdout
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace")
            if on_progress and duration and line.startswith(("out_time_us=", "out_time_ms=")):
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                    on_progress(max(0.0, min(1.0, seconds / duration)))
                except ValueError:
                    pass
            elif not (on_progress and _KV.match(line)):
                stdout_lines.append(line)

    async def read_stderr() -> None:
        assert proc.stderr
        async for raw in proc.stderr:
            stderr_tail.append(raw.decode("utf-8", "replace"))
            del stderr_tail[:-40]

    async def kill() -> None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), 5)

    try:
        await asyncio.wait_for(asyncio.gather(read_stdout(), read_stderr(), proc.wait()), timeout)
    except asyncio.TimeoutError:
        await kill()
        raise ToolError("media.timeout", tool=Path(args[0]).name) from None
    except BaseException:
        await kill()
        raise
    if proc.returncode:
        raise ToolError(error_code, "".join(stderr_tail)[-1500:], tool=Path(args[0]).name)
    return "".join(stdout_lines)


def ffmpeg_base() -> list[str]:
    return [settings.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]


def progress_args() -> list[str]:
    return ["-progress", "pipe:1", "-nostats"]


@dataclass
class MediaInfo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    pix_fmt: str = ""
    color_transfer: str = ""
    rotation: int = 0
    has_video: bool = False
    has_audio: bool = False
    sample_rate: int = 0
    size: int = 0
    format: str = ""

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def orientation(self) -> str:
        if not self.width:
            return ""
        return "portrait" if self.height > self.width else "landscape" if self.width > self.height else "square"

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(is_hdr=self.is_hdr, orientation=self.orientation)
        return data


def _rotation(stream: dict[str, Any]) -> int:
    rotate = stream.get("tags", {}).get("rotate")
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rotate = side["rotation"]
    try:
        return int(float(rotate or 0)) % 360
    except (TypeError, ValueError):
        return 0


def _fps(value: str) -> float:
    try:
        num, _, den = (value or "0/1").partition("/")
        return round(float(num) / float(den or 1), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_probe(data: dict[str, Any], size: int = 0) -> MediaInfo:
    info = MediaInfo(size=size)
    fmt = data.get("format", {})
    info.format = fmt.get("format_name", "")
    with contextlib.suppress(TypeError, ValueError):
        info.duration = float(fmt.get("duration") or 0)
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and not info.has_video and stream.get("disposition", {}).get("attached_pic", 0) == 0:
            info.has_video = True
            info.vcodec = stream.get("codec_name", "")
            info.pix_fmt = stream.get("pix_fmt", "")
            info.color_transfer = stream.get("color_transfer", "")
            info.rotation = _rotation(stream)
            w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
            # iPhone portrait clips are stored landscape + a rotation flag; ffmpeg
            # auto-rotates when decoding, so report the *display* size.
            if info.rotation in (90, 270):
                w, h = h, w
            info.width, info.height = w, h
            info.fps = _fps(stream.get("avg_frame_rate")) or _fps(stream.get("r_frame_rate"))
            if not info.duration:
                with contextlib.suppress(TypeError, ValueError):
                    info.duration = float(stream.get("duration") or 0)
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.acodec = stream.get("codec_name", "")
            with contextlib.suppress(TypeError, ValueError):
                info.sample_rate = int(stream.get("sample_rate") or 0)
    return info


async def probe(path: Path) -> MediaInfo:
    out = await run_cmd(
        [settings.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        error_code="media.probe_failed",
    )
    try:
        data = json.loads(out)
    except ValueError as exc:
        raise ToolError("media.probe_failed") from exc
    info = parse_probe(data, path.stat().st_size if path.exists() else 0)
    if not info.has_video and not info.has_audio:
        raise ToolError("media.no_streams")
    return info


async def thumbnail(src: Path, dest: Path, at: float = 1.0, width: int = 360) -> bool:
    try:
        await run_cmd(ffmpeg_base() + ["-ss", f"{at:.2f}", "-i", str(src), "-frames:v", "1",
                                       "-vf", f"scale={width}:-2", "-q:v", "4", str(dest)])
    except ToolError:
        try:
            await run_cmd(ffmpeg_base() + ["-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)])
        except ToolError:
            return False
    return dest.is_file()


@lru_cache(maxsize=1)
def ffmpeg_caps() -> dict[str, Any]:
    """Which filters/encoders this ffmpeg build has (Ubuntu 20.04 ships 4.2)."""
    import subprocess

    caps: dict[str, Any] = {"available": False, "version": "", "filters": set(), "encoders": set()}
    exe = shutil.which(settings.ffmpeg) or settings.ffmpeg
    try:
        version = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=15)
        filters = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15)
        encoders = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return caps
    caps["available"] = version.returncode == 0
    first = version.stdout.splitlines()[0] if version.stdout else ""
    caps["version"] = first.replace("ffmpeg version ", "").split(" ")[0]
    caps["filters"] = {m.group(1) for m in re.finditer(r"^\s*[TSC.]{2,3}\s+(\w+)\s", filters.stdout, re.M)}
    caps["encoders"] = {m.group(1) for m in re.finditer(r"^\s*V[\w.]{5}\s+(\w+)\s", encoders.stdout, re.M)}
    return caps


def video_encoder_args(encoder: str | None = None, crf: int | None = None, preset: str | None = None) -> list[str]:
    rt = settings.runtime
    encoder = encoder or rt.video_encoder
    if encoder not in ffmpeg_caps()["encoders"] and ffmpeg_caps()["encoders"]:
        encoder = "libx264"
    if encoder == "libx264":
        args = ["-c:v", "libx264", "-preset", preset or rt.x264_preset, "-crf", str(crf if crf is not None else rt.x264_crf),
                "-profile:v", "high"]
    else:  # hardware encoders take a bitrate instead of CRF
        args = ["-c:v", encoder, "-b:v", rt.hw_bitrate]
    if rt.ffmpeg_threads:
        args += ["-threads", str(rt.ffmpeg_threads)]
    return args + ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


def tonemap_chain() -> str | None:
    """HDR (HLG/PQ) -> SDR BT.709. Needs zscale (libzimg); None if unavailable."""
    if "zscale" not in ffmpeg_caps()["filters"] or "tonemap" not in ffmpeg_caps()["filters"]:
        return None
    return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


async def extract_audio(src: Path, dest: Path, *, start: float = 0.0, end: float = 0.0, rate: int = 16000,
                        channels: int = 1, codec: str = "pcm_s16le", extra: list[str] | None = None,
                        duration: float | None = None, on_progress: Callable[[float], None] | None = None) -> Path:
    args = ffmpeg_base()
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    if end and end > start:
        args += ["-to", f"{end:.3f}"] if start <= 0 else ["-t", f"{end - start:.3f}"]
    args += ["-i", str(src), "-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", str(channels), "-ar", str(rate), "-c:a", codec]
    args += (extra or []) + ["-map_metadata", "-1"] + progress_args() + [str(dest)]
    await run_cmd(args, duration=duration, on_progress=on_progress, error_code="media.no_audio")
    return dest
