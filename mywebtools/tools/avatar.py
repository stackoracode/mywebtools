"""Digital avatar picture-in-picture for vertical vlogs (YouTube Shorts).

Pipeline: iPhone footage (portrait or landscape, HDR or SDR)
  -> voice track (from the footage or a separate file, optionally trimmed)
  -> DUIX lip-syncs the chosen *base model video* to that voice
  -> the talking clip is masked/bordered and overlaid onto the footage.

Re-compositing an already generated clip ("Existing clip") costs no credits,
and "Preview frame" renders one composited frame with the base model video
as a stand-in so the layout can be tuned before spending credits.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from .. import composite, duix, subtitles
from ..composite import ANCHORS, SHAPES, Layout, Timing
from ..config import settings
from ..errors import ToolError
from ..i18n import WHISPER_LANGUAGES
from ..llm import EFFORTS, LLM
from ..media import (
    MediaInfo,
    extract_audio,
    ffmpeg_base,
    ffmpeg_caps,
    probe,
    progress_args,
    run_cmd,
    thumbnail,
    tonemap_chain,
    video_encoder_args,
)
from ..storage import new_id
from ..subtitles import TARGET_LANGUAGES, Cue
from .base import Group, Param, Tool, file_param, register
from .video_ocr import VIDEO_ACCEPT

AUDIO_ACCEPT = "audio/*,.wav,.mp3,.m4a,.aac,.flac"
OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
                 "marin", "cedar"]


def _labels(values: list[str]) -> list[list[str]]:
    return [[v, v.replace("_", " ").capitalize()] for v in values]


def layout_from(params: dict[str, Any]) -> Layout:
    names = {f.name for f in fields(Layout)}
    return Layout(**{k: v for k, v in params.items() if k in names})


def tonemap_for(info: MediaInfo, mode: str) -> tuple[str | None, str]:
    if mode == "off" or (mode == "auto" and not info.is_hdr):
        return None, ""
    chain = tonemap_chain()
    return chain, "" if chain else "avatar.no_tonemap"


async def conform_model(model: Path, cache: Path, *, fps: int, max_seconds: float, fit: str,
                        cloud: bool, height: int = 720) -> Path:
    """Prepare the base model video for the engine, cached next to the library item.

    cloud: 9:16 or 16:9, short side ``height`` px (DUIX API rules), H.264, no audio.
    local: keep the frame; only re-encode (H.264/SDR, short side <= 1080) when the
           source is HEVC / HDR / not MP4 / trimmed, else use the file as is.
    """
    if cache.is_file():
        return cache
    info = await probe(model)
    tm = tonemap_chain() if info.is_hdr else None
    if cloud:
        portrait = info.height >= info.width
        w, h = (height, composite.even(height * 16 / 9)) if portrait else (composite.even(height * 16 / 9), height)
        if fit == "pad":
            vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        else:
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    else:
        if (info.vcodec == "h264" and not info.is_hdr and model.suffix.lower() == ".mp4" and not max_seconds
                and min(info.width, info.height) <= 1080):
            return model
        short = min(info.width, info.height, 1080)
        vf = f"scale={-2 if info.width > info.height else short}:{short if info.width > info.height else -2}"
    if tm:
        vf = tm + "," + vf
    args = ffmpeg_base() + ["-i", str(model)]
    if max_seconds > 0:
        args += ["-t", f"{max_seconds:.2f}"]
    args += ["-vf", vf + ",setsar=1", "-r", str(fps), "-an"] + video_encoder_args("libx264", 18, "veryfast")
    tmp = cache.with_name(cache.stem + ".part.mp4")
    await run_cmd(args + ["-movflags", "+faststart", str(tmp)])
    tmp.replace(cache)
    return cache


@register
class AvatarPip(Tool):
    id = "avatar_pip"
    category = "avatar"
    icon = "user-video"
    name = "Digital avatar PiP"
    description = "Lip-sync a digital avatar to your voice and overlay it on your vlog (Shorts ready)."
    lane = "net"  # mostly waits on the cloud; CPU-heavy steps take the shared CPU slot
    actions = ("check", "preview", "voices")
    groups = [
        Group("source", "Source"),
        Group("canvas", "Output frame"),
        Group("place", "Avatar position & size"),
        Group("style", "Mask & style"),
        Group("crop", "Avatar crop", collapsed=True),
        Group("subs", "Subtitles"),
        Group("substyle", "Subtitle style", collapsed=True),
        Group("timing", "Timing & audio", collapsed=True),
        Group("encode", "Encoding", collapsed=True),
    ]

    def params(self) -> list[Param]:
        L = Layout()
        duix_only = {"avatar_source": "duix"}
        return [
            # --- source
            file_param("main_video", "Main video (your footage)", VIDEO_ACCEPT, group="source"),
            Param("avatar_source", "select", "Avatar", default="duix", group="source",
                  options=[["duix", "Generate with Duix"], ["clip", "Existing lip-sync clip (re-composite)"]]),
            Param("model", "library", "Base model video", library="avatar_models", group="source", show_if=duix_only,
                  default=settings.runtime.duix_default_model,
                  help="A short clip of the presenter: face visible, mouth closed, little head movement. "
                       "The default is set in Admin → Engine."),
            Param("clip", "library", "Lip-sync clip", library="avatar_clips", group="source",
                  show_if={"avatar_source": "clip"}),
            Param("audio_source", "select", "Voice", default="main", group="source", show_if=duix_only,
                  options=[["main", "Audio of the main video"], ["file", "Separate audio file"],
                           ["tts", "Type a script (AI voice)"]]),
            file_param("audio_file", "Voice audio file", AUDIO_ACCEPT, required=False, group="source",
                       show_if={"audio_source": "file"}),
            Param("audio_start", "number", "Voice from", default=0, min=0, step=0.1, unit="s", group="source",
                  show_if={"audio_source": ["main", "file"]}, help="Lip-sync only part of the voice (faster)."),
            Param("audio_end", "number", "Voice to", default=0, min=0, step=0.1, unit="s", group="source",
                  show_if={"audio_source": ["main", "file"]}, help="0 = until the end"),
            Param("script", "textarea", "Script", default="", group="source", remember=False,
                  help="Text the avatar speaks (AI voice) and/or the exact subtitle text."),
            Param("tts_voice", "text", "AI voice", default="", group="source", show_if={"audio_source": "tts"},
                  help="Empty = default from Admin → AI model (e.g. alloy, coral, marin)."),
            Param("tts_instructions", "textarea", "Speaking style", default="", group="source",
                  show_if={"audio_source": "tts"}, help="e.g. warm and upbeat vlogger, natural pace"),
            Param("tts_speed", "range", "Speech speed", default=1.0, min=0.5, max=2.0, step=0.05, group="source",
                  show_if={"audio_source": "tts"}),
            Param("credits_info", "info", "", group="source", show_if=duix_only),
            # --- subtitles
            Param("subtitles", "bool", "Add subtitles", default=True, group="subs"),
            Param("sub_source", "select", "Subtitle text from", default="auto", group="subs", show_if={"subtitles": True},
                  options=[["auto", "Auto (script if typed, else speech recognition)"], ["voice", "Speech recognition"],
                           ["script", "My script (timed to the voice)"]]),
            Param("sub_language", "select", "Spoken language", default="auto", group="subs",
                  show_if={"subtitles": True, "sub_source": ["auto", "voice"]},
                  options=[["auto", "Detect automatically"]] + [[c, c] for c in WHISPER_LANGUAGES]),
            Param("sub_ai", "select", "AI editing", default="fix", group="subs", show_if={"subtitles": True},
                  options=[["none", "None"], ["fix", "Fix recognition errors & punctuation"],
                           ["translate", "Translate"], ["bilingual", "Bilingual (original + translation)"]]),
            Param("sub_target", "select", "Translate to", default="en", group="subs",
                  show_if={"subtitles": True, "sub_ai": ["translate", "bilingual"]},
                  options=[[k, v] for k, v in TARGET_LANGUAGES.items()]),
            Param("sub_review", "bool", "Review the draft before rendering", default=True, group="subs",
                  show_if={"subtitles": True},
                  help="The job pauses at “Review”; edit the lines (or ask the AI to revise) and approve."),
            Param("sub_burn", "bool", "Burn into the video", default=True, group="subs", show_if={"subtitles": True},
                  help="An .srt file is always included."),
            Param("sub_context", "textarea", "Names & terms", default="", group="subs", show_if={"subtitles": True},
                  advanced=True, help="Helps recognition and AI editing spell names correctly."),
            Param("sub_effort", "select", "AI reasoning effort", default="", group="subs", show_if={"subtitles": True},
                  advanced=True, options=[["", "From settings"]] + [[e, e] for e in EFFORTS if e]),
            Param("sub_font", "text", "Font", default="", group="substyle", show_if={"subtitles": True},
                  help="Empty = default from settings (Noto Sans CJK SC covers Chinese/Japanese/Korean)."),
            Param("sub_size", "int", "Font size", default=64, min=16, max=200, unit="px", group="substyle",
                  show_if={"subtitles": True}),
            Param("sub_color", "color", "Text colour", default="#ffffff", group="substyle", show_if={"subtitles": True}),
            Param("sub_outline_color", "color", "Outline colour", default="#000000", group="substyle",
                  show_if={"subtitles": True, "sub_box": False}),
            Param("sub_outline", "int", "Outline width", default=4, min=0, max=20, unit="px", group="substyle",
                  show_if={"subtitles": True, "sub_box": False}),
            Param("sub_bold", "bool", "Bold", default=True, group="substyle", show_if={"subtitles": True}),
            Param("sub_box", "bool", "Background box", default=False, group="substyle", show_if={"subtitles": True}),
            Param("sub_box_color", "color", "Box colour", default="#000000", group="substyle",
                  show_if={"subtitles": True, "sub_box": True}),
            Param("sub_box_opacity", "int", "Box opacity", default=55, min=0, max=100, unit="%", group="substyle",
                  show_if={"subtitles": True, "sub_box": True}),
            Param("sub_position", "select", "Position", default="bottom", group="substyle", show_if={"subtitles": True},
                  options=[["bottom", "Bottom"], ["middle", "Middle"], ["top", "Top"]]),
            Param("sub_margin", "int", "Distance from edge", default=0, min=0, max=1500, unit="px", group="substyle",
                  show_if={"subtitles": True}, help="0 = automatic (above the Shorts caption area)"),
            Param("sub_max_chars", "int", "Max characters per line", default=0, min=0, max=80, group="substyle",
                  show_if={"subtitles": True}, help="0 = automatic from font size and frame width"),
            # --- canvas
            Param("canvas", "select", "Output size", default="shorts", group="canvas",
                  options=[["shorts", "Shorts 1080×1920 (9:16)"], ["shorts_720", "Shorts 720×1280"],
                           ["portrait_45", "Portrait 1080×1350 (4:5)"], ["square", "Square 1080×1080"],
                           ["landscape", "Landscape 1920×1080"], ["landscape_720", "Landscape 1280×720"],
                           ["source", "Same as main video"], ["custom", "Custom"]]),
            Param("canvas_w", "int", "Width", default=1080, min=240, max=3840, step=2, unit="px", group="canvas",
                  show_if={"canvas": "custom"}),
            Param("canvas_h", "int", "Height", default=1920, min=240, max=3840, step=2, unit="px", group="canvas",
                  show_if={"canvas": "custom"}),
            Param("fit", "select", "Fit main video", default="auto", group="canvas",
                  options=[["auto", "Auto (fill if same shape, else blurred background)"],
                           ["blur", "Fit on blurred background"], ["cover", "Fill and crop"],
                           ["color", "Fit on solid colour"]],
                  help="Landscape footage on a 9:16 canvas looks best on a blurred background."),
            Param("main_align", "select", "Letterboxed video position", default="center", group="canvas",
                  options=[["top", "Top"], ["center", "Center"], ["bottom", "Bottom"]],
                  show_if={"fit": ["blur", "color", "auto"]}),
            Param("blur", "int", "Background blur", default=L.blur, min=4, max=80, group="canvas",
                  show_if={"fit": ["blur", "auto"]}),
            Param("bg_color", "color", "Background colour", default=L.bg_color, group="canvas", show_if={"fit": "color"}),
            Param("hdr", "select", "HDR to SDR (iPhone HDR)", default="auto", group="canvas",
                  options=[["auto", "Auto-detect"], ["on", "Always"], ["off", "Never"]]),
            # --- placement
            Param("anchor", "select", "Position", default=L.anchor, options=_labels(ANCHORS), group="place"),
            Param("size", "int", "Avatar width", default=L.size, min=64, max=2160, step=2, unit="px", group="place"),
            Param("size_h", "int", "Avatar height", default=L.size_h, min=0, max=3840, step=2, unit="px", group="place",
                  help="0 = automatic (square, or source shape for 'Full frame' crop)"),
            Param("margin_x", "int", "Horizontal margin", default=L.margin_x, min=0, max=1000, unit="px", group="place"),
            Param("margin_y", "int", "Vertical margin", default=L.margin_y, min=0, max=1500, unit="px", group="place"),
            Param("safe_area", "bool", "Keep clear of Shorts buttons & captions", default=L.safe_area, group="place"),
            # --- style
            Param("shape", "select", "Mask shape", default=L.shape, options=_labels(SHAPES), group="style"),
            Param("radius", "int", "Corner radius", default=L.radius, min=0, max=1000, unit="px", group="style",
                  show_if={"shape": "rounded"}),
            Param("border_w", "int", "Border width", default=L.border_w, min=0, max=60, unit="px", group="style"),
            Param("border_color", "color", "Border colour", default=L.border_color, group="style"),
            Param("feather", "int", "Edge softness", default=L.feather, min=0, max=40, unit="px", group="style"),
            Param("shadow", "bool", "Drop shadow", default=L.shadow, group="style"),
            Param("shadow_blur", "int", "Shadow size", default=L.shadow_blur, min=2, max=80, unit="px", group="style",
                  show_if={"shadow": True}),
            Param("shadow_opacity", "int", "Shadow opacity", default=L.shadow_opacity, min=0, max=100, unit="%",
                  group="style", show_if={"shadow": True}),
            Param("opacity", "range", "Avatar opacity", default=L.opacity, min=10, max=100, unit="%", group="style"),
            Param("mirror", "bool", "Mirror avatar", default=L.mirror, group="style"),
            Param("chroma", "bool", "Remove green screen", default=L.chroma, group="style"),
            Param("key_color", "color", "Screen colour", default=L.key_color, group="style", show_if={"chroma": True}),
            Param("key_similarity", "range", "Key similarity", default=L.key_similarity, min=0.01, max=0.8, step=0.01,
                  group="style", show_if={"chroma": True}),
            Param("key_blend", "range", "Key edge blend", default=L.key_blend, min=0, max=0.5, step=0.01,
                  group="style", show_if={"chroma": True}),
            # --- crop
            Param("crop_mode", "select", "Crop of avatar video", default=L.crop_mode, group="crop",
                  options=[["auto", "Auto (head & shoulders)"], ["full", "Full frame"], ["custom", "Custom"]]),
            Param("crop_cx", "range", "Crop centre X", default=L.crop_cx, min=0, max=1, step=0.01, group="crop",
                  show_if={"crop_mode": "custom"}),
            Param("crop_cy", "range", "Crop centre Y", default=L.crop_cy, min=0, max=1, step=0.01, group="crop",
                  show_if={"crop_mode": "custom"}),
            Param("crop_zoom", "range", "Zoom", default=L.crop_zoom, min=1, max=4, step=0.05, group="crop",
                  show_if={"crop_mode": "custom"}),
            # --- timing & audio
            Param("avatar_delay", "number", "Avatar appears at", default=0, min=0, step=0.1, unit="s", group="timing",
                  help="Extra offset. With 'Voice from', the avatar starts at that time automatically."),
            Param("end_mode", "select", "When the avatar clip ends", default="hide", group="timing",
                  options=[["hide", "Hide avatar"], ["freeze", "Freeze last frame"], ["loop", "Loop clip"]]),
            Param("fade_in", "number", "Fade in", default=L.fade_in, min=0, max=5, step=0.1, unit="s", group="timing"),
            Param("fade_out", "number", "Fade out", default=L.fade_out, min=0, max=5, step=0.1, unit="s", group="timing"),
            Param("audio_mix", "select", "Final soundtrack", default="auto", group="timing",
                  options=[["auto", "Auto (main video; mixed with the AI voice)"], ["main", "Main video"], ["avatar", "Avatar clip"],
                           ["mix", "Mix both"], ["none", "Silent"]]),
            Param("main_volume", "range", "Main volume", default=1.0, min=0, max=2, step=0.05, group="timing",
                  show_if={"audio_mix": "mix"}),
            Param("avatar_volume", "range", "Avatar volume", default=1.0, min=0, max=2, step=0.05, group="timing",
                  show_if={"audio_mix": ["mix", "avatar"]}),
            Param("max_duration", "number", "Limit output length", default=0, min=0, step=1, unit="s", group="timing",
                  help="0 = full length. YouTube Shorts allow up to 180 s."),
            # --- encoding
            Param("fps", "select", "Frame rate", default="30", group="encode",
                  options=[["source", "Same as main"], ["24", "24"], ["25", "25"], ["30", "30"], ["60", "60"]]),
            Param("crf", "int", "Quality (CRF, lower = better)", default=20, min=14, max=32, group="encode"),
            Param("preset", "select", "Encoder speed", default="veryfast", group="encode",
                  options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"]),
            Param("encoder", "select", "Encoder", default="default", group="encode",
                  options=[["default", "From admin settings"], ["libx264", "libx264 (CPU)"],
                           ["h264_rkmpp", "h264_rkmpp (Rockchip HW)"], ["h264_v4l2m2m", "h264_v4l2m2m (V4L2 HW)"]]),
            Param("model_fit", "select", "Model video shape (cloud engine)", default="crop", group="encode",
                  show_if=duix_only, advanced=True,
                  options=[["crop", "Crop to 9:16 / 16:9"], ["pad", "Pad with black"]]),
            Param("model_max_seconds", "number", "Use first seconds of model video", default=0, min=0, step=1,
                  unit="s", group="encode", show_if=duix_only, help="0 = whole model video (smaller = faster upload)"),
            Param("save_clip", "bool", "Save lip-sync clip to library", default=True, group="encode", show_if=duix_only),
            Param("output_clip", "bool", "Also download the raw lip-sync clip", default=False, group="encode",
                  show_if=duix_only),
            Param("preview_at", "number", "Preview at", default=2, min=0, step=0.5, unit="s", group="encode"),
        ]

    # ------------------------------------------------------------------
    async def run(self, ctx, params):
        main = ctx.input("main_video")
        minfo = await probe(main)
        if not minfo.has_video:
            raise ToolError("media.no_video")
        library = ctx.manager.library
        delay = params["avatar_delay"]
        credits_used = 0
        notes: list[str] = []
        cw, ch = composite.canvas_size(layout_from(params), minfo.width, minfo.height)

        if params["avatar_source"] == "clip":
            if not params["clip"]:
                raise ToolError("avatar.no_clip")
            clip = library.path(library.get("avatar_clips", params["clip"]))
            cues, sub_offset = [], 0.0
            if params["subtitles"]:
                # subtitles follow the soundtrack of the main video (or the clip if the main one is silent)
                source, sub_offset = (main, 0.0) if minfo.has_audio else (clip, delay)
                voice = ctx.workdir / "voice.wav"
                await extract_audio(source, voice)
                cues = await self._subtitles(ctx, params, voice, (await probe(voice)).duration, cw, notes)
            scale = lambda p: 30 + p * 0.7  # noqa: E731
        else:
            if params["audio_source"] in ("main", "file"):
                delay += params["audio_start"]
            sub_offset = delay
            clip, cues, credits_used = await self._lipsync_with_subtitles(ctx, params, main, minfo, cw, notes)
            scale = lambda p: 75 + p * 0.25  # noqa: E731

        ass_name = None
        if cues:
            srt = ctx.out_path(f"{main.stem}_avatar.srt")
            srt.write_text(subtitles.to_srt(cues, sub_offset), encoding="utf-8")
            ctx.add_output(srt, "subtitles")
            if params["sub_burn"]:
                if "ass" in ffmpeg_caps()["filters"] or not ffmpeg_caps()["filters"]:
                    (ctx.workdir / "subs.ass").write_text(
                        subtitles.to_ass(cues, subtitles.style_from(params), cw, ch, sub_offset), encoding="utf-8")
                    ass_name = "subs.ass"
                else:
                    notes.append("subs.no_libass")

        out = ctx.out_path(f"{main.stem}_avatar.mp4")
        notes += await self._compose(ctx, params, main, minfo, clip, out, delay,
                                     lambda f: ctx.progress(scale(100 * f), "avatar.compose"), ass_name)
        ctx.add_output(out, "video")
        ctx.set_result({"credits_used": credits_used, "subtitles": len(cues), "notes": notes,
                        "duration": round(minfo.duration, 1)})

    async def _lipsync_with_subtitles(self, ctx, params, main: Path, minfo: MediaInfo, canvas_w: int,
                                      notes: list[str]) -> tuple[Path, list[Cue], int]:
        """Prepare the voice, then run Duix and the subtitle draft/review *concurrently*:
        the user reviews subtitles while the PC renders the lip-sync."""
        rt = settings.runtime
        library = ctx.manager.library
        if not params["model"]:
            raise ToolError("avatar.no_model")
        item = library.get("avatar_models", params["model"])
        engine = duix.engine()
        resume = ctx.meta.get("resume_duix_task", "") if ctx.meta.get("resume_engine") == engine.name else ""
        cloud = engine.name == "duix_cloud"

        voice = ctx.workdir / "voice.wav"
        async with ctx.manager.cpu_slot:
            ctx.progress(1, "avatar.conform")
            fps = rt.duix_upload_fps
            key = (f"{'cloud' if cloud else 'local'}_{rt.duix_upload_height}p_{fps}fps_{params['model_fit']}"
                   f"_{params['model_max_seconds']:g}.mp4")
            model = await conform_model(library.path(item), library.cache_path(item, key), fps=fps,
                                        max_seconds=params["model_max_seconds"], fit=params["model_fit"],
                                        cloud=cloud, height=rt.duix_upload_height)
        ctx.progress(4, "avatar.audio")
        if params["audio_source"] == "tts":
            ctx.progress(4, "avatar.tts")
            spoken = ctx.workdir / "tts.wav"
            await subtitles.synthesize(params["script"], spoken, ctx.workdir, voice=params["tts_voice"],
                                       instructions=params["tts_instructions"], speed=params["tts_speed"])
            await extract_audio(spoken, voice)
        else:
            source = main if params["audio_source"] == "main" else ctx.input("audio_file")
            if params["audio_source"] == "main" and not minfo.has_audio:
                raise ToolError("media.no_audio")
            await extract_audio(source, voice, start=params["audio_start"], end=params["audio_end"])
        voice_info = await probe(voice)
        account = await engine.account()
        cost = engine.check(account, voice_info.duration)
        ctx.log(f"voice {voice_info.duration:.1f}s, credits needed {cost['credits_needed']}, left {cost['credits_left']}")
        ctx.save_meta(_keep_uploads=[u for ids in ctx.job.files.values() for u in ids])

        clip = ctx.workdir / "avatar_clip.mp4"

        async def lipsync() -> Path:
            await engine.generate(
                model, voice, clip,
                progress=lambda p, stage, kw: ctx.progress(8 + 0.67 * p if p is not None else None, stage, **kw),
                save_task=lambda task: ctx.save_meta(resume_duix_task=task, resume_engine=engine.name),
                task_id=resume,
                model_name=f"mwt_model_{item.id}_{model.stat().st_size}{model.suffix}",
            )
            return clip

        async def subs() -> list[Cue]:
            if not params["subtitles"]:
                return []
            return await self._subtitles(ctx, params, voice, voice_info.duration, canvas_w, notes)

        tasks = [asyncio.create_task(lipsync()), asyncio.create_task(subs())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()  # re-raise the first failure
            await asyncio.gather(*pending)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        cues = tasks[1].result()

        if params["save_clip"]:
            cinfo = await probe(clip)
            saved = library.add("avatar_clips", clip, f"{item.name} · {time.strftime('%m-%d %H:%M')}",
                                {"duration": round(cinfo.duration, 1), "width": cinfo.width, "height": cinfo.height,
                                 "model": item.id, "job": ctx.job.id})
            if await thumbnail(library.path(saved), library.thumb_path(saved), at=min(1.0, cinfo.duration / 2)):
                library.update(saved, has_thumb=True)
            ctx.save_meta(clip_id=saved.id)
        if params["output_clip"]:
            raw = ctx.out_path(f"{main.stem}_lipsync.mp4")
            raw.write_bytes(clip.read_bytes())
            ctx.add_output(raw, "clip")
        return clip, cues, 0 if resume else cost["credits_needed"]

    async def _subtitles(self, ctx, params, voice: Path, duration: float, canvas_w: int,
                         notes: list[str]) -> list[Cue]:
        """Build the subtitle draft (speech recognition or script), optional AI edit, then review."""
        draft = ctx.meta.get("resume_review_draft")
        style = subtitles.style_from(params)
        script = params["script"].strip()
        source = params["sub_source"]
        if source == "auto":
            source = "script" if script else "voice"
        if draft and draft.get("cues"):
            cues = subtitles.cues_from(draft["cues"])  # resumed job: show the last edited draft again
        else:
            ctx.log(f"subtitles from {source}, asr engine {settings.runtime.asr_engine}")
            context = params["sub_context"]
            lang = params["sub_language"]
            if source == "script":
                if not script:
                    raise ToolError("subs.no_script")
                try:
                    segments = await subtitles.recognize(voice, lang, context)
                except ToolError as exc:
                    ctx.log(f"no speech recognition ({exc.code}): spreading the script over the voice")
                    notes.append("subs.proportional")
                    segments = []
                max_chars = style.max_chars or subtitles.auto_max_chars(script, canvas_w, style.size)
                cues = subtitles.align_script(subtitles.split_script(script, max_chars), segments, duration)
            else:
                segments = await subtitles.recognize(voice, lang, context)
                cues = [Cue(s["start"], s["end"], s["text"]) for s in segments]
            mode = params["sub_ai"]
            if mode == "fix" and source == "script":
                mode = "none"  # typed text is already exact
            if mode != "none" and cues:
                try:
                    cues = await subtitles.ai_edit(cues, mode, target=params["sub_target"], context=context,
                                                   effort=params["sub_effort"] or None)
                except ToolError as exc:
                    if mode in ("translate", "bilingual"):
                        raise
                    ctx.log(f"AI fix skipped: {exc.code} {exc.detail[:200]}")
                    notes.append("subs.ai_skipped")
            sample = " ".join(c.text for c in cues[:20])
            max_chars = style.max_chars or subtitles.auto_max_chars(sample, canvas_w, style.size)
            cues = subtitles.fit_cues(cues, max_chars)
        if params["sub_review"] and cues:
            result = await ctx.request_review("subtitles", subtitles.draft_payload(
                cues, source=source, ai=params["sub_ai"], target=params["sub_target"], canvas_w=canvas_w,
                max_chars=style.max_chars or subtitles.auto_max_chars(" ".join(c.text for c in cues[:20]),
                                                                       canvas_w, style.size)))
            cues = subtitles.cues_from(result.get("cues", []))
        return cues

    async def _compose(self, ctx, params, main: Path, minfo: MediaInfo, clip: Path, out: Path, delay: float,
                       on_progress, ass_name: str | None = None) -> list[str]:
        cinfo = await probe(clip)
        layout = layout_from(params)
        geo = composite.compute_geometry(layout, minfo.width, minfo.height, cinfo.width, cinfo.height)
        fps = minfo.fps if params["fps"] == "source" and minfo.fps else float(params["fps"] if params["fps"] != "source" else 30)
        timing = Timing(minfo.duration, cinfo.duration, delay, params["end_mode"], params["max_duration"], min(fps, 60))
        masks = composite.build_masks(layout, geo, ctx.workdir / "masks")
        tonemap, note = tonemap_for(minfo, params["hdr"])
        audio = params["audio_mix"]
        if audio == "auto":
            ai_voice = params["avatar_source"] == "duix" and params["audio_source"] == "tts"
            audio = ("mix" if ai_voice else "main") if minfo.has_audio else ("avatar" if cinfo.has_audio else "none")
        if audio in ("avatar", "mix") and not cinfo.has_audio:
            audio = "main" if minfo.has_audio else "none"
        if audio == "mix" and not minfo.has_audio:
            audio = "avatar"
        graph, maps = composite.build_filter(layout, geo, timing, masks, tonemap=tonemap, audio=audio,
                                             main_volume=params["main_volume"], avatar_volume=params["avatar_volume"],
                                             subtitles=ass_name)
        encoder = None if params["encoder"] == "default" else params["encoder"]
        args = ffmpeg_base() + composite.build_inputs(main, clip, masks, timing)
        args += ["-filter_complex", graph] + maps + ["-t", f"{timing.total:.3f}", "-r", f"{timing.fps:g}"]
        args += video_encoder_args(encoder, params["crf"], params["preset"])
        if audio != "none":
            args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        args += ["-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart"] + progress_args() + [str(out)]
        ctx.log(f"canvas {geo.canvas_w}x{geo.canvas_h} fit={geo.fit} avatar {geo.box} at {geo.pos} crop {geo.crop}")
        async with ctx.manager.cpu_slot:
            await run_cmd(args, duration=timing.total, on_progress=on_progress, cwd=ctx.workdir)
        return [n for n in geo.notes + [note] if n]

    # ------------------------------------------------------------------
    async def action(self, name, params, files, app):
        if name == "check":
            return await self._check(params, files, app)
        if name == "preview":
            return await self._preview(params, files, app)
        if name == "voices":
            return {"voices": OPENAI_VOICES, "default": settings.runtime.llm_tts_voice}
        return await super().action(name, params, files, app)

    async def _check(self, params, files, app) -> dict[str, Any]:
        out: dict[str, Any] = {"warnings": []}
        warn = out["warnings"].append
        main_ids = files.get("main_video") or []
        minfo = await probe(app.uploads.get(main_ids[0]).path) if main_ids else None
        if minfo:
            out["main"] = minfo.public()
            if minfo.is_hdr:
                warn("avatar.hdr_main" if tonemap_chain() else "avatar.no_tonemap")
            cw, ch = composite.canvas_size(layout_from(params), minfo.width, minfo.height)
            if composite.resolve_fit(params["fit"], minfo.width, minfo.height, cw, ch) != "cover":
                warn("avatar.letterbox")
        if params["avatar_source"] == "clip":
            if params["clip"]:
                item = app.library.get("avatar_clips", params["clip"])
                out["clip"] = (await probe(app.library.path(item))).public()
            self._check_subtitles(params, warn)
            return out
        if params["model"]:
            item = app.library.get("avatar_models", params["model"])
            model = await probe(app.library.path(item))
            out["model"] = model.public()
            if not duix.ratio_ok(model.width, model.height):
                warn("avatar.model_ratio")
            if min(model.width, model.height) < 720:
                warn("avatar.model_small")
        # voice length -> credits
        seconds = 0.0
        if params["audio_source"] == "file" and files.get("audio_file"):
            seconds = (await probe(app.uploads.get(files["audio_file"][0]).path)).duration
        elif minfo:
            seconds = minfo.duration
        end = params["audio_end"] or seconds
        seconds = max(0.0, min(end, seconds) - params["audio_start"]) if seconds else 0.0
        out["voice_seconds"] = round(seconds, 1)
        out["credits_needed"] = duix.credits_for(seconds) if seconds else 0
        try:
            account = await duix.engine().account()
            out["account"] = account
            if account.get("plan") == "local":
                out["credits_needed"] = 0
            if account.get("max_minutes") and seconds > account["max_minutes"] * 60:
                warn("duix.too_long")
            if account.get("credits_left") is not None and out["credits_needed"] > account["credits_left"]:
                warn("duix.no_credits")
        except ToolError as exc:
            out["account_error"] = exc.public()
        if params["audio_source"] == "tts":
            if not params["script"].strip():
                warn("subs.no_script")
            if not LLM().configured:
                warn("llm.not_configured")
        self._check_subtitles(params, warn)
        return out

    @staticmethod
    def _check_subtitles(params, warn) -> None:
        if not params["subtitles"]:
            return
        try:
            subtitles.asr_engine()
        except ToolError as exc:
            if not (params["sub_source"] == "script" and params["script"].strip()):
                warn(exc.code)
        if params["sub_ai"] != "none" and not LLM().configured:
            warn("llm.not_configured")
        if params["sub_burn"] and ffmpeg_caps()["filters"] and "ass" not in ffmpeg_caps()["filters"]:
            warn("subs.no_libass")

    async def _preview(self, params, files, app) -> dict[str, Any]:
        main_ids = files.get("main_video") or []
        if not main_ids:
            raise ToolError("input.missing", field="main_video")
        main = app.uploads.get(main_ids[0]).path
        if params["avatar_source"] == "clip" and params["clip"]:
            stand_in = app.library.path(app.library.get("avatar_clips", params["clip"]))
        elif params["model"]:
            stand_in = app.library.path(app.library.get("avatar_models", params["model"]))
        else:
            raise ToolError("avatar.no_model")
        minfo, cinfo = await probe(main), await probe(stand_in)
        layout = layout_from(params)
        geo = composite.compute_geometry(layout, minfo.width, minfo.height, cinfo.width, cinfo.height)
        folder = settings.tmp / f"preview_{new_id()}"
        masks = composite.build_masks(layout, geo, folder)
        at = min(max(params["preview_at"], 0), max(minfo.duration - 0.1, 0))
        timing = Timing(minfo.duration, cinfo.duration, 0.0, "freeze", 0, 30)
        tonemap, _ = tonemap_for(minfo, params["hdr"])
        ass_name = None
        if params["subtitles"] and params["sub_burn"] and "ass" in (ffmpeg_caps()["filters"] or {"ass"}):
            style = subtitles.style_from(params)
            sample = (params["script"].strip() or "Subtitle preview · 字幕预览 · This is how a two-line caption looks")
            max_chars = style.max_chars or subtitles.auto_max_chars(sample, geo.canvas_w, style.size)
            chunk = (subtitles.split_script(sample, max_chars) or [sample])[0]
            cue = subtitles.fit_cues([Cue(0, 3600, chunk)], max_chars)[:1]
            (folder / "subs.ass").write_text(subtitles.to_ass(cue, style, geo.canvas_w, geo.canvas_h), encoding="utf-8")
            ass_name = "subs.ass"
        graph, _ = composite.build_filter(layout, geo, timing, masks, tonemap=tonemap, preview=True, subtitles=ass_name)
        name = f"{folder.name}.jpg"
        args = ffmpeg_base() + composite.build_inputs(main, stand_in, masks, timing, preview_at=at)
        args += ["-filter_complex", graph, "-map", "[v]", "-frames:v", "1", "-q:v", "3", str(settings.tmp / name)]
        async with app.jobs.cpu_slot:
            await run_cmd(args, timeout=120, cwd=folder)
        return {"image": f"/api/previews/{name}", "canvas": [geo.canvas_w, geo.canvas_h], "fit": geo.fit,
                "box": list(geo.box), "pos": list(geo.pos), "notes": geo.notes}

