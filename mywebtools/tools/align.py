"""Time an existing script to speech (forced alignment with the MMS aligner on the GPU PC)."""

from __future__ import annotations

import json

from .. import speech, subtitles
from ..errors import ToolError
from ..i18n import WHISPER_LANGUAGES
from ..media import extract_audio, probe
from .base import Param, Tool, file_param, register
from .transcribe import MEDIA_ACCEPT, vtt_time


@register
class AlignSubtitles(Tool):
    id = "align_subs"
    category = "video"
    icon = "captions"
    name = "Time a script to speech"
    description = "Exact subtitle text from your script, timed to the voice (MMS forced alignment)."
    lane = "net"  # waits on the GPU PC

    def params(self):
        return [
            file_param("media", "Video or audio", MEDIA_ACCEPT),
            Param("script", "textarea", "Script", default="", remember=False,
                  help="What is said, in order. Or upload a .txt/.srt below."),
            file_param("script_file", "Script file (.txt or .srt)", ".txt,.srt,text/plain", required=False,
                       help="An .srt keeps its lines and gets new times."),
            Param("language", "select", "Spoken language", default="auto",
                  options=[["auto", "Detect from the text"]] + [[c, c] for c in WHISPER_LANGUAGES]),
            Param("engine", "select", "Timing", default="auto",
                  options=[["auto", "Auto (forced alignment if available)"], ["mms", "MMS forced alignment (GPU PC)"],
                           ["basic", "Speech recognition + proportional"]]),
            Param("max_chars", "int", "Max characters per line", default=42, min=8, max=120,
                  help="Script lines are split into subtitles of at most two such lines."),
            Param("formats", "multiselect", "Outputs", default=["srt"],
                  options=[["srt", "SRT"], ["vtt", "VTT"], ["json", "JSON (with word timings)"]]),
        ]

    async def run(self, ctx, params):
        source = ctx.input("media")
        info = await probe(source)
        if not info.has_audio:
            raise ToolError("media.no_audio")
        script_file = ctx.input("script_file", required=False)
        if script_file:
            raw = script_file.read_text(encoding="utf-8-sig", errors="replace")
            parsed = subtitles.parse_srt(raw) if script_file.suffix.lower() == ".srt" else []
            chunks = [c.text.replace("\n", " ") for c in parsed] or subtitles.split_script(raw, params["max_chars"])
        else:
            chunks = subtitles.split_script(params["script"], params["max_chars"])
        if not chunks:
            raise ToolError("subs.no_script")
        ctx.progress(2, "stt.audio")
        wav = await extract_audio(source, ctx.workdir / "voice.wav", duration=info.duration)
        ctx.progress(10, "align.running")
        notes: list[str] = []
        cues, words = await speech.align_script(chunks, wav, info.duration, language=params["language"],
                                                engine=params["engine"], notes=notes, log=ctx.log)
        cues = subtitles.fit_cues(cues, params["max_chars"])
        stem = source.stem
        formats = params["formats"] or ["srt"]
        if "srt" in formats:
            path = ctx.out_path(f"{stem}.srt")
            path.write_text(subtitles.to_srt(cues), encoding="utf-8")
            ctx.add_output(path, "subtitles")
        if "vtt" in formats:
            path = ctx.out_path(f"{stem}.vtt")
            path.write_text("WEBVTT\n\n" + "\n".join(f"{vtt_time(c.start)} --> {vtt_time(c.end)}\n{c.text}\n" for c in cues),
                            encoding="utf-8")
            ctx.add_output(path, "subtitles")
        if "json" in formats:
            path = ctx.out_path(f"{stem}.json")
            path.write_text(json.dumps({"cues": [c.public() for c in cues], "words": words}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            ctx.add_output(path, "json")
        ctx.set_result({"cues": len(cues), "duration": round(info.duration, 1), "notes": notes,
                        "engine": "mms" if words else "basic"})
