"""Text to speech / voice cloning (IndexTTS 2.5 on the GPU PC, or an OpenAI voice).

The voice parameters are shared with the avatar tool (``voice_params``), so the
same settings that work here work in the avatar flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import speech, subtitles
from ..errors import ToolError
from ..media import extract_audio, ffmpeg_base, probe, run_cmd
from ..nodes import Node
from .base import Group, Param, Tool, file_param, register

AUDIO_ACCEPT = "audio/*,video/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.mp4,.mov"
OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
                 "marin", "cedar"]
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def voice_params(group: str, ref_options: list[list[str]], when: dict[str, Any] | None = None,
                 emotion_group: str | None = None) -> list[Param]:
    """Engine, reference voice, language, emotion and pace (``when`` = extra show_if conditions)."""
    base = dict(when or {})
    ix = {**base, "tts_engine": "indextts"}
    oa = {**base, "tts_engine": "openai"}
    emo = emotion_group or group
    return [
        Param("tts_engine", "select", "Voice engine", default="indextts", group=group, show_if=base or None,
              options=[["indextts", "IndexTTS 2.5 voice clone (GPU PC)"], ["openai", "OpenAI voice (AI API)"]]),
        Param("tts_ref", "select", "Voice to clone", default=ref_options[0][0], group=group, show_if=ix,
              options=ref_options, help="3–15 s of clean speech works best; only the first 15 s are used."),
        Param("tts_ref_item", "library", "Voice sample", library="voices", group=group,
              show_if={**ix, "tts_ref": "library"}),
        Param("tts_lang", "select", "Language", default="auto", group=group, show_if=ix,
              options=[["auto", "Detect from the text"], ["ZH", "Chinese"], ["EN", "English"], ["JA", "Japanese"],
                       ["ES", "Spanish"], ["AR", "Arabic"]]),
        Param("tts_duration", "range", "Speech length", default=1.0, min=0.5, max=2.0, step=0.05, group=group,
              show_if=ix, help="1 = natural; above 1 = slower and longer, below 1 = faster."),
        Param("tts_emotion", "select", "Emotion", default="speaker", group=emo, show_if=ix,
              options=[["speaker", "Like the voice sample"], ["text", "Detect from the text (Qwen emotion model)"],
                       ["vector", "Set by hand"], ["audio", "From another recording"]]),
        Param("tts_emo_item", "library", "Emotion recording", library="voices", group=emo,
              show_if={**ix, "tts_emotion": "audio"}),
        Param("tts_emo_text", "text", "Emotion description", default="", group=emo, show_if={**ix, "tts_emotion": "text"},
              help="Empty = the script itself, e.g. “excited, a little nervous”."),
        Param("tts_emo_alpha", "range", "Emotion strength", default=0.65, min=0, max=1, step=0.05, group=emo,
              show_if={**ix, "tts_emotion": ["text", "audio"]}),
        *[Param(f"emo_{name}", "range", name.capitalize(), default=0.0, min=0, max=1, step=0.05, group=emo,
                show_if={**ix, "tts_emotion": "vector"}) for name in EMOTIONS],
        Param("tts_random", "bool", "More varied delivery", default=False, group=emo, show_if=ix, advanced=True,
              help="Random sampling: livelier, but less like the voice sample."),
        Param("tts_voice", "text", "AI voice", default="", group=group, show_if=oa,
              help="Empty = default from Admin → AI model (e.g. alloy, coral, marin)."),
        Param("tts_instructions", "textarea", "Speaking style", default="", group=group, show_if=oa,
              help="e.g. warm and upbeat vlogger, natural pace"),
        Param("tts_speed", "range", "Speech speed", default=1.0, min=0.5, max=2.0, step=0.05, group=group, show_if=oa),
    ]


def indextts_options(params: dict[str, Any]) -> dict[str, Any]:
    opts: dict[str, Any] = {"lang": params.get("tts_lang", "auto"), "emotion": params.get("tts_emotion", "speaker"),
                            "duration_factor": params.get("tts_duration", 1.0),
                            "use_random": params.get("tts_random", False)}
    if opts["emotion"] in ("text", "audio"):
        opts["emo_alpha"] = params.get("tts_emo_alpha", 0.65)
    if opts["emotion"] == "text":
        opts["emo_text"] = params.get("tts_emo_text", "")
    if opts["emotion"] == "vector":
        opts["emo_vector"] = [float(params.get(f"emo_{name}", 0.0)) for name in EMOTIONS]
        if not any(opts["emo_vector"]):
            opts["emotion"] = "speaker"
    return opts


def tts_language(params: dict[str, Any], text: str) -> str:
    """Language code (en, zh, ...) of the synthesised speech, for alignment."""
    return speech.TTS_LANGS.get(params.get("tts_lang", "auto"), "") or speech.guess_language(text)


async def synthesize_from(ctx, params: dict[str, Any], text: str, dest: Path, sources: dict[str, Path],
                          on_progress=None) -> Path:
    """Run the chosen voice engine. ``sources``: reference choices besides the library (model/main/upload)."""
    library = ctx.manager.library
    kwargs: dict[str, Any] = {"engine": params["tts_engine"], "voice": params["tts_voice"],
                              "instructions": params["tts_instructions"], "speed": params["tts_speed"]}
    if params["tts_engine"] == "indextts":
        choice = params["tts_ref"]
        if choice == "library":
            if not params["tts_ref_item"]:
                raise ToolError("tts.no_reference")
            source = library.path(library.get("voices", params["tts_ref_item"]))
        elif choice in sources:
            source = sources[choice]
        else:
            raise ToolError("tts.no_reference")
        try:
            kwargs["reference"] = await speech.voice_sample(source, ctx.workdir / "voice_ref.wav")
        except ToolError as exc:
            raise ToolError("tts.reference_no_audio", exc.detail) from exc
        if params["tts_emotion"] == "audio":
            if not params["tts_emo_item"]:
                raise ToolError("tts.no_emotion_audio")
            kwargs["emo_reference"] = await speech.voice_sample(
                library.path(library.get("voices", params["tts_emo_item"])), ctx.workdir / "emotion_ref.wav")
        kwargs["options"] = indextts_options(params)
        ctx.log(f"IndexTTS: voice from {choice}, options {kwargs['options']}")
    return await speech.synthesize(text, dest, ctx.workdir, on_progress=on_progress, **kwargs)


async def check_voice(params: dict[str, Any], warn) -> None:
    """Warnings for the voice settings, including the live state of the IndexTTS service."""
    from ..llm import LLM

    if params["tts_engine"] == "openai":
        if not LLM().configured:
            warn("llm.not_configured")
        return
    if params["tts_ref"] == "library" and not params["tts_ref_item"]:
        warn("tts.no_reference")
    health = await Node("tts").health()
    if not health.get("reachable"):
        warn("node.offline")
    elif not health.get("ok"):
        warn("node.files_missing" if health.get("model", {}).get("exists") else "node.model_missing")
    elif params["tts_emotion"] == "text" and not health["model"].get("optional", {}).get("emotion_text", {}).get("ok"):
        warn("node.emotion_model_missing")


@register
class TextToSpeech(Tool):
    id = "tts"
    category = "voice"
    icon = "speaker"
    name = "Text to speech / voice clone"
    description = "Speak a script in a cloned voice (IndexTTS 2.5 on the GPU PC) or an OpenAI voice."
    lane = "net"  # waits on the GPU PC
    groups = [Group("main", ""), Group("voice", "Voice"), Group("emotion", "Emotion & delivery", collapsed=True),
              Group("output", "Output", collapsed=True)]

    def params(self):
        return [
            Param("text", "textarea", "Text", default="", remember=False, required=True,
                  help="Long texts are spoken in parts and joined."),
            *voice_params("voice", [["library", "Voice library sample"], ["upload", "Upload a recording"]],
                          emotion_group="emotion"),
            file_param("ref_file", "Voice recording", AUDIO_ACCEPT, required=False, group="voice",
                       show_if={"tts_engine": "indextts", "tts_ref": "upload"}),
            Param("save_ref", "bool", "Add the recording to the voice library", default=True, group="voice",
                  show_if={"tts_engine": "indextts", "tts_ref": "upload"}),
            Param("format", "select", "Format", default="wav", group="output",
                  options=[["wav", "WAV"], ["mp3", "MP3"], ["m4a", "M4A"]]),
            Param("subtitles", "bool", "Also make subtitles (SRT)", default=False, group="output",
                  help="The text, timed to the generated speech (forced alignment)."),
            Param("max_chars", "int", "Max characters per subtitle line", default=42, min=8, max=120, group="output",
                  show_if={"subtitles": True}),
        ]

    def title_for(self, params, file_names):
        text = " ".join(params.get("text", "").split())
        return (text[:40] + "…") if len(text) > 40 else text or self.name

    async def run(self, ctx, params):
        text = params["text"].strip()
        if not text:
            raise ToolError("subs.no_script")
        sources = {}
        if params["tts_engine"] == "indextts" and params["tts_ref"] == "upload":
            upload = ctx.input("ref_file")
            sources["upload"] = upload
            if params["save_ref"]:
                info = await probe(upload)
                item = ctx.manager.library.add("voices", upload, upload.stem,
                                               {"duration": round(info.duration, 1)})
                ctx.log(f"saved voice sample {item.id}")
        spoken = ctx.workdir / "speech.wav"
        ctx.progress(2, "tts.speaking")
        await synthesize_from(ctx, params, text, spoken, sources,
                              on_progress=lambda f: ctx.progress(5 + 80 * f, "tts.speaking"))
        info = await probe(spoken)
        stem = "speech"
        target = ctx.out_path(f"{stem}.{params['format']}")
        if params["format"] == "wav":
            spoken.replace(target)
            spoken = target
        else:
            codec = ["-c:a", "libmp3lame", "-q:a", "2"] if params["format"] == "mp3" else ["-c:a", "aac", "-b:a", "192k"]
            await run_cmd(ffmpeg_base() + ["-i", str(spoken)] + codec + [str(target)])
        ctx.add_output(target, "audio")
        result: dict[str, Any] = {"duration": round(info.duration, 1), "engine": params["tts_engine"]}
        if params["subtitles"]:
            ctx.progress(90, "tts.aligning")
            voice16 = await extract_audio(spoken, ctx.workdir / "speech16k.wav")
            notes: list[str] = []
            chunks = subtitles.split_script(text, params["max_chars"])
            cues, _ = await speech.align_script(chunks, voice16, info.duration, language=tts_language(params, text),
                                                notes=notes, log=ctx.log)
            srt = ctx.out_path(f"{stem}.srt")
            srt.write_text(subtitles.to_srt(cues), encoding="utf-8")
            ctx.add_output(srt, "subtitles")
            result.update(subtitles=len(cues), notes=notes)
        ctx.set_result(result)
