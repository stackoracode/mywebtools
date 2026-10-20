"""Speech to text (video/audio -> TXT/SRT/VTT/JSON) with faster-whisper on the CPU."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from typing import Any

from ..config import settings
from ..errors import ToolError
from ..i18n import CHINESE_PROMPTS, WHISPER_LANGUAGES
from ..llm import LLM
from ..media import extract_audio, probe
from ..subtitles import TARGET_LANGUAGES, Cue, ai_edit
from .base import Group, Param, Tool, file_param, register
from .video_ocr import srt_time

MEDIA_ACCEPT = "video/*,audio/*,.mp4,.mov,.mkv,.webm,.m4a,.mp3,.wav,.aac,.flac,.ogg,.opus"
MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]

_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def _load_model(name: str):
    from faster_whisper import WhisperModel  # type: ignore

    with _model_lock:
        if name not in _model_cache:
            _model_cache.clear()  # keep one model in RAM on a small board
            _model_cache[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                              cpu_threads=settings.runtime.whisper_threads,
                                              download_root=str(settings.models / "whisper"))
        return _model_cache[name]


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


@register
class Transcribe(Tool):
    id = "transcribe"
    category = "video"
    icon = "mic"
    name = "Speech to text"
    description = "Transcribe or translate speech from video/audio into subtitles and text."
    groups = [Group("main", ""), Group("advanced", "Advanced", collapsed=True)]

    def params(self):
        return [
            file_param("media", "Video or audio", MEDIA_ACCEPT, multiple=True, batch=True),
            Param("language", "select", "Spoken language", default="auto",
                  options=[["auto", "Detect automatically"]] + [[c, c] for c in WHISPER_LANGUAGES]),
            Param("chinese_script", "select", "Chinese characters", default="simplified",
                  options=[["simplified", "Simplified"], ["traditional", "Traditional"], ["any", "As recognised"]],
                  show_if={"language": "zh"}),
            Param("task", "select", "Task", default="transcribe",
                  options=[["transcribe", "Transcribe (same language)"], ["translate", "Translate to English"]]),
            Param("model", "select", "Model", default="base", options=MODELS, show_if={"engine": ["auto", "local"]},
                  help="tiny/base are fast on ARM; small is more accurate; bigger needs lots of RAM."),
            Param("engine", "select", "Recognition engine", default="auto",
                  options=[["auto", "Auto (local if installed, else AI API)"], ["local", "Local faster-whisper"],
                           ["api", "AI API (Admin → AI model)"]]),
            Param("ai", "select", "AI editing", default="none",
                  options=[["none", "None"], ["fix", "Fix recognition errors & punctuation"], ["translate", "Translate"],
                           ["bilingual", "Bilingual (original + translation)"]]),
            Param("ai_target", "select", "Translate to", default="en", show_if={"ai": ["translate", "bilingual"]},
                  options=[[k, v] for k, v in TARGET_LANGUAGES.items()]),
            Param("formats", "multiselect", "Outputs", default=["txt", "srt"],
                  options=[["txt", "TXT"], ["srt", "SRT"], ["vtt", "VTT"], ["json", "JSON"]]),
            Param("vad", "bool", "Skip silence (VAD)", default=True, group="advanced"),
            Param("beam_size", "int", "Beam size", default=2, min=1, max=8, group="advanced"),
            Param("word_timestamps", "bool", "Word timestamps (JSON)", default=False, group="advanced"),
            Param("max_line", "int", "Max characters per subtitle line", default=42, min=0, max=200, group="advanced",
                  help="0 = no wrapping"),
            Param("prompt", "textarea", "Vocabulary hint", default="", group="advanced",
                  help="Names or terms that appear in the audio."),
        ]

    def availability(self):
        if importlib.util.find_spec("faster_whisper") is None and not LLM().configured:
            return False, "dependency.faster_whisper_or_api"
        return True, ""

    @staticmethod
    def _engine(choice: str) -> str:
        local = importlib.util.find_spec("faster_whisper") is not None
        if choice == "local" or (choice == "auto" and local):
            if not local:
                raise ToolError("dependency.faster_whisper")
            return "local"
        if not LLM().configured:
            raise ToolError("llm.not_configured")
        return "api"

    async def run(self, ctx, params):
        source = ctx.input("media")
        info = await probe(source)
        if not info.has_audio:
            raise ToolError("media.no_audio")
        wav = ctx.workdir / "audio.wav"
        ctx.progress(0, "stt.audio")
        await extract_audio(source, wav, duration=info.duration, on_progress=lambda f: ctx.progress(5 * f, "stt.audio"))
        language = None if params["language"] == "auto" else params["language"]
        prompt = params["prompt"].strip()
        if language == "zh" and params["chinese_script"] in CHINESE_PROMPTS:
            prompt = (CHINESE_PROMPTS[params["chinese_script"]] + " " + prompt).strip()
        duration = info.duration or 1.0
        if self._engine(params["engine"]) == "api":
            ctx.progress(10, "stt.transcribing")
            segments = await LLM().transcribe(wav, language, prompt)
            meta = type("Meta", (), {"language": language or "", "duration": duration})()
        else:
            segments, meta = await self._local(ctx, params, wav, language, prompt, duration)
        mode, target = params["ai"], params["ai_target"]
        if params["task"] == "translate" and self._engine(params["engine"]) == "api" and mode == "none":
            mode, target = "translate", "en"  # the API transcribes; the model translates
        if mode != "none" and segments:
            ctx.progress(97, "stt.ai")
            cues = await ai_edit([Cue(s["start"], s["end"], s["text"]) for s in segments], mode,
                                 target=target, context=params["prompt"])
            segments = [{**seg, "text": cue.text} for seg, cue in zip(segments, cues, strict=True)]
        lang = getattr(meta, "language", language or "")
        self._write(ctx, source.stem, segments, params, lang)
        ctx.set_result({"language": lang, "segments": len(segments),
                        "duration": round(getattr(meta, "duration", duration), 1)})

    async def _local(self, ctx, params, wav, language, prompt, duration):
        ctx.progress(5, "stt.loading_model", model=params["model"])
        model = await asyncio.to_thread(_load_model, params["model"])
        stop = threading.Event()
        loop = asyncio.get_running_loop()

        def work():
            segments, meta = model.transcribe(
                str(wav), language=language, task=params["task"], vad_filter=params["vad"],
                beam_size=params["beam_size"], word_timestamps=params["word_timestamps"],
                initial_prompt=prompt or None,
            )
            items = []
            for seg in segments:
                if stop.is_set():
                    raise asyncio.CancelledError()
                item = {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
                if params["word_timestamps"] and seg.words:
                    item["words"] = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word} for w in seg.words]
                items.append(item)
                loop.call_soon_threadsafe(ctx.progress, 10 + 85 * min(1.0, seg.end / duration), "stt.transcribing")
            return items, meta

        try:
            return await asyncio.to_thread(work)
        except asyncio.CancelledError:
            stop.set()
            raise

    @staticmethod
    def _write(ctx, stem: str, segments: list[dict], params: dict, lang: str) -> None:
        from ..i18n import join_words

        cjk = lang in ("zh", "ja", "ko", "th")
        wrap = params["max_line"]

        def wrapped(text: str) -> str:
            if not wrap or len(text) <= wrap:
                return text
            if cjk:
                return "\n".join(text[i:i + wrap] for i in range(0, len(text), wrap))
            lines, line = [], ""
            for word in text.split():
                if line and len(line) + 1 + len(word) > wrap:
                    lines.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            return "\n".join(lines + [line])

        formats = params["formats"] or ["txt"]
        if "txt" in formats:
            body = join_words([s["text"] for s in segments]) if cjk else "\n".join(s["text"] for s in segments)
            path = ctx.out_path(f"{stem}.txt")
            path.write_text(body + "\n", encoding="utf-8")
            ctx.add_output(path, "text")
        if "srt" in formats:
            path = ctx.out_path(f"{stem}.srt")
            path.write_text("\n".join(f"{n}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{wrapped(s['text'])}\n"
                                      for n, s in enumerate(segments, 1)), encoding="utf-8")
            ctx.add_output(path, "subtitles")
        if "vtt" in formats:
            path = ctx.out_path(f"{stem}.vtt")
            path.write_text("WEBVTT\n\n" + "\n".join(f"{vtt_time(s['start'])} --> {vtt_time(s['end'])}\n{wrapped(s['text'])}\n"
                                                    for s in segments), encoding="utf-8")
            ctx.add_output(path, "subtitles")
        if "json" in formats:
            path = ctx.out_path(f"{stem}.json")
            path.write_text(json.dumps({"language": lang, "segments": segments}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            ctx.add_output(path, "json")
