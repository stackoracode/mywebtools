"""Speech engines: recognition (ASR), synthesis (TTS) and forced alignment.

    recognise   node      faster-whisper service on the GPU PC (Admin → Model services)
                local     faster-whisper in this process (small models on the board)
                api       OpenAI-compatible /audio/transcriptions (Admin → AI model)
    synthesise  indextts  IndexTTS 2.5 voice clone on the GPU PC (needs a voice sample)
                openai    OpenAI-compatible /audio/speech
    align       mms       MMS CTC forced aligner on the GPU PC: the exact script, precisely timed
                basic     the script scaled onto recognised speech (or spread over the voice)

"auto" tries the engines in that order and moves on when a service is offline.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import subtitles
from .config import settings
from .errors import ToolError
from .llm import LLM
from .media import extract_audio, ffmpeg_base, run_cmd
from .nodes import Node
from .subtitles import Cue

# a service that cannot do the job right now: "auto" moves on to the next engine
FALLBACK = {"node.offline", "node.not_configured", "node.model_missing", "node.files_missing", "node.import_failed",
            "node.model_outside_root"}
TTS_LANGS = {"ZH": "zh", "EN": "en", "JA": "ja", "ES": "es", "AR": "ar"}
Log = Callable[[str], None]

_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def load_local_model(name: str):
    from faster_whisper import WhisperModel  # type: ignore

    with _model_lock:
        if name not in _model_cache:
            _model_cache.clear()  # keep one model in RAM on a small board
            _model_cache[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                              cpu_threads=settings.runtime.whisper_threads,
                                              download_root=str(settings.models / "whisper"))
        return _model_cache[name]


def local_whisper() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def guess_language(text: str) -> str:
    """Rough language of a script from its characters (for alignment / voice synthesis)."""
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[㐀-鿿]", text):
        return "zh"
    if re.search(r"[؀-ۿ]", text):
        return "ar"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    if re.search(r"[ñ¿¡]", text, re.I):
        return "es"
    return "en"


def _quiet(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# recognition
# ---------------------------------------------------------------------------
def asr_order(choice: str | None = None) -> list[str]:
    choice = choice or settings.runtime.asr_engine
    if choice == "node":
        return ["node"]
    if choice == "local":
        if not local_whisper():
            raise ToolError("dependency.faster_whisper")
        return ["local"]
    if choice == "api":
        if not LLM().configured:
            raise ToolError("llm.not_configured")
        return ["api"]
    order = [name for name, ok in (("node", Node("whisper").configured), ("local", local_whisper()),
                                   ("api", LLM().configured)) if ok]
    if not order:
        raise ToolError("subs.no_asr")
    return order


async def transcribe(audio: Path, *, engine: str | None = None, language: str | None = None, task: str = "transcribe",
                     prompt: str = "", vad: bool = True, beam_size: int = 0, word_timestamps: bool = False,
                     local_model: str = "", duration: float = 0.0, on_progress: Callable[[float], None] | None = None,
                     log: Log = _quiet) -> dict[str, Any]:
    """Timed text of a 16 kHz WAV: {"engine", "language", "duration", "segments": [{start, end, text[, words]}]}."""
    order = asr_order(engine)
    report = on_progress or (lambda _: None)
    for n, name in enumerate(order):
        try:
            if name == "node":
                def seen(seg: dict[str, Any], total: float) -> None:
                    report(min(1.0, seg["end"] / (total or duration or 1.0)))

                opts: dict[str, Any] = {"language": language or "", "task": task, "vad": vad,
                                        "word_timestamps": word_timestamps, "initial_prompt": prompt}
                if beam_size:
                    opts["beam_size"] = beam_size
                result = await Node("whisper").transcribe(audio, opts, on_segment=seen)
                segments = [{"start": s["start"], "end": s["end"], "text": s["text"], **({"words": s["words"]} if
                             word_timestamps and s.get("words") else {})} for s in result["segments"] if s["text"]]
                return {"engine": "node", "language": result["language"] or language or "",
                        "duration": result["duration"] or duration, "segments": segments}
            if name == "local":
                return await _local(audio, local_model or settings.runtime.whisper_model, language, task, prompt, vad,
                                    beam_size or 2, word_timestamps, duration, report)
            segments = await LLM().transcribe(audio, language, prompt)
            return {"engine": "api", "language": language or "", "duration": duration, "segments": segments}
        except ToolError as exc:
            if n + 1 < len(order) and exc.code in FALLBACK:
                log(f"speech recognition: {name} unavailable ({exc.code}), using {order[n + 1]}")
                continue
            raise
    raise ToolError("subs.no_asr")


async def _local(audio, model_name, language, task, prompt, vad, beam_size, word_timestamps, duration, report):
    model = await asyncio.to_thread(load_local_model, model_name)
    stop = threading.Event()
    loop = asyncio.get_running_loop()

    def work():
        segments, meta = model.transcribe(str(audio), language=language, task=task, vad_filter=vad,
                                          beam_size=beam_size, word_timestamps=word_timestamps,
                                          initial_prompt=prompt or None)
        items = []
        for seg in segments:
            if stop.is_set():
                raise asyncio.CancelledError()
            item = {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
            if word_timestamps and seg.words:
                item["words"] = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word} for w in seg.words]
            if item["text"]:
                items.append(item)
            loop.call_soon_threadsafe(report, min(1.0, seg.end / (duration or meta.duration or 1.0)))
        return items, meta

    try:
        items, meta = await asyncio.to_thread(work)
    except asyncio.CancelledError:
        stop.set()
        raise
    return {"engine": "local", "language": meta.language, "duration": meta.duration, "segments": items}


async def recognize(audio: Path, language: str = "", prompt: str = "", log: Log = _quiet) -> list[dict[str, Any]]:
    """Segments for subtitles, with the engine from the settings."""
    lang = None if language in ("", "auto") else language.split("-")[0]
    return (await transcribe(audio, language=lang, prompt=prompt, log=log))["segments"]


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
async def voice_sample(source: Path, dest: Path, *, start: float = 0.0, seconds: float = 15.0) -> Path:
    """Reference clip for voice cloning (IndexTTS uses up to 15 s)."""
    return await extract_audio(source, dest, start=start, end=start + seconds, rate=24000)


def _chunks(script: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for sentence in subtitles._SENTENCE_END.split(script):
        sentence = (sentence or "").strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip() if subtitles.cjk_ratio(sentence) <= 0.3 else current + sentence
    return chunks + ([current] if current else [])


async def synthesize(script: str, dest: Path, workdir: Path, *, engine: str = "", reference: Path | None = None,
                     emo_reference: Path | None = None, options: dict[str, Any] | None = None, voice: str = "",
                     instructions: str = "", speed: float = 1.0,
                     on_progress: Callable[[float], None] | None = None) -> Path:
    """Speak ``script`` into ``dest`` (WAV). Long scripts are spoken in chunks and joined."""
    script = script.strip()
    if not script:
        raise ToolError("subs.no_script")
    engine = engine if engine in ("indextts", "openai") else settings.runtime.tts_engine
    if engine == "indextts" and reference is None:
        raise ToolError("tts.no_reference")
    chunks = _chunks(script, 600 if engine == "indextts" else 3000)
    parts = []
    for n, chunk in enumerate(chunks):
        part = workdir / f"tts_{n:03d}.wav"
        if engine == "indextts":
            await Node("tts").tts(chunk, reference, part, options or {}, emo_audio=emo_reference)
        else:
            await LLM().speech(chunk, part, voice=voice, instructions=instructions, speed=speed)
        parts.append(part)
        if on_progress:
            on_progress((n + 1) / len(chunks))
    if len(parts) == 1:
        parts[0].replace(dest)
        return dest
    inputs = [arg for p in parts for arg in ("-i", str(p))]
    graph = "".join(f"[{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[a]"
    await run_cmd(ffmpeg_base() + inputs + ["-filter_complex", graph, "-map", "[a]", str(dest)])
    return dest


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
async def align_script(chunks: list[str], audio: Path, duration: float, *, language: str = "", prompt: str = "",
                       engine: str = "", notes: list[str] | None = None,
                       log: Log = _quiet) -> tuple[list[Cue], list[dict[str, Any]]]:
    """Time typed subtitle chunks to the voice in ``audio`` (16 kHz WAV). Returns (cues, word timings)."""
    if not chunks:
        return [], []
    notes = notes if notes is not None else []
    engine = engine if engine in ("auto", "mms", "basic") else settings.runtime.align_engine
    lang = language if language not in ("", "auto") else guess_language(" ".join(chunks))
    node = Node("align")
    if engine == "mms" or (engine == "auto" and node.configured):
        try:
            result = await node.align(audio, chunks, lang)
            cues = [Cue(float(c["start"]), float(c["end"]), text) for c, text in zip(result["cues"], chunks, strict=True)]
            log(f"forced alignment ({lang}): {sum(c.get('aligned', True) for c in result['cues'])}/{len(chunks)} lines")
            return subtitles.tidy_cues(cues, duration), result.get("words", [])
        except ToolError as exc:
            if engine == "mms" or not exc.code.startswith("node."):
                raise
            log(f"forced alignment unavailable ({exc.code}): timing the script by speech recognition")
            notes.append("subs.align_fallback")
    try:
        segments = await recognize(audio, lang, prompt, log=log)
    except ToolError as exc:
        log(f"no speech recognition ({exc.code}): spreading the script over the voice")
        notes.append("subs.proportional")
        segments = []
    return subtitles.align_script(chunks, segments, duration), []
