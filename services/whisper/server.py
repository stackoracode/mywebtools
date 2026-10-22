"""faster-whisper speech-to-text service (default port 8391).

Model: a CTranslate2 Whisper folder, e.g. C:\\Users\\Administrator\\Models\\faster-whisper-turbo
(model.bin, config.json, tokenizer.json, vocabulary.json, preprocessor_config.json).

    POST /transcribe   multipart: file=<audio/video>, options=<JSON>
        options: language ("" = detect), task (transcribe|translate), beam_size, vad,
                 word_timestamps, initial_prompt, hotwords, model_dir, device, compute_type
        reply:   NDJSON lines  {"type":"info",language,duration} {"type":"segment",start,end,text[,words]}...
                 {"type":"done",segments}   (or {"type":"error",code,detail})

Test without the web server:
    python server.py --test clip.wav [language]
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import File, Form, UploadFile
from fastapi.responses import StreamingResponse

sys.path.append(str(Path(__file__).resolve().parent.parent))  # svc_common.py; appended so it never shadows packages

from svc_common import (  # noqa: E402
    Folders,
    ModelSlot,
    ServiceError,
    add_cuda_dll_dirs,
    base_parser,
    error_payload,
    folder_report,
    make_app,
    parse_options,
    save_upload,
    serve,
    work_dir,
)

NAME = "whisper"
VERSION = "1.0.0"
FILES: list[Any] = ["model.bin", "config.json", "tokenizer.json", ("vocabulary.json", "vocabulary.txt")]


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def resolve_compute(device: str, compute: str) -> str:
    if compute and compute != "auto":
        return compute
    return "float16" if device == "cuda" else "int8"


def load_model(key: tuple) -> Any:
    model_dir, device, compute, threads = key
    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    return WhisperModel(model_dir, device=device, compute_type=compute, cpu_threads=threads)


def transcribe_iter(model: Any, audio: Path, opts: dict[str, Any]) -> Iterator[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "language": opts.get("language") or None,
        "task": opts.get("task") or "transcribe",
        "beam_size": int(opts.get("beam_size") or 5),
        "vad_filter": bool(opts.get("vad", True)),
        "word_timestamps": bool(opts.get("word_timestamps", False)),
        "initial_prompt": opts.get("initial_prompt") or None,
    }
    if opts.get("hotwords"):
        kwargs["hotwords"] = opts["hotwords"]
    segments, info = model.transcribe(str(audio), **kwargs)
    yield {"type": "info", "language": info.language, "language_probability": round(info.language_probability, 3),
           "duration": round(info.duration, 3)}
    count = 0
    for seg in segments:
        item: dict[str, Any] = {"type": "segment", "start": round(seg.start, 3), "end": round(seg.end, 3),
                                "text": seg.text.strip()}
        if seg.words:
            item["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                              "p": round(w.probability, 3)} for w in seg.words]
        count += 1
        yield item
    yield {"type": "done", "segments": count}


def build(args) -> tuple[Any, ModelSlot]:
    folders = Folders(args.model_dir, args.models_root)
    slot = ModelSlot(load_model, args.idle_unload)

    def key_for(opts: dict[str, Any]) -> tuple:
        folder = folders.require(opts.get("model_dir"), FILES)
        device = resolve_device(opts.get("device") or args.device)
        return (str(folder), device, resolve_compute(device, opts.get("compute_type", "auto")), int(opts.get("threads") or 0))

    def health(model_dir: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {"model": folder_report(folders, model_dir, FILES), "device": resolve_device(args.device)}
        try:
            import ctranslate2
            import faster_whisper

            out["engine"] = {"faster_whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__,
                             "cuda_devices": ctranslate2.get_cuda_device_count()}
        except Exception as exc:  # noqa: BLE001
            out["import_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def preload(opts: dict[str, Any]) -> None:
        with slot.use(key_for(opts)):
            pass

    app = make_app(NAME, VERSION, args, slot, health, preload)

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...), options: str = Form("{}")):
        opts = parse_options(options)
        key = key_for(opts)  # bad folder -> HTTP error before streaming starts
        tmp = work_dir(NAME)
        audio = await save_upload(file, tmp / ("input" + Path(file.filename or "a.wav").suffix))

        def stream() -> Iterator[str]:
            try:
                with slot.use(key) as model:
                    for item in transcribe_iter(model, audio, opts):
                        yield json.dumps(item, ensure_ascii=False) + "\n"
            except Exception as exc:  # noqa: BLE001 - reported in-band, the HTTP status is already sent
                yield json.dumps({"type": "error", **error_payload(exc)}) + "\n"
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app, slot


def main(argv: list[str] | None = None) -> None:
    parser = base_parser(NAME, 8391, "faster-whisper-turbo", "faster-whisper speech-to-text service for mywebtools")
    args = parser.parse_args(argv)
    if args.test:
        audio = Path(args.test[0])
        if not audio.is_file():
            raise SystemExit(f"no such file: {audio}")
        device = resolve_device(args.device)
        folder = Folders(args.model_dir, "*").require(None, FILES)
        model = load_model((str(folder), device, resolve_compute(device, "auto"), 0))
        try:
            for item in transcribe_iter(model, audio, {"language": args.test[1] if len(args.test) > 1 else ""}):
                print(json.dumps(item, ensure_ascii=False))
        except ServiceError as exc:
            raise SystemExit(f"{exc.code}: {exc.detail}") from exc
        return
    app, _ = build(args)
    serve(app, args, NAME)


if __name__ == "__main__":
    main()
