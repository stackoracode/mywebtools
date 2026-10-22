"""MMS CTC forced-alignment service (default port 8393): exact subtitle text, timed to the voice.

Model: MahmoudAshraf/mms-300m-1130-forced-aligner, e.g.
C:\\Users\\Administrator\\Models\\mms-300m-1130-forced-aligner (config.json, model.safetensors, vocab.json).

    POST /align   multipart: file=<audio>, options=<JSON>
        options: cues [subtitle texts, in order], language (en, zh, ja, ... or ISO 639-3),
                 star (true: tolerate words that are not in the script), model_dir, device
        reply:   {"cues": [{start, end, score, aligned}], "words": [{cue, text, start, end, score}], "duration"}

Test without the web server:
    python server.py --test speech.wav script.txt [language]      (one subtitle per line)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import File, Form, UploadFile

sys.path.append(str(Path(__file__).resolve().parent.parent))  # svc_common.py; appended so it never shadows packages
sys.path.append(str(Path(__file__).resolve().parent))

import ctc  # noqa: E402
from svc_common import (  # noqa: E402
    Folders,
    ModelSlot,
    ServiceError,
    base_parser,
    folder_report,
    make_app,
    parse_options,
    save_upload,
    serve,
    work_dir,
)

NAME = "aligner"
VERSION = "1.0.0"
FILES: list[Any] = ["config.json", ("model.safetensors", "pytorch_model.bin"), "vocab.json"]
RATE = 16000
_romanizer: Any = None


def read_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32 in [-1, 1]; other formats are converted with ffmpeg."""
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getframerate() == RATE and wav.getnchannels() == 1 and wav.getsampwidth() == 2:
                return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    if not shutil.which("ffmpeg"):
        raise ServiceError("bad_request", "send 16 kHz mono 16-bit WAV, or install ffmpeg on this PC")
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(RATE), "-"], capture_output=True, check=False)
    if out.returncode != 0:
        raise ServiceError("bad_request", out.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def romanizer(language: str | None):
    """uroman (romanises Chinese, Japanese, Arabic, Cyrillic, ...) bound to the language."""
    global _romanizer
    if _romanizer is None:
        import uroman

        _romanizer = uroman.Uroman()
    code = ctc.iso3(language)
    return lambda text: _romanizer.romanize_string(text, lcode=code)


def load_model(key: tuple) -> Any:
    folder, device = key
    import torch
    from transformers import AutoModelForCTC

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCTC.from_pretrained(folder, torch_dtype=dtype).to(device).eval()
    vocab = json.loads((Path(folder) / "vocab.json").read_text(encoding="utf-8"))
    return {"model": model, "vocab": {k.lower(): v for k, v in vocab.items()}, "device": device, "dtype": dtype}


def emissions(bundle: dict[str, Any], audio: np.ndarray, window: int = 30, context: int = 2,
              batch: int = 4) -> tuple[np.ndarray, float]:
    """Log-probabilities per 20 ms frame, computed in overlapping 30 s windows."""
    import torch

    model = bundle["model"]
    ratio = int(getattr(model.config, "inputs_to_logits_ratio", 320))
    wave_t = torch.from_numpy(audio).to(bundle["device"], bundle["dtype"])
    win, ctx_len = window * RATE, context * RATE
    if wave_t.numel() < win:
        chunks, ctx_frames, extension = wave_t.unsqueeze(0), 0, 0
    else:
        extension = -wave_t.numel() % win
        padded = torch.nn.functional.pad(wave_t, (ctx_len, ctx_len + extension))
        chunks, ctx_frames = padded.unfold(0, win + 2 * ctx_len, win), ctx_len // ratio
    outs = []
    with torch.inference_mode():
        for i in range(0, chunks.size(0), batch):
            outs.append(model(chunks[i:i + batch]).logits)
    logits = torch.cat(outs, 0)
    if ctx_frames:
        logits = logits[:, ctx_frames:ctx_frames + win // ratio]
    logits = logits.flatten(0, 1)
    if extension:
        logits = logits[: -(extension // ratio)]
    return torch.log_softmax(logits.float(), dim=-1).cpu().numpy(), ratio / RATE


def run_align(bundle: dict[str, Any], audio: np.ndarray, cues: list[str], language: str | None,
              star: bool = True) -> dict[str, Any]:
    log_probs, frame = emissions(bundle, audio)
    star_index = None
    if star:
        log_probs, star_index = ctc.emissions_with_star(log_probs)
    try:
        result = ctc.align(log_probs, frame, cues, bundle["vocab"], romanizer(language), star=star_index)
    except ctc.AlignError as exc:
        raise ServiceError(exc.code, exc.detail) from exc
    result["language"] = ctc.iso3(language)
    return result


def build(args) -> tuple[Any, ModelSlot]:
    folders = Folders(args.model_dir, args.models_root)
    slot = ModelSlot(load_model, args.idle_unload)

    def key_for(opts: dict[str, Any]) -> tuple:
        return (str(folders.require(opts.get("model_dir"), FILES)), opts.get("device") or args.device)

    def health(model_dir: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {"model": folder_report(folders, model_dir, FILES), "device": args.device}
        try:
            import torch
            import transformers
            import uroman  # noqa: F401

            out["engine"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                             "cuda": torch.cuda.is_available()}
        except Exception as exc:  # noqa: BLE001
            out["import_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def preload(opts: dict[str, Any]) -> None:
        with slot.use(key_for(opts)):
            romanizer(None)

    app = make_app(NAME, VERSION, args, slot, health, preload)

    @app.post("/align")
    async def align(file: UploadFile = File(...), options: str = Form("{}")):
        opts = parse_options(options)
        cues = [str(c) for c in opts.get("cues") or [] if str(c).strip()]
        if not cues:
            raise ServiceError("bad_request", "no cues")
        key = key_for(opts)
        tmp = work_dir(NAME)
        try:
            audio = read_audio(await save_upload(file, tmp / ("input" + Path(file.filename or "a.wav").suffix)))

            def work() -> dict[str, Any]:
                with slot.use(key) as bundle:
                    return run_align(bundle, audio, cues, opts.get("language"), bool(opts.get("star", True)))

            return await asyncio.to_thread(work)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return app, slot


def main(argv: list[str] | None = None) -> None:
    parser = base_parser(NAME, 8393, "mms-300m-1130-forced-aligner", "MMS CTC forced-alignment service for mywebtools")
    args = parser.parse_args(argv)
    if args.test:
        if len(args.test) < 2:
            raise SystemExit("usage: --test speech.wav script.txt [language]")
        audio = read_audio(Path(args.test[0]))
        cues = [ln.strip() for ln in Path(args.test[1]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        folder = Folders(args.model_dir, "*").require(None, FILES)
        result = run_align(load_model((str(folder), args.device)), audio, cues, args.test[2] if len(args.test) > 2 else "en")
        for text, cue in zip(cues, result["cues"], strict=True):
            print(f"{cue['start']:8.2f} {cue['end']:8.2f}  {cue['score']:.2f}  {text}")
        return
    app, _ = build(args)
    serve(app, args, NAME)


if __name__ == "__main__":
    main()
