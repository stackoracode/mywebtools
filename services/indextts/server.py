"""IndexTTS 2.5 voice-cloning text-to-speech service (default port 8392).

Runs inside the environment of an IndexTTS checkout (https://github.com/index-tts/index-tts,
``uv sync``), see ../README.md. Model: the IndexTTS-2.5 folder, e.g.
C:\\Users\\Administrator\\Models\\IndexTTS-2.5 (gpt.pth, s2mel.pth, codec.pth, config.yaml,
the tiktoken file and hf_cache/{w2v-bert-2.0,bigvgan,campplus_cn_common.bin}).
Emotion-from-text additionally needs the files in qwen0.6bemo4-merge/.

    POST /tts   multipart: text=<script>, voice=<reference audio>, [emo_audio=<emotion reference>],
                options=<JSON>
        options: lang (auto|ZH|EN|JA|ES|AR), emotion (speaker|audio|vector|text), emo_alpha,
                 emo_vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm],
                 emo_text, duration_factor (0.5-2), use_random, interval_silence (ms),
                 max_text_tokens_per_segment, temperature, top_p, top_k, model_dir, device, bf16
        reply:   audio/wav (headers X-Seconds, X-Lang)

Test without the web server:
    python server.py --test voice.wav "Text to speak" [out.wav]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import File, Form, UploadFile
from fastapi.responses import Response

sys.path.append(str(Path(__file__).resolve().parent.parent))  # svc_common.py; appended so it never shadows packages

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
    wav_seconds,
    work_dir,
)

NAME = "indextts"
VERSION = "1.0.0"
FILES: list[Any] = [
    "config.yaml", "gpt.pth", "s2mel.pth", "codec.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "hf_cache/w2v-bert-2.0/config.json", "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    "hf_cache/w2v-bert-2.0/model.safetensors",
    "hf_cache/bigvgan/config.json", "hf_cache/bigvgan/bigvgan_generator.pt",
    "hf_cache/campplus_cn_common.bin",
]
EMOTION_FILES: list[Any] = ["qwen0.6bemo4-merge/config.json", "qwen0.6bemo4-merge/model.safetensors",
                            "qwen0.6bemo4-merge/tokenizer.json"]
LANGS = ("ZH", "EN", "JA", "ES", "AR")
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u3400-\u9fff]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_SPANISH = re.compile(r"[ñ¿¡áéíóú]", re.I)


def detect_lang(text: str) -> str:
    if _KANA.search(text):
        return "JA"
    if _HAN.search(text):
        return "ZH"
    if _ARABIC.search(text):
        return "AR"
    if _SPANISH.search(text):
        return "ES"
    return "EN"


def import_engine(repo: str) -> Any:
    """IndexTTS2 class of the checkout (``indextts`` must be importable)."""
    if repo and Path(repo).is_dir() and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ImportError as exc:
        raise ServiceError("import_failed", f"cannot import indextts.infer_v2_5 ({exc}); run this service with the "
                                            "IndexTTS environment, see services/README.md", 500) from exc
    return IndexTTS2


def create_engine(repo: str, folder: str, device: str, bf16: bool, emotion: bool) -> Any:
    engine_cls = import_engine(repo)
    os.environ["HF_HUB_CACHE"] = str(Path(folder) / "hf_cache")  # the module sets a cwd-relative one
    return engine_cls(cfg_path=str(Path(folder) / "config.yaml"), model_dir=folder, use_bf16=bf16,
                      device=None if device == "auto" else device, use_cuda_kernel=False, use_qwen_emo=emotion)


def build_kwargs(engine: Any, opts: dict[str, Any], emo_audio: Path | None) -> dict[str, Any]:
    """infer() keyword arguments for the requested emotion mode and generation settings."""
    kwargs: dict[str, Any] = {
        "emo_alpha": float(opts.get("emo_alpha", 1.0)),
        "duration_factor": min(2.0, max(0.5, float(opts.get("duration_factor", 1.0)))),
        "use_random": bool(opts.get("use_random", False)),
        "interval_silence": int(opts.get("interval_silence", 200)),
        "max_text_tokens_per_segment": int(opts.get("max_text_tokens_per_segment", 120)),
    }
    mode = opts.get("emotion") or "speaker"
    if mode == "audio":
        if emo_audio is None:
            raise ServiceError("bad_request", "emotion=audio needs an emo_audio file")
        kwargs["emo_audio_prompt"] = str(emo_audio)
    elif mode == "vector":
        vector = [float(x) for x in (opts.get("emo_vector") or [])][:8]
        if len(vector) != 8:
            raise ServiceError("bad_request", f"emo_vector needs 8 values: {', '.join(EMOTIONS)}")
        kwargs["emo_vector"] = engine.normalize_emo_vec(vector, apply_bias=True)
    elif mode == "text":
        if getattr(engine, "qwen_emo", None) is None:
            raise ServiceError("emotion_model_missing", "emotion from text needs the qwen0.6bemo4-merge model files")
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = (opts.get("emo_text") or "").strip() or None
    for name in ("temperature", "top_p", "top_k"):
        if opts.get(name) not in (None, ""):
            kwargs[name] = int(opts[name]) if name == "top_k" else float(opts[name])
    if kwargs.get("temperature") or kwargs.get("top_p") or kwargs.get("top_k"):
        kwargs["do_sample"] = True
    return kwargs


def build(args) -> tuple[Any, ModelSlot]:
    folders = Folders(args.model_dir, args.models_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    slot = ModelSlot(lambda key: create_engine(args.repo, *key), args.idle_unload)

    def key_for(opts: dict[str, Any]) -> tuple:
        folder = folders.require(opts.get("model_dir"), FILES)
        emotion_ok = folder_report(folders, str(folder), [], {"emotion_text": EMOTION_FILES})["optional"]["emotion_text"]["ok"]
        emotion = {"on": True, "off": False}.get(args.emotion_model, emotion_ok)
        if opts.get("emotion") == "text" and not emotion:
            raise ServiceError("emotion_model_missing", f"{folder / 'qwen0.6bemo4-merge'} is empty or incomplete")
        bf16 = args.bf16 if opts.get("bf16") is None else bool(opts["bf16"])
        return (str(folder), opts.get("device") or args.device, bf16, emotion)

    def health(model_dir: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {"model": folder_report(folders, model_dir, FILES, {"emotion_text": EMOTION_FILES}),
                               "device": args.device, "repo": args.repo, "languages": list(LANGS),
                               "emotions": list(EMOTIONS)}
        try:
            import_engine(args.repo)
            import torch

            out["engine"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
        except ServiceError as exc:
            out["import_error"] = exc.detail
        except Exception as exc:  # noqa: BLE001
            out["import_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def preload(opts: dict[str, Any]) -> None:
        with slot.use(key_for(opts)):
            pass

    async def cached(upload: UploadFile) -> Path:
        """Reference audio stored by content hash: IndexTTS caches the speaker features by path."""
        tmp = await save_upload(upload, work_dir(NAME) / "ref")
        digest = hashlib.sha1(tmp.read_bytes()).hexdigest()[:20]
        dest = cache / f"ref_{digest}{Path(upload.filename or 'a.wav').suffix or '.wav'}"
        if not dest.is_file():
            shutil.move(str(tmp), dest)
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return dest

    app = make_app(NAME, VERSION, args, slot, health, preload)

    @app.post("/tts")
    async def tts(text: str = Form(...), options: str = Form("{}"), voice: UploadFile = File(...),
                  emo_audio: UploadFile | None = File(None)):
        opts = parse_options(options)
        text = text.strip()
        if not text:
            raise ServiceError("bad_request", "empty text")
        lang = str(opts.get("lang") or "auto").upper()
        lang = detect_lang(text) if lang == "AUTO" else lang
        if lang not in LANGS:
            raise ServiceError("bad_request", f"lang must be one of {', '.join(LANGS)}")
        key = key_for(opts)
        voice_path = await cached(voice)
        emo_path = await cached(emo_audio) if emo_audio is not None and emo_audio.filename else None
        tmp = work_dir(NAME)
        out = tmp / "speech.wav"

        def work() -> None:
            with slot.use(key) as engine:
                engine.infer(spk_audio_prompt=str(voice_path), text=text, output_path=str(out), lang=lang,
                             **build_kwargs(engine, opts, emo_path))

        try:
            await asyncio.to_thread(work)
            if not out.is_file():
                raise ServiceError("internal", "IndexTTS produced no audio")
            return Response(out.read_bytes(), media_type="audio/wav",
                            headers={"X-Seconds": str(wav_seconds(out)), "X-Lang": lang})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return app, slot


def main(argv: list[str] | None = None) -> None:
    parser = base_parser(NAME, 8392, "IndexTTS-2.5", "IndexTTS 2.5 text-to-speech service for mywebtools")
    parser.add_argument("--repo", default=os.environ.get("INDEXTTS_REPO", r"C:\AI\index-tts"),
                        help="IndexTTS source checkout (contains the 'indextts' package)")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="bfloat16 inference (less GPU memory)")
    parser.add_argument("--emotion-model", choices=("auto", "on", "off"), default="auto",
                        help="load the Qwen emotion-from-text model (auto = when its files are present)")
    parser.add_argument("--cache-dir", default=str(Path(os.environ.get("TEMP", "/tmp")) / "mwt-indextts-refs"))
    args = parser.parse_args(argv)
    if args.test:  # resolve the test paths before moving into the checkout
        if len(args.test) < 2:
            raise SystemExit('usage: --test voice.wav "Text to speak" [out.wav]')
        voice, text = Path(args.test[0]).resolve(), args.test[1]
        out = Path(args.test[2] if len(args.test) > 2 else "indextts_test.wav").resolve()
    args.model_dir = str(Path(args.model_dir).resolve())
    if args.repo and Path(args.repo).is_dir():
        os.chdir(args.repo)  # IndexTTS resolves a few paths relative to its checkout
    if args.test:
        folder = Folders(args.model_dir, "*").require(None, FILES)
        emotion = folder_report(Folders(str(folder), "*"), None, [], {"e": EMOTION_FILES})["optional"]["e"]["ok"]
        engine = create_engine(args.repo, str(folder), args.device, args.bf16, emotion)
        engine.infer(spk_audio_prompt=str(voice), text=text, output_path=str(out), lang=detect_lang(text))
        print(f"wrote {out} ({wav_seconds(out)} s)")
        return
    print("importing IndexTTS ...")
    try:
        import_engine(args.repo)  # slow (torch, transformers): do it once before serving
    except ServiceError as exc:
        print(f"  {exc.detail}")
    app, _ = build(args)
    serve(app, args, NAME)


if __name__ == "__main__":
    main()
