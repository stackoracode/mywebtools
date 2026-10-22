# Model services (run on the Windows PC with the graphics card)

Each local model runs as its own small web service in its own Python environment, so their dependencies
never conflict and each one can be started, checked and tested on its own. mywebtools (on the same PC now,
on the board later) calls them over HTTP and sends the model folder configured in **Admin → Model services**
with every request.

| Service | Port | Model folder (default) | Used for |
| --- | --- | --- | --- |
| `whisper` | 8391 | `C:\Users\Administrator\Models\faster-whisper-turbo` | Speech to text, avatar subtitles from speech |
| `indextts` | 8392 | `C:\Users\Administrator\Models\IndexTTS-2.5` | Voice cloning / text to speech, the avatar's voice from a typed script |
| `aligner` | 8393 | `C:\Users\Administrator\Models\mms-300m-1130-forced-aligner` | Timing an existing script exactly to the voice (subtitles) |

The avatar combines them: **script → IndexTTS (cloned voice) → Duix lip-sync**, while **MMS** times the script to
the generated voice for the subtitles. With your own recording instead of a script, **faster-whisper** writes the
subtitle draft.

## Install (once)

Needs [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`), Git, ffmpeg on PATH and the NVIDIA driver.
In a checkout of this repository:

```powershell
powershell -ExecutionPolicy Bypass -File services\install.ps1               # all three
powershell -ExecutionPolicy Bypass -File services\install.ps1 -Only whisper # or one at a time
```

* `whisper` → `services\.venv-whisper` (faster-whisper, plus the CUDA 12 cuBLAS / cuDNN 9 wheels it needs on Windows)
* `aligner` → `services\.venv-aligner` (PyTorch CUDA 12.8, transformers, uroman for non-Latin scripts)
* `indextts` → clones https://github.com/index-tts/index-tts to `C:\AI\index-tts` and runs `uv sync` there.
  The model folder is passed by path, so nothing is downloaded into its `checkpoints\`.

Run it from an elevated PowerShell once to also open TCP 8391–8393 for the board (private networks only).

## Start

```powershell
powershell -ExecutionPolicy Bypass -File services\start.ps1                     # one window per service
powershell -ExecutionPolicy Bypass -File services\start.ps1 -Only whisper -Here # one service, this window
```

Options: `-ModelsRoot` (default `%USERPROFILE%\Models`; requested model folders must be inside it),
`-BindHost 127.0.0.1` (this PC only), `-Token <secret>` (then set the same **Service token** in
Admin → Model services), `-IdleUnload 900` (seconds before an idle model is unloaded to free GPU memory for Duix).

## Test one service on its own

Without the web server (loads the model from the default folder and runs once):

```powershell
services\.venv-whisper\Scripts\python services\whisper\server.py --test C:\clips\vlog.wav
services\.venv-aligner\Scripts\python services\aligner\server.py --test C:\clips\voice.wav C:\clips\script.txt zh
uv run --project C:\AI\index-tts --with fastapi --with uvicorn --with python-multipart python services\indextts\server.py --repo C:\AI\index-tts --test C:\clips\me.wav "大家好，今天去海边。" out.wav
```

With the service running: `http://127.0.0.1:8391/health` (and `/docs`) shows the model folder check (every required
file, with sizes), the GPU and whether the model is loaded. In mywebtools, **Admin → Model services → Check** shows
the same, and each service has a tool of its own: *Speech to text* (engine: GPU PC), *Text to speech / voice clone*,
*Time a script to speech*.

## API

All endpoints accept an optional `Authorization: Bearer <token>`. Errors are `{"error": {"code", "detail"}}`
(`model_missing`, `files_missing`, `model_outside_root`, `emotion_model_missing`, `out_of_memory`, ...).

| Endpoint | Body | Reply |
| --- | --- | --- |
| `GET /health?model_dir=` | – | service, version, GPU, model folder + file check, loaded state |
| `POST /load` | JSON options | loads the model now (otherwise on first use) |
| `POST /unload` | – | frees the model and GPU memory |
| `POST /transcribe` (whisper) | multipart `file`, `options` JSON: `language`, `task`, `beam_size`, `vad`, `word_timestamps`, `initial_prompt`, `hotwords`, `model_dir`, `device`, `compute_type` | NDJSON: `info`, `segment`…, `done` |
| `POST /tts` (indextts) | multipart `text`, `voice` (reference audio), optional `emo_audio`, `options` JSON: `lang` (auto/ZH/EN/JA/ES/AR), `emotion` (speaker/audio/vector/text), `emo_alpha`, `emo_vector` (8: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm), `emo_text`, `duration_factor` (0.5–2), `use_random`, `model_dir`, `device`, `bf16` | `audio/wav` |
| `POST /align` (aligner) | multipart `file` (16 kHz WAV, or anything with ffmpeg), `options` JSON: `cues` (texts in order), `language`, `star`, `model_dir`, `device` | `{cues: [{start, end, score, aligned}], words, duration}` |

## Notes on your model folders

* **IndexTTS-2.5**: `qwen0.6bemo4-merge\` is empty on the PC today. Everything works except *Emotion → Detect from the
  text*, which needs that model. Download it into the same folder:
  `hf download IndexTeam/IndexTTS-2.5 --include "qwen0.6bemo4-merge/*" --local-dir C:\Users\Administrator\Models\IndexTTS-2.5`
  (Check in Admin → Model services shows when it is complete.)
* IndexTTS loads `hf_cache\w2v-bert-2.0`, `hf_cache\bigvgan` and `hf_cache\campplus_cn_common.bin` from the model
  folder, so it does not download them again.
* **mms-300m-1130-forced-aligner**: works for any language: text is romanised (uroman) before alignment, Chinese and
  Japanese character by character. Digits are not spoken letters: lines that are only numbers get the gap between
  their neighbours.
* All three fit on one card together with Duix only if it has plenty of memory. Admin → Model services →
  *Free GPU memory before Duix renders* unloads IndexTTS after speaking; idle models unload after 15 minutes.
