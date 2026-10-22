# mywebtools

A personal, mobile-first web toolbox. It runs natively on Windows (where it is tested first, next to the graphics
card) and on a small ARM Linux board (target: LubanCat 5, RK3588, Ubuntu 20.04 aarch64). Open it on your phone on
the LAN. Local models (faster-whisper-turbo, IndexTTS 2.5, the MMS forced aligner) run as separate
[model services](services/README.md) on the Windows PC, so later the board can run the app and use the PC as its
compute node.

| Category | Tools |
| --- | --- |
| Digital avatar | **Avatar PiP**: lip-sync a digital avatar to your vlog voice (or to a typed script spoken in your cloned voice by IndexTTS 2.5, or by an OpenAI voice) with Duix, add reviewed subtitles (MMS-aligned script or faster-whisper), and composite it picture-in-picture onto iPhone footage (portrait or landscape, HDR or SDR) for YouTube Shorts |
| Voice | **Text to speech / voice clone** (IndexTTS 2.5: voice sample library, 5 languages, emotion by sample / sliders / text, speech length; or OpenAI voices), optional aligned SRT |
| Video & speech | Video on-screen text OCR (TXT / timed SRT, optional AI clean-up) · **Speech to text** with faster-whisper-turbo on the GPU PC, faster-whisper on this computer, or the AI API (TXT/SRT/VTT/JSON, AI fix / translate / bilingual) · **Time a script to speech** (MMS forced alignment of a .txt/.srt to a recording) |
| PDF | Extract text + images (OCR fallback) · Merge · Split · PDF → images · Images (incl. HEIC) → PDF · Compress |
| Media | Extract audio · Media info (resolution, orientation, rotation, HDR, codecs) |

Planned additions such as Jupyter notebook → PDF, cover images, memes, HyperFrames rendering, or YOLOv8 can each be
added as one more module (see [Adding a tool](#adding-a-tool)).

## Architecture

```
 phone / browser ──HTTP──▶ FastAPI (single process, single asyncio event loop)
                            ├─ /static         vanilla-JS SPA (no build step), i18n en / zh-CN
                            ├─ /api/uploads    raw-body streaming uploads (start while you fill the form)
                            ├─ /api/jobs       in-memory job queue ──▶ lanes (asyncio.Queue + workers)
                            │                     cpu   (1)  ffmpeg / OCR / Whisper
                            │                     net   (2)  Duix waits (heavy steps borrow the CPU slot)
                            │                     light (2)  quick jobs
                            ├─ /api/events     Server-Sent Events: live progress on every open page
                            └─ data/           uploads · outputs · libraries · prefs.json · jobs.json
                                    │
                                    │ HTTP (LAN or localhost)
                                    ▼
   Windows PC with the GPU ── Duix.Avatar Docker :8383   lip-sync
                            ├─ whisper  :8391           faster-whisper-turbo      speech → text
                            ├─ indextts :8392           IndexTTS 2.5              text + voice sample → speech
                            └─ aligner  :8393           MMS-300m forced aligner   speech + script → timings
```

* Each model service is its own process and Python environment ([services/README.md](services/README.md)), with
  `/health`, `/load`, `/unload`, one work endpoint and a `--test` command-line mode. The app sends the model folder
  from its settings with every request, so the services need no configuration of their own.

* External programs run through `asyncio.create_subprocess_exec` and Python CPU work through `asyncio.to_thread`,
  so the loop never blocks. Cancelling a job kills its ffmpeg/tesseract process.
* Batch: choosing several files for a batch-capable tool creates one job per file under a shared `batch_id`.
* Job history is snapshotted to `data/jobs.json`. After a restart, unfinished jobs show as *interrupted*. **Resume**
  on an avatar job continues polling the existing Duix task instead of starting over.
* **Form memory:** every form starts with the values you last submitted (stored on the server, so the phone and the
  laptop share them). Unsent edits are kept as a local draft, and you can save named **presets** too.
* An optional password (Admin → Security) protects the whole site with a signed cookie.

## Digital avatar engine (default: Duix.Avatar Docker on your Windows PC)

The default engine is the self-hosted `guiji2025/duix.avatar` container (`duix-avatar-gen-video`, port 8383) on a
Windows PC in the same LAN. The hosted DUIX API is still available as an option (Admin → Engine), but its free plan
is only 36 credits (≈ 3 min) valid for 30 days.

How it talks to the container (the same protocol the Duix.Avatar desktop client uses):

1. The board writes `voice.wav` and the base model video into the folder that is mounted into the container
   (`C:\AI\Duix\data\face2face\temp` → `/code/data/temp`). The board reaches that folder through an SMB share.
2. `POST http://192.168.8.106:8383/easy/submit` `{audio_url, video_url, code, chaofen:0, watermark_switch:0, pn:1}`
3. Poll `GET /easy/query?code=…` (`status` 1 = running, 2 = done, 3 = failed; codes 9999/10002/10003 = error)
4. Read `<code>-r.mp4` back from the share, then delete the voice/result files from the share (the model video stays
   cached there).

Default settings (Admin → Engine):

| Setting | Default |
| --- | --- |
| API address | `http://192.168.8.106:8383` |
| Docker mount directory (on Windows) | `C:\AI\Duix\data\face2face\temp` |
| Path to that folder on this server | `/mnt/duix_temp` on Linux (the SMB mount); the Windows folder itself when running on Windows |
| Timeout | `3600` s |
| Default base model video | chosen from the *Base model videos* library |
| Wake-on-LAN MAC | empty (fill it in to get a **Wake PC** button) |

**Test connection** checks both that the API answers and that the shared folder is writable. Every page checks the
engine (on load and every minute). If the PC or Docker is off, a banner asks you to **start your Windows PC and Docker
Desktop** (the container restarts by itself). A job submitted while the PC is still booting waits up to 10 minutes for
the engine.

### One-time setup on the Windows PC

```powershell
# elevated PowerShell, in a checkout of this repo
powershell -ExecutionPolicy Bypass -File deploy\windows\setup-duix-host.ps1
```

This shares `C:\AI\Duix\data\face2face\temp` as `\\PC\duix_temp` (read/write for your account), opens TCP 8383 and
file sharing on private networks, sets the container's restart policy to `unless-stopped` (it then comes back whenever
Docker starts), and prints the adapter MAC for Wake-on-LAN. In Docker Desktop, enable *Start Docker Desktop when you
sign in*.

### Avatar PiP options

Base model video (library, reusable) · voice from the main video or a separate file, with an optional from/to range
(the avatar is placed at that time automatically) · output frame (Shorts 1080×1920, 720×1280, 4:5, square, landscape,
same as source, custom) · fit mode for landscape footage (blurred background / fill-and-crop / solid colour,
letterbox position) · automatic iPhone **HDR → SDR** tone-mapping · position (9 anchors) · avatar width/height in px ·
horizontal/vertical margins in px · **keep clear of the Shorts buttons/caption area** · mask shape (circle, rounded
rect with radius, rect, ellipse, squircle, hexagon, diamond, heart) · border width/colour · edge feather · drop shadow
(size, opacity) · opacity · mirror · green-screen removal (colour, similarity, blend) · crop of the avatar video (auto
head-and-shoulders, full frame, custom centre/zoom) · appear time · fade in/out · what happens when the clip ends
(hide / freeze / loop) · soundtrack (main / avatar / mix with volumes / silent) · max output length · fps · CRF ·
x264 preset · encoder (libx264, `h264_rkmpp`, `h264_v4l2m2m`).

* The **layout preview** sketches the result live as you change settings, with the model's thumbnail clipped to the
  mask and the Shorts UI zones shaded.
* **Render preview frame** runs the real ffmpeg graph on one frame, using the model video as a stand-in, so you can
  tune the layout before running Duix.
* Every generated lip-sync clip is saved to the *Lip-sync clips* library. Choose **Existing lip-sync clip** to
  re-composite it with different settings without running Duix again.
* Masks, borders and shadows are drawn once with Pillow and merged with `alphamerge`. The reference script's per-pixel
  `geq` mask was far too slow on ARM.

## Local models on the Windows PC (Admin → Model services)

| Service | Default address (app on Windows / on the board) | Default model folder |
| --- | --- | --- |
| Speech recognition · faster-whisper | `http://127.0.0.1:8391` / `http://192.168.8.106:8391` | `C:\Users\Administrator\Models\faster-whisper-turbo` |
| Voice cloning · IndexTTS 2.5 | `http://127.0.0.1:8392` / `http://192.168.8.106:8392` | `C:\Users\Administrator\Models\IndexTTS-2.5` |
| Forced alignment · MMS | `http://127.0.0.1:8393` / `http://192.168.8.106:8393` | `C:\Users\Administrator\Models\mms-300m-1130-forced-aligner` |

Per service you can set the address, the **model folder on the PC**, the device (auto / GPU / CPU) and, for Whisper,
precision and beam size, or bf16 for IndexTTS. **Check** lists every required model file with ✓/✗ and its size, the GPU
and its free memory, and whether the model is loaded. **Load model** / **Unload** warm it up or free the GPU. Engine
choices: speech recognition (auto = GPU PC → this computer → AI API), voice synthesis (IndexTTS / OpenAI) and script
timing (auto = MMS if reachable, else speech recognition). An optional **service token** protects the services on the
LAN. *Free GPU memory before Duix renders* unloads IndexTTS after speaking.

Install and start them on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File services\install.ps1
powershell -ExecutionPolicy Bypass -File services\start.ps1
```

Each one can be tested alone: from its tool (*Speech to text* with engine *GPU PC*, *Text to speech / voice clone*,
*Time a script to speech*), from `http://127.0.0.1:839x/health`, or without any web server with `--test` (see
[services/README.md](services/README.md)). Your IndexTTS-2.5 folder has an empty `qwen0.6bemo4-merge\`, so only
*Emotion → Detect from the text* is unavailable until that model is downloaded; Check says so.

## AI model (LLM & speech) and subtitles

Admin → **AI model** configures one OpenAI-compatible API (default: OpenAI):

| Setting | Notes |
| --- | --- |
| API style | *OpenAI Responses API* (default) or *Chat Completions* for compatible servers (DeepSeek, Qwen/DashScope, Ollama, vLLM, LM Studio, OpenRouter …) |
| API URL / key | `https://api.openai.com/v1`; the key is stored only in `.env` (`LLM_API_KEY`) |
| Model | **Load models** fetches `GET /models`; pick one or type any id (e.g. a new model such as `gpt-6-luna`) |
| Reasoning effort | model default, none, minimal, low, medium, high, **xhigh**. It is dropped automatically when a model rejects it |
| Speech | transcription model (`whisper-1`, needs timestamps), TTS model (`gpt-4o-mini-tts`) and default voice |

**Test connection** sends a one-word prompt and shows the reply, model, effort and latency.

### Subtitles in the avatar flow

1. **Input as text:** type a **script**. With *Voice → Type a script (synthesized voice)* the script is spoken by
   **IndexTTS 2.5 in a cloned voice** (sample: the base model video's own voice, a *Voice samples* library item, or
   your main video; language, emotion and speech length are adjustable) or by an OpenAI voice, and drives the
   lip-sync. With your own recorded voice, the script is used as the exact subtitle text.
2. **Draft:** subtitle text comes from your script, timed to the voice by **MMS forced alignment** (falls back to
   timing against recognised speech when the aligner is offline), or from **faster-whisper** speech recognition. **AI editing** can fix recognition errors and punctuation, translate, or make
   bilingual subtitles. A *Names & terms* field keeps spellings right.
3. **Review:** the job pauses in a **Review** state, and the page shows "subtitles ready for review". Meanwhile Duix
   keeps rendering on the PC. In the editor you can change text and timing, insert/merge/delete lines, edit raw SRT,
   or ask the AI to *fix*, *translate*, *make bilingual*, or follow a free-form instruction ("shorter, casual tone").
   Approve, and rendering continues.
4. **Output:** subtitles are burned in with libass using the chosen font, size, colour, outline or background box,
   position and margin. On 9:16 they sit above the Shorts caption area by default, and lines wrap per language. An
   `.srt` file is always included. **Render preview frame** shows the subtitle style too.

Resuming an interrupted job shows your last edited draft again instead of recomputing it.

## Install on the board (Ubuntu 20.04 aarch64, no Docker)

```bash
git clone https://github.com/<you>/mywebtools.git && cd mywebtools
deploy/install.sh                 # add --with-whisper for speech-to-text
sudo deploy/mount-duix-share.sh //192.168.8.106/duix_temp <windows-user>
```

`install.sh` installs ffmpeg, Tesseract (eng, chi_sim, chi_tra, jpn, kor), Noto CJK fonts (for subtitles) and
cifs-utils, then uses
[uv](https://docs.astral.sh/uv/) to create a Python 3.11 virtualenv. Ubuntu 20.04's own Python 3.8 is too old for
current wheels. Finally it installs and starts the `mywebtools` systemd service on port 8080.

The SMB mount is on-demand and `soft`, so the board boots and the UI keeps working while the PC is off.

Open `http://<board-ip>:8080` on your iPhone and use *Share → Add to Home Screen* for an app-like icon.

Run manually: `.venv/bin/python -m mywebtools` (settings come from `.env`, see `.env.example`).

### Run on Windows (current setup)

`deploy\windows\run-mywebtools.ps1` creates a venv and starts the server on `http://<pc-ip>:8080`. On the Duix PC it
uses `C:\AI\Duix\data\face2face\temp` directly (no share needed) and the model services on `127.0.0.1`. ffmpeg
has to be on PATH (`winget install Gyan.FFmpeg`); for OCR install Tesseract (UB Mannheim build). Start the model
services first (`services\start.ps1`).

Moving the app to the board later: run `deploy/install.sh` there, keep the services running on the PC (started with
the default `-BindHost 0.0.0.0`, firewall rule from `services\install.ps1`), and the board's defaults already point at
`http://192.168.8.106:839x`.

### Hardware video encoding (optional)

Ubuntu's ffmpeg has no Rockchip encoder, so libx264 (`veryfast`) is the default. If you build or install an
ffmpeg with `h264_rkmpp` (ffmpeg-rockchip), point `FFMPEG_CMD` at it and select the encoder in Admin → Settings.
Admin → System lists the hardware encoders it detects.

## Internationalisation

* UI catalogs live in `mywebtools/static/i18n/` (`en.json`, `zh-CN.json`). The language follows the browser and can be
  switched in the header. Tool labels, help texts and options fall back to the English text in the tool schema.
  `tests/test_api.py` fails if `zh-CN.json` misses any key.
* The server never returns translated prose: errors and progress use codes plus parameters
  (`err.duix.too_long {seconds, limit}`), and the browser translates them.
* OCR handles more than English: you pick Tesseract language packs (shown with localised names via
  `Intl.DisplayNames`). CJK tokens are joined without spaces, and the reference's forced-ASCII filter only applies
  when the OCR language is English (configurable). Date/time stripping also understands `2024年6月20日` and `下午3:15`.
* Speech to text: choose the spoken language or auto-detect, with a Simplified/Traditional Chinese output hint and
  CJK-aware subtitle wrapping.
* The DUIX cloud API answers in Chinese; its messages are mapped to translatable error codes, and the original text
  is kept as detail.

## Adding a tool

Create `mywebtools/tools/my_tool.py` and import it in `mywebtools/tools/__init__.py`:

```python
from .base import Param, Tool, file_param, register

@register
class NotebookToPdf(Tool):
    id = "nb_to_pdf"
    category = "pdf"
    name = "Notebook to PDF"
    description = "Render a Jupyter notebook as PDF."
    lane = "cpu"

    def params(self):
        return [file_param("nb", "Notebooks", ".ipynb", multiple=True, batch=True),
                Param("toc", "bool", "Table of contents", default=True)]

    async def run(self, ctx, params):
        src = ctx.input("nb")
        out = ctx.out_path(src.stem + ".pdf")
        ...  # await media.run_cmd([...]) or await asyncio.to_thread(...)
        ctx.progress(50, "my.stage")
        ctx.add_output(out)
```

The form, its memory and presets, batching, the queue, progress, downloads and admin status all work automatically.
Add `tool.nb_to_pdf.*` keys to `zh-CN.json`.

## Development

```bash
pip install -e ".[dev]"
pytest            # ffmpeg-based composite tests run when ffmpeg is installed; the model services are
                  # tested with stand-in models (no torch / GPU needed)
ruff check .
```

CI (GitHub Actions) runs lint, a JS syntax check and the test suite on Linux (with real ffmpeg/tesseract) and
Windows. It also runs on arm64 when the repository is public, since GitHub's arm64 runners are free only for public
repos. Pushing a `v*` tag builds a release with a deploy tarball.

API docs: `http://<host>:8080/api/docs`.

## Notes

* Secrets (`APP_PASSWORD`, `LLM_API_KEY`, `NODE_TOKEN`, optional `DUIX_API_KEY`) are only stored in `.env` (mode
  600), never in the repo or `settings.json`.
* Results are kept for 7 days and unused uploads for 24 h (Admin → Settings). Metadata such as iPhone GPS tags is
  stripped from composited videos.
