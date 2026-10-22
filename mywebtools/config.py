"""Configuration.

Three layers, later wins:
1. Built-in defaults below.
2. Environment / ``.env`` file (deployment settings and secrets).
3. ``data/settings.json`` (runtime settings edited from the admin page).

Secrets (API keys, password) live only in the ``.env`` file, never in settings.json.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.getenv("MWT_ENV_FILE", str(ROOT / ".env")))
load_dotenv(ENV_FILE)

SECRET_KEYS = ("DUIX_API_KEY", "APP_PASSWORD", "LLM_API_KEY", "NODE_TOKEN")
# The GPU PC with the model services (services/): this PC when running on Windows,
# else the Windows PC in the LAN (the same one that runs Duix).
NODE_HOST = "127.0.0.1" if os.name == "nt" else "192.168.8.106"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Runtime:
    """Settings that can be changed live from the admin page."""

    retention_days: int = 7
    upload_ttl_hours: int = 24
    delete_inputs_on_success: bool = True
    max_upload_mb: int = 4096
    default_language: str = "en"
    # Video encoding (the board's CPU is the bottleneck: keep presets fast).
    video_encoder: str = "libx264"  # libx264 | h264_rkmpp | h264_v4l2m2m
    x264_preset: str = "veryfast"
    x264_crf: int = 20
    hw_bitrate: str = "8M"
    ffmpeg_threads: int = 0
    ocr_parallel: int = 3
    whisper_threads: int = 4
    # Digital avatar engine. Default: the self-hosted Duix.Avatar Docker container
    # (guiji2025/duix.avatar, "duix-avatar-gen-video") on a Windows PC in the LAN.
    avatar_engine: str = "duix_local"  # duix_local | duix_cloud
    duix_local_url: str = "http://192.168.8.106:8383"
    # Host folder mounted into the container (…/face2face/temp); the API takes
    # file names relative to it and writes "<code>-r.mp4" there.
    duix_windows_dir: str = r"C:\AI\Duix\data\face2face\temp"
    # The same folder as seen by *this* server: empty = duix_windows_dir when
    # running on Windows, else the SMB mount point below.
    duix_share_path: str = ""
    duix_timeout_seconds: int = 3600
    duix_poll_seconds: int = 5
    duix_default_model: str = ""  # library item id of the default base model video
    duix_cleanup_share: bool = True  # remove our voice/result files from the share afterwards
    duix_wol_mac: str = ""  # optional: Wake-on-LAN MAC of the Windows PC
    # Hosted DUIX API (optional; free plan = 36 credits / 30 days)
    duix_base_url: str = "https://meta.guiji.cn"
    duix_upload_height: int = 720
    duix_upload_fps: int = 25
    duix_delete_remote: bool = False

    # Large language model + speech APIs (OpenAI by default; any compatible server).
    # The API key is the LLM_API_KEY secret in .env.
    llm_provider: str = "openai"  # openai (Responses API) | openai_compatible (Chat Completions)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = ""  # picked from the list loaded from {base}/models, or typed
    llm_reasoning_effort: str = "medium"  # "" | none | minimal | low | medium | high | xhigh
    llm_timeout_seconds: int = 600
    llm_transcribe_model: str = "whisper-1"  # needs segment timestamps for subtitles
    llm_tts_model: str = "gpt-4o-mini-tts"
    llm_tts_voice: str = "alloy"
    # Speech engines. auto = the model service on the GPU PC, falling back to local
    # faster-whisper (if installed) and then the AI API.
    asr_engine: str = "auto"  # auto | node | local | api
    whisper_model: str = "base"  # local engine (this computer): name or model folder
    tts_engine: str = "indextts"  # indextts (voice clone on the GPU PC) | openai
    align_engine: str = "auto"  # auto | mms (forced alignment on the GPU PC) | basic
    subtitle_font: str = "Noto Sans CJK SC"

    # Model services on the GPU PC (services/README.md). Model folders are paths on *that* PC.
    node_whisper_url: str = f"http://{NODE_HOST}:8391"
    node_tts_url: str = f"http://{NODE_HOST}:8392"
    node_align_url: str = f"http://{NODE_HOST}:8393"
    node_timeout_seconds: int = 1800
    node_free_vram: bool = True  # unload the voice model before Duix renders on the same GPU
    whisper_model_dir: str = r"C:\Users\Administrator\Models\faster-whisper-turbo"
    whisper_device: str = "auto"  # auto | cuda | cpu
    whisper_compute_type: str = "auto"  # auto (float16 on GPU, int8 on CPU) | float16 | int8_float16 | int8
    whisper_beam_size: int = 5
    indextts_model_dir: str = r"C:\Users\Administrator\Models\IndexTTS-2.5"
    indextts_device: str = "auto"
    indextts_bf16: bool = True
    mms_model_dir: str = r"C:\Users\Administrator\Models\mms-300m-1130-forced-aligner"
    mms_device: str = "auto"

    def duix_share(self) -> Path:
        if self.duix_share_path:
            return Path(self.duix_share_path)
        return Path(self.duix_windows_dir) if os.name == "nt" else Path("/mnt/duix_temp")

    @classmethod
    def editable(cls) -> list[dict[str, Any]]:
        out = []
        for f in fields(cls):
            kind = "bool" if isinstance(f.default, bool) else "int" if isinstance(f.default, int) else "text"
            out.append({"name": f.name, "type": kind, "default": f.default})
        return out


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("MWT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("MWT_PORT", 8080))
    data_dir: Path = field(default_factory=lambda: Path(_env("MWT_DATA_DIR", str(ROOT / "data"))).resolve())
    ffmpeg: str = field(default_factory=lambda: _env("FFMPEG_CMD", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _env("FFPROBE_CMD", "ffprobe"))
    tesseract: str = field(default_factory=lambda: _env("TESSERACT_CMD", "tesseract"))
    lane_cpu: int = field(default_factory=lambda: _env_int("MWT_LANE_CPU", 1))
    lane_net: int = field(default_factory=lambda: _env_int("MWT_LANE_NET", 3))
    lane_light: int = field(default_factory=lambda: _env_int("MWT_LANE_LIGHT", 2))
    runtime: Runtime = field(default_factory=Runtime)

    # ---- derived paths -------------------------------------------------
    @property
    def uploads(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def work(self) -> Path:
        return self.data_dir / "work"

    @property
    def library(self) -> Path:
        return self.data_dir / "library"

    @property
    def models(self) -> Path:
        return self.data_dir / "models"

    @property
    def tmp(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def runtime_file(self) -> Path:
        return self.data_dir / "settings.json"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads, self.outputs, self.work, self.library, self.models, self.tmp):
            path.mkdir(parents=True, exist_ok=True)

    # ---- secrets -------------------------------------------------------
    @staticmethod
    def secret(name: str) -> str:
        return _env(name)

    def session_key(self) -> bytes:
        path = self.data_dir / ".session_key"
        if not path.is_file():
            path.write_text(secrets.token_hex(32), encoding="ascii")
            os.chmod(path, 0o600)
        return path.read_text(encoding="ascii").strip().encode()

    # ---- runtime settings ---------------------------------------------
    def load_runtime(self) -> None:
        if self.runtime_file.is_file():
            try:
                stored = json.loads(self.runtime_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = {}
            self.runtime = coerce_runtime({**asdict(self.runtime), **stored})

    def save_runtime(self, values: dict[str, Any]) -> Runtime:
        merged = coerce_runtime({**asdict(self.runtime), **values})
        tmp = self.runtime_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(merged), indent=2), encoding="utf-8")
        tmp.replace(self.runtime_file)
        self.runtime = merged
        return merged


def coerce_runtime(values: dict[str, Any]) -> Runtime:
    base = Runtime()
    clean: dict[str, Any] = {}
    for f in fields(Runtime):
        value = values.get(f.name, getattr(base, f.name))
        default = getattr(base, f.name)
        try:
            if isinstance(default, bool):
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                value = int(value)
            else:
                value = str(value).strip()
        except (TypeError, ValueError):
            value = default
        clean[f.name] = value
    return Runtime(**clean)


def write_secrets(values: dict[str, str]) -> None:
    """Merge secrets into the .env file (mode 600) and the live environment."""
    existing: dict[str, str] = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip()
    for key, value in values.items():
        if key not in SECRET_KEYS:
            raise ValueError(f"unknown secret {key}")
        if value and (not value.isascii() or re.search(r"[\s\"'\\]", value)):
            raise ValueError("invalid secret format")
        if value:
            existing[key] = value
            os.environ[key] = value
        else:
            existing.pop(key, None)
            os.environ.pop(key, None)
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ENV_FILE.open("w", encoding="utf-8") as out:
        for key, value in existing.items():
            out.write(f"{key}={value}\n")
    os.chmod(ENV_FILE, 0o600)


settings = Settings()
