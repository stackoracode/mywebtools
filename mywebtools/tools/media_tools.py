"""Small ffmpeg utilities: audio extraction and media inspection."""

from __future__ import annotations

from ..media import extract_audio, probe
from .base import Param, Tool, file_param, register
from .transcribe import MEDIA_ACCEPT

AUDIO_FORMATS = {
    "wav": ("pcm_s16le", []),
    "mp3": ("libmp3lame", ["-q:a", "2"]),
    "m4a": ("aac", ["-b:a", "192k"]),
    "flac": ("flac", []),
}


@register
class AudioExtract(Tool):
    id = "audio_extract"
    category = "media"
    icon = "music"
    name = "Extract audio"
    description = "Save the soundtrack of a video as WAV, MP3, M4A or FLAC."

    def params(self):
        return [
            file_param("video", "Videos", MEDIA_ACCEPT, multiple=True, batch=True),
            Param("format", "select", "Format", default="wav", options=[[k, k.upper()] for k in AUDIO_FORMATS]),
            Param("rate", "select", "Sample rate", default="16000",
                  options=[["16000", "16 kHz (speech)"], ["44100", "44.1 kHz"], ["48000", "48 kHz"]]),
            Param("channels", "select", "Channels", default="1", options=[["1", "Mono"], ["2", "Stereo"]]),
            Param("start", "number", "Start", default=0, min=0, step=0.1, unit="s"),
            Param("end", "number", "End", default=0, min=0, step=0.1, unit="s", help="0 = until the end"),
            Param("loudnorm", "bool", "Normalize loudness", default=False),
        ]

    async def run(self, ctx, params):
        source = ctx.input("video")
        info = await probe(source)
        codec, extra = AUDIO_FORMATS[params["format"]]
        if params["loudnorm"]:
            extra = extra + ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
        target = ctx.out_path(f"{source.stem}.{params['format']}")
        span = (params["end"] or info.duration) - params["start"]
        await extract_audio(source, target, start=params["start"], end=params["end"], rate=int(params["rate"]),
                            channels=int(params["channels"]), codec=codec, extra=extra, duration=span or info.duration,
                            on_progress=lambda f: ctx.progress(100 * f, "media.encoding"))
        ctx.add_output(target)


@register
class MediaInfoTool(Tool):
    id = "media_info"
    category = "media"
    icon = "info"
    name = "Media info"
    description = "Resolution, orientation, HDR, codecs and duration of a file."
    lane = "light"

    def params(self):
        return [file_param("media", "Files", MEDIA_ACCEPT, multiple=True, batch=True)]

    async def run(self, ctx, params):
        info = await probe(ctx.input("media"))
        ctx.set_result(info.public())
