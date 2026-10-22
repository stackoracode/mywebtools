"""Tool registry. Importing this package registers every built-in tool.

To add a feature, create a module here with a ``Tool`` subclass decorated with
``@register`` and add it to the import list below (see README "Adding a tool").
"""

from . import align, avatar, media_tools, pdf_tools, transcribe, video_ocr, voice  # noqa: F401
from .base import REGISTRY, Param, Tool, register  # noqa: F401
