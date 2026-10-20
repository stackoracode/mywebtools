from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """An expected failure with a translatable message code.

    ``code`` maps to ``err.<code>`` in the browser's i18n catalog; ``params``
    are interpolated into the translated string; ``detail`` is free-form,
    untranslated diagnostic text (stderr tail, remote message, ...).
    """

    def __init__(self, code: str, detail: str = "", **params: Any) -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
        self.params = params

    def public(self) -> dict[str, Any]:
        return {"code": self.code, "params": self.params, "detail": self.detail}
