"""Tool plug-in API.

A tool declares its form as a list of ``Param`` objects; the browser renders
the form from that schema (labels are translated via
``tool.<id>.p.<param>`` keys, with the English ``label`` as fallback). Adding a
feature = one new module in ``mywebtools/tools/`` with a ``Tool`` subclass
decorated with ``@register``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..errors import ToolError

if TYPE_CHECKING:
    from ..jobs import JobContext

PARAM_TYPES = {"file", "select", "multiselect", "number", "int", "bool", "text", "textarea", "color", "range",
               "library", "info"}


@dataclass
class Param:
    name: str
    type: str
    label: str
    default: Any = None
    help: str = ""
    options: list[Any] | None = None  # [value, ...] or [[value, english_label], ...]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    accept: str = ""
    multiple: bool = False
    required: bool = False
    group: str = "main"
    advanced: bool = False
    show_if: dict[str, Any] | None = None  # {"other_param": value | [values]}
    library: str = ""
    remember: bool = True
    batch: bool = False  # file param that fans out into one job per file

    def schema(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if v not in (None, "", False) or k in ("default",)}
        if self.options is not None:
            data["options"] = [o if isinstance(o, list | tuple) else [o, str(o)] for o in self.options]
        return data


@dataclass
class Group:
    name: str
    label: str
    collapsed: bool = False


class Tool:
    id: str = ""
    category: str = "misc"
    icon: str = "tool"
    name: str = ""
    description: str = ""
    lane: str = "cpu"
    groups: list[Group] = []
    actions: tuple[str, ...] = ()  # extra instant endpoints, e.g. ("check", "preview")

    def params(self) -> list[Param]:
        return []

    def availability(self) -> tuple[bool, str]:
        """(available, reason-code) — e.g. missing optional dependency."""
        return True, ""

    async def run(self, ctx: JobContext, params: dict[str, Any]) -> None:
        raise NotImplementedError

    async def action(self, name: str, params: dict[str, Any], files: dict[str, list[Any]], app: Any) -> dict[str, Any]:
        raise ToolError("tool.no_action", action=name)

    def title_for(self, params: dict[str, Any], file_names: list[str]) -> str:
        return file_names[0] if file_names else self.name

    # ---- schema / validation -------------------------------------------
    def schema(self) -> dict[str, Any]:
        ok, reason = self.availability()
        return {
            "id": self.id, "category": self.category, "icon": self.icon, "name": self.name,
            "description": self.description, "lane": self.lane, "available": ok, "unavailable_reason": reason,
            "groups": [asdict(g) for g in self.groups] or [asdict(Group("main", ""))],
            "params": [p.schema() for p in self.params()], "actions": list(self.actions),
        }

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Validate submitted values against the schema; unknown keys are dropped."""
        clean: dict[str, Any] = {}
        for p in self.params():
            if p.type in ("file", "info"):
                continue
            value = raw.get(p.name, p.default)
            clean[p.name] = _coerce(p, value)
        return clean


def _option_values(p: Param) -> list[Any]:
    return [o[0] if isinstance(o, list | tuple) else o for o in (p.options or [])]


def _coerce(p: Param, value: Any) -> Any:
    try:
        if p.type in ("number", "range"):
            value = float(value if value not in (None, "") else p.default or 0)
            if p.min is not None:
                value = max(p.min, value)
            if p.max is not None:
                value = min(p.max, value)
            return value
        if p.type == "int":
            value = int(float(value if value not in (None, "") else p.default or 0))
            if p.min is not None:
                value = max(int(p.min), value)
            if p.max is not None:
                value = min(int(p.max), value)
            return value
        if p.type == "bool":
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
        if p.type == "select":
            values = _option_values(p)
            return value if value in values or not values else p.default
        if p.type == "multiselect":
            items = value if isinstance(value, list) else [v for v in str(value or "").split(",") if v]
            values = _option_values(p)
            items = [v for v in items if v in values] if values else items
            return items or (p.default or [])
        if p.type == "color":
            text = str(value or p.default or "#000000")
            return text if len(text) == 7 and text.startswith("#") else p.default
        if p.type == "library":
            return str(value or "")
        return str(value if value is not None else (p.default or ""))[:5000]
    except (TypeError, ValueError):
        return p.default


REGISTRY: dict[str, Tool] = {}


def register(cls: type[Tool]) -> type[Tool]:
    tool = cls()
    if not tool.id:
        raise ValueError(f"{cls.__name__} has no id")
    REGISTRY[tool.id] = tool
    return cls


def file_param(name: str, label: str, accept: str, *, multiple: bool = False, batch: bool = False,
               required: bool = True, **kw: Any) -> Param:
    return Param(name, "file", label, accept=accept, multiple=multiple, batch=batch, required=required,
                 remember=False, **kw)


def lang_options(provider: Callable[[], list[str]]) -> list[Any]:
    try:
        return provider()
    except Exception:  # noqa: BLE001 - schema must render even if a binary is missing
        return []

