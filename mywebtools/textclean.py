"""OCR text clean-up, aware of the OCR language.

Derived from the reference implementation, which forced ASCII and stripped
dates. Here both are options so Chinese/Japanese/Korean text survives.
"""

from __future__ import annotations

import calendar
import html
import re

from .i18n import normalize_width

_MONTHS = [m for m in tuple(calendar.month_name)[1:] + tuple(calendar.month_abbr)[1:] if m]
_MONTH = "(?:" + "|".join(re.escape(m) for m in sorted(_MONTHS, key=len, reverse=True)) + ")"

_DATE_PATTERNS = [
    # March 5, 2024 / Mar 5th
    re.compile(r"\b" + _MONTH + r"\.?[ \t]+[0-9]{1,2}(?:st|nd|rd|th)?(?:,?[ \t]+[12][0-9]{3})?\b", re.I),
    # 5 March 2024
    re.compile(r"\b[0-9]{1,2}(?:st|nd|rd|th)?[ \t]+" + _MONTH + r"\.?(?:[ \t]+[12][0-9]{3})?\b", re.I),
    # 2024-06-20, 20/06/2024, 20240620, optional time
    re.compile(
        r"(?<![0-9])(?:[12][0-9]{3}[-/._][0-9]{1,2}[-/._][0-9]{1,2}|[0-9]{1,2}[-/._][0-9]{1,2}[-/._][12][0-9]{3}|[12][0-9]{7})"
        r"(?:[T _-][0-9]{1,2}[-:._][0-9]{2}(?:[-:._][0-9]{2})?)?(?![0-9])"
    ),
    # CJK: 2024年6月20日 / 6月20日 / 2024年6月 / 2024년 6월 20일
    re.compile(r"(?:[12][0-9]{3}\s*[年년]\s*)?[0-9]{1,2}\s*[月월]\s*(?:[0-9]{1,2}\s*[日号號일])?"),
    re.compile(r"[12][0-9]{3}\s*[年년]"),
    # clock 12:30, 12:30:15.5, 下午3:15
    # (?<!\d) rather than \b: CJK characters count as word characters, so "下午3:15" has no \b
    re.compile(r"(?:[上下]午|AM|PM)?\s*(?<![0-9])[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:\.[0-9]+)?(?![0-9])(?:\s*(?:AM|PM))?", re.I),
    re.compile(r"(?<![0-9])[0-9]{1,2}[-_][0-9]{2}[-_][0-9]{2}(?![0-9])"),
    re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])"),
    re.compile(r"\b" + _MONTH + r"\b", re.I),
]
_ENCODED = re.compile(r"\\(?:u[0-9a-f]{4}|U[0-9a-f]{8}|x[0-9a-f]{2}|N\{[^}]*\})", re.I)


def clean_text(value: str, *, ascii_only: bool = False, remove_dates: bool = False, normalize: bool = True) -> str:
    text = html.unescape(str(value))
    text = _ENCODED.sub(" ", text)
    if normalize:
        text = normalize_width(text)
    if ascii_only:
        text = text.encode("ascii", "ignore").decode("ascii")
    if remove_dates:
        for pattern in _DATE_PATTERNS:
            text = pattern.sub(" ", text)
    # drop control characters but keep newlines and every printable script
    text = "".join(ch if ch == "\n" or ch.isprintable() else " " for ch in text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
