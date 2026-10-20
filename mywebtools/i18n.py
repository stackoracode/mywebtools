"""Server-side internationalisation helpers.

User-facing text is translated in the browser: the server sends stable message
codes plus parameters (``ToolError("duix.too_long", seconds=200)``) and English
fallbacks. This module holds the language data the processing code itself needs:
OCR / speech language codes and script-aware text joining.
"""

from __future__ import annotations

import unicodedata

UI_LANGUAGES = {"en": "English", "zh-CN": "简体中文"}

# Tesseract traineddata code -> BCP-47 tag (the browser renders the display name
# with Intl.DisplayNames, so names are localised for free).
TESSERACT_TO_BCP47 = {
    "eng": "en", "chi_sim": "zh-Hans", "chi_tra": "zh-Hant", "chi_sim_vert": "zh-Hans",
    "chi_tra_vert": "zh-Hant", "jpn": "ja", "jpn_vert": "ja", "kor": "ko", "kor_vert": "ko",
    "fra": "fr", "deu": "de", "spa": "es", "ita": "it", "por": "pt", "rus": "ru", "ukr": "uk",
    "pol": "pl", "nld": "nl", "swe": "sv", "nor": "no", "dan": "da", "fin": "fi", "tur": "tr",
    "ara": "ar", "heb": "he", "hin": "hi", "ben": "bn", "tha": "th", "vie": "vi", "ind": "id",
    "msa": "ms", "fil": "fil", "ell": "el", "ces": "cs", "hun": "hu", "ron": "ro", "bul": "bg",
    "fas": "fa", "urd": "ur", "tam": "ta", "tel": "te", "kan": "kn", "mal": "ml", "mar": "mr",
}

# Languages faster-whisper understands (subset most people need; "auto" detects).
WHISPER_LANGUAGES = [
    "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "uk", "pl", "nl", "sv", "tr",
    "ar", "he", "hi", "bn", "th", "vi", "id", "ms", "tl", "el", "cs", "hu", "ro", "fa", "ur", "ta",
]

# Hints that steer Whisper's Chinese output script (it otherwise mixes both).
CHINESE_PROMPTS = {
    "simplified": "以下是普通话的句子，使用简体中文。",
    "traditional": "以下是普通話的句子，使用繁體中文。",
}

CJK_TESSERACT = {"chi_sim", "chi_tra", "chi_sim_vert", "chi_tra_vert", "jpn", "jpn_vert", "kor", "kor_vert", "tha"}


def is_cjk_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # kana
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xAC00 <= code <= 0xD7AF  # hangul
        or 0x0E00 <= code <= 0x0E7F  # thai
        or 0x3000 <= code <= 0x303F  # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def join_words(words: list[str]) -> str:
    """Join OCR tokens: no space between CJK tokens, one space elsewhere."""
    out = ""
    for word in words:
        if not word:
            continue
        if out and not (is_cjk_char(out[-1]) and is_cjk_char(word[0])):
            out += " "
        out += word
    return out


def uses_cjk(languages: list[str]) -> bool:
    return any(lang in CJK_TESSERACT for lang in languages)


def only_latin_english(languages: list[str]) -> bool:
    return bool(languages) and all(lang == "eng" for lang in languages)


def normalize_width(text: str) -> str:
    """Full-width ASCII -> half-width (NFKC), keeping CJK characters intact."""
    return unicodedata.normalize("NFKC", text)
