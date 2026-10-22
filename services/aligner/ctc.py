"""CTC forced alignment of known text to speech (numpy only).

The MMS-300m forced-aligner model scores 20 ms frames against 28 letters
(a-z, apostrophe) plus blank. Text in any script is romanised first (uroman),
then the letters are placed on the frames with a Viterbi pass. A free "star"
token between words soaks up speech or noise that is not in the script.

Units: CJK / kana characters are aligned one by one, other scripts by word.
The result keeps the caller's cue boundaries: every cue gets the time span of
its words; a cue without alignable letters (e.g. only digits) is placed in the
gap between its neighbours.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

import numpy as np

_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_UNIT = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")
ISO3 = {"en": "eng", "zh": "zho", "ja": "jpn", "ko": "kor", "es": "spa", "ar": "ara", "fr": "fra", "de": "deu",
        "ru": "rus", "pt": "por", "it": "ita", "hi": "hin", "vi": "vie", "th": "tha", "id": "ind", "tr": "tur",
        "uk": "ukr", "nl": "nld", "pl": "pol", "ms": "msa", "fa": "fas", "he": "heb", "el": "ell"}
MAX_CELLS = 600_000_000  # frames x states kept for the back-trace (int8): ~600 MB


class AlignError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def iso3(language: str | None) -> str | None:
    lang = (language or "").lower().replace("_", "-").split("-")[0]
    return ISO3.get(lang) or (lang if len(lang) == 3 else None)


def letters(text: str, vocab: dict[str, int]) -> list[str]:
    """Lower-case, strip accents, keep only the model's characters."""
    plain = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return [c for c in plain if c in vocab and c not in ("<blank>", "<pad>", "</s>", "<unk>", "|")]


def units(cues: list[str], vocab: dict[str, int], romanize: Callable[[str], str] | None) -> list[dict[str, Any]]:
    """Alignable pieces of every cue: [{cue, text, tokens}] (pieces without letters are skipped)."""
    out = []
    for index, cue in enumerate(cues):
        for match in _UNIT.finditer(cue):
            piece = match.group(0)
            spelled = piece if piece.isascii() or romanize is None else romanize(piece)
            tokens = letters(spelled, vocab)
            if tokens:
                out.append({"cue": index, "text": piece, "tokens": tokens})
    return out


def viterbi(log_probs: np.ndarray, targets: np.ndarray, blank: int = 0) -> np.ndarray:
    """Best CTC path; returns, per frame, the index into the blank-expanded target sequence."""
    frames = log_probs.shape[0]
    ext = np.full(2 * len(targets) + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    states = len(ext)
    repeats = int(np.sum(targets[1:] == targets[:-1])) if len(targets) > 1 else 0
    if frames < len(targets) + repeats:
        raise AlignError("audio_too_short", f"{frames} frames for {len(targets)} letters")
    if frames * states > MAX_CELLS:
        raise AlignError("too_long", f"{frames} frames x {states} states")
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    neg = np.float32(-1e30)
    alpha = np.full(states, neg, dtype=np.float32)
    alpha[0] = log_probs[0, ext[0]]
    alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    from1 = np.empty(states, dtype=np.float32)
    from2 = np.empty(states, dtype=np.float32)
    for t in range(1, frames):
        from1[0] = neg
        from1[1:] = alpha[:-1]
        from2[:2] = neg
        from2[2:] = alpha[:-2]
        from2[~skip] = neg
        best = alpha.copy()
        step = np.zeros(states, dtype=np.int8)
        better = from1 > best
        best[better], step[better] = from1[better], 1
        better = from2 > best
        best[better], step[better] = from2[better], 2
        alpha = best + log_probs[t, ext]
        back[t] = step
    state = states - 1 if states == 1 or alpha[states - 1] >= alpha[states - 2] else states - 2
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return path


def align(log_probs: np.ndarray, frame_seconds: float, cues: list[str], vocab: dict[str, int],
          romanize: Callable[[str], str] | None = None, star: int | None = None) -> dict[str, Any]:
    """Time every cue. ``log_probs``: [frames, classes] log-softmax output; ``star``: index of an extra
    always-zero column (free filler between words), or None."""
    pieces = units(cues, vocab, romanize)
    if not pieces:
        raise AlignError("no_letters", "nothing alignable in the text")
    blank = vocab.get("<blank>", vocab.get("<pad>", 0))
    seq: list[int] = []
    positions: list[list[int]] = []  # piece -> its target positions
    for piece in pieces:
        if star is not None:
            seq.append(star)
        positions.append(list(range(len(seq), len(seq) + len(piece["tokens"]))))
        seq += [vocab[ch] for ch in piece["tokens"]]
    if star is not None:
        seq.append(star)
    path = viterbi(log_probs, np.asarray(seq, dtype=np.int64), blank)
    first = np.full(len(seq), -1, dtype=np.int64)
    last = np.full(len(seq), -1, dtype=np.int64)
    labelled = path % 2 == 1
    frames_idx = np.nonzero(labelled)[0]
    target_idx = path[labelled] // 2
    for f, k in zip(frames_idx.tolist(), target_idx.tolist(), strict=True):
        if first[k] < 0:
            first[k] = f
        last[k] = f
    words = []
    for piece, ks in zip(pieces, positions, strict=True):
        start_f, end_f = int(first[ks[0]]), int(last[ks[-1]]) + 1
        frame_scores = [float(log_probs[f, seq[k]]) for k in ks for f in range(int(first[k]), int(last[k]) + 1)]
        words.append({"cue": piece["cue"], "text": piece["text"], "start": round(start_f * frame_seconds, 3),
                      "end": round(end_f * frame_seconds, 3),
                      "score": round(float(np.exp(np.mean(frame_scores))) if frame_scores else 0.0, 3)})
    duration = log_probs.shape[0] * frame_seconds
    timed: list[dict[str, Any]] = []
    for index in range(len(cues)):
        mine = [w for w in words if w["cue"] == index]
        if mine:
            timed.append({"start": mine[0]["start"], "end": mine[-1]["end"], "aligned": True,
                          "score": round(sum(w["score"] for w in mine) / len(mine), 3)})
        else:
            timed.append({"start": None, "end": None, "aligned": False, "score": 0.0})
    for index, cue in enumerate(timed):  # unalignable cues: the gap between their neighbours
        if cue["aligned"]:
            continue
        prev_end = next((c["end"] for c in reversed(timed[:index]) if c["end"] is not None), 0.0)
        next_start = next((c["start"] for c in timed[index + 1:] if c["aligned"]), duration)
        cue["start"], cue["end"] = prev_end, max(next_start, prev_end + 0.3)
    return {"cues": timed, "words": words, "duration": round(duration, 3)}


def emissions_with_star(log_probs: np.ndarray) -> tuple[np.ndarray, int]:
    """Append the free "star" column (log p = 0)."""
    return np.concatenate([log_probs, np.zeros((log_probs.shape[0], 1), dtype=log_probs.dtype)], axis=1), \
        log_probs.shape[1]
