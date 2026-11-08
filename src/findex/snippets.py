"""Result snippets: a ±80-character window around the best match, terms highlighted."""

from __future__ import annotations

import gzip
import unicodedata
from collections.abc import Callable, Collection
from itertools import islice
from pathlib import Path

from findex.models import DocMeta
from findex.tokenize import iter_token_spans

Highlighter = Callable[[str], str]


def brackets(s: str) -> str:
    return f"[{s}]"


def load_text(meta: DocMeta) -> str:
    """Re-read the document's text from the corpus (the index stores no text)."""
    path, _, line = meta.path.partition("#L")
    p = Path(path)
    if line:  # JSON-lines corpus: read just that line, lazily
        import json

        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            raw = next(islice(f, int(line) - 1, None), "{}")
        return str(json.loads(raw).get("text", ""))
    return p.read_text(encoding="utf-8", errors="replace")


def _best_window(hits: list[tuple[str, int, int]], width: int) -> int:
    """Index of the hit whose ±width window holds the most DISTINCT query terms.

    Sliding window, O(h): both window edges only move forward as the center advances.
    (Lab 3's version re-scanned every hit for every center -- O(h²): 1.3 s for 10
    snippets of ``the OR a OR of``, found with the profiler in Lab 8.)
    """
    counts: dict[str, int] = {}
    lo = hi = 0  # window = hits[lo:hi]
    best, best_cov = 0, -1
    for i, (_tok, center, _end) in enumerate(hits):
        while hi < len(hits) and hits[hi][1] <= center + width:
            counts[hits[hi][0]] = counts.get(hits[hi][0], 0) + 1
            hi += 1
        while hits[lo][1] < center - width:
            t = hits[lo][0]
            counts[t] -= 1
            if not counts[t]:
                del counts[t]
            lo += 1
        if len(counts) > best_cov:  # strict: the first best wins on ties
            best, best_cov = i, len(counts)
    return best


def make_snippet(
    text: str,
    terms: Collection[str],
    width: int = 80,
    highlight: Highlighter = brackets,
) -> str:
    """Pick the match position whose ±width window covers the most distinct query terms."""
    # Token offsets refer to normalize(text); NFC alone gives the same length (apostrophe
    # unification is 1:1), so we can slice the NFC text and show the original characters.
    text = unicodedata.normalize("NFC", text)
    wanted = set(terms)
    hits = [(tok, s, e) for tok, s, e in iter_token_spans(text) if tok in wanted]
    if not hits:
        return " ".join(text[: 2 * width].split()) + ("…" if len(text) > 2 * width else "")

    center = hits[_best_window(hits, width)][1]
    lo, hi = max(0, center - width), min(len(text), center + width)
    # widen to word boundaries so we never cut a word in half
    while lo > 0 and not text[lo - 1].isspace():
        lo -= 1
    while hi < len(text) and not text[hi].isspace():
        hi += 1

    out: list[str] = []
    cursor = lo
    for _tok, s, e in hits:
        if s < lo or e > hi:
            continue
        out.append(text[cursor:s])
        out.append(highlight(text[s:e]))
        cursor = e
    out.append(text[cursor:hi])
    snippet = " ".join("".join(out).split())  # collapse newlines/indentation
    return ("…" if lo > 0 else "") + snippet + ("…" if hi < len(text) else "")
