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

    def coverage(i: int) -> int:
        center = hits[i][1]
        return len({t for t, s, _ in hits if abs(s - center) <= width})

    best = max(range(len(hits)), key=coverage)  # first best wins on ties
    center = hits[best][1]
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
