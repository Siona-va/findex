"""Streaming tokenizer.

Pipeline for one text:  NFC-normalize  ->  re.finditer  ->  casefold each match.

Tokenization policy (documented on purpose -- it decides what users can find):

* **Words** are runs of Unicode word characters (``\\w``: letters of any script,
  digits, underscore). Cyrillic, accented Latin, CJK all work.
* **Apostrophes** *inside* a word are kept and unified to ``'``:
  ``don't``, ``п'ять``, ``п’ять``, ``пʼять`` -> one token each (``п'ять``).
  Leading/trailing apostrophes (quotes) are dropped.
* **Hyphens** split words: ``free-threaded`` -> ``free``, ``threaded``.
  A user who types either half still finds the document.
* **Digits** are tokens: ``Python 3.13`` -> ``python``, ``3``, ``13``.
* **Underscores** stay inside identifiers (``__init__``, ``sys_path``) because
  the corpus is technical; tokens made only of underscores are dropped.
* **Case**: ``str.casefold()`` (not ``lower()``): ``Straße`` == ``STRASSE``.
* **Unicode**: NFC first, so ``café`` (1 code point) == ``cafe`` + U+0301.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Apostrophe look-alikes: U+2019 right single quote, U+02BC modifier letter apostrophe,
# U+2018 left single quote, U+0060 grave accent.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'"})

WORD_RE = re.compile(r"\w+(?:'\w+)*")


def normalize(text: str) -> str:
    """NFC-normalize and unify apostrophes. Length-preserving except for NFC itself."""
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES)


def iter_token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield ``(token, start, end)``; offsets point into ``normalize(text)``."""
    for m in WORD_RE.finditer(normalize(text)):
        word = m.group()
        if not word.strip("_"):
            continue
        yield word.casefold(), m.start(), m.end()


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens of ``text`` (a generator).

    Same tokens as ``iter_token_spans``, but without building a Match object and a
    3-tuple per token: ``findall`` collects one *document's* words in C, then we
    casefold them. Memory stays bounded by the largest document, not the corpus.
    (Lab 8: 1.04 s -> see README for the measured gain on 1.8M tokens.)
    """
    for word in WORD_RE.findall(normalize(text)):
        if "_" not in word or word.strip("_"):
            yield word.casefold()
