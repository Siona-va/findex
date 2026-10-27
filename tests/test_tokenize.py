"""Tokenizer: parametrized examples + Hypothesis properties."""

from __future__ import annotations

import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.tokenize import iter_token_spans, normalize, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Hello WORLD hello", ["hello", "world", "hello"], id="mixed-case"),
        pytest.param("Straße STRASSE", ["strasse", "strasse"], id="casefold"),
        pytest.param(
            "Привіт, Світе! Їжак і ґанок", ["привіт", "світе", "їжак", "і", "ґанок"], id="cyrillic"
        ),
        pytest.param("cafe\u0301 café", ["café", "café"], id="combining-mark-nfc"),
        pytest.param("Hi!!! (yes), no... ok?", ["hi", "yes", "no", "ok"], id="punctuation"),
        pytest.param("", [], id="empty"),
        pytest.param("  \n\t ... ", [], id="only-punctuation"),
        pytest.param(
            "don't п'ять п’ять пʼять 'quoted'",
            ["don't", "п'ять", "п'ять", "п'ять", "quoted"],
            id="apostrophes",
        ),
        pytest.param("free-threaded e-mail", ["free", "threaded", "e", "mail"], id="hyphens"),
        pytest.param("Python 3.13 has 2x", ["python", "3", "13", "has", "2x"], id="digits"),
        pytest.param("__init__ sys_path ___", ["__init__", "sys_path"], id="underscores"),
        pytest.param("東京 tower", ["東京", "tower"], id="cjk"),
    ],
)
def test_tokenize_examples(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_a_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)


def test_spans_point_into_normalized_text() -> None:
    text = "Мій п’ять КОТІВ"
    norm = normalize(text)
    for tok, s, e in iter_token_spans(text):
        assert norm[s:e].casefold() == tok


@given(st.text())
def test_never_yields_empty_or_whitespace_tokens(text: str) -> None:
    for tok in tokenize(text):
        assert tok
        assert not any(ch.isspace() for ch in tok)


@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    """Property: re-tokenizing the joined tokens gives the same tokens."""
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once
