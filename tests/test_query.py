"""Lab 3: the parser is tested with TREES, not strings."""

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse


def test_implicit_and_binds_tighter_than_or() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_explicit_and_and_parentheses() -> None:
    assert parse("(a OR b) AND c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_not_and_phrase() -> None:
    assert parse('python NOT java "event loop"') == And(
        And(Term("python"), Not(Term("java"))), Phrase(("event", "loop"))
    )


def test_operators_build_the_same_tree() -> None:
    q = Term("python") & (Term("async") | Term("await")) & ~Term("java")
    assert q == parse("python (async OR await) NOT java")


def test_terms_are_normalized_like_the_index() -> None:
    assert parse("Straße") == Term("strasse")
    assert parse("free-threaded") == Phrase(("free", "threaded"))


@pytest.mark.parametrize("bad", ["", "a OR", "(a b", "a )", "NOT", "...", "OR b"])
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_evaluate() -> None:
    ix = build_index(
        [
            Document(0, "0", "the event loop runs", "0"),
            Document(1, "1", "loop the event", "1"),
            Document(2, "2", "java event loop", "2"),
        ],
        positions=True,
    )
    assert parse('"event loop"').evaluate(ix) == [0, 2]
    assert parse('"event loop" NOT java').evaluate(ix) == [0]
    assert parse("NOT java").evaluate(ix) == [0, 1]
    assert parse("runs OR java").evaluate(ix) == [0, 2]
