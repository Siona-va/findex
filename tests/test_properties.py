"""Hypothesis properties: oracles, invariants, round trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.merge import merge_and, merge_not, merge_or
from findex.query import And, Node, Not, Or, Term
from findex.store import load, save

sorted_ids = st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted)

words = st.sampled_from(["cat", "dog", "кіт", "пес", "the", "a", "zebra", "loop"])
docs_strategy = st.lists(st.lists(words, max_size=12).map(" ".join), min_size=1, max_size=8)


@given(sorted_ids, sorted_ids)
def test_merges_agree_with_set_oracle(a: list[int], b: list[int]) -> None:
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(docs_strategy)
def test_postings_are_always_sorted_and_complete(texts: list[str]) -> None:
    ix = build_index(Document(i, str(i), t) for i, t in enumerate(texts))
    for term in ix:
        ids = [p.doc_id for p in ix[term]]
        assert ids == sorted(set(ids))
        assert all(p.tf > 0 for p in ix[term])
    assert sum(p.tf for t in ix for p in ix[t]) == sum(ix.doc_length(d) for d in ix.all_doc_ids)


@pytest.mark.slow
@settings(max_examples=40, deadline=None)
@given(docs_strategy, st.booleans(), st.sampled_from(["pkl", "json"]))
def test_load_save_round_trip(texts: list[str], positions: bool, ext: str) -> None:
    ix = build_index((Document(i, str(i), t) for i, t in enumerate(texts)), positions=positions)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"ix.{ext}"
        save(ix, path)
        assert load(path) == ix


def trees(depth: int = 3) -> st.SearchStrategy[Node]:
    leaf = words.map(Term)
    return st.recursive(
        leaf,
        lambda kids: st.one_of(
            st.builds(And, kids, kids), st.builds(Or, kids, kids), st.builds(Not, kids)
        ),
        max_leaves=8,
    )


@given(docs_strategy, trees())
def test_query_evaluation_matches_set_semantics(texts: list[str], tree: Node) -> None:
    """Oracle: evaluating a random query tree == the same logic on Python sets."""
    ix = build_index(Document(i, str(i), t) for i, t in enumerate(texts))
    universe = set(ix.all_doc_ids)

    def oracle(node: Node) -> set[int]:
        match node:
            case Term(term=t):
                return {p.doc_id for p in ix.get(t, [])}
            case And(left=lhs, right=rhs):
                return oracle(lhs) & oracle(rhs)
            case Or(left=lhs, right=rhs):
                return oracle(lhs) | oracle(rhs)
            case Not(child=c):
                return universe - oracle(c)
            case _:
                raise AssertionError(node)

    assert tree.evaluate(ix) == sorted(oracle(tree))
