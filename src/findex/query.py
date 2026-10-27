"""Query language: a recursive-descent parser producing a tree of node objects.

Grammar (implicit AND binds tighter than OR; NOT binds tightest)::

    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*
    not_expr := "NOT" not_expr | atom
    atom     := WORD | "quoted phrase" | "(" or_expr ")"

Operators are recognized only in UPPER case; ``and``/``or``/``not`` are ordinary words.

Nodes overload ``&``, ``|`` and ``~``, so a query can be built without any string::

    Term("python") & (Term("async") | Term("await")) & ~Term("java")

Every node has ``evaluate(index) -> list[int]`` (sorted doc_ids).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator
from dataclasses import dataclass

from findex.index import Index
from findex.merge import DocIds, merge_and, merge_not, merge_or
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """The query string cannot be parsed."""


class Node:
    """Base of all query nodes: operator overloading lives here."""

    def __and__(self, other: Node) -> And:
        return And(self, other)

    def __or__(self, other: Node) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> DocIds:
        raise NotImplementedError

    def positive_terms(self) -> Iterator[str]:
        """Terms that contribute to the score (everything not under a NOT)."""
        return iter(())


@dataclass(frozen=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index) -> DocIds:
        return [p.doc_id for p in index.get(self.term, [])]

    def positive_terms(self) -> Iterator[str]:
        yield self.term

    def __str__(self) -> str:
        return self.term


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> DocIds:
        if not index.has_positions:
            raise QuerySyntaxError(
                "phrase queries need positions: rebuild the index with --positions"
            )
        # candidate documents contain every term of the phrase
        candidates = Term(self.terms[0]).evaluate(index)
        for t in self.terms[1:]:
            candidates = merge_and(candidates, Term(t).evaluate(index))
        if not candidates:
            return []
        positions = [{p.doc_id: p.positions for p in index[t]} for t in self.terms]
        return [d for d in candidates if _phrase_in_doc([pm[d] for pm in positions])]

    def positive_terms(self) -> Iterator[str]:
        yield from self.terms

    def __str__(self) -> str:
        return '"' + " ".join(self.terms) + '"'


def _phrase_in_doc(position_lists: list[tuple[int, ...]]) -> bool:
    """Two-pointer merge on positions: shift the candidate starts by one each step."""
    cand: list[int] = list(position_lists[0])
    for plist in position_lists[1:]:
        cand = merge_and([c + 1 for c in cand], plist)
        if not cand:
            return False
    return True


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> DocIds:
        if isinstance(self.right, Not):  # a NOT b: no need to build the complement
            return merge_not(self.left.evaluate(index), self.right.child.evaluate(index))
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def positive_terms(self) -> Iterator[str]:
        yield from self.left.positive_terms()
        yield from self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> DocIds:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def positive_terms(self) -> Iterator[str]:
        yield from self.left.positive_terms()
        yield from self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> DocIds:
        return merge_not(index.all_doc_ids, self.child.evaluate(index))

    def __str__(self) -> str:
        return f"NOT {self.child}"


# --- lexer + parser ---------------------------------------------------------------------

_LEX_RE = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"?|([^\s()"]+))')
_OPERATORS = {"AND", "OR", "NOT"}


def _lex(query: str) -> list[tuple[str, str]]:
    """Split into (kind, value): kind is '(' ')' 'OP' 'WORD' 'PHRASE'."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        m = _LEX_RE.match(query, pos)
        if m is None or m.end() == pos:  # pragma: no cover - the regex always advances
            raise QuerySyntaxError(f"cannot read query at position {pos}")
        lpar, rpar, phrase, word = m.groups()
        if lpar:
            tokens.append(("(", lpar))
        elif rpar:
            tokens.append((")", rpar))
        elif phrase is not None:
            tokens.append(("PHRASE", phrase))
        elif word in _OPERATORS:
            tokens.append(("OP", word))
        else:
            tokens.append(("WORD", word))
        pos = m.end()
    return tokens


def _atom_from_text(text: str) -> Node | None:
    """Normalize exactly like the indexer: one token -> Term, several -> Phrase."""
    toks = tuple(tokenize(text))
    if not toks:
        return None
    return Term(toks[0]) if len(toks) == 1 else Phrase(toks)


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_op(self, op: str) -> bool:
        return self.peek() == ("OP", op)

    def or_expr(self) -> Node | None:
        node = self.and_expr()
        while self.at_op("OR"):
            self.take()
            right = self.and_expr()
            if node is None or right is None:
                raise QuerySyntaxError("OR needs a term on both sides")
            node = Or(node, right)
        return node

    def and_expr(self) -> Node | None:
        node = self.not_expr()
        while (tok := self.peek()) is not None and tok != (")", ")") and not self.at_op("OR"):
            if self.at_op("AND"):
                self.take()
            right = self.not_expr()
            if right is None:
                continue  # e.g. a word that tokenizes to nothing ("...")
            node = right if node is None else And(node, right)
        return node

    def not_expr(self) -> Node | None:
        if self.at_op("NOT"):
            self.take()
            child = self.not_expr()
            if child is None:
                raise QuerySyntaxError("NOT needs a term after it")
            return Not(child)
        return self.atom()

    def atom(self) -> Node | None:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("unexpected end of query")
        kind, value = self.take()
        if kind == "(":
            node = self.or_expr()
            if self.peek() != (")", ")"):
                raise QuerySyntaxError("missing closing parenthesis")
            self.take()
            return node
        if kind in ("WORD", "PHRASE"):
            return _atom_from_text(value)
        raise QuerySyntaxError(f"unexpected {value!r}")


@functools.lru_cache(maxsize=512)
def parse(query: str) -> Node:
    """Parse a query string into a node tree. Raises ``QuerySyntaxError``.

    Cached: the trees are immutable (frozen dataclasses), so sharing them is safe.
    """
    parser = _Parser(_lex(query))
    node = parser.or_expr()
    if (leftover := parser.peek()) is not None:
        raise QuerySyntaxError(f"unexpected {leftover[1]!r}")
    if node is None:
        raise QuerySyntaxError("empty query")
    return node
