"""Record types of the index.

Both are ``frozen`` (immutable -> hashable: ``__hash__`` is generated from the
fields, consistently with ``__eq__``) and ``slots`` (no per-instance ``__dict__``,
which saves ~100 bytes per object -- we have hundreds of thousands of postings).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Posting:
    """One (term, document) pair: the term occurs ``tf`` times in ``doc_id``."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()  # token offsets; empty unless built with positions


@dataclass(frozen=True, slots=True)
class DocMeta:
    """What we need to *display* a result."""

    doc_id: int
    path: str
    title: str


@dataclass(frozen=True)
class PlainPosting:
    """Same as ``Posting`` but WITHOUT slots -- only for the Lab 2 memory study."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()
