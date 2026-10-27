"""Set operations on doc_id lists sorted ascending.

Two interchangeable families:

* ``merge_*`` -- the classic two-pointer merge. O(len(a) + len(b)), no hashing,
  works on streams and compressed postings; also used for phrase positions.
* ``set_*``   -- via Python sets. Simple; implemented in C, so often faster in CPython
  on short lists, but needs both lists materialized and hashed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

type DocIds = list[int]
type Op = Callable[[Sequence[int], Sequence[int]], DocIds]
EngineName = Literal["merge", "set"]


def merge_and(a: Sequence[int], b: Sequence[int]) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: Sequence[int], b: Sequence[int]) -> DocIds:
    """Elements of ``a`` that are not in ``b`` (``a NOT b``)."""
    out: DocIds = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def set_and(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) | set(b))


def set_not(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) - set(b))


ENGINES: dict[EngineName, dict[str, Op]] = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}
