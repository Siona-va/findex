"""Parallel index construction: map (``build_partial``) + reduce (``merge``).

Map:    the corpus is split into chunks of *paths* (never loaded text); every chunk owns
        a contiguous doc_id range assigned up front, so ids are globally unique.
        ``build_partial`` is a module-level function, therefore picklable, so the very
        same code runs in a plain loop, a ThreadPoolExecutor or a ProcessPoolExecutor.
        It returns a COMPACT ``PartialIndex``: flat ``array('I')`` buffers instead of
        hundreds of thousands of ``Posting`` objects. Pickling objects back to the parent
        cost more than building them (measured: 2.1 s of pickling vs 1.4 s of work);
        pickling arrays is a memcpy (0.01 s).
Reduce: ``merge`` turns the buffers into ``Posting`` objects, chunk by chunk in doc_id
        order. It is sequential -- the Amdahl ceiling of the whole build.

Workers share NO mutable state: each returns its own partial. That is why this code is
also correct on the free-threaded build (scripts/lab05_race.py shows the shared-Counter
version that is not).
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents, iter_text_files, read_document
from findex.index import Index, build_index
from findex.models import DocMeta, Posting
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[ExecutorName, ...] = ("serial", "threads", "processes")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of work. Small and picklable: a root, a first doc_id and file paths."""

    root: str
    start: int
    paths: tuple[str, ...]
    positions: bool = False


def _u32() -> array[int]:
    return array("I")


@dataclass(frozen=True, slots=True)
class PartialIndex:
    """Postings of one chunk in flat buffers (cheap to pickle).

    Term ``terms[i]`` owns postings ``offsets[i]:offsets[i+1]`` of ``doc_ids``/``tfs``.
    With positions, posting ``j`` owns ``positions[pos_offsets[j]:pos_offsets[j+1]]``.
    """

    start: int
    terms: tuple[str, ...]
    offsets: array[int]
    doc_ids: array[int]
    tfs: array[int]
    doc_lengths: dict[int, int]
    doc_meta: tuple[tuple[int, str, str], ...]
    has_positions: bool = False
    pos_offsets: array[int] = field(default_factory=_u32)
    positions: array[int] = field(default_factory=_u32)


@dataclass(frozen=True, slots=True)
class BuildStats:
    chunks: int
    map_seconds: float
    merge_seconds: float


def make_chunks(root: Path, n_chunks: int, positions: bool = False) -> list[Chunk]:
    """Split the sorted file list into ``n_chunks`` contiguous slices.

    doc_id of a file == its position in the sorted list, exactly as the serial reader
    numbers them, so every build numbers documents identically.
    """
    paths = [str(p) for p in iter_text_files(root)]
    n = max(1, min(n_chunks, len(paths)))
    size, extra = divmod(len(paths), n)
    chunks: list[Chunk] = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        chunks.append(Chunk(str(root), start, tuple(paths[start:end]), positions))
        start = end
    return chunks


def _chunk_documents(chunk: Chunk) -> Iterator[Document]:
    root = Path(chunk.root)
    for offset, p in enumerate(chunk.paths):
        doc = read_document(Path(p), chunk.start + offset, root)
        if doc is not None:
            yield doc


def build_partial(chunk: Chunk) -> PartialIndex:
    """MAP: index one chunk into flat buffers. Runs inside a worker thread or process."""
    ids: defaultdict[str, array[int]] = defaultdict(_u32)
    tfs: defaultdict[str, array[int]] = defaultdict(_u32)
    pos: defaultdict[str, list[list[int]]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    meta: list[tuple[int, str, str]] = []
    for doc in _chunk_documents(chunk):
        if chunk.positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for i, tok in enumerate(tokenize(doc.text)):
                pos_map[tok].append(i)
                length = i + 1
            for term, plist in pos_map.items():
                ids[term].append(doc.doc_id)
                tfs[term].append(len(plist))
                pos[term].append(plist)
        else:
            counts = Counter(tokenize(doc.text))
            length = counts.total()
            for term, tf in counts.items():
                ids[term].append(doc.doc_id)
                tfs[term].append(tf)
        doc_lengths[doc.doc_id] = length
        meta.append((doc.doc_id, doc.path, doc.title))

    terms = tuple(ids)  # first-appearance order == the serial build's dict order
    offsets, flat_ids, flat_tfs = array("I", [0]), _u32(), _u32()
    pos_offsets, flat_pos = array("I", [0]), _u32()
    for term in terms:
        flat_ids.extend(ids[term])
        flat_tfs.extend(tfs[term])
        offsets.append(len(flat_ids))
        for plist in pos.get(term, ()):
            flat_pos.extend(plist)
            pos_offsets.append(len(flat_pos))
    return PartialIndex(
        chunk.start,
        terms,
        offsets,
        flat_ids,
        flat_tfs,
        doc_lengths,
        tuple(meta),
        chunk.positions,
        pos_offsets,
        flat_pos,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    """REDUCE: rebuild ``Posting`` objects, chunk by chunk in doc_id order.

    Chunks cover disjoint, increasing doc_id ranges, so appending keeps every postings
    list sorted, and term insertion order equals the serial build's: the result is
    byte-identical to ``build_index`` over the whole corpus (tested).
    """
    ordered = sorted(partials, key=lambda p: p.start)
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    has_positions = bool(ordered) and ordered[0].has_positions
    for part in ordered:
        off, ids, tfs = part.offsets, part.doc_ids, part.tfs
        po, flat_pos = part.pos_offsets, part.positions
        for i, term in enumerate(part.terms):
            lo, hi = off[i], off[i + 1]
            if has_positions:
                new = [
                    Posting(ids[j], tfs[j], tuple(flat_pos[po[j] : po[j + 1]]))
                    for j in range(lo, hi)
                ]
            else:
                new = [Posting(d, f) for d, f in zip(ids[lo:hi], tfs[lo:hi], strict=True)]
            target = postings.get(term)
            if target is None:
                postings[term] = new
            else:
                target.extend(new)
        doc_lengths.update(part.doc_lengths)
        for doc_id, path, title in part.doc_meta:
            doc_meta[doc_id] = DocMeta(doc_id, path, title)
    return Index(postings, doc_lengths, doc_meta, has_positions)


def make_pool(executor: ExecutorName, workers: int, start_method: str | None = None) -> Executor:
    if executor == "threads":
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    if start_method is None:
        # Never plain `fork`: forking a multi-threaded parent (rich's progress thread,
        # a thread pool) can deadlock the child. forkserver (Linux default from 3.14)
        # or spawn (macOS/Windows default) start clean interpreters instead.
        methods = multiprocessing.get_all_start_methods()
        start_method = "forkserver" if "forkserver" in methods else "spawn"
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context(start_method)
    )


def build_parallel(
    root: Path,
    executor: ExecutorName = "processes",
    workers: int = 4,
    positions: bool = False,
    chunks_per_worker: int = 4,
    start_method: str | None = None,
    on_chunk_done: Callable[[int], None] | None = None,
) -> tuple[Index, BuildStats]:
    """Build an index with the chosen executor. Same bytes for all three (tested).

    A worker exception propagates here with the worker's original traceback attached
    (``__cause__``); pending chunks are cancelled, so the build fails loudly, never hangs.
    """
    if not root.is_dir():
        log.info("%s is not a directory: building serially from the stream", root)
        t0 = time.perf_counter()
        ix = build_index(iter_documents(root), positions=positions)
        return ix, BuildStats(1, time.perf_counter() - t0, 0.0)

    if executor == "serial":
        # one chunk == merge([build_partial(all_paths)]); more only so a progress bar moves
        n_chunks = 1 if on_chunk_done is None else 4 * chunks_per_worker
    else:
        n_chunks = workers * chunks_per_worker
    chunks = make_chunks(root, n_chunks, positions)
    log.info("%s: %d chunks, %d workers", executor, len(chunks), workers)

    t0 = time.perf_counter()
    partials: list[PartialIndex] = []
    if executor == "serial":
        for chunk in chunks:  # the plain loop
            partials.append(build_partial(chunk))
            if on_chunk_done:
                on_chunk_done(len(chunk.paths))
    else:
        with make_pool(executor, workers, start_method) as pool:
            futures: dict[Future[PartialIndex], Chunk] = {
                pool.submit(build_partial, c): c for c in chunks
            }
            try:
                for fut in as_completed(futures):
                    partials.append(fut.result())  # re-raises a worker's exception here
                    if on_chunk_done:
                        on_chunk_done(len(futures[fut].paths))
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    t_map = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = merge(partials)
    t_merge = time.perf_counter() - t0
    return index, BuildStats(len(chunks), t_map, t_merge)
