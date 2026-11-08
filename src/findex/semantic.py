"""Semantic and hybrid search (Lab 8 M3).

Documents are cut into ~200-word chunks; every chunk becomes a unit-length vector
(an *embedding*). A query is embedded the same way, and ``E @ q`` -- one matrix-vector
product -- gives the cosine similarity to every chunk. A document scores as its BEST
chunk. Hybrid search fuses the keyword ranking and the semantic ranking with
Reciprocal Rank Fusion: ``score(d) = Σ 1 / (60 + rank_i(d))``.

Two embedders behind one Protocol:

* ``MiniLMEmbedder`` -- ``sentence-transformers`` (``all-MiniLM-L6-v2``, 384-d): a neural
  model, the real thing. ``pip install 'findex[semantic]'``; downloads ~80 MB once.
* ``LSAEmbedder`` -- Latent Semantic Analysis (TF-IDF + truncated SVD, 1990): learned
  from OUR corpus, offline, no model download. Words that co-occur get nearby vectors,
  so it catches some synonymy; much weaker than a neural model.

Files written by ``findex embed`` next to the index, in ``<index>.emb/``:
``vectors.npy`` float32[N_chunks, d] (L2-normalized), ``chunk_doc.npy`` int32[N_chunks],
``chunk_start.npy`` int32[N_chunks] (first word of the chunk), ``meta.json``,
and ``lsa.pkl`` for the LSA model.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.snippets import load_text
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]
Mode = Literal["keyword", "semantic", "hybrid"]
DEFAULT_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> F32: ...  # (n, d), rows L2-normalized


def _untyped(module: str) -> Any:
    """Import an optional, untyped third-party module. ``Any`` marks the one place where
    findex talks to code pyright cannot check (scikit-learn, sentence-transformers)."""
    import importlib

    return importlib.import_module(module)


def _normalize(m: npt.NDArray[Any]) -> F32:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def analyzer(text: str) -> list[str]:
    """findex's own tokenizer, so LSA sees exactly the tokens the index sees."""
    return list(tokenize(text))


class LSAEmbedder:
    """TF-IDF (sublinear tf) -> TruncatedSVD -> L2-normalize. Needs scikit-learn."""

    name = "lsa"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._model: Any = None

    def fit(self, texts: Sequence[str]) -> LSAEmbedder:
        decomposition = _untyped("sklearn.decomposition")
        text = _untyped("sklearn.feature_extraction.text")
        pipeline = _untyped("sklearn.pipeline")
        # min_df=2 drops typos and one-off identifiers; a tiny corpus may need min_df=1
        vectorizer: Any = None
        x: Any = None
        for min_df in (2, 1):
            vectorizer = text.TfidfVectorizer(
                analyzer=analyzer, sublinear_tf=True, min_df=min_df, max_df=0.5
            )
            try:
                x = vectorizer.fit_transform(list(texts))
                break
            except ValueError:  # "After pruning, no terms remain"
                continue
        dim = max(1, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        svd = decomposition.TruncatedSVD(n_components=dim, random_state=0)
        svd.fit(x)
        self._model = pipeline.make_pipeline(vectorizer, svd)  # both steps already fitted
        return self

    def encode(self, texts: Sequence[str]) -> F32:
        return _normalize(self._model.transform(list(texts)))

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self._model))

    @classmethod
    def load(cls, path: Path) -> LSAEmbedder:
        emb = cls()
        emb._model = pickle.loads(path.read_bytes())  # our own file, written by `embed`
        return emb


class MiniLMEmbedder:
    """A sentence-transformers model (CPU is fine for a corpus of this size)."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        st = _untyped("sentence_transformers")  # optional dependency: findex[semantic]
        self.name = model
        self._model: Any = st.SentenceTransformer(model)

    def encode(self, texts: Sequence[str]) -> F32:
        vecs: Any = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


# --- chunking ---------------------------------------------------------------------------


def chunks(text: str, words: int = 200) -> Iterator[tuple[int, str]]:
    """(first word index, chunk text) for consecutive ``words``-word windows."""
    ws = text.split()
    for start in range(0, max(len(ws), 1), words):
        piece = ws[start : start + words]
        if piece:
            yield start, " ".join(piece)


@dataclass
class Corpus:
    texts: list[str]
    doc: list[int]
    start: list[int]


def chunk_index(index: Index, words: int = 200) -> Corpus:
    texts: list[str] = []
    doc: list[int] = []
    start: list[int] = []
    for d in index.all_doc_ids:
        meta = index.doc_meta(d)
        try:
            text = load_text(meta)
        except OSError:
            log.warning("no text for doc %d (%s); not embedded", d, meta.path)
            continue
        for s, piece in chunks(f"{meta.title}\n{text}", words):
            texts.append(piece)
            doc.append(d)
            start.append(s)
    return Corpus(texts, doc, start)


# --- build + load -----------------------------------------------------------------------


def emb_dir(index_path: Path) -> Path:
    return index_path.with_name(index_path.name + ".emb")


def build_embeddings(
    index: Index,
    out: Path,
    model: str = DEFAULT_MODEL,
    words: int = 200,
    batch: int = 256,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Chunk, embed, normalize, save. Returns the number of chunks."""
    corpus = chunk_index(index, words)
    embedder: Embedder = (
        LSAEmbedder().fit(corpus.texts) if model == "lsa" else MiniLMEmbedder(model)
    )
    parts: list[F32] = []
    for i in range(0, len(corpus.texts), batch):
        parts.append(embedder.encode(corpus.texts[i : i + batch]))
        if on_progress:
            on_progress(min(i + batch, len(corpus.texts)), len(corpus.texts))
    vectors = np.vstack(parts) if parts else np.zeros((0, 1), dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    np.save(out / "chunk_doc.npy", np.asarray(corpus.doc, dtype=np.int32))
    np.save(out / "chunk_start.npy", np.asarray(corpus.start, dtype=np.int32))
    if isinstance(embedder, LSAEmbedder):
        embedder.save(out / "lsa.pkl")
    meta = {
        "model": model,
        "dim": int(vectors.shape[1]),
        "chunks": len(corpus.texts),
        "chunk_words": words,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(corpus.texts)


class SemanticIndex:
    """Brute-force vector search: fine up to a few hundred thousand chunks."""

    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        self.meta: dict[str, Any] = json.loads((path / "meta.json").read_text())
        self.vectors: F32 = np.load(path / "vectors.npy", mmap_mode="r")
        self.chunk_doc: I32 = np.load(path / "chunk_doc.npy")
        self.chunk_start: I32 = np.load(path / "chunk_start.npy")
        self.words = int(self.meta["chunk_words"])
        if embedder is not None:
            self.embedder: Embedder = embedder
        elif self.meta["model"] == "lsa":
            self.embedder = LSAEmbedder.load(path / "lsa.pkl")
        else:
            self.embedder = MiniLMEmbedder(self.meta["model"])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float, int]]:
        """Top ``k`` documents as (doc_id, cosine, best chunk row)."""
        q = self.embedder.encode([query])[0]
        sims = self.vectors @ q  # cosine: rows and q are unit vectors
        n_docs = int(self.chunk_doc.max()) + 1 if len(self.chunk_doc) else 0
        best = np.full(n_docs, -np.inf, dtype=np.float32)
        np.maximum.at(best, self.chunk_doc, sims)  # best chunk per document
        present = np.flatnonzero(np.isfinite(best))
        if len(present) > k:
            present = present[np.argpartition(-best[present], k - 1)[:k]]
        order = present[np.lexsort((present, -best[present]))]
        # which chunk was the best one, for the snippet
        rows: list[tuple[int, float, int]] = []
        for d in order:
            idx = np.flatnonzero(self.chunk_doc == d)
            rows.append((int(d), float(best[d]), int(idx[np.argmax(sims[idx])])))
        return rows

    def chunk_text(self, index: Index, row: int, words: int = 40) -> str:
        d, start = int(self.chunk_doc[row]), int(self.chunk_start[row])
        meta = index.doc_meta(d)
        try:
            ws = f"{meta.title}\n{load_text(meta)}".split()
        except OSError:
            return ""
        piece = " ".join(ws[start : start + words])
        return piece + ("…" if start + words < len(ws) else "")


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: no score normalization, only ranks (1-based)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


# --- the three modes behind one call ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ranked:
    doc_id: int
    score: float
    title: str
    path: str
    chunk_row: int | None = None  # set when the semantic side found this document


def search_mode(
    engine: Any,  # a findex.search.SearchEngine (Any: avoids an import cycle)
    sem: SemanticIndex | None,
    query: str,
    k: int = 10,
    mode: Mode = "keyword",
    depth: int = 100,
) -> list[Ranked]:
    """``keyword`` = the query language + BM25/TF-IDF; ``semantic`` = embeddings;
    ``hybrid`` = RRF over the top ``depth`` of each. Keyword errors (a bad query)
    propagate; a semantic mode without embeddings raises ``LookupError``."""
    index: Index = engine.index
    if mode == "keyword":
        return [Ranked(r.doc_id, r.score, r.title, r.path) for r in engine.search(query, k)]
    if sem is None:
        raise LookupError("no embeddings for this index: run `findex embed` first")
    semantic = sem.search(query, k=max(k, depth) if mode == "hybrid" else k)
    rows = {d: row for d, _, row in semantic}
    if mode == "semantic":
        return [
            Ranked(d, s, index.doc_meta(d).title, index.doc_meta(d).path, row)
            for d, s, row in semantic
        ]
    from findex.query import QuerySyntaxError

    try:
        keyword = [r.doc_id for r in engine.search(query, depth)]
    except QuerySyntaxError:
        keyword = []  # free text that is not a valid query: semantic side only
    fused = rrf([keyword, [d for d, _, _ in semantic]])[:k]
    return [
        Ranked(d, s, index.doc_meta(d).title, index.doc_meta(d).path, rows.get(d)) for d, s in fused
    ]


def snippet_for(
    engine: Any,
    sem: SemanticIndex | None,
    r: Ranked,
    query: str,
    highlight: Callable[[str], str],
) -> str:
    """A document that contains a query term gets the term-window snippet (Lab 3);
    one found ONLY by meaning shows its best-matching chunk instead."""
    from findex.query import QuerySyntaxError, parse
    from findex.search import SearchResult

    index: Index = engine.index
    try:
        terms = set(parse(query).positive_terms())
    except QuerySyntaxError:
        terms = set[str]()
    has_term = any(p.doc_id == r.doc_id for t in terms for p in index.get(t, []))
    if sem is not None and r.chunk_row is not None and not has_term:
        return sem.chunk_text(index, r.chunk_row)
    as_result = SearchResult(r.score, r.doc_id, r.title, r.path)
    return str(engine.snippet(as_result, query, highlight=highlight))
