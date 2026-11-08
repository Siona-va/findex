"""The ``findex`` command-line tool (typer + rich).

    findex index data/pydocs --out index.pkl --positions
    findex index data/pydocs --out index.pkl --executor processes --workers 8
    findex search index.pkl 'asyncio "event loop"' --k 5
    findex search index.pkl 'asyncio' --json | jq .title
    findex stats index.pkl
    findex -vv search ...          # DEBUG logs (timings, cache hits) on stderr

Rules: data goes to stdout, diagnostics (logs, progress, errors) go to stderr;
expected errors print one clear line and exit with code 1 -- no traceback.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import Document, iter_documents, iter_text_files
from findex.index import build_index
from findex.parallel import build_parallel
from findex.query import QuerySyntaxError, parse
from findex.scoring import ScorerName, make_scorer
from findex.search import SearchEngine
from findex.snippets import Highlighter
from findex.store import IndexFormatError, open_index, save

log = logging.getLogger(__name__)

app = typer.Typer(
    help="findex -- full-text search over your own corpus.",
    no_args_is_help=True,
    add_completion=False,
)
out = Console()  # stdout: data
err = Console(stderr=True)  # stderr: diagnostics

_HL_START, _HL_END = "\x02", "\x03"  # sentinels around highlighted terms in snippets


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


def _fail(message: str) -> typer.Exit:
    err.print(f"[bold red]error:[/] {message}", highlight=False)
    return typer.Exit(code=1)


def _version(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v = INFO, -vv = DEBUG")
    ] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version")
    ] = False,
) -> None:
    """Configure logging ONCE, here, at the entry point (never in library code)."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False)],  # -> stderr
        force=True,
    )


def _with_progress(
    docs: Iterable[Document], progress: Progress, total: int | None
) -> Iterator[Document]:
    task = progress.add_task("indexing", total=total)
    for doc in docs:
        yield doc
        progress.advance(task)


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Directory of text files or a .jsonl file")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Index file (.pkl or .json)")],
    positions: Annotated[bool, typer.Option(help="Store token positions (phrases)")] = False,
    limit: Annotated[int | None, typer.Option(help="Only the first N documents")] = None,
    executor: Annotated[
        ExecutorChoice, typer.Option(help="How to parallelize the build (Lab 5)")
    ] = ExecutorChoice.processes,
    workers: Annotated[
        int, typer.Option(min=1, help="Worker threads/processes (default: CPU count)")
    ] = os.cpu_count() or 1,
) -> None:
    """Build an index from a corpus and save it."""
    if not corpus.exists():
        raise _fail(f"corpus not found: {corpus}")
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=err,
        transient=True,
    )
    if limit is not None or not corpus.is_dir():
        # streaming serial build (Lab 1-4 path): needed for --limit and for .jsonl corpora
        total = sum(1 for _ in iter_text_files(corpus)) if corpus.is_dir() else None
        if total is not None and limit is not None:
            total = min(total, limit)
        docs: Iterable[Document] = iter_documents(corpus)
        if limit is not None:
            docs = islice(docs, limit)
        with progress:
            ix = build_index(_with_progress(docs, progress, total), positions=positions)
    else:
        total = sum(1 for _ in iter_text_files(corpus))
        with progress:
            task = progress.add_task(f"indexing ({executor.value} × {workers})", total=total)
            ix, stats = build_parallel(
                corpus,
                executor.value,
                workers=workers,
                positions=positions,
                on_chunk_done=lambda n: progress.advance(task, n),
            )
        log.info(
            "map %.2f s, merge %.2f s, %d chunks",
            stats.map_seconds,
            stats.merge_seconds,
            stats.chunks,
        )
    save(ix, out_path)
    err.print(f"[green]✓[/] {ix!r} → {out_path}", highlight=False)


def _sentinels(s: str) -> str:
    return f"{_HL_START}{s}{_HL_END}"


def _plain(s: str) -> str:
    return s


def _snippet_text(snippet: str) -> Text:
    """Turn sentinel-marked snippet into rich Text (no markup parsing of document text)."""
    text = Text()
    for i, part in enumerate(snippet.replace(_HL_END, _HL_START).split(_HL_START)):
        text.append(part, style="bold magenta" if i % 2 else "")
    return text


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Index file")],
    query: Annotated[str, typer.Argument(help='e.g. python (async OR await) "event loop"')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Number of results")] = 10,
    scorer: Annotated[ScorerChoice, typer.Option(help="Ranking function")] = ScorerChoice.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines on stdout")] = False,
) -> None:
    """Ranked search with highlighted snippets."""
    scorer_name: ScorerName = "bm25" if scorer is ScorerChoice.bm25 else "tfidf"
    try:
        parse(query)
        with open_index(index_path) as ix:
            engine = SearchEngine(ix, make_scorer(scorer_name))
            results = engine.search(query, k=k)
            mark: Highlighter = _sentinels if not as_json else _plain
            snippets = [engine.snippet(r, query, highlight=mark) for r in results]
    except (FileNotFoundError, IndexFormatError, QuerySyntaxError) as exc:
        raise _fail(str(exc)) from None

    if as_json:
        for rank, (r, snip) in enumerate(zip(results, snippets, strict=True), start=1):
            row = {
                "rank": rank,
                "score": round(r.score, 4),
                "doc_id": r.doc_id,
                "title": r.title,
                "path": r.path,
                "snippet": snip,
            }
            print(json.dumps(row, ensure_ascii=False))
        return
    if not results:
        err.print("no documents match")
        return
    table = Table(title=f"{query}  ·  {scorer_name}", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("document", overflow="fold", max_width=40)
    table.add_column("snippet", overflow="fold")
    for rank, (r, snip) in enumerate(zip(results, snippets, strict=True), start=1):
        doc = Text(r.title, style="bold")
        doc.append(f"\n{r.path}", style="dim")
        table.add_row(str(rank), f"{r.score:.2f}", doc, _snippet_text(snip))
    out.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Index file")],
    top: Annotated[int, typer.Option(help="How many top terms")] = 20,
) -> None:
    """Corpus statistics from a saved index."""
    try:
        with open_index(index_path) as ix:
            cf = ix.collection_frequencies()
            summary = Table(show_header=False, title="findex stats")
            summary.add_row("documents", f"{ix.num_docs:,}")
            summary.add_row("tokens", f"{ix.total_tokens:,}")
            summary.add_row("vocabulary", f"{len(ix):,}")
            summary.add_row("avg doc length", f"{ix.avg_doc_length:,.1f}")
            summary.add_row("positions", "yes" if ix.has_positions else "no")
            terms = Table(title=f"top-{top} terms")
            terms.add_column("#", justify="right")
            terms.add_column("term")
            terms.add_column("count", justify="right")
            terms.add_column("df", justify="right")
            for rank, (term, n) in enumerate(cf.most_common(top), start=1):
                terms.add_row(str(rank), term, f"{n:,}", f"{ix.df(term):,}")
    except (FileNotFoundError, IndexFormatError) as exc:
        raise _fail(str(exc)) from None
    out.print(summary)
    out.print(terms)
