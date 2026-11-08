"""The ``findex`` command-line tool (typer + rich).

    findex index data/pydocs --out index.pkl --positions
    findex index data/pydocs --out index.pkl --executor processes --workers 8
    findex search index.pkl 'asyncio "event loop"' --k 5
    findex search index.pkl 'asyncio' --json | jq .title
    findex stats index.pkl
    findex crawl https://docs.example.org/ --max-pages 200 --out data/crawl.jsonl
    findex -vv search ...          # DEBUG logs (timings, cache hits) on stderr

Rules: data goes to stdout, diagnostics (logs, progress, errors) go to stderr;
expected errors print one clear line and exit with code 1 -- no traceback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import Document, iter_documents, iter_text_files
from findex.crawler import USER_AGENT, CrawlStats, FetchResult
from findex.crawler import crawl as crawl_pages
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


def _crawl_panel(stats: CrawlStats, max_pages: int) -> Table:
    t = Table(title="findex crawl", show_header=False, min_width=40)
    t.add_row("pages", f"{stats.pages} / {max_pages}")
    t.add_row("pages/s", f"{stats.pages_per_s:.1f}")
    t.add_row("in flight", str(stats.in_flight))
    t.add_row("queue depth", str(stats.queued))
    t.add_row("fetched", str(stats.fetched))
    t.add_row("errors", f"[red]{stats.errors}[/]" if stats.errors else "0")
    t.add_row("robots.txt skips", str(stats.skipped_robots))
    return t


async def _run_crawl(
    seeds: list[str],
    out_path: Path,
    log_path: Path,
    stats: CrawlStats,
    live: Live | None,
    **options: Any,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("w", encoding="utf-8") as out_f,
        log_path.open("w", encoding="utf-8") as log_f,
    ):
        log_f.write("time\tstatus\tbytes\tms\tattempts\turl\tnote\n")

        def log_sink(res: FetchResult, note: str) -> None:
            ts = datetime.now(UTC).strftime("%H:%M:%S")
            log_f.write(
                f"{ts}\t{res.status}\t{res.nbytes}\t{res.elapsed * 1000:.0f}\t"
                f"{res.attempts}\t{res.url}\t{note}\n"
            )

        async for page in crawl_pages(seeds, stats=stats, log_sink=log_sink, **options):
            # one small buffered write per page: streaming, never buffering all pages
            out_f.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")
            if live is not None:
                live.update(_crawl_panel(stats, options["max_pages"]))


@app.command(name="crawl")
def crawl_cmd(
    seeds: Annotated[list[str], typer.Argument(help="Seed URL(s)")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="JSON-lines output")] = Path(
        "data/crawl.jsonl"
    ),
    max_pages: Annotated[int, typer.Option(min=1, help="Stop after N pages")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, help="Requests in flight, total")] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="Requests in flight, per host")] = 2,
    min_delay: Annotated[
        float, typer.Option(min=0, help="Seconds between requests to one host")
    ] = 0.25,
    timeout: Annotated[float, typer.Option(min=0.1, help="Per-request timeout (s)")] = 10.0,
    allow_domain: Annotated[
        list[str] | None, typer.Option(help="Allowed domain (repeatable; default: seeds')")
    ] = None,
    log_file: Annotated[Path, typer.Option(help="Per-URL log")] = Path("crawl.log"),
    user_agent: Annotated[str, typer.Option(help="Identify yourself honestly")] = USER_AGENT,
    debug: Annotated[bool, typer.Option(help="asyncio debug mode (slow callbacks)")] = False,
    stats_json: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Crawl politely (robots.txt, per-host limits) and write pages as JSON lines."""
    stats = CrawlStats()
    options: dict[str, Any] = {
        "max_pages": max_pages,
        "concurrency": concurrency,
        "per_host": per_host,
        "min_delay": min_delay,
        "timeout": timeout,
        "allowed_domains": allow_domain,
        "user_agent": user_agent,
    }
    t0 = time.perf_counter()
    try:
        with Live(
            _crawl_panel(stats, max_pages), console=err, refresh_per_second=4, transient=False
        ) as live:
            asyncio.run(_run_crawl(seeds, out_path, log_file, stats, live, **options), debug=debug)
    except KeyboardInterrupt:
        err.print("[yellow]interrupted[/] -- pages so far are saved")
    wall = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary = {
        "pages": stats.pages,
        "fetched": stats.fetched,
        "errors": stats.errors,
        "robots_skipped": stats.skipped_robots,
        "wall": round(wall, 3),
        "pages_per_s": round(stats.pages / wall, 2) if wall else 0.0,
        "peak_rss_mb": round(rss_mb, 1),
    }
    if stats_json is not None:
        stats_json.write_text(json.dumps(summary))
    err.print(
        f"[green]✓[/] {stats.pages} pages in {wall:.1f} s → {out_path}  "
        f"(errors: {stats.errors}, log: {log_file})",
        highlight=False,
    )
