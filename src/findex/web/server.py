"""The findex web service: JSON API + server-rendered UI, one FastAPI app.

    uvicorn findex.web:app            # INDEX_PATH from the environment
    findex serve --index index.pkl    # the CLI wrapper (Lab 4's entry point)

Request flow: uvicorn (sockets, event loop) -> ASGI -> Starlette (routing, middleware)
-> FastAPI (parameter parsing/validation from type hints, dependencies) -> our function
-> Pydantic (response validation + JSON) -> back out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__
from findex.fast import NumpyPostings, NumpySearchEngine
from findex.index import Index
from findex.query import QuerySyntaxError, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchEngine
from findex.semantic import Mode, Ranked, SemanticIndex, emb_dir, search_mode, snippet_for
from findex.snippets import load_text
from findex.store import IndexFormatError, load
from findex.web.models import DocumentOut, Health, SearchHit, SearchResponse, StatsOut
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
_HL0, _HL1 = "\x02", "\x03"


# --- logging (configured once per server process, at startup) ---------------------------


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()  # stdout/stderr -> the platform's log viewer
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger("findex")
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    root.propagate = False


# --- lifespan: load the index ONCE --------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings)
    app.state.started = time.monotonic()
    app.state.index = None
    app.state.semantic = None
    app.state.engines = {}
    app.state.np_postings = {}
    try:
        index = await asyncio.to_thread(load, settings.index_path)  # 1 s of unpickling
    except (FileNotFoundError, IndexFormatError) as exc:
        log.error("index not loaded: %s -- /health will answer 503", exc)
    else:
        app.state.index = index
        log.info("loaded %r from %s", index, settings.index_path)
        emb = settings.embeddings_path or emb_dir(settings.index_path)
        if emb.is_dir():
            app.state.semantic = await asyncio.to_thread(SemanticIndex, emb)
            log.info("loaded embeddings (%s) from %s", app.state.semantic.meta["model"], emb)
    yield
    if app.state.index is not None:  # shutdown (SIGTERM from the platform)
        app.state.index.close()
        log.info("index released, bye")


# --- dependencies -----------------------------------------------------------------------


def get_index(request: Request) -> Index:
    index: Index | None = request.app.state.index
    if index is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return index


def get_engine(
    request: Request,
    index: Annotated[Index, Depends(get_index)],
    scorer: Literal["bm25", "tfidf"] = "bm25",
) -> SearchEngine:
    """One SearchEngine (with its query cache) per scorer, built on first use."""
    engines: dict[tuple[int, str], SearchEngine] = request.app.state.engines
    key = (id(index), scorer)
    if key not in engines:
        impl: Scorer = BM25() if scorer == "bm25" else TfIdf()
        shared: dict[int, NumpyPostings] = request.app.state.np_postings
        if id(index) not in shared:  # the NumPy arrays are shared by both scorers
            shared[id(index)] = NumpyPostings(index)
        engines[key] = NumpySearchEngine(index, impl, postings=shared[id(index)])
    return engines[key]


def get_semantic(request: Request) -> SemanticIndex | None:
    return request.app.state.semantic


IndexDep = Annotated[Index, Depends(get_index)]
SemanticDep = Annotated[SemanticIndex | None, Depends(get_semantic)]
ModeParam = Literal["keyword", "semantic", "hybrid"]
EngineDep = Annotated[SearchEngine, Depends(get_engine)]
QueryParam = Annotated[str, Query(min_length=1, max_length=200, description="findex query")]
KParam = Annotated[int, Query(ge=1, le=100, description="results per page")]
PageParam = Annotated[int, Query(ge=1, le=1000)]


# --- the shared search logic --------------------------------------------------------------


def _bracket(s: str) -> str:
    return f"[{s}]"


def _sentinel(s: str) -> str:
    return f"{_HL0}{s}{_HL1}"


def run_search(
    engine: SearchEngine,
    q: str,
    k: int,
    page: int,
    mark: Callable[[str], str] = _bracket,
    mode: Mode = "keyword",
    sem: SemanticIndex | None = None,
) -> tuple[list[tuple[Ranked, str]], int, float]:
    """CPU-bound: parse, match, score, top-k, snippets. Returns (hits, total, took_ms)."""
    t0 = time.perf_counter()
    try:
        if mode == "keyword":
            total = len(engine.match(q))
            ranked = search_mode(engine, sem, q, page * k, mode)
        else:
            ranked = search_mode(engine, sem, q, max(page * k, 100), mode)
            total = len(ranked)
        ranked = ranked[(page - 1) * k : page * k]
        hits = [(r, snippet_for(engine, sem, r, q, mark)) for r in ranked]
    except QuerySyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"bad query: {exc}") from None
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip("'\"")) from None
    return hits, total, (time.perf_counter() - t0) * 1000


def _response(
    q: str,
    scorer: str,
    k: int,
    page: int,
    hits: list[tuple[Ranked, str]],
    total: int,
    took: float,
    mode: str = "keyword",
) -> SearchResponse:
    return SearchResponse(
        query=q,
        scorer=scorer,
        mode=mode,
        total=total,
        page=page,
        k=k,
        took_ms=round(took, 3),
        results=[
            SearchHit(doc_id=r.doc_id, title=r.title, score=round(r.score, 4), snippet=s)
            for r, s in hits
        ],
    )


def highlight_html(snippet: str) -> Markup:
    """Escape the document text, THEN turn our sentinels into <mark> -- no HTML injection."""
    return Markup(str(escape(snippet)).replace(_HL0, "<mark>").replace(_HL1, "</mark>"))


# --- the app ------------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="findex",
        version=__version__,
        description="Full-text search over the Python documentation (a student project).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.index = None
    app.state.semantic = None
    app.state.engines = {}
    app.state.np_postings = {}
    app.state.started = time.monotonic()
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["hl"] = highlight_html

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "unhandled error, request_id=%s %s %s", rid, request.method, request.url.path
            )
            response = JSONResponse(
                {"detail": "internal server error", "request_id": rid}, status_code=500
            )
        response.headers["x-request-id"] = rid
        log.info(
            "%s %s %s %d %.1fms",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
        )
        return response

    # ---- JSON API ----

    @app.get("/search", response_model=SearchResponse, tags=["api"])
    def search(
        engine: EngineDep,
        sem: SemanticDep,
        q: QueryParam,
        k: KParam = 10,
        page: PageParam = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: ModeParam = "keyword",
    ) -> SearchResponse:
        """Ranked search. A plain `def`: FastAPI runs it in its thread pool, so the CPU
        work never blocks the event loop (measured in the README).
        `mode=semantic|hybrid` needs embeddings (`findex embed`)."""
        hits, total, took = run_search(engine, q, k, page, mode=mode, sem=sem)
        return _response(q, scorer, k, page, hits, total, took, mode)

    if settings.enable_bench_routes:

        @app.get("/_bench/search-async", response_model=SearchResponse, include_in_schema=False)
        async def search_async(
            engine: EngineDep,
            q: QueryParam,
            k: KParam = 10,
            page: PageParam = 1,
            scorer: Literal["bm25", "tfidf"] = "bm25",
        ) -> SearchResponse:
            """The WRONG way, kept for the benchmark: CPU work directly in `async def`."""
            hits, total, took = run_search(engine, q, k, page)
            return _response(q, scorer, k, page, hits, total, took)

        @app.get("/_bench/search-to-thread", response_model=SearchResponse, include_in_schema=False)
        async def search_to_thread(
            engine: EngineDep,
            q: QueryParam,
            k: KParam = 10,
            page: PageParam = 1,
            scorer: Literal["bm25", "tfidf"] = "bm25",
        ) -> SearchResponse:
            hits, total, took = await asyncio.to_thread(run_search, engine, q, k, page)
            return _response(q, scorer, k, page, hits, total, took)

    @app.get("/docs/{doc_id}", response_model=DocumentOut, tags=["api"])
    def document(doc_id: int, index: IndexDep) -> DocumentOut:
        try:
            meta = index.doc_meta(doc_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}") from None
        try:
            text = load_text(meta)
        except OSError:
            raise HTTPException(status_code=404, detail="document text not available") from None
        return DocumentOut(
            doc_id=doc_id,
            title=meta.title,
            path=meta.path,
            length=index.doc_length(doc_id),
            text=text,
        )

    @app.get("/stats", response_model=StatsOut, tags=["api"])
    def stats(request: Request, index: IndexDep) -> StatsOut:
        path = settings.index_path
        return StatsOut(
            version=__version__,
            documents=index.num_docs,
            terms=len(index),
            tokens=index.total_tokens,
            avg_doc_length=round(index.avg_doc_length, 2),
            positions=index.has_positions,
            index_path=str(path),
            index_bytes=path.stat().st_size if path.is_file() else 0,
            uptime_s=round(time.monotonic() - request.app.state.started, 1),
            semantic=request.app.state.semantic.meta["model"]
            if request.app.state.semantic
            else None,
        )

    @app.get("/health", response_model=Health, tags=["ops"])
    async def health(request: Request) -> Health | JSONResponse:
        """`async def` on purpose: it never blocks, so it runs on the event loop and answers
        even while the thread pool is saturated by searches (see the README's load test)."""
        if request.app.state.index is None:
            return JSONResponse(
                {"status": "unavailable", "detail": "index not loaded"}, status_code=503
            )
        return Health(status="ok")

    # ---- HTML UI ----

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        q: str = "",
        page: PageParam = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: ModeParam = "keyword",
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", _ui_context(request, q, page, scorer, mode)
        )

    @app.get("/fragment/results", response_class=HTMLResponse, include_in_schema=False)
    def results_fragment(
        request: Request,
        q: str = "",
        page: PageParam = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: ModeParam = "keyword",
    ) -> HTMLResponse:
        """htmx target: just the results block, for search-as-you-type."""
        return templates.TemplateResponse(
            request, "_results.html", _ui_context(request, q, page, scorer, mode)
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def doc_page(request: Request, doc_id: int, index: IndexDep, q: str = "") -> HTMLResponse:
        doc = document(doc_id, index)
        text = doc.text
        if q.strip():
            terms: set[str] = set()
            with contextlib.suppress(QuerySyntaxError):
                terms = set(parse(q).positive_terms())
            text = _mark_terms(text, terms)
        return templates.TemplateResponse(request, "doc.html", {"doc": doc, "text": text, "q": q})

    def _ui_context(
        request: Request, q: str, page: int, scorer: str, mode: Mode = "keyword"
    ) -> dict[str, object]:
        sem: SemanticIndex | None = request.app.state.semantic
        ctx: dict[str, object] = {
            "q": q,
            "page": page,
            "scorer": scorer,
            "mode": mode,
            "has_semantic": sem is not None,
            "results": [],
            "total": 0,
            "took": 0.0,
            "error": "",
            "pages": 0,
            "k": 10,
        }
        if not q.strip():
            return ctx
        try:
            index = get_index(request)
        except HTTPException as exc:
            ctx["error"] = str(exc.detail)
            return ctx
        engine = get_engine(request, index, scorer)  # type: ignore[arg-type]
        try:
            hits, total, took = run_search(engine, q, 10, page, _sentinel, mode, sem)
        except HTTPException as exc:
            ctx["error"] = str(exc.detail)
            return ctx
        ctx.update(results=hits, total=total, took=took, pages=(total + 9) // 10)
        return ctx

    return app


def _mark_terms(text: str, terms: set[str]) -> str:
    """Wrap every occurrence of a query term in sentinels (rendered as <mark>)."""
    import unicodedata

    from findex.tokenize import iter_token_spans

    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    cursor = 0
    for tok, s, e in iter_token_spans(text):
        if tok in terms:
            out.append(text[cursor:s])
            out.append(_sentinel(text[s:e]))
            cursor = e
    out.append(text[cursor:])
    return "".join(out)
