"""The crawler: an ``asyncio.Queue`` frontier drained by worker tasks in a ``TaskGroup``.

    async for page in crawl(["https://example.org/"], max_pages=200, concurrency=10):
        ...

Politeness, enforced in ``Crawler._polite_fetch``:
* robots.txt is consulted before every URL (``RobotsCache``);
* a GLOBAL semaphore bounds requests in flight (our sockets, our memory);
* a PER-HOST semaphore bounds requests to any one server (their capacity);
* a per-host minimum delay between request starts (``max(min_delay, Crawl-delay)``);
* only ``allowed_domains``; ``max_pages`` successful pages, then a clean stop.

CPU work stays out of the loop: parsing a page is ~1 ms of ``html.parser``; tokenizing
and indexing happen later, in ``findex index``, never here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.html import parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, domain_allowed, host_of

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str

    def to_json(self) -> dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
        }


@dataclass
class CrawlStats:
    """Live counters for the progress display (read by the UI, written by workers)."""

    started: float = field(default_factory=time.perf_counter)
    pages: int = 0
    fetched: int = 0
    errors: int = 0
    skipped_robots: int = 0
    in_flight: int = 0
    queued: int = 0

    @property
    def pages_per_s(self) -> float:
        elapsed = time.perf_counter() - self.started
        return self.pages / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True)
class CrawlConfig:
    max_pages: int = 500
    concurrency: int = 10
    per_host: int = 2
    min_delay: float = 0.0
    timeout: float = 10.0
    allowed_domains: frozenset[str] = frozenset()
    user_agent: str = USER_AGENT


LogSink = Callable[[FetchResult, str], None]
PageSink = Callable[[Page], Awaitable[None]]


class Crawler:
    def __init__(
        self,
        client: httpx.AsyncClient,
        config: CrawlConfig,
        stats: CrawlStats | None = None,
        log_sink: LogSink | None = None,
    ) -> None:
        self.client = client
        self.cfg = config
        self.stats = stats or CrawlStats()
        self.log_sink = log_sink
        self.robots = RobotsCache(client, config.user_agent, config.timeout)
        self.frontier: asyncio.Queue[str] = asyncio.Queue()
        self.seen: set[str] = set()
        self.global_sem = asyncio.Semaphore(config.concurrency)
        self.host_sems: dict[str, asyncio.Semaphore] = {}
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.host_next: dict[str, float] = {}
        self.done = asyncio.Event()

    def enqueue(self, url: str) -> None:
        """Add a URL if it is new and in scope. Runs between awaits: no lock needed."""
        if url in self.seen or not domain_allowed(url, self.cfg.allowed_domains):
            return
        self.seen.add(url)
        self.frontier.put_nowait(url)
        self.stats.queued = self.frontier.qsize()

    async def _wait_turn(self, host: str, url: str) -> None:
        """Per-host minimum delay between request STARTS."""
        delay = max(self.cfg.min_delay, await self.robots.crawl_delay(url) or 0.0)
        if delay <= 0:
            return
        lock = self.host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            wait = self.host_next.get(host, now) - now
            if wait > 0:
                await asyncio.sleep(wait)  # the one sleep that does not block the loop
            self.host_next[host] = time.monotonic() + delay

    async def _polite_fetch(self, url: str) -> FetchResult:
        host = host_of(url)
        host_sem = self.host_sems.setdefault(host, asyncio.Semaphore(self.cfg.per_host))
        async with self.global_sem, host_sem:
            await self._wait_turn(host, url)
            self.stats.in_flight += 1
            try:
                return await fetch(self.client, url, timeout=self.cfg.timeout)
            finally:
                self.stats.in_flight -= 1

    async def _process(self, url: str, on_page: PageSink) -> None:
        if not await self.robots.allowed(url):
            self.stats.skipped_robots += 1
            self._log(FetchResult(url, url, 0, "", "", 0, 0.0, 0), "disallowed by robots.txt")
            return
        res = await self._polite_fetch(url)
        self.stats.fetched += 1
        if res.error:
            self.stats.errors += 1
        retried = f"ok after {res.attempts} attempts" if res.attempts > 1 else ""
        self._log(res, res.error or retried)
        if not res.ok_html or self.done.is_set():
            return
        parsed = parse_html(res.text)
        final = canonicalize(res.final_url) or url
        for href in parsed.links:
            if (link := canonicalize(href, base=res.final_url)) is not None:
                self.enqueue(link)
        if self.stats.pages >= self.cfg.max_pages:
            return
        self.stats.pages += 1
        now = datetime.now(UTC).isoformat(timespec="seconds")
        await on_page(Page(final, parsed.title or final, parsed.text, now))
        if self.stats.pages >= self.cfg.max_pages:
            self.done.set()

    def _log(self, res: FetchResult, note: str) -> None:
        if self.log_sink is not None:
            self.log_sink(res, note)

    async def _worker(self, on_page: PageSink) -> None:
        while True:
            url = await self.frontier.get()
            try:
                if not self.done.is_set():
                    await self._process(url, on_page)
            except Exception:  # a bug in one page must not kill the crawl...
                log.exception("unexpected error on %s", url)
                self.stats.errors += 1
            # ...but CancelledError is a BaseException: it passes through, as it must.
            finally:
                self.frontier.task_done()
                self.stats.queued = self.frontier.qsize()

    async def run(self, seeds: Iterable[str], on_page: PageSink) -> CrawlStats:
        """Crawl until ``max_pages`` pages or an empty frontier. Structured: every task
        started here has finished (or been cancelled) when this returns."""
        for seed in seeds:
            if (url := canonicalize(seed)) is not None:
                self.enqueue(url)
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(self._worker(on_page)) for _ in range(self.cfg.concurrency)]
            drained = tg.create_task(self.frontier.join())
            budget = tg.create_task(self.done.wait())
            await asyncio.wait({drained, budget}, return_when=asyncio.FIRST_COMPLETED)
            self.done.set()
            for t in (*workers, drained, budget):
                t.cancel()  # stop cleanly: in-flight fetches are cancelled at their await
        return self.stats


@contextlib.asynccontextmanager
async def make_client(
    user_agent: str = USER_AGENT,
    timeout: float = 10.0,
) -> AsyncGenerator[httpx.AsyncClient]:
    """ONE client per crawl: its connection pool is reused for every request."""
    async with httpx.AsyncClient(
        headers={"User-Agent": user_agent},
        follow_redirects=True,
        max_redirects=5,
        timeout=timeout,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ) as client:
        yield client


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Iterable[str] | None = None,
    min_delay: float = 0.0,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    log_sink: LogSink | None = None,
) -> AsyncIterator[Page]:
    """Async generator of crawled pages, yielded as they arrive.

    The crawl runs in a task owned by this generator; the pages flow through a bounded
    queue (backpressure: a slow consumer slows the crawl instead of growing memory).
    Closing the generator early (``break``) cancels the crawl task -- nothing leaks.
    """
    seeds = list(seeds)
    domains = (
        frozenset(d.lower() for d in allowed_domains)
        if allowed_domains
        else frozenset((httpx.URL(s).host or "").lower() for s in seeds)
    )
    cfg = CrawlConfig(max_pages, concurrency, per_host, min_delay, timeout, domains, user_agent)
    out: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=2 * concurrency)

    async def run() -> None:
        try:
            async with contextlib.AsyncExitStack() as stack:
                c = client or await stack.enter_async_context(make_client(user_agent, timeout))
                await Crawler(c, cfg, stats, log_sink).run(seeds, out.put)
        finally:
            with contextlib.suppress(asyncio.QueueFull):  # if full, the consumer will see
                out.put_nowait(None)  # task.done() once it has drained the queue

    task = asyncio.create_task(run())  # keep the reference: unreferenced tasks can vanish
    try:
        while (page := await out.get()) is not None:
            yield page
            if out.empty() and task.done():
                break
        await task  # re-raise anything the crawl raised
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
