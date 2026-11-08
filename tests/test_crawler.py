"""Lab 6: the crawler against fake sites served by httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Coroutine
from typing import Any

import httpx
import pytest

from findex.crawler import FetchResult, RobotsCache, canonicalize, crawl, fetch
from findex.crawler.fetch import backoff_delay, retry_after_seconds
from findex.crawler.html import parse_html


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class Sleeps:
    """Records requested sleeps instead of sleeping (fast, deterministic tests)."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def client_for(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,  # type: ignore[arg-type]
        max_redirects=5,
    )


# --- urls -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM", "http://example.com/"),
        ("https://example.com:443/a/", "https://example.com/a/"),
        ("http://example.com:8080/a#frag", "http://example.com:8080/a"),
        ("http://example.com/x/index.html", "http://example.com/x/"),
        ("http://example.com/s?b=2&a=1", "http://example.com/s?a=1&b=2"),
        ("mailto:someone@example.com", None),
        ("javascript:void(0)", None),
    ],
)
def test_canonicalize(raw: str, expected: str | None) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_relative() -> None:
    assert canonicalize("../b.html#x", base="http://h/docs/a/page.html") == "http://h/docs/b.html"


# --- fetch ------------------------------------------------------------------------------


def test_retries_5xx_then_succeeds() -> None:
    calls = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        calls[req.url.path] += 1
        if calls[req.url.path] < 3:
            return httpx.Response(503)
        return httpx.Response(200, html="<p>ok</p>")

    sleeps = Sleeps()

    async def go() -> FetchResult:
        async with client_for(handler) as c:
            return await fetch(c, "http://h/p", sleep=sleeps, base_delay=0.5)

    res = run(go())
    assert res.status == 200 and res.attempts == 3 and res.ok_html
    assert len(sleeps.delays) == 2
    assert 0.5 <= sleeps.delays[0] <= 1.5 and 1.0 <= sleeps.delays[1] <= 2.0  # 2**n + jitter


def test_retry_after_is_honored() -> None:
    hits = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="fine")

    sleeps = Sleeps()

    async def go() -> FetchResult:
        async with client_for(handler) as c:
            return await fetch(c, "http://h/", sleep=sleeps)

    assert run(go()).status == 200
    assert sleeps.delays == [7.0]


def test_404_is_not_retried() -> None:
    sleeps = Sleeps()

    async def go() -> FetchResult:
        async with client_for(lambda req: httpx.Response(404)) as c:
            return await fetch(c, "http://h/missing", sleep=sleeps)

    res = run(go())
    assert (res.status, res.attempts, res.error) == (404, 1, "HTTP 404")
    assert sleeps.delays == []


def test_gives_up_after_max_attempts() -> None:
    sleeps = Sleeps()

    async def go() -> FetchResult:
        async with client_for(lambda req: httpx.Response(500)) as c:
            return await fetch(c, "http://h/", sleep=sleeps, max_attempts=3)

    res = run(go())
    assert res.attempts == 3 and res.error == "gave up: HTTP 500" and len(sleeps.delays) == 2


def test_timeout_is_retried_then_reported() -> None:
    async def slow(req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    async def go() -> FetchResult:
        async with client_for(slow) as c:
            return await fetch(c, "http://h/", timeout=0.05, sleep=Sleeps(), max_attempts=2)

    res = run(go())
    assert res.status == 0 and res.error == "gave up: timeout after 0.05s"


def test_redirect_loop_is_reported_not_retried() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        nxt = "/b" if req.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": nxt})

    async def go() -> FetchResult:
        async with client_for(handler) as c:
            return await fetch(c, "http://h/a", sleep=Sleeps())

    res = run(go())
    assert res.error == "redirect loop" and res.attempts == 1


def test_backoff_and_retry_after_parsing() -> None:
    assert 4.0 <= backoff_delay(3, base=0.5) <= 5.0
    assert 8.0 <= backoff_delay(10, base=0.5, cap=8.0) <= 9.0
    assert retry_after_seconds("12") == 12.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
    assert retry_after_seconds("soon") is None


# --- robots -----------------------------------------------------------------------------


def test_robots_cache() -> None:
    hits = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        hits[req.url.host] += 1
        if req.url.host == "a":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nCrawl-delay: 2")
        if req.url.host == "b":
            return httpx.Response(404)
        return httpx.Response(503)

    async def go() -> list[object]:
        async with client_for(handler) as c:
            robots = RobotsCache(c, "findex-test")
            results = await asyncio.gather(*(robots.allowed("http://a/x") for _ in range(10)))
            return [
                all(results),
                await robots.allowed("http://a/private/secret"),
                await robots.crawl_delay("http://a/"),
                await robots.allowed("http://b/anything"),
                await robots.allowed("http://c/anything"),
            ]

    assert run(go()) == [True, False, 2.0, True, False]
    assert hits["a"] == 1  # 10 concurrent askers, ONE robots.txt fetch


# --- html -------------------------------------------------------------------------------


def test_parse_html() -> None:
    page = parse_html(
        "<html><head><title> Hi  there </title><style>p{}</style></head>"
        "<body><h1>Head</h1><p>Text &amp; more</p><script>x()</script>"
        "<a href='/a'>A</a><a>no href</a></body></html>"
    )
    assert page.title == "Hi there"
    assert page.links == ("/a",)
    assert "Text & more" in page.text and "x()" not in page.text and "p{}" not in page.text


# --- crawl ------------------------------------------------------------------------------


def site(n_pages: int, *, latency: float = 0.0, track: Counter[str] | None = None) -> object:
    """A fake site: page i links to i+1, i+2 and back to 0; /private/ is robots-disallowed."""

    async def handler(req: httpx.Request) -> httpx.Response:
        if track is not None:
            track["now"] += 1
            track["max"] = max(track["max"], track["now"])
            track[req.url.path] += 1
        try:
            if latency:
                await asyncio.sleep(latency)
            if req.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
            i = int(req.url.path.strip("/").removeprefix("p") or 0)
            links = "".join(
                f"<a href='/p{j}#top'>p{j}</a>" for j in (i + 1, i + 2, 0) if j < n_pages
            )
            links += "<a href='/private/x'>secret</a><a href='http://other.org/'>ext</a>"
            return httpx.Response(200, html=f"<title>Page {i}</title><p>body {i}</p>{links}")
        finally:
            if track is not None:
                track["now"] -= 1

    return handler


def collect(handler: object, **kwargs: object) -> list[str]:
    async def go() -> list[str]:
        async with client_for(handler) as c:
            return [p.url async for p in crawl(["http://site.test/"], client=c, **kwargs)]  # type: ignore[arg-type]

    return run(go())


def test_crawl_respects_max_pages_and_never_refetches() -> None:
    track: Counter[str] = Counter()
    urls = collect(site(50, track=track), max_pages=20, concurrency=5, per_host=5)
    assert len(urls) == 20 == len(set(urls))
    assert all(v == 1 for k, v in track.items() if k.startswith("/p"))
    assert track["/private/x"] == 0  # robots.txt obeyed
    assert all(u.startswith("http://site.test/") for u in urls)  # domain allowlist


def test_crawl_stops_when_frontier_is_empty() -> None:
    urls = collect(site(6), max_pages=100, concurrency=3)
    expected = ["http://site.test/"] + [f"http://site.test/p{i}" for i in range(6)]
    assert sorted(urls) == sorted(expected)


def test_per_host_limit_is_never_exceeded() -> None:
    track: Counter[str] = Counter()
    collect(site(40, latency=0.01, track=track), max_pages=30, concurrency=20, per_host=3)
    assert 1 < track["max"] <= 3


def test_breaking_out_early_cancels_cleanly() -> None:
    async def go() -> int:
        async with client_for(site(100, latency=0.01)) as c:
            n = 0
            async for _ in crawl(["http://site.test/"], client=c, max_pages=100):
                n += 1
                if n == 3:
                    break
            await asyncio.sleep(0.05)
            return len([t for t in asyncio.all_tasks() if t is not asyncio.current_task()])

    assert run(go()) == 0  # no task left running after the generator was closed
