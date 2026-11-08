"""One polite HTTP fetch: timeout, bounded retries with backoff + jitter, Retry-After."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from findex import __version__

log = logging.getLogger(__name__)

USER_AGENT = (
    f"findex-crawler/{__version__} (student search-engine project; "
    "+https://github.com/USERNAME/findex)"
)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    """What happened to one URL -- recorded for every URL in crawl.log."""

    url: str
    final_url: str
    status: int  # HTTP status; 0 = no response (timeout, connection error, redirect loop)
    content_type: str
    text: str
    nbytes: int
    elapsed: float
    attempts: int
    error: str = ""

    @property
    def ok_html(self) -> bool:
        return self.status == 200 and "html" in self.content_type


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """Exponential backoff with jitter: ``min(cap, base * 2**attempt) + U(0, 1)``."""
    return min(cap, base * 2**attempt) + random.uniform(0, 1)


def retry_after_seconds(value: str | None, cap: float = 60.0) -> float | None:
    """Parse ``Retry-After``: either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(cap, float(value))
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(cap, when.timestamp() - time.time()))


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Sleep = asyncio.sleep,
) -> FetchResult:
    """GET ``url``. Never raises for network/HTTP problems -- they become a FetchResult.

    Retried (up to ``max_attempts`` in total): 429, 5xx, timeouts, connection errors.
    Not retried: other 4xx (a 404 will still be a 404), redirect loops.
    ``CancelledError`` is NOT caught: cancellation must always propagate.
    """
    t0 = time.perf_counter()
    last_error = ""
    status = 0
    for attempt in range(max_attempts):
        delay: float | None = None
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
                body = await resp.aread()
        except TimeoutError:
            last_error = f"timeout after {timeout:g}s"
        except httpx.TooManyRedirects:
            return FetchResult(
                url, url, 0, "", "", 0, time.perf_counter() - t0, attempt + 1, "redirect loop"
            )
        except httpx.TransportError as exc:  # connection refused/reset, DNS, protocol
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            status = resp.status_code
            if status not in RETRY_STATUSES:
                ctype = resp.headers.get("content-type", "")
                text = (
                    body.decode(resp.encoding or "utf-8", errors="replace")
                    if "text" in ctype or "html" in ctype
                    else ""
                )
                return FetchResult(
                    url,
                    str(resp.url),
                    status,
                    ctype,
                    text,
                    len(body),
                    time.perf_counter() - t0,
                    attempt + 1,
                    "" if status < 400 else f"HTTP {status}",
                )
            last_error = f"HTTP {status}"
            delay = retry_after_seconds(resp.headers.get("retry-after"))
        if attempt + 1 < max_attempts:
            wait = delay if delay is not None else backoff_delay(attempt, base_delay)
            log.info("retry %s in %.2fs (%s, attempt %d)", url, wait, last_error, attempt + 1)
            await sleep(wait)
    return FetchResult(
        url,
        url,
        status,
        "",
        "",
        0,
        time.perf_counter() - t0,
        max_attempts,
        f"gave up: {last_error}",
    )
