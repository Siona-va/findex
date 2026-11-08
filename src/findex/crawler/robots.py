"""robots.txt: fetched once per host, cached, consulted before every request."""

from __future__ import annotations

import asyncio
import logging
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.urls import host_of

log = logging.getLogger(__name__)


class RobotsCache:
    """``await robots.allowed(url)`` -> bool.

    Policy (as Google documents it): robots.txt 4xx -> everything allowed;
    5xx / timeout / network error -> everything DISALLOWED (when unsure, stay out).
    One fetch per host even if 20 workers ask at once (a lock per host).
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str, timeout: float = 10.0) -> None:
        self._client = client
        self._agent = user_agent
        self._timeout = timeout
        # a parser, or True (robots.txt missing: allow all) / False (unavailable: deny all)
        self._rules: dict[str, RobotFileParser | bool] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _rules_for(self, host: str) -> RobotFileParser | bool:
        if (cached := self._rules.get(host)) is not None:
            return cached
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:  # the first worker fetches; the others wait for its result
            if (cached := self._rules.get(host)) is not None:
                return cached
            res = await fetch(self._client, f"{host}/robots.txt", timeout=self._timeout)
            rules: RobotFileParser | bool
            if res.status == 200:
                rules = RobotFileParser(f"{host}/robots.txt")
                rules.parse(res.text.splitlines())
            elif 400 <= res.status < 500:
                rules = True
            else:
                rules = False
                log.warning(
                    "robots.txt for %s unavailable (%s): not crawling it",
                    host,
                    res.error or res.status,
                )
            self._rules[host] = rules
            return rules

    async def allowed(self, url: str) -> bool:
        rules = await self._rules_for(host_of(url))
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        rules = await self._rules_for(host_of(url))
        if isinstance(rules, bool):
            return None
        delay = rules.crawl_delay(self._agent)
        return float(delay) if delay is not None else None
