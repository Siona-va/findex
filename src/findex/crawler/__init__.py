"""findex.crawler -- a polite asyncio web crawler (Lab 6)."""

from findex.crawler.crawl import CrawlConfig, Crawler, CrawlStats, Page, crawl, make_client
from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize

__all__ = [
    "USER_AGENT",
    "CrawlConfig",
    "CrawlStats",
    "Crawler",
    "FetchResult",
    "Page",
    "RobotsCache",
    "canonicalize",
    "crawl",
    "fetch",
    "make_client",
]
