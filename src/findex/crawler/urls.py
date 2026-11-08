"""URL canonicalization: one spelling per page, so the ``seen`` set really dedupes."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Return the canonical form of ``url`` (resolved against ``base``), or ``None``.

    * only ``http``/``https`` (``mailto:``, ``javascript:``, ``ftp:`` -> None);
    * scheme and host lower-case, default port dropped;
    * fragment stripped (``#section`` is the same page);
    * empty path -> ``/``; ``/x/index.html`` -> ``/x/``. A trailing ``/`` is KEPT: for most
      servers ``/x`` and ``/x/`` are different URLs (``/x`` usually redirects to ``/x/``),
      and stripping it once sent every section link to a 404;
    * query parameters sorted (``?b=2&a=1`` == ``?a=1&b=2``).
    """
    raw = urljoin(base, url.strip()) if base else url.strip()
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:  # e.g. an invalid port number
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if port is not None and port != DEFAULT_PORTS[scheme]:
        host = f"{host}:{port}"
    path = parts.path or "/"
    if path.endswith("/index.html"):
        path = path.removesuffix("index.html")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def host_of(url: str) -> str:
    """``scheme://host[:port]`` -- the unit for politeness limits and robots.txt."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def domain_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    """True if the URL's hostname is one of ``allowed_domains`` or a subdomain of one."""
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)
