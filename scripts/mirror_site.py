"""A LOCAL mirror site to crawl (Lab 6): our own copy of the Python docs, served by us.

    uv run python scripts/mirror_site.py build data/pydocs data/site
    uv run python scripts/mirror_site.py serve data/site --port 8765 --latency 0.05 [--chaos]

``build`` turns every .rst file into a small HTML page whose links follow the docs'
own cross-references (toctree entries, :doc:, :mod:), so the site is a real link graph.
``serve`` adds what the internet adds: latency per request, a robots.txt, and with
``--chaos`` deterministic failures to prove the crawler handles them:

* ~1 in 30 pages answers 503 on its first request, ~1 in 30 answers 429 + Retry-After: 1;
* /slow takes 30 s (longer than any sane timeout);
* /loop-a <-> /loop-b redirect forever;  /old/... -> 301 -> /...;  /missing -> 404.

Ethics: crawling a site you run yourself, on your own machine, needs nobody's permission.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

ROBOTS = "User-agent: *\nDisallow: /whatsnew/\nDisallow: /private/\n"
TOCTREE_RE = re.compile(r"^\.\. toctree::\n((?:[ \t]+.*\n|\n)+)", re.M)
DOC_RE = re.compile(r":doc:`(?:[^<`]*<)?([^>`]+)>?`")
MOD_RE = re.compile(r":mod:`!?~?([\w.]+)`")
TITLE_RE = re.compile(r"^(?P<t>[^\n].*)\n(?P<u>[=\-~*^#]{3,})\s*$", re.M)


def _title(text: str, fallback: str) -> str:
    m = TITLE_RE.search(text)
    t = m.group("t").strip() if m else fallback
    return re.sub(r":\w+:`!?~?([^`]+)`", r"\1", t)


def _resolve(target: str, here: str, known: set[str]) -> str | None:
    """A docs-relative target ('library/re', '../howto/regex', '/tutorial/index')."""
    target = target.strip().removesuffix(".rst")
    if target.startswith("/"):
        cand = target.lstrip("/")
    else:
        parts = (Path(here).parent / target).as_posix().split("/")
        stack: list[str] = []
        for p in parts:
            if p == "..":
                if stack:
                    stack.pop()
            elif p not in (".", ""):
                stack.append(p)
        cand = "/".join(stack)
    return cand if cand in known else None


def build(src: Path, dst: Path) -> None:
    files = sorted(src.rglob("*.rst"))
    known = {f.relative_to(src).with_suffix("").as_posix() for f in files}
    edges = 0
    for f in files:
        rel = f.relative_to(src).with_suffix("").as_posix()
        text = f.read_text(encoding="utf-8", errors="replace")
        links: list[str] = []
        for block in TOCTREE_RE.findall(text):
            for line in block.splitlines():
                entry = line.strip()
                if entry and not entry.startswith(":"):
                    entry = entry.split("<")[-1].rstrip(">") if "<" in entry else entry
                    if (t := _resolve(entry, rel, known)) is not None:
                        links.append(t)
        for target in DOC_RE.findall(text):
            if (t := _resolve(target, rel, known)) is not None:
                links.append(t)
        for mod in MOD_RE.findall(text):
            for cand in (f"library/{mod}", f"library/{mod.split('.')[0]}"):
                if cand in known:
                    links.append(cand)
                    break
        parent = rel.rsplit("/", 1)[0] + "/index" if "/" in rel else "index"
        if parent in known and parent != rel:
            links.append(parent)
        uniq = list(dict.fromkeys(x for x in links if x != rel))
        edges += len(uniq)
        title = _title(text, rel)
        nav = "".join(f'<li><a href="/{u}.html">{html.escape(u)}</a></li>' for u in uniq)
        page = (
            f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}"
            f"</title></head><body><nav><a href='/'>home</a><ul>{nav}</ul></nav>"
            f"<main><h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre></main>"
            "</body></html>"
        )
        out = dst / f"{rel}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
    roots = [
        k
        for k in (
            "index",
            "contents",
            "glossary",
            "library/index",
            "tutorial/index",
            "howto/index",
            "reference/index",
            "faq/index",
        )
        if k in known
    ]
    extra = ["/slow", "/loop-a", "/old/library/re.html", "/missing", "/private/secret.html"]
    items = "".join(f'<li><a href="/{r}.html">{r}</a></li>' for r in roots)
    items += "".join(f'<li><a href="{e}">{e}</a></li>' for e in extra)
    (dst / "_home.html").write_text(
        f"<!doctype html><html><head><title>Local Python docs mirror</title></head>"
        f"<body><h1>Local Python docs mirror</h1><ul>{items}</ul></body></html>",
        encoding="utf-8",
    )
    print(f"built {len(files)} pages, {edges} links -> {dst}")


def make_app(root: Path, latency: float, chaos: bool):  # noqa: ANN201 (starlette types)
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
    from starlette.routing import Route

    hits: Counter[str] = Counter()

    async def handler(request: Request) -> Response:
        path = request.url.path
        await asyncio.sleep(latency)  # the "network"
        if path == "/robots.txt":
            return PlainTextResponse(ROBOTS)
        if chaos:
            hits[path] += 1
            bucket = zlib.crc32(path.encode()) % 30
            if path == "/slow":
                await asyncio.sleep(30)
            if path == "/loop-a":
                return RedirectResponse("/loop-b", status_code=302)
            if path == "/loop-b":
                return RedirectResponse("/loop-a", status_code=302)
            if path.startswith("/old/"):
                return RedirectResponse(path.removeprefix("/old"), status_code=301)
            if bucket == 0 and hits[path] == 1:
                return PlainTextResponse("busy", status_code=503)
            if bucket == 1 and hits[path] == 1:
                return PlainTextResponse("slow down", status_code=429, headers={"Retry-After": "1"})
        name = "_home.html" if path == "/" else path.lstrip("/")
        if name.endswith("/"):
            name += "index.html"  # /library/ -> library/index.html, like any web server
        f = (root / name).resolve()
        if root.resolve() not in f.parents or not f.is_file():
            return PlainTextResponse("not found", status_code=404)
        return HTMLResponse(f.read_text(encoding="utf-8"))

    return Starlette(routes=[Route("/{path:path}", handler)])


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("src", type=Path)
    b.add_argument("dst", type=Path)
    s = sub.add_parser("serve")
    s.add_argument("root", type=Path)
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--latency", type=float, default=0.05)
    s.add_argument("--chaos", action="store_true")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.src, args.dst)
        return
    import uvicorn

    print(
        f"serving {args.root} on http://127.0.0.1:{args.port} "
        f"(latency {args.latency * 1000:.0f} ms, chaos={args.chaos})",
        file=sys.stderr,
    )
    uvicorn.run(
        make_app(args.root, args.latency, args.chaos),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
