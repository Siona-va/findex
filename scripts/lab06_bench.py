"""Lab 6 benchmark: the same crawl at concurrency 1 / 5 / 20, plus a polite row.

    uv run python scripts/lab06_bench.py data/site

Starts the local mirror (50 ms latency per request, no chaos), then runs `findex crawl`
in a fresh process per measurement (so peak RSS belongs to that crawl).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8799
MAX_PAGES = 300


def crawl_once(concurrency: int, per_host: int, min_delay: float) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "findex",
                "crawl",
                f"http://127.0.0.1:{PORT}/",
                "--max-pages",
                str(MAX_PAGES),
                "--concurrency",
                str(concurrency),
                "--per-host",
                str(per_host),
                "--min-delay",
                str(min_delay),
                "--out",
                f"{tmp}/c.jsonl",
                "--log-file",
                f"{tmp}/c.log",
                "--stats-json",
                str(stats),
            ],
            check=True,
            capture_output=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
        return json.loads(stats.read_text())


def main() -> None:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else "data/site")
    server = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "mirror_site.py"),
            "serve",
            str(site),
            "--port",
            str(PORT),
            "--latency",
            "0.05",
        ],
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/robots.txt", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        configs = [
            ("1 (effectively sequential)", 1, 1, 0.0),
            ("5", 5, 5, 0.0),
            ("20", 20, 20, 0.0),
            ("20, polite: per-host 2, delay 0.1 s", 20, 2, 0.1),
        ]
        print(f"{MAX_PAGES} pages, local mirror with 50 ms latency per request, 3 runs, median\n")
        print("| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |")
        print("|---|---|---|---|---|---|")
        for label, conc, per_host, delay in configs:
            runs = [crawl_once(conc, per_host, delay) for _ in range(3)]
            wall = statistics.median(r["wall"] for r in runs)
            print(
                f"| {label} | {runs[0]['pages']} | {wall:.1f} | "
                f"{statistics.median(r['pages_per_s'] for r in runs):.1f} | "
                f"{max(r['errors'] for r in runs)} | "
                f"{statistics.median(r['peak_rss_mb'] for r in runs):.0f} MB |"
            )
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
