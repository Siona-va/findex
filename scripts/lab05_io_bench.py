"""Lab 5 counter-benchmark: an I/O-bound job where threads DO scale.

    uv run python scripts/lab05_io_bench.py

Starts a local HTTP server whose every response takes 50 ms (a stand-in for network
latency), then fetches 64 URLs with 1..32 threads. While a thread waits in
``socket.recv`` it has released the GIL, so the waits overlap.
"""

from __future__ import annotations

import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LATENCY = 0.05
N_URLS = 64


class SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        time.sleep(LATENCY)  # the "network"
        body = b"x" * 2048
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # silence
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 128  # default listen backlog is 5: more clients -> SYN retries
    daemon_threads = True


def fetch(url: str) -> int:
    with urllib.request.urlopen(url, timeout=10) as r:
        return len(r.read())


def main() -> None:
    server = Server(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    urls = [f"{base}/page/{i}" for i in range(N_URLS)]

    print(f"{N_URLS} requests, {LATENCY * 1000:.0f} ms latency each\n")
    print("| Threads | Wall (s, median of 3) | Speedup |")
    print("|---|---|---|")
    serial = 0.0
    for workers in (1, 2, 4, 8, 16, 32):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                assert sum(pool.map(fetch, urls)) == 2048 * N_URLS
            times.append(time.perf_counter() - t0)
        wall = statistics.median(times)
        serial = serial or wall
        print(f"| {workers} | {wall:.2f} | {serial / wall:.1f}× |")
    server.shutdown()


if __name__ == "__main__":
    main()
