"""Lab 5: the bug the GIL was hiding.

    uv run python scripts/lab05_race.py data/pydocs
    "$(uv python find 3.14t)" scripts/lab05_race.py data/pydocs      # PYTHONPATH=src

A *naive* threaded statistics pass: several threads count tokens into ONE shared
Counter with ``counts[tok] += 1``. That line is read-modify-write (see the ``dis``
output this script prints): a thread can be paused between reading the old value and
storing the new one, and another thread's increment is lost.

findex's real build does not do this: every worker returns its own partial index and
only the parent merges (no shared mutable state). This script exists to show WHY.
"""

from __future__ import annotations

import dis
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from findex.corpus import iter_documents  # noqa: E402
from findex.tokenize import tokenize  # noqa: E402


def count_naive(texts: list[str], counts: Counter[str]) -> None:
    for text in texts:
        for tok in tokenize(text):
            counts[tok] += 1  # RACE: load, add, store are separate steps


def count_locked(texts: list[str], counts: Counter[str], lock: threading.Lock) -> None:
    for text in texts:
        for tok in tokenize(text):
            with lock:  # fix 1: mutual exclusion (correct, but serializes the hot loop)
                counts[tok] += 1


def count_local(texts: list[str], out: list[Counter[str]]) -> None:
    local: Counter[str] = Counter()  # fix 2: no shared state; merge at the end
    for text in texts:
        local.update(tokenize(text))
    out.append(local)  # list.append of a finished object: one step, once per thread


def run(mode: str, parts: list[list[str]]) -> tuple[int, float]:
    counts: Counter[str] = Counter()
    lock = threading.Lock()
    locals_: list[Counter[str]] = []
    if mode == "naive":
        targets = [threading.Thread(target=count_naive, args=(p, counts)) for p in parts]
    elif mode == "lock":
        targets = [threading.Thread(target=count_locked, args=(p, counts, lock)) for p in parts]
    else:
        targets = [threading.Thread(target=count_local, args=(p, locals_)) for p in parts]
    t0 = time.perf_counter()
    for t in targets:
        t.start()
    for t in targets:
        t.join()
    if mode == "local":
        for c in locals_:
            counts.update(c)
    return counts.total(), time.perf_counter() - t0


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pydocs")
    texts = [d.text for d in iter_documents(root)]
    expected = sum(1 for t in texts for _ in tokenize(t))
    n_threads = 4
    parts = [texts[i::n_threads] for i in range(n_threads)]
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        f"Python {sys.version.split()[0]}, GIL enabled: {gil}, {n_threads} threads, "
        f"expected {expected:,} tokens\n"
    )
    print("| Mode | Counted | Lost updates | Time (s) |")
    print("|---|---|---|---|")
    for mode in ("naive", "lock", "local"):
        lost_runs = []
        for _ in range(3):
            total, secs = run(mode, parts)
            lost_runs.append(expected - total)
        print(f"| {mode} | {total:,} | {max(lost_runs):,} (worst of 3) | {secs:.2f} |")
    print("\n`counts[tok] += 1` compiles to:")
    for ins in dis.get_instructions("counts[tok] += 1"):
        if ins.opname not in ("RESUME", "RETURN_CONST", "RETURN_VALUE", "LOAD_CONST"):
            print(f"    {ins.opname} {ins.argrepr}")


if __name__ == "__main__":
    main()
