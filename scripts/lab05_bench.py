"""Lab 5 benchmark: serial vs threads vs processes (+ free-threaded threads).

    uv run python scripts/lab05_bench.py data/pydocs                      # GIL build only
    uv run python scripts/lab05_bench.py data/pydocs --ft "$(uv python find 3.13t)"

Every measurement runs in a FRESH subprocess, so peak RSS and CPU time belong to that
one build. Writes docs/lab05_bench.json, docs/lab05_speedup.png and prints a Markdown
table for the README.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def one(corpus: Path, executor: str, workers: int) -> None:
    """Child mode: one build, print one JSON line."""
    from findex.parallel import build_parallel

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    # spawn: workers are OUR children, so RUSAGE_CHILDREN sees their CPU time
    # (forkserver workers are children of the fork server and would be invisible).
    _ix, stats = build_parallel(
        corpus,
        executor,
        workers=workers,  # type: ignore[arg-type]
        start_method="spawn",
    )
    wall = time.perf_counter() - t0
    cpu_parent = time.process_time() - cpu0
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        json.dumps(
            {
                "wall": wall,
                "cpu": cpu_parent + children.ru_utime + children.ru_stime,
                "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "merge": stats.merge_seconds,
                "gil": gil,
            }
        )
    )


def run(python: str, corpus: Path, executor: str, workers: int) -> dict[str, float]:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run(
        [python, __file__, "--one", executor, str(workers), str(corpus)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def plot(results: list[dict[str, object]], counts: list[int], base: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = os.cpu_count() or 1
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    ax.plot(
        counts,
        [min(c, cores) for c in counts],
        linestyle="--",
        color="#999999",
        label=f"ideal (linear up to {cores} cores)",
    )
    for label, color in (
        ("threads", "#d62728"),
        ("processes", "#1f77b4"),
        ("threads (3.13t)", "#2ca02c"),
        ("threads (3.14t)", "#9467bd"),
    ):
        rows = [r for r in results if r["label"] == label]
        if rows:
            ax.plot(
                [r["workers"] for r in rows],
                [base / float(r["wall"]) for r in rows],  # type: ignore[arg-type]
                marker="o",
                color=color,
                label=label,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial (×)")
    ax.set_title(f"findex index: speedup vs workers ({cores}-core machine)")
    ax.set_ylim(0, cores + 0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    print(f"plot: {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("--one", nargs=2, metavar=("EXECUTOR", "WORKERS"))
    p.add_argument(
        "--ft",
        action="append",
        default=[],
        help="free-threaded python (python3.13t); may repeat (3.14t)",
    )
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--workers", default=f"1,2,4,8,{os.cpu_count()}")
    p.add_argument("--plot-only", action="store_true", help="redraw from docs/lab05_bench.json")
    args = p.parse_args()
    if args.plot_only:
        data = json.loads((ROOT / "docs" / "lab05_bench.json").read_text())
        res = data["results"]
        plot(
            res,
            sorted({int(r["workers"]) for r in res}),
            float(res[0]["wall"]),
            ROOT / "docs" / "lab05_speedup.png",
        )
        return
    if args.one:
        one(args.corpus, args.one[0], int(args.one[1]))
        return

    counts = sorted({int(w) for w in args.workers.split(",")})
    configs: list[tuple[str, str, str, int]] = [("serial", sys.executable, "serial", 1)]
    configs += [("threads", sys.executable, "threads", w) for w in counts]
    configs += [("processes", sys.executable, "processes", w) for w in counts]
    for ft in args.ft:
        ver = subprocess.run(
            [ft, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        configs += [(f"serial ({ver}t)", ft, "serial", 1)]
        configs += [(f"threads ({ver}t)", ft, "threads", w) for w in counts]

    run(sys.executable, args.corpus, "serial", 1)  # warm-up (page cache, .pyc files)
    results: list[dict[str, object]] = []
    for label, python, executor, workers in configs:
        runs = [run(python, args.corpus, executor, workers) for _ in range(args.runs)]
        walls = [r["wall"] for r in runs]
        results.append(
            {
                "label": label,
                "workers": workers,
                "wall": statistics.median(walls),
                "wall_min": min(walls),
                "wall_max": max(walls),
                "cpu": statistics.median(r["cpu"] for r in runs),
                "rss_mb": statistics.median(r["rss_mb"] for r in runs),
                "merge": statistics.median(r["merge"] for r in runs),
                "gil": runs[0]["gil"],
            }
        )
        print(f"  {label:<16} {workers:>2}  {results[-1]['wall']:.2f} s", file=sys.stderr)

    base = float(results[0]["wall"])  # type: ignore[arg-type]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    machine = (
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}"
    )
    (docs / "lab05_bench.json").write_text(
        json.dumps({"machine": machine, "results": results}, indent=2)
    )

    print(f"\nMachine: {machine}; {args.runs} runs each, median (min–max)\n")
    print(
        "| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS (parent) | Merge (s) "
        "| Speedup |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['label']} | {r['workers']} | {r['wall']:.2f} ({r['wall_min']:.2f}–"
            f"{r['wall_max']:.2f}) | {r['cpu']:.2f} | {r['rss_mb']:.0f} MB | "
            f"{r['merge']:.3f} | {base / float(r['wall']):.2f}× |"
        )  # type: ignore[arg-type]

    plot(results, counts, base, docs / "lab05_speedup.png")


if __name__ == "__main__":
    main()
