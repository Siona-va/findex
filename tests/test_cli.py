"""The CLI, driven in-process through typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from findex import __version__
from findex.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_index_then_search_then_stats(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.pkl"
    r = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out), "--positions"])
    assert r.exit_code == 0, r.output
    assert out.exists()

    r = runner.invoke(app, ["search", str(out), '"event loop"', "-k", "3"])
    assert r.exit_code == 0, r.output
    assert "asyncio" in r.stdout

    r = runner.invoke(app, ["stats", str(out), "--top", "3"])
    assert r.exit_code == 0
    assert "documents" in r.stdout and "vocabulary" in r.stdout


def test_search_json_is_pipeable(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "zebra", "--json", "--scorer", "tfidf"])
    assert r.exit_code == 0
    rows = [json.loads(line) for line in r.stdout.splitlines()]
    assert {row["rank"] for row in rows} == {1, 2}
    assert all("[zebra]" not in row["snippet"] for row in rows)  # plain text in JSON mode


def test_missing_index_is_one_clear_line(tmp_path: Path) -> None:
    r = runner.invoke(app, ["search", str(tmp_path / "nope.pkl"), "x"])
    assert r.exit_code == 1
    assert "index file not found" in r.stderr
    assert "Traceback" not in r.output


def test_bad_query_exits_nonzero(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "(a OR"])
    assert r.exit_code == 1
    assert "error" in r.stderr


def test_missing_corpus(tmp_path: Path) -> None:
    r = runner.invoke(app, ["index", str(tmp_path / "nope"), "--out", str(tmp_path / "x.pkl")])
    assert r.exit_code == 1
    assert "corpus not found" in r.stderr


def test_no_results_message(index_file: Path) -> None:
    r = runner.invoke(app, ["-vv", "search", str(index_file), "nonexistentword"])
    assert r.exit_code == 0
    assert "no documents match" in r.stderr
