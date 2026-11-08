"""Lab 7: every route through TestClient; the tiny fixture index injected via Depends."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web import create_app
from findex.web.server import get_index, highlight_html
from findex.web.settings import Settings


@pytest.fixture
def api(index: Index, tmp_path: Path) -> Iterator[TestClient]:
    """App whose real index fails to load (no file); routes get the fixture via override."""
    app = create_app(Settings(index_path=tmp_path / "missing.pkl"))
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as client:
        yield client


@pytest.fixture
def site(index_file: Path) -> Iterator[TestClient]:
    """App that really loads an index file at startup (lifespan), for the UI."""
    with TestClient(create_app(Settings(index_path=index_file))) as client:
        yield client


def test_search_ok(api: TestClient) -> None:
    r = api.get("/search", params={"q": "event loop", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and body["page"] == 1 and len(body["results"]) == 2
    assert body["results"][0]["title"].startswith("Asyncio")
    assert "[event]" in body["results"][0]["snippet"]
    assert r.headers["x-request-id"]


def test_pagination(api: TestClient) -> None:
    all_ = api.get("/search", params={"q": "the OR loop OR zebra", "k": 10}).json()["results"]
    p1 = api.get("/search", params={"q": "the OR loop OR zebra", "k": 2, "page": 1}).json()
    p2 = api.get("/search", params={"q": "the OR loop OR zebra", "k": 2, "page": 2}).json()
    assert [h["doc_id"] for h in p1["results"] + p2["results"]] == [h["doc_id"] for h in all_[:4]]
    assert p1["total"] == p2["total"] == len(all_)


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "x", "k": "abc"},
        {"q": "x", "k": 0},
        {"q": "x", "k": 101},
        {"q": "x", "scorer": "pagerank"},
        {"q": "x", "page": 0},
        {},
        {"q": "x" * 201},
    ],
)
def test_validation_422(api: TestClient, params: dict[str, object]) -> None:
    r = api.get("/search", params=params)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_bad_query_400(api: TestClient) -> None:
    r = api.get("/search", params={"q": "(a OR"})
    assert r.status_code == 400 and r.json()["detail"].startswith("bad query")


def test_document_and_404(api: TestClient) -> None:
    doc = api.get("/docs/0").json()
    assert doc["doc_id"] == 0 and doc["text"] and doc["length"] > 0
    r = api.get("/docs/999")
    assert r.status_code == 404 and r.json() == {"detail": "no document 999"}
    assert api.get("/docs/abc").status_code == 422


def test_stats(api: TestClient) -> None:
    s = api.get("/stats").json()
    assert s["documents"] == 7 and s["terms"] > 10 and s["positions"] is True


def test_health_503_when_index_not_loaded(api: TestClient) -> None:
    r = api.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


def test_health_ok_and_lifespan_loads(site: TestClient) -> None:
    assert site.get("/health").json() == {"status": "ok"}


def test_openapi_lists_the_api(site: TestClient) -> None:
    paths = site.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/search", "/docs/{doc_id}", "/stats", "/health"}
    assert site.get("/docs").status_code == 200  # Swagger UI


def test_ui_search_page_and_doc_view(site: TestClient) -> None:
    page = site.get("/", params={"q": "zebra"}).text
    assert "2 results" in page and "<mark>zebra</mark>" in page
    doc = site.get("/doc/0", params={"q": "event"}).text
    assert "<mark>" in doc
    frag = site.get("/fragment/results", params={"q": "zebra"}).text
    assert "<html" not in frag and "zebra" in frag
    assert site.get("/static/style.css").status_code == 200


def test_ui_reports_bad_query(site: TestClient) -> None:
    assert "bad query" in site.get("/", params={"q": "(a OR"}).text


def test_snippets_cannot_inject_html() -> None:
    assert str(highlight_html("<script>x</script> \x02hit\x03")) == (
        "&lt;script&gt;x&lt;/script&gt; <mark>hit</mark>"
    )


def test_unexpected_error_is_clean_500(index: Index, tmp_path: Path) -> None:
    app = create_app(Settings(index_path=tmp_path / "none.pkl"))

    def broken() -> Index:
        raise RuntimeError("boom")

    app.dependency_overrides[get_index] = broken
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/stats")
    assert r.status_code == 500
    assert r.json()["detail"] == "internal server error" and r.json()["request_id"]


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "x.pkl"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.index_path == tmp_path / "x.pkl" and s.log_level == "DEBUG"
    monkeypatch.setenv("PORT", "not-a-number")
    with pytest.raises(ValueError, match="port"):
        Settings()
