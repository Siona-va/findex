"""findex.web -- the HTTP API and web UI (Lab 7). ``uvicorn findex.web:app``."""

from findex.web.server import create_app

app = create_app()

__all__ = ["app", "create_app"]
