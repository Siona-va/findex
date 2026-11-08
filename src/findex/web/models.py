"""Pydantic models at the HTTP boundary. Inside, findex keeps its own dataclasses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = Field(description="Plain text; matched terms are wrapped in [brackets]")


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="All documents matching the query")
    page: int
    k: int
    took_ms: float
    results: list[SearchHit]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="Number of tokens")
    text: str


class StatsOut(BaseModel):
    version: str
    documents: int
    terms: int
    tokens: int
    avg_doc_length: float
    positions: bool
    index_path: str
    index_bytes: int
    uptime_s: float


class Health(BaseModel):
    status: str
