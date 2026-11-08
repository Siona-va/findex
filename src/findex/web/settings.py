"""12-factor configuration: everything comes from the environment (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """``INDEX_PATH=/app/index.pkl LOG_LEVEL=INFO PORT=8000 findex serve``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    index_path: Path = Field(default=Path("index.pkl"), description="Index file to serve")
    embeddings_path: Path | None = Field(
        default=None, description="`findex embed` output (default: <INDEX_PATH>.emb if present)"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    port: int = Field(default=8000, ge=1, le=65535)
    page_size_max: int = Field(default=100, ge=1, le=1000)
    enable_bench_routes: bool = False  # exposes the deliberately-wrong async route (Lab 7 M2)


@lru_cache
def get_settings() -> Settings:
    """Read the environment ONCE per process (and a dependency tests can override)."""
    return Settings()
