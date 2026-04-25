"""Shared configuration helpers for CLI scripts."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()


def get_required_env(name: str) -> str:
    """Return required environment variable or raise a clear error."""
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        "Add it to your shell environment or .env file."
    )


def get_pageindex_client() -> PageIndexClient:
    """Build a PageIndex client from environment configuration."""
    return PageIndexClient(api_key=get_required_env("PAGEINDEX_API_KEY"))


def get_openai_api_key() -> str:
    """Return the configured OpenAI API key."""
    return get_required_env("OPENAI_API_KEY")
