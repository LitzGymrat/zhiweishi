from __future__ import annotations

import httpx
from openai import OpenAI


def create_openai_compatible_client(
    *,
    api_key: str,
    base_url: str,
    max_connections: int = 4,
    timeout_seconds: float = 120.0,
    connect_timeout_seconds: float = 30.0,
    max_retries: int = 2,
) -> OpenAI:
    """Create a bounded OpenAI-compatible client for generation or evaluation.

    Callers own the returned client and should use it as a context manager (or
    close it explicitly) so its HTTP connection pool is released predictably.
    """
    if max_connections <= 0:
        raise ValueError("max_connections 必须为正数。")
    timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        max_retries=max_retries,
        timeout=timeout,
        http_client=httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=timeout,
        ),
    )
