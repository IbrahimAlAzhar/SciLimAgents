"""
llm_client.py
-------------

Thin wrapper around the OpenAI Chat Completions API with automatic retries on
transient errors (rate limit / 5xx / timeouts / connection blips).

Used by every agent in the limitation-generation pipeline. The code is
intentionally minimal -- one class, one method (`chat`) -- so the same client
instance can be shared across all agents.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

# OpenAI v1.x SDK. We re-raise a friendlier ImportError so users know what to
# install instead of seeing a low-level traceback.
try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'openai' package (>=1.0.0) is required. "
        "Install it with: pip install 'openai>=1.0.0'"
    ) from e

logger = logging.getLogger(__name__)

class GPTClient:
    """Minimal OpenAI Chat Completions client with retry-on-transient logic."""

    # Substrings that indicate a *retriable* error (case-insensitive match
    # against str(exception)). Anything else is raised immediately.
    _RETRIABLE_SUBSTRINGS = (
        "rate limit", "rate_limit", "429",
        "timeout", "timed out",
        "500", "502", "503", "504",
        "connection", "connect",
        "internal server error", "server error",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: int = 120,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        retry_initial_delay: float = 1.0,
    ) -> None:
        """
        Args:
            api_key: OpenAI API key. If None, falls back to OPENAI_API_KEY env var.
            model:   OpenAI model name (e.g. gpt-4o-mini).
            timeout: HTTP timeout per request in seconds.
            max_retries: Max retries on transient errors before giving up.
            retry_backoff: Multiplicative backoff between retries.
            retry_initial_delay: First retry delay in seconds.
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Either pass api_key=... or "
                "export OPENAI_API_KEY before running."
            )
        # Single shared client. The OpenAI SDK is thread-safe for ordinary
        # chat.completions.create calls, which is what we use.
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_initial_delay = retry_initial_delay

    # ------------------------------------------------------------------

    def _is_retriable(self, exc: Exception) -> bool:
        """Decide whether an exception should trigger a retry."""
        s = str(exc).lower()
        return any(sub in s for sub in self._RETRIABLE_SUBSTRINGS)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Run a single chat completion and return the assistant's text.

        Retries on transient errors with exponential backoff + 10% jitter.
        Non-retriable errors (e.g. invalid_request, auth errors) bubble up
        immediately so they are not silently retried.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # The OpenAI SDK can return None for content in rare edge
                # cases (e.g. content filter). Coerce to "" so downstream
                # JSON-parsing logic can handle it gracefully.
                return resp.choices[0].message.content or ""

            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries - 1 and self._is_retriable(e):
                    delay = self.retry_initial_delay * (
                        self.retry_backoff ** attempt
                    )
                    delay += random.uniform(0, delay * 0.1)  # jitter
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
                    continue
                # Either non-retriable or out of retries -> raise.
                raise

        # Defensive fallback (shouldn't reach here).
        raise last_err if last_err else RuntimeError("LLM call failed without exception")