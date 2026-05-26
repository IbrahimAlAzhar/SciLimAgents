"""
llm_client.py
=============
Thin, retry-safe wrapper around OpenAI / Azure OpenAI chat-completion APIs.

Why a custom wrapper? The original AgentReview repo also wraps OpenAI calls
because the same prompt has to be issued to many different "agents" with
different system messages. We do the same here, but stay within the
dependencies the user is most likely to already have installed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Dict, Optional

from config import LLMConfig

logger = logging.getLogger(__name__)

class LLMClient:
    """A simple chat-completion client for OpenAI and Azure OpenAI."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Backend bootstrap
    # ------------------------------------------------------------------
    def _build_client(self):
        if self.cfg.provider == "azure":
            try:
                from openai import AzureOpenAI
            except ImportError as e:
                raise ImportError(
                    "Please `pip install openai>=1.0.0` to use the Azure backend."
                ) from e
            if not (self.cfg.azure_endpoint and self.cfg.azure_api_key):
                raise ValueError(
                    "Azure backend selected but AZURE_ENDPOINT / AZURE_OPENAI_KEY "
                    "are not set."
                )
            return AzureOpenAI(
                azure_endpoint=self.cfg.azure_endpoint,
                api_key=self.cfg.azure_api_key,
                api_version=self.cfg.azure_api_version,
            )

        # default: OpenAI
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Please `pip install openai>=1.0.0` to use the OpenAI backend."
            ) from e
        if not self.cfg.openai_api_key:
            raise ValueError(
                "OpenAI backend selected but OPENAI_API_KEY is not set."
            )
        return OpenAI(api_key=self.cfg.openai_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format_json: bool = False,
    ) -> str:
        """Issue a chat-completion request with retries and return the text."""
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens

        # On Azure, the "model" parameter is the deployment name.
        model = (
            self.cfg.azure_deployment
            if self.cfg.provider == "azure"
            else self.cfg.model
        )

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.cfg.top_p,
            timeout=self.cfg.request_timeout,
        )
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # broad on purpose — many transient errors
                last_err = exc
                wait = self.cfg.retry_backoff_seconds * attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ------------------------------------------------------------------
    # JSON helper
    # ------------------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Same as `chat` but tries to enforce a JSON object response. Falls back
        gracefully to best-effort JSON extraction if the model returns prose.
        """
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        return _safe_json_loads(raw)

# ----------------------------------------------------------------------------
# JSON parsing helpers
# ----------------------------------------------------------------------------

def _safe_json_loads(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return {}

    # Strip code fences if any
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove a leading "json" tag if present
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # try to grab the first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM JSON response. Returning raw text.")
    return {"_raw_text": text} 
