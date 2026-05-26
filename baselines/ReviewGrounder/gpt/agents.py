"""
agents.py
---------

Five-agent limitation-generation pipeline (adapted from ReviewGrounder).

Pipeline:
    DrafterAgent             -> initial limitations from paper alone
    InsightMinerAgent        -> method/contribution-grounded limitations
    ResultsAnalyzerAgent     -> experiment/results-grounded limitations
    RelatedWorkAnalyzerAgent -> related-work-grounded limitations
    RefinerAgent             -> final consolidated limitations

Each agent shares the same `BaseAgent.call(...)` retry-on-bad-JSON loop, so
adding a new agent is just: subclass + define a `generate(...)` that builds
the user prompt and calls `self._call(prompt)`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from llm_client import GPTClient
from prompts import (
    SYSTEM,
    DRAFTER_PROMPT,
    INSIGHT_MINER_PROMPT,
    RESULTS_ANALYZER_PROMPT,
    RW_ANALYZER_PROMPT,
    REFINER_PROMPT,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# Robust JSON parsing
# ===========================================================================

def parse_json_loose(text: str, fallback: Any = None) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Handles:
      - plain JSON
      - JSON wrapped in ``` or ```json fences
      - leading/trailing prose around the JSON object
      - trailing commas inside the JSON

    Returns the parsed Python object, or `fallback` on failure.
    """
    if text is None:
        return fallback
    s = text.strip()
    if not s:
        return fallback

    # 1) Strip ```json ... ``` or ``` ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence:
        s = fence.group(1).strip()

    # 2) Direct parse.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3) Fallback: take the substring from the first '{' to the last '}',
    #    drop trailing commas, and try again. This handles models that
    #    occasionally leak text before/after the JSON object.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return fallback

# ===========================================================================
# Base agent: shared LLM call + JSON-parse loop
# ===========================================================================

class BaseAgent:
    """Common LLM invocation logic for every agent.

    `_call(prompt)` returns a dict:
        {
          "parsed": <dict | {} on failure>,
          "raw":    <raw assistant text>,
          "ok":     <bool: did we get a parseable dict?>
        }
    """

    NAME = "base"  # subclasses override for nicer logs

    def __init__(
        self,
        client: GPTClient,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_parse_retries: int = 3,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_parse_retries = max_parse_retries

    def _call(self, user_prompt: str) -> Dict[str, Any]:
        """Call the LLM, retrying up to `max_parse_retries` if the response
        is not parseable as JSON. Network/transient retries are handled
        inside GPTClient.chat().
        """
        last_raw = ""
        for attempt in range(self.max_parse_retries):
            raw = self.client.chat(
                system=SYSTEM,
                user=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            last_raw = raw
            parsed = parse_json_loose(raw, fallback=None)
            if isinstance(parsed, dict):
                return {"parsed": parsed, "raw": raw, "ok": True}
            logger.warning(
                "[%s] Could not parse JSON (attempt %d/%d). First 200 chars: %r",
                self.NAME, attempt + 1, self.max_parse_retries, raw[:200],
            )
        # All retries failed -- return the raw text so we don't lose info.
        return {"parsed": {}, "raw": last_raw, "ok": False}

# ===========================================================================
# Five concrete agents
# ===========================================================================

class DrafterAgent(BaseAgent):
    """Generate the initial limitations draft from the paper text alone."""
    NAME = "drafter"

    def generate(self, paper_text: str) -> Dict[str, Any]:
        prompt = DRAFTER_PROMPT.format(paper_text=paper_text)
        return self._call(prompt)

class InsightMinerAgent(BaseAgent):
    """Mine method/contribution-grounded limitations and flag bad draft items."""
    NAME = "insight_miner"

    def generate(self, paper_text: str, candidate_limitations: str) -> Dict[str, Any]:
        prompt = INSIGHT_MINER_PROMPT.format(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
        )
        return self._call(prompt)

class ResultsAnalyzerAgent(BaseAgent):
    """Analyze the experiments/results and flag bad draft items."""
    NAME = "results_analyzer"

    def generate(self, paper_text: str, candidate_limitations: str) -> Dict[str, Any]:
        prompt = RESULTS_ANALYZER_PROMPT.format(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
        )
        return self._call(prompt)

class RelatedWorkAnalyzerAgent(BaseAgent):
    """Surface related-work-grounded limitations using the citation columns."""
    NAME = "related_work_analyzer"

    def generate(
        self,
        paper_text: str,
        candidate_limitations: str,
        cited_in_text: str,
        cited_in_ret: str,
    ) -> Dict[str, Any]:
        # We pass non-empty placeholders even when one of the two citation
        # columns is missing -- the orchestrator decides whether to call this
        # agent at all based on whether there's any usable citation context.
        prompt = RW_ANALYZER_PROMPT.format(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
            cited_in_text=cited_in_text or "(none provided)",
            cited_in_ret=cited_in_ret or "(none provided)",
        )
        return self._call(prompt)

class RefinerAgent(BaseAgent):
    """Synthesize the final, deduplicated, evidence-grounded list."""
    NAME = "refiner"

    def generate(
        self,
        paper_text: str,
        candidate_limitations: str,
        insight_miner_json: str,
        results_analyzer_json: str,
        related_work_json: str,
    ) -> Dict[str, Any]:
        prompt = REFINER_PROMPT.format(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
            insight_miner_json=insight_miner_json,
            results_analyzer_json=results_analyzer_json,
            related_work_json=related_work_json,
        )
        return self._call(prompt)