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

Supports two output formats (JSON or structured text), chosen via
`--output-format` in main.py. The text parser converts the structured-text
format into the same dict shape as the JSON schema, so downstream code is
format-agnostic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from prompts import build_prompt, SYSTEM

logger = logging.getLogger(__name__)

# ===========================================================================
# JSON parser (lenient)
# ===========================================================================

def parse_json_loose(text: str, fallback: Any = None) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Handles ``` fences, leading/trailing prose, and trailing commas.
    Returns the parsed object or `fallback` on failure.
    """
    if text is None:
        return fallback
    s = text.strip()
    if not s:
        return fallback

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence:
        s = fence.group(1).strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Substring from first '{' to last '}', drop trailing commas, retry.
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
# Text-format parser
#
# Format produced by the prompts in prompts.py (text mode):
#
#   SUMMARY: <summary text>             # refiner only
#
#   === LIMITATION 1 ===
#   FieldName: value (can span lines)
#   FieldName: value
#
#   === LIMITATION 2 ===
#   ...
#
# `parse_text_items` splits on `=== LIMITATION N ===` and `parse_kv_block`
# parses each section into a {field_lower: value} dict, supporting multi-line
# values by treating any line that does NOT start with `Field:` as a
# continuation of the previous field.
# ===========================================================================

# Regex matching a single `=== LIMITATION N ===` (or similar) header line.
_HEADER_RE = re.compile(
    r"^\s*={2,}\s*LIMITATION\s+\d+\s*={2,}\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

# Regex matching a `Field: value` line. We restrict field names to
# 1-30 chars of word/space/hyphen so we don't match arbitrary "abc:" inside
# free-form descriptions. Field is captured in group(1), value in group(2).
_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _\-/]{0,29}?)\s*:\s*(.*)$")

def parse_kv_block(text: str) -> Dict[str, str]:
    """Parse a block of `Field: value` lines, supporting multi-line values."""
    item: Dict[str, str] = {}
    current_field: str | None = None
    current_value: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = _FIELD_RE.match(line)
        if m:
            # Flush previous field.
            if current_field is not None:
                item[current_field] = "\n".join(current_value).strip()
            current_field = (
                m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            )
            current_value = [m.group(2)]
        else:
            if current_field is not None and line.strip():
                current_value.append(line)

    if current_field is not None:
        item[current_field] = "\n".join(current_value).strip()

    return item

def parse_text_items(text: str) -> List[Dict[str, str]]:
    """Split `text` on `=== LIMITATION N ===` headers and parse each chunk.

    Returns an empty list if the explicit "NO LIMITATIONS" sentinel is found.
    """
    if not text:
        return []
    if re.search(r"\bNO\s+LIMITATIONS\b", text, re.IGNORECASE):
        # The model explicitly said it has nothing to report.
        return []

    parts = _HEADER_RE.split(text)
    # parts[0] is preamble (anything before the first header). Discard it.
    items: List[Dict[str, str]] = []
    for part in parts[1:]:
        kv = parse_kv_block(part)
        if kv:
            items.append(kv)
    return items

def parse_text_summary(text: str) -> str:
    """Extract the leading `SUMMARY: ...` block (refiner only)."""
    if not text:
        return ""
    # Match SUMMARY: ... up to the first `=== LIMITATION` header or end.
    m = re.search(
        r"(?im)^\s*SUMMARY\s*:\s*(.+?)(?=\n\s*={2,}\s*LIMITATION|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""

# ===========================================================================
# Unified parser dispatcher
# ===========================================================================

def parse_response(
    raw: str, agent_name: str, output_format: str
) -> Dict[str, Any]:
    """Parse an LLM response into the canonical dict schema for `agent_name`.

    Output schema matches the JSON_FORMAT specs in prompts.py exactly, so
    downstream code (refiner inputs, CSV columns, render_*_text helpers)
    works identically regardless of which format the model produced.
    """
    if output_format == "json":
        parsed = parse_json_loose(raw, fallback=None)
        return parsed if isinstance(parsed, dict) else {}

    # ---------------- text mode ----------------
    items = parse_text_items(raw)

    if agent_name == "drafter":
        # Each item should already have description/category/rationale.
        return {"limitations": items}

    if agent_name == "insight_miner":
        # Map "status" -> "draft_status" for schema parity with JSON mode.
        normalized = []
        for it in items:
            d = dict(it)
            if "status" in d and "draft_status" not in d:
                d["draft_status"] = d.pop("status")
            normalized.append(d)
        return {"method_limitations": normalized, "draft_issues": []}

    if agent_name == "results_analyzer":
        normalized = []
        for it in items:
            d = dict(it)
            if "status" in d and "draft_status" not in d:
                d["draft_status"] = d.pop("status")
            normalized.append(d)
        return {"results_limitations": normalized, "draft_issues": []}

    if agent_name == "rw_analyzer":
        # Map "type" -> "comparison_type" for schema parity.
        normalized = []
        for it in items:
            d = dict(it)
            if "type" in d and "comparison_type" not in d:
                d["comparison_type"] = d.pop("type")
            normalized.append(d)
        return {"related_work_limitations": normalized}

    if agent_name == "refiner":
        return {
            "final_limitations": items,
            "summary": parse_text_summary(raw),
        }

    return {}

# ===========================================================================
# BaseAgent + concrete agents
# ===========================================================================

class BaseAgent:
    """Shared LLM call + parse-retry loop for every agent."""

    AGENT_NAME = "base"

    def __init__(
        self,
        client,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_parse_retries: int = 3,
        output_format: str = "json",
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_parse_retries = max_parse_retries
        self.output_format = output_format

    @staticmethod
    def _is_useful(parsed: Dict[str, Any]) -> bool:
        """Return True if the parsed dict has at least one non-empty value
        from the agent's expected keys. Used to decide whether to retry."""
        if not parsed:
            return False
        for v in parsed.values():
            if isinstance(v, list) and v:
                return True
            if isinstance(v, str) and v.strip():
                return True
            if isinstance(v, dict) and v:
                return True
        return False

    def _build_prompt(self, **kwargs) -> str:
        return build_prompt(self.AGENT_NAME, self.output_format, **kwargs)

    def _call(self, user_prompt: str) -> Dict[str, Any]:
        """Call LLM and parse output. Retries up to max_parse_retries on
        unusable parses. Network/transient retries are handled in the client.
        """
        last_raw = ""
        last_parsed: Dict[str, Any] = {}
        for attempt in range(self.max_parse_retries):
            raw = self.client.chat(
                system=SYSTEM,
                user=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            last_raw = raw
            parsed = parse_response(raw, self.AGENT_NAME, self.output_format)
            last_parsed = parsed
            if self._is_useful(parsed):
                return {"parsed": parsed, "raw": raw, "ok": True}
            logger.warning(
                "[%s] Parsed result was empty/unusable (attempt %d/%d). "
                "First 200 chars of raw response: %r",
                self.AGENT_NAME, attempt + 1, self.max_parse_retries, raw[:200],
            )
        # Return whatever we have, even if empty -- caller can audit `raw`.
        return {"parsed": last_parsed, "raw": last_raw, "ok": False}

class DrafterAgent(BaseAgent):
    """Generate the initial limitations draft from paper text alone."""
    AGENT_NAME = "drafter"

    def generate(self, paper_text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(paper_text=paper_text)
        return self._call(prompt)

class InsightMinerAgent(BaseAgent):
    """Method/contribution-grounded limitations."""
    AGENT_NAME = "insight_miner"

    def generate(
        self, paper_text: str, candidate_limitations: str
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
        )
        return self._call(prompt)

class ResultsAnalyzerAgent(BaseAgent):
    """Experiments/results-grounded limitations."""
    AGENT_NAME = "results_analyzer"

    def generate(
        self, paper_text: str, candidate_limitations: str
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
        )
        return self._call(prompt)

class RelatedWorkAnalyzerAgent(BaseAgent):
    """Related-work-grounded limitations using cited / retrieved context."""
    AGENT_NAME = "rw_analyzer"

    def generate(
        self,
        paper_text: str,
        candidate_limitations: str,
        cited_in_text: str,
        cited_in_ret: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
            cited_in_text=cited_in_text or "(none provided)",
            cited_in_ret=cited_in_ret or "(none provided)",
        )
        return self._call(prompt)

class RefinerAgent(BaseAgent):
    """Final consolidated, deduplicated, evidence-grounded limitations."""
    AGENT_NAME = "refiner"

    def generate(
        self,
        paper_text: str,
        candidate_limitations: str,
        insight_miner_json: str,
        results_analyzer_json: str,
        related_work_json: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            paper_text=paper_text,
            candidate_limitations=candidate_limitations,
            insight_miner_json=insight_miner_json,
            results_analyzer_json=results_analyzer_json,
            related_work_json=related_work_json,
        )
        return self._call(prompt)