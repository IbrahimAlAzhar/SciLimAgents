"""
utils.py
--------
Small helpers used by the limitation-generation pipeline.

The most important utility here is `extract_json_between_markers`, which is a
direct port of the helper of the same name from
    https:/github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py
It looks for a fenced ```json ... ``` block (or any {...} block) in the LLM
output and parses it.  We need it because AI-Scientist's review template asks
the model to return its review inside a fenced JSON block.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# JSON extraction -----------------------------------------------------------
# ---------------------------------------------------------------------------
def extract_json_between_markers(llm_output: str) -> Optional[dict]:
    """
    Extract the first JSON object found in `llm_output`.

    AI-Scientist asks the model to wrap its review in:

        REVIEW JSON:
        ```json
        { ... }
        ```

    so we first try to grab anything between ```json ... ``` fences, then fall
    back to the first balanced {...} block.  Returns ``None`` if nothing
    parseable is found.
    """
    if not llm_output:
        return None

    # 1) Prefer fenced ```json ... ``` blocks (most reliable).
    fenced = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        llm_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates: List[str] = list(fenced)

    # 2) Fallback: any balanced-looking {...} blob.
    if not candidates:
        # Greedy-but-bounded search for outermost JSON object.
        for match in re.finditer(r"\{[\s\S]*\}", llm_output):
            candidates.append(match.group(0))

    for raw in candidates:
        # Strip stray newlines / control chars that break json.loads.
        cleaned = raw.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to repair by removing trailing commas, common LLM bug.
            repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None

# ---------------------------------------------------------------------------
# Limitation post-processing ------------------------------------------------
# ---------------------------------------------------------------------------
def normalize_limitations(value: Any) -> str:
    """
    The "Limitations" field in the review JSON is supposed to be a list of
    strings, but LLMs occasionally return a single string or a dict.  This
    helper coerces any reasonable shape into a clean newline-separated string
    (one limitation per line, prefixed with "- ").
    """
    if value is None:
        return ""

    # Already a list -> bullet it.
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(f"- {item}" for item in items)

    # A dict -> flatten values.
    if isinstance(value, dict):
        items = [str(v).strip() for v in value.values() if str(v).strip()]
        return "\n".join(f"- {item}" for item in items)

    # A string -> return as-is.
    return str(value).strip()

# ---------------------------------------------------------------------------
# Citation context builder --------------------------------------------------
# ---------------------------------------------------------------------------
_EMPTY_CITATION_MARKERS = {
    "",
    "nan",
    "none",
    "null",
    "no citations found",
    "no citation found",
    "no citations",
}

def _is_meaningful(value: Any) -> bool:
    """Return True iff `value` looks like real citation text."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    return s.lower() not in _EMPTY_CITATION_MARKERS

def build_citation_block(
    cited_in_text: Any,
    cited_in_ret: Any,
    max_chars: int = 6000,
) -> str:
    """
    Build a "## Cited works" context block that we prepend to the paper text.

    Both `cited_in_text` (the paper's own bibliographic mentions) and
    `cited_in_ret` (text retrieved from OpenAlex) are optional - the function
    silently skips any column that is empty / NaN / "No citations found".

    The combined block is truncated to `max_chars` characters so we don't blow
    past the model's context window when the retrieved text is huge.
    """
    blocks: List[str] = []
    if _is_meaningful(cited_in_text):
        blocks.append(
            "### Citations from the paper\n" + str(cited_in_text).strip()
        )
    if _is_meaningful(cited_in_ret):
        blocks.append(
            "### Retrieved abstracts of cited works (OpenAlex)\n"
            + str(cited_in_ret).strip()
        )

    if not blocks:
        return ""

    out = "## Cited works\n" + "\n\n".join(blocks)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n... [truncated]"
    return out 
