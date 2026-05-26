# =============================================================================
# retrieval.py
# -----------------------------------------------------------------------------
# Retrieval-augmented context construction for the ReviewRL-style baseline.
#
# In the original ReviewRL paper (Section 3.2, "Context Retrieval"), the
# pipeline:
#   (1) generates 3 query questions about the target paper,
#   (2) routes them to ArXiv-MCP (https:/github.com/blazickjp/arxiv-mcp-server),
#   (3) post-processes the retrieved papers into "(query, response, excerpts)"
#       triplets, and
#   (4) concatenates this context with the paper before review generation.
#
# Our user already has *pre-retrieved* citation context in two CSV columns:
#     * cited_in_text  -> citations the authors actually cite in the paper
#     * cited_in_ret   -> citations retrieved from OpenAlex
#
# This module substitutes the live ArXiv-MCP step with a deterministic
# "format & merge" function that turns those two columns into the same
# structured context block the ReviewRL paper feeds into the policy model.
# Empty / NaN / "No citations found" values are handled gracefully so the
# pipeline still runs when no citations are available.
# =============================================================================

from __future__ import annotations

import math
from typing import Optional

# Sentinel strings that the user's CSV may contain to indicate "no retrieval"
_EMPTY_TOKENS = {"", "no citations found", "nan", "none", "null", "n/a"}

def _is_missing(value) -> bool:
    """Return True if a citation field is effectively empty/missing."""
    if value is None:
        return True
    # pandas may pass NaN floats here
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _EMPTY_TOKENS:
        return True
    return False

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to `max_chars` characters, keeping the start."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...truncated]"

def build_retrieval_context(
    cited_in_text: Optional[str],
    cited_in_ret: Optional[str],
    max_chars_per_source: int = 4000,
    use_citations: bool = True,
) -> str:
    """
    Build a single context string from the two citation columns.

    The output mimics the post-processed "(query-response pairs, bibliography,
    relevance-ranked excerpts)" structure described in Section 3.2 of the
    ReviewRL paper.

    Parameters
    ----------
    cited_in_text : str | None
        Free-form text concatenating the snippets the paper itself cites.
    cited_in_ret : str | None
        Free-form text from OpenAlex retrieval.
    max_chars_per_source : int
        Hard cap on each source so the prompt fits in the context window.
        With Qwen2.5-3B-Instruct (32K context) we comfortably allow 4K each.
    use_citations : bool
        If False, returns a placeholder string and ignores both inputs.
        This is the equivalent of ReviewRL's "w/o Retrieval" ablation
        (Figure 1 / Section 5 of the paper) and is wired to the
        `--no-citations` CLI flag.

    Returns
    -------
    str
        A formatted context block ready for `LIMITATION_GENERATION_PROMPT`.
    """
    if not use_citations:
        return "[No retrieval context provided - running in 'w/o Retrieval' ablation mode.]"

    blocks = []

    # Block 1: in-text citations (analogous to "papers cited by the target paper").
    if not _is_missing(cited_in_text):
        blocks.append(
            "### In-text citations (snippets the paper cites)\n"
            + _truncate(str(cited_in_text).strip(), max_chars_per_source)
        )

    # Block 2: retrieved-from-OpenAlex citations (analogous to ArXiv-MCP excerpts).
    if not _is_missing(cited_in_ret):
        blocks.append(
            "### Retrieved related work (OpenAlex excerpts)\n"
            + _truncate(str(cited_in_ret).strip(), max_chars_per_source)
        )

    if not blocks:
        # Both columns were missing - keep a placeholder so the prompt is still well-formed.
        return "[No citation context available for this paper.]"

    return "\n\n".join(blocks)