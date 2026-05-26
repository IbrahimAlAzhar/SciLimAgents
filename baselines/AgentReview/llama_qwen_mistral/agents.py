"""
agents.py
=========
LLM-powered agents for the limitation-generation pipeline.

Three roles, mirroring AgentReview (Jin et al., 2024):

  * `ReviewerAgent`   — identifies limitations of the manuscript.
  * `AuthorAgent`     — writes a rebuttal addressing those limitations.
  * `AreaChairAgent`  — synthesises the final consolidated list of limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from config import ReviewerProfile, AreaChairProfile
from llm_client import LLMClient
from prompts import (
    GLOBAL_PROMPT,
    reviewer_persona,
    area_chair_persona,
    AUTHOR_PERSONA,
    PHASE1_REVIEWER_TASK,
    PHASE2_AUTHOR_REBUTTAL,
    PHASE3_UPDATED_REVIEW,
    PHASE4_META_LIMITATIONS,
    build_citation_block,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Output containers
# ============================================================================

@dataclass
class LimitationItem:
    category: str
    limitation: str
    evidence: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, str]:
        d = {"category": self.category, "limitation": self.limitation}
        if self.evidence:
            d["evidence"] = self.evidence
        if self.rationale:
            d["rationale"] = self.rationale
        return d

def _parse_limitations(payload: dict, key: str) -> List[LimitationItem]:
    """Convert the LLM JSON output into a list of LimitationItem."""
    items = []
    raw_list = payload.get(key, []) if isinstance(payload, dict) else []
    if not isinstance(raw_list, list):
        return items
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        items.append(LimitationItem(
            category=str(entry.get("category", "Other")).strip() or "Other",
            limitation=str(entry.get("limitation", "")).strip(),
            evidence=str(entry.get("evidence", "")).strip(),
            rationale=str(entry.get("rationale", "")).strip(),
        ))
    # Drop empty entries
    return [it for it in items if it.limitation]

# ============================================================================
# Reviewer agent
# ============================================================================

class ReviewerAgent:
    """A single reviewer that produces a JSON list of limitations."""

    def __init__(self, profile: ReviewerProfile, client: LLMClient):
        self.profile = profile
        self.client = client
        self._persona = reviewer_persona(profile)

    # ----- system message -------------------------------------------------
    def _system_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": GLOBAL_PROMPT},
            {"role": "system", "content": self._persona},
        ]

    # ----- Phase I -------------------------------------------------------
    def assess_limitations(
        self,
        paper_text: str,
        citation_text: Optional[str] = None,
        max_citation_chars: int = 4000,
    ) -> List[LimitationItem]:
        citation_block = build_citation_block(
            citation_text or "", max_chars=max_citation_chars
        )
        user = PHASE1_REVIEWER_TASK.format(
            paper_text=paper_text,
            citation_block=citation_block,
        )
        messages = self._system_messages() + [{"role": "user", "content": user}]
        payload = self.client.chat_json(messages)
        return _parse_limitations(payload, "limitations")

    # ----- Phase III -----------------------------------------------------
    def update_after_rebuttal(
        self,
        paper_text: str,
        prior_limitations: List[LimitationItem],
        author_rebuttal: str,
    ) -> List[LimitationItem]:
        prior_str = _format_limitations_for_prompt(prior_limitations)
        user = PHASE3_UPDATED_REVIEW.format(
            paper_text=paper_text,
            prior_limitations=prior_str,
            author_rebuttal=author_rebuttal,
        )
        messages = self._system_messages() + [{"role": "user", "content": user}]
        payload = self.client.chat_json(messages)
        return _parse_limitations(payload, "limitations")

# ============================================================================
# Author agent
# ============================================================================

class AuthorAgent:
    """Writes a rebuttal that addresses each reviewer's limitations."""

    def __init__(self, client: LLMClient):
        self.client = client

    def _system_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": GLOBAL_PROMPT},
            {"role": "system", "content": AUTHOR_PERSONA},
        ]

    def write_rebuttal(
        self,
        paper_text: str,
        reviewer_limitations: Dict[str, List[LimitationItem]],
    ) -> str:
        block_chunks = []
        for reviewer_name, items in reviewer_limitations.items():
            block_chunks.append(
                f"### {reviewer_name}\n" + _format_limitations_for_prompt(items)
            )
        reviewer_block = "\n\n".join(block_chunks)
        user = PHASE2_AUTHOR_REBUTTAL.format(
            n_reviewers=len(reviewer_limitations),
            paper_text=paper_text,
            reviewer_limitations_block=reviewer_block,
        )
        messages = self._system_messages() + [{"role": "user", "content": user}]
        return self.client.chat(messages)

# ============================================================================
# Area-chair agent
# ============================================================================

class AreaChairAgent:
    """Synthesises a final, deduplicated list of limitations."""

    def __init__(self, profile: AreaChairProfile, client: LLMClient):
        self.profile = profile
        self.client = client
        self._persona = area_chair_persona(profile)

    def _system_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": GLOBAL_PROMPT},
            {"role": "system", "content": self._persona},
        ]

    def synthesise(
        self,
        paper_text: str,
        all_reviewer_limitations: Dict[str, List[LimitationItem]],
    ) -> List[LimitationItem]:
        block_chunks = []
        for reviewer_name, items in all_reviewer_limitations.items():
            block_chunks.append(
                f"### {reviewer_name}\n" + _format_limitations_for_prompt(items)
            )
        review_block = "\n\n".join(block_chunks)

        user = PHASE4_META_LIMITATIONS.format(
            paper_text=paper_text,
            all_reviewer_limitations=review_block,
        )
        messages = self._system_messages() + [{"role": "user", "content": user}]
        payload = self.client.chat_json(messages)
        return _parse_limitations(payload, "final_limitations")

# ============================================================================
# Helpers
# ============================================================================

def _format_limitations_for_prompt(items: List[LimitationItem]) -> str:
    if not items:
        return "(none)"
    lines = []
    for i, it in enumerate(items, start=1):
        lines.append(
            f"{i}. [{it.category}] {it.limitation}"
            + (f"\n   Evidence: {it.evidence}" if it.evidence else "")
        )
    return "\n".join(lines) 