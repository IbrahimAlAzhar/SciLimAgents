"""
pipeline.py
===========
Five-phase pipeline (adapted from AgentReview, Jin et al. 2024) for
generating *limitations* of a single scientific manuscript.

Phases (compared to the original AgentReview):

  I.   Reviewer Limitation Assessment       (kept, focused on limitations)
  II.  Author Rebuttal                      (kept)
  III. Reviewer-AC Discussion / Update      (kept)
  IV.  Meta-review Compilation              (kept, outputs final limitation list)
  V.   Paper Decision                       (REMOVED — not needed for our task)

The decision phase is intentionally dropped because we are not classifying
papers; we only want a structured list of their limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from agents import (
    ReviewerAgent, AuthorAgent, AreaChairAgent, LimitationItem,
)
from config import ExperimentConfig, LLMConfig
from llm_client import LLMClient

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Result container for a single paper
# ----------------------------------------------------------------------------

@dataclass
class PaperResult:
    paper_id: str
    initial_reviewer_limitations: Dict[str, List[dict]] = field(default_factory=dict)
    author_rebuttal: str = ""
    final_reviewer_limitations: Dict[str, List[dict]] = field(default_factory=dict)
    final_limitations: List[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

# ----------------------------------------------------------------------------
# Pipeline class
# ----------------------------------------------------------------------------

class AgentReviewLimitationPipeline:
    """End-to-end pipeline running phases I–IV for one paper at a time."""

    def __init__(self, exp_cfg: ExperimentConfig, llm_cfg: LLMConfig):
        self.exp_cfg = exp_cfg
        self.llm_cfg = llm_cfg
        self.client = LLMClient(llm_cfg)

        # Build agents once and reuse them across papers
        self.reviewers: List[ReviewerAgent] = [
            ReviewerAgent(profile=p, client=self.client)
            for p in exp_cfg.reviewers[: exp_cfg.num_reviewers]
        ]
        self.author = AuthorAgent(client=self.client)
        self.area_chair = AreaChairAgent(
            profile=exp_cfg.area_chair, client=self.client
        )

    # ------------------------------------------------------------------
    # Per-paper execution
    # ------------------------------------------------------------------
    def run_one(
        self,
        paper_id: str,
        paper_text: str,
        citation_text: Optional[str] = None,
    ) -> PaperResult:
        """Run all phases on a single paper and return a structured result."""
        result = PaperResult(paper_id=paper_id)

        # Truncate inputs defensively
        paper_text = _truncate(paper_text, self.exp_cfg.max_paper_chars)
        if not self.exp_cfg.use_citation_context:
            citation_text = None

        # ------------------------------------------------------------------
        # Phase I — Reviewer Limitation Assessment
        # ------------------------------------------------------------------
        reviewer_limits: Dict[str, List[LimitationItem]] = {}
        for rev in self.reviewers:
            try:
                items = rev.assess_limitations(
                    paper_text=paper_text,
                    citation_text=citation_text,
                    max_citation_chars=self.exp_cfg.max_citation_chars,
                )
                reviewer_limits[rev.profile.name] = items
                _log(self.exp_cfg, f"  [{paper_id}] {rev.profile.name}: "
                                   f"{len(items)} limitation(s) (Phase I).")
            except Exception as exc:
                logger.exception("Phase I failed for %s/%s: %s",
                                 paper_id, rev.profile.name, exc)
                reviewer_limits[rev.profile.name] = []

        result.initial_reviewer_limitations = {
            name: [it.to_dict() for it in items]
            for name, items in reviewer_limits.items()
        }

        # ------------------------------------------------------------------
        # Phase II — Author Rebuttal
        # Phase III — Reviewer-AC Discussion / Update
        # ------------------------------------------------------------------
        if self.exp_cfg.enable_rebuttal and any(reviewer_limits.values()):
            try:
                rebuttal = self.author.write_rebuttal(
                    paper_text=paper_text,
                    reviewer_limitations=reviewer_limits,
                )
                result.author_rebuttal = rebuttal
                _log(self.exp_cfg, f"  [{paper_id}] author rebuttal generated "
                                   f"({len(rebuttal)} chars).")
            except Exception as exc:
                logger.exception("Phase II failed for %s: %s", paper_id, exc)
                result.author_rebuttal = ""

            updated_limits: Dict[str, List[LimitationItem]] = {}
            for rev in self.reviewers:
                try:
                    items = rev.update_after_rebuttal(
                        paper_text=paper_text,
                        prior_limitations=reviewer_limits[rev.profile.name],
                        author_rebuttal=result.author_rebuttal,
                    )
                    updated_limits[rev.profile.name] = items
                    _log(self.exp_cfg, f"  [{paper_id}] {rev.profile.name}: "
                                       f"{len(items)} limitation(s) (Phase III).")
                except Exception as exc:
                    logger.exception("Phase III failed for %s/%s: %s",
                                     paper_id, rev.profile.name, exc)
                    # fall back to the initial list rather than dropping entirely
                    updated_limits[rev.profile.name] = reviewer_limits[rev.profile.name]
        else:
            updated_limits = reviewer_limits  # rebuttal disabled

        result.final_reviewer_limitations = {
            name: [it.to_dict() for it in items]
            for name, items in updated_limits.items()
        }

        # ------------------------------------------------------------------
        # Phase IV — Meta-review (Limitation Synthesis)
        # ------------------------------------------------------------------
        try:
            final_items = self.area_chair.synthesise(
                paper_text=paper_text,
                all_reviewer_limitations=updated_limits,
            )
            result.final_limitations = [it.to_dict() for it in final_items]
            _log(self.exp_cfg,
                 f"  [{paper_id}] AC produced {len(final_items)} "
                 f"final limitation(s) (Phase IV).")
        except Exception as exc:
            logger.exception("Phase IV failed for %s: %s", paper_id, exc)
            result.error = f"Phase IV failed: {exc}"
            # Fallback: union of reviewer limitations, deduplicated by sentence.
            union: List[dict] = []
            seen = set()
            for items in updated_limits.values():
                for it in items:
                    key = it.limitation.lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        union.append(it.to_dict())
            result.final_limitations = union

        return result

# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    keep_head = int(max_chars * 0.7)
    keep_tail = max_chars - keep_head - 50
    return (
        text[:keep_head]
        + "\n\n[... manuscript truncated for length ...]\n\n"
        + text[-keep_tail:]
    )

def _log(cfg: ExperimentConfig, msg: str) -> None:
    if cfg.verbose:
        logger.info(msg)
        print(msg, flush=True) 