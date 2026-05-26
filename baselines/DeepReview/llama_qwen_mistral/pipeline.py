"""
pipeline.py
===========
Sequential orchestrator: per row, runs the four DeepReview-style stages
in order and returns a structured result dict.

Mirrors DeepReview's three inference modes:
    fast      ->  Stage 2 only (single reviewer, no verification, no z1)
    standard  ->  Stage 2 (multi-reviewer) + Stage 3 (verification) + meta
                  (NO Stage 1 / no citation use)
    best      ->  Stage 1 (novelty + citations) + Stage 2 + Stage 3 + meta
"""

from __future__ import annotations

from typing import Dict

# Support both packaged and flat layouts.
try:
    from .agents import (
        NoveltyAgent, MultiReviewerAgent,
        VerificationAgent, MetaReviewerAgent, to_cell,
    )
except ImportError:                                 # flat layout
    from agents import (                            # type: ignore
        NoveltyAgent, MultiReviewerAgent,
        VerificationAgent, MetaReviewerAgent, to_cell,
    )

class DeepReviewLimitationPipeline:
    """End-to-end limitation generator (one row at a time)."""

    def __init__(self, llm, args):
        self.llm = llm
        self.args = args
        self.store_format = getattr(args, "store_format", "json")

        self.novelty_agent = NoveltyAgent(
            llm,
            max_new_tokens=args.max_new_tokens_stage,
            temperature=args.temperature, top_p=args.top_p,
        )
        self.reviewer_agent = MultiReviewerAgent(
            llm,
            reviewer_num=args.reviewer_num if args.mode != "fast" else 1,
            max_new_tokens=args.max_new_tokens_stage,
            temperature=args.temperature, top_p=args.top_p,
        )
        self.verifier_agent = VerificationAgent(
            llm,
            max_new_tokens=args.max_new_tokens_stage + 400,
            temperature=args.temperature, top_p=args.top_p,
        )
        self.meta_agent = MetaReviewerAgent(
            llm,
            max_new_tokens=args.max_new_tokens_final,
            temperature=args.temperature, top_p=args.top_p,
        )

    def run(self, paper: str, citations: str) -> Dict[str, str]:
        """Run the full pipeline on one paper. Never raises."""
        mode = self.args.mode
        out = {
            "deepreview_questions": "",
            "deepreview_novelty_analysis": "",
            "deepreview_reviewer_weaknesses": "",
            "deepreview_verification": "",
            "deepreview_final_limitations": "",
            "deepreview_status": "ok",
        }

        try:
            # ---- Stage 1 (z1) -----------------------------------------
            novelty_analysis = ""
            if mode == "best" and not self.args.no_citations:
                stage1 = self.novelty_agent.run(paper, citations)
                novelty_analysis = stage1["analysis"]
                out["deepreview_questions"] = stage1["questions"]
                out["deepreview_novelty_analysis"] = novelty_analysis

            # ---- Stage 2 (z2) -----------------------------------------
            reviewer_results = self.reviewer_agent.run(paper, novelty_analysis)
            out["deepreview_reviewer_weaknesses"] = to_cell(
                reviewer_results, self.store_format
            )

            # ---- Stage 3 (z3) -----------------------------------------
            verification = ""
            if mode in {"standard", "best"}:
                verification = self.verifier_agent.run(paper, reviewer_results)
                out["deepreview_verification"] = verification

            # ---- Final meta ------------------------------------------
            verification_for_meta = (
                "(fast mode: no verification)" if mode == "fast"
                else verification
            )
            meta = self.meta_agent.run(
                paper=paper,
                novelty_analysis=novelty_analysis,
                verification=verification_for_meta,
            )
            out["deepreview_final_limitations"] = to_cell(
                meta["limitations"], self.store_format
            )

        except Exception as e:                       # safety net
            out["deepreview_status"] = f"error: {type(e).__name__}: {e}"

        return out