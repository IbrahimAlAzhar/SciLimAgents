"""
pipeline.py
===========
Orchestrator that runs the four DeepReview-style stages in sequence and
returns a structured per-paper result.

The pipeline mirrors DeepReview's three inference modes:

    fast      ->  Stage 2 only (single reviewer, no verification, no z1)
    standard  ->  Stage 2 (multi-reviewer) + Stage 3 (verification) + meta
                  (NO Stage 1 / no citation use)
    best      ->  Stage 1 (novelty + citations) + Stage 2 + Stage 3 + meta

This keeps the public API identical to the trained DeepReviewer-14B
(`mode="Fast Mode" | "Standard Mode" | "Best Mode"`) while only outputting
the limitation list.
"""

from __future__ import annotations

from typing import Dict

from .agents import (
    NoveltyAgent,
    MultiReviewerAgent,
    VerificationAgent,
    MetaReviewerAgent,
    to_json_cell,
)

class DeepReviewLimitationPipeline:
    """End-to-end limitation generator."""

    def __init__(self, llm, args):
        self.llm = llm
        self.args = args

        # Build the four agents up-front so the LLM is only loaded once
        self.novelty_agent = NoveltyAgent(
            llm,
            max_new_tokens=args.max_new_tokens_stage,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        self.reviewer_agent = MultiReviewerAgent(
            llm,
            reviewer_num=args.reviewer_num if args.mode != "fast" else 1,
            max_new_tokens=args.max_new_tokens_stage,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        self.verifier_agent = VerificationAgent(
            llm,
            max_new_tokens=args.max_new_tokens_stage + 400,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        self.meta_agent = MetaReviewerAgent(
            llm,
            max_new_tokens=args.max_new_tokens_final,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # ------------------------------------------------------------------
    # Single-row inference
    # ------------------------------------------------------------------
    def run(self, paper: str, citations: str) -> Dict[str, str]:
        """Run the full pipeline on a single paper.

        Returns a dict matching the new CSV columns in
        `data_utils.NEW_COLUMNS`.
        """
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
            # ---------------------------------------------- Stage 1 (z1)
            novelty_analysis = ""
            if mode == "best" and not self.args.no_citations:
                stage1 = self.novelty_agent.run(paper, citations)
                novelty_analysis = stage1["analysis"]
                out["deepreview_questions"] = stage1["questions"]
                out["deepreview_novelty_analysis"] = novelty_analysis

            # ---------------------------------------------- Stage 2 (z2)
            reviewer_results = self.reviewer_agent.run(paper, novelty_analysis)
            out["deepreview_reviewer_weaknesses"] = to_json_cell(
                reviewer_results
            )

            # ---------------------------------------------- Stage 3 (z3)
            verification = ""
            if mode in {"standard", "best"}:
                verification = self.verifier_agent.run(paper, reviewer_results)
                out["deepreview_verification"] = verification

            # ---------------------------------------------- Final meta
            if mode == "fast":
                # Fast mode: no z3, just deduplicate Stage-2 weaknesses via meta.
                verification_for_meta = "(fast mode: no verification)"
            else:
                verification_for_meta = verification

            meta = self.meta_agent.run(
                paper=paper,
                novelty_analysis=novelty_analysis,
                verification=verification_for_meta,
            )
            out["deepreview_final_limitations"] = to_json_cell(
                meta["limitations"]
            )

        except Exception as e:                       # never crash the run
            out["deepreview_status"] = f"error: {type(e).__name__}: {e}"

        return out 