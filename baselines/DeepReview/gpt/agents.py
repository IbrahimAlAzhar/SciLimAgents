"""
agents.py
=========
Per-stage agents for the DeepReview-baseline limitation-generation pipeline.

Each agent is a small class with a single `run(...)` method.  The agent
formats its prompt(s), calls the shared LLM backend, parses the boxed
output and returns a typed result dict.  Pipeline orchestration is in
`pipeline.py`.

Stages (from the DeepReview paper):
    NoveltyAgent       -> z1 : question generation + literature analysis
    MultiReviewerAgent -> z2 : N simulated reviewers, each emits weaknesses
    VerificationAgent  -> z3 : evidence-based reliability check on weaknesses
    MetaReviewerAgent  -> final: aggregate KEEP/REVISE weaknesses into
                                 the limitation list
"""

from __future__ import annotations

import json
from typing import Dict, List

from . import prompts
from .data_utils import extract_boxed, extract_numbered_list

# ---------------------------------------------------------------------------
# Stage 1 - Novelty Verification
# ---------------------------------------------------------------------------
class NoveltyAgent:
    """Mirrors DeepReview Stage 1 (novelty verification z1).

    Consists of two LLM calls:
      1. Generate three research questions.
      2. Generate the novelty analysis using the questions and the
         (optional) cited-literature context.
    """

    def __init__(self, llm, max_new_tokens: int = 800,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # -------------------------------------------------------------- run
    def run(self, paper: str, citations: str) -> Dict[str, str]:
        """Run question generation + novelty analysis.

        Returns:
            {"questions": str, "analysis": str}
        """
        # ----- Step 1: question generation
        q_messages = [
            {"role": "system", "content":
                "You are a meticulous academic reviewer."},
            {"role": "user",
             "content": prompts.NOVELTY_QUESTION_PROMPT.format(paper=paper)},
        ]
        q_raw = self.llm.chat(
            q_messages,
            max_new_tokens=self.max_new_tokens / 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        questions_block = extract_boxed(q_raw, "questions") or q_raw

        # ----- Step 2: novelty analysis
        a_messages = [
            {"role": "system", "content":
                "You are a meticulous academic reviewer."},
            {"role": "user",
             "content": prompts.NOVELTY_ANALYSIS_PROMPT.format(
                 paper=paper,
                 questions=questions_block,
                 citations=citations or "(no citation context available)",
             )},
        ]
        a_raw = self.llm.chat(
            a_messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        analysis = extract_boxed(a_raw, "analysis") or a_raw

        return {
            "questions": questions_block.strip(),
            "analysis": analysis.strip(),
        }

# ---------------------------------------------------------------------------
# Stage 2 - Multi-dimensional Review
# ---------------------------------------------------------------------------
class MultiReviewerAgent:
    """Simulates `reviewer_num` reviewers, each emitting weaknesses only.

    Each reviewer is given a different "expertise focus" so that we get
    multi-dimensional coverage (cf. DeepReview's `\\boxed_simreviewers{}`).
    """

    def __init__(self, llm, reviewer_num: int = 4,
                 max_new_tokens: int = 800,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.reviewer_num = reviewer_num
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # -------------------------------------------------------------- run
    def run(self, paper: str, novelty_analysis: str
            ) -> List[Dict[str, object]]:
        """Run the simulated reviewers sequentially.

        Returns a list of `reviewer_num` dicts:
            {"reviewer_id": int,
             "focus": str,
             "raw": str,
             "weaknesses": list[str]}
        """
        results = []
        for i in range(self.reviewer_num):
            focus = prompts.reviewer_focus(i)
            sys_prompt = prompts.REVIEWER_SYSTEM_PROMPT.format(
                reviewer_id=i + 1, focus=focus)
            user_prompt = prompts.REVIEWER_USER_PROMPT.format(
                paper=paper,
                novelty_analysis=novelty_analysis or "(skipped)",
            )
            raw = self.llm.chat(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user_prompt}],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            weaknesses_block = extract_boxed(raw, "weaknesses") or raw
            weaknesses = extract_numbered_list(weaknesses_block)
            results.append({
                "reviewer_id": i + 1,
                "focus": focus,
                "raw": raw.strip(),
                "weaknesses": weaknesses,
            })
        return results

# ---------------------------------------------------------------------------
# Stage 3 - Reliability Verification
# ---------------------------------------------------------------------------
class VerificationAgent:
    """Evidence-based reliability check (DeepReview z3).

    Concatenates all candidate weaknesses from the reviewers and asks the
    LLM to verify each one against the paper.  Output is a single block
    containing per-weakness verification with KEEP / DROP / REVISE verdicts.
    """

    def __init__(self, llm, max_new_tokens: int = 1200,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # -------------------------------------------------------------- run
    def run(self, paper: str,
            reviewer_results: List[Dict[str, object]]) -> str:
        """Return the verification report text (concatenated boxed blocks)."""
        # Flatten and number the candidate weaknesses
        all_weaknesses: list[str] = []
        for r in reviewer_results:
            for w in r["weaknesses"]:
                all_weaknesses.append(f"(R{r['reviewer_id']} | {r['focus']}) {w}")

        if not all_weaknesses:
            return ""

        numbered = "\n".join(
            f"{i + 1}. {w}" for i, w in enumerate(all_weaknesses)
        )
        prompt = prompts.VERIFICATION_PROMPT.format(
            paper=paper,
            n_reviewers=len(reviewer_results),
            weaknesses=numbered,
        )
        raw = self.llm.chat(
            [{"role": "system",
              "content": "You are a careful evidence-grounded verifier."},
             {"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return raw.strip()

# ---------------------------------------------------------------------------
# Final - Meta-Review limitation aggregator
# ---------------------------------------------------------------------------
class MetaReviewerAgent:
    """Synthesise the verified weaknesses into the final limitation list."""

    def __init__(self, llm, max_new_tokens: int = 900,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # -------------------------------------------------------------- run
    def run(self, paper: str, novelty_analysis: str,
            verification: str) -> Dict[str, object]:
        """Return {"raw": str, "limitations": list[str]}"""
        prompt = prompts.META_LIMITATIONS_PROMPT.format(
            paper=paper,
            novelty_analysis=novelty_analysis or "(skipped)",
            verification=verification or "(no verification report available)",
        )
        raw = self.llm.chat(
            [{"role": "system",
              "content": "You are the DeepReview meta-reviewer."},
             {"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        block = extract_boxed(raw, "limitations") or raw
        limitations = extract_numbered_list(block)
        return {"raw": raw.strip(), "limitations": limitations}

# ---------------------------------------------------------------------------
# Convenience: serialise list[dict] -> json string for a CSV cell
# ---------------------------------------------------------------------------
def to_json_cell(obj) -> str:
    """JSON-encode a Python object so it can be safely stored in a CSV cell."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj) 
    