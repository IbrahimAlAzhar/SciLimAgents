"""
agents.py
=========
Per-stage agents for the DeepReview-baseline limitation-generation pipeline.

Each agent has a single `run(...)` method.  It formats prompt(s), calls the
shared LLM backend, parses the boxed output, and returns a typed result
dict.  Pipeline orchestration is in `pipeline.py`.

Stages (from the DeepReview paper):
    NoveltyAgent       -> z1 : question generation + literature analysis
    MultiReviewerAgent -> z2 : N simulated reviewers, each emits weaknesses
    VerificationAgent  -> z3 : evidence-based reliability check on weaknesses
    MetaReviewerAgent  -> final: aggregate KEEP/REVISE weaknesses into
                                 the limitation list

JSON / text output:
    The LLM is *never* asked to produce JSON — every prompt requests a
    plain `<boxed_*>...</boxed_*>` block with a numbered list, which 3B-8B
    models handle well.  The PARSED Python list can then be serialised to
    either JSON (default) or newline-separated text via `to_cell()`.
"""

from __future__ import annotations

import json
from typing import Dict, List

# Make the package importable both as `deepreview_baseline.agents` AND as
# a flat module (when the user dumps the files into a single directory).
try:
    from . import prompts
    from .data_utils import extract_boxed, extract_numbered_list
except ImportError:                                 # flat layout fallback
    import prompts                                  # type: ignore
    from data_utils import extract_boxed, extract_numbered_list  # type: ignore

# ---------------------------------------------------------------------------
# Stage 1 - Novelty Verification
# ---------------------------------------------------------------------------
class NoveltyAgent:
    """Mirrors DeepReview Stage 1 (novelty verification z1).

    Two LLM calls:
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

    def run(self, paper: str, citations: str) -> Dict[str, str]:
        # ----- Step 1: question generation
        q_messages = [
            {"role": "system", "content":
                "You are a meticulous academic reviewer."},
            {"role": "user",
             "content": prompts.NOVELTY_QUESTION_PROMPT.format(paper=paper)},
        ]
        q_raw = self.llm.chat(
            q_messages,
            max_new_tokens=max(300, self.max_new_tokens / 2),
            temperature=self.temperature, top_p=self.top_p,
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
            temperature=self.temperature, top_p=self.top_p,
        )
        analysis = extract_boxed(a_raw, "analysis") or a_raw

        return {
            "questions": questions_block.strip(),
            "analysis":  analysis.strip(),
        }

# ---------------------------------------------------------------------------
# Stage 2 - Multi-dimensional Review (sequential reviewers)
# ---------------------------------------------------------------------------
class MultiReviewerAgent:
    """Simulates `reviewer_num` reviewers, each emitting weaknesses only.

    Each reviewer is given a different "expertise focus" so we get
    multi-dimensional coverage (cf. DeepReview's `\\boxed_simreviewers{}`).
    Reviewers run *sequentially* — there is no batching, which keeps the
    code simple and memory-light on a 40 GB GPU.
    """

    def __init__(self, llm, reviewer_num: int = 4,
                 max_new_tokens: int = 800,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.reviewer_num = reviewer_num
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def run(self, paper: str, novelty_analysis: str
            ) -> List[Dict[str, object]]:
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
                temperature=self.temperature, top_p=self.top_p,
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
    LLM to verify each one against the paper.  Returns the verification
    text (one block per weakness with KEEP / DROP / REVISE verdicts).
    """

    def __init__(self, llm, max_new_tokens: int = 1200,
                 temperature: float = 0.4, top_p: float = 0.95):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def run(self, paper: str,
            reviewer_results: List[Dict[str, object]]) -> str:
        all_weaknesses: list[str] = []
        for r in reviewer_results:
            for w in r["weaknesses"]:
                all_weaknesses.append(
                    f"(R{r['reviewer_id']} | {r['focus']}) {w}")

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
            temperature=self.temperature, top_p=self.top_p,
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

    def run(self, paper: str, novelty_analysis: str,
            verification: str) -> Dict[str, object]:
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
            temperature=self.temperature, top_p=self.top_p,
        )
        block = extract_boxed(raw, "limitations") or raw
        limitations = extract_numbered_list(block)
        return {"raw": raw.strip(), "limitations": limitations}

# ---------------------------------------------------------------------------
# Convenience: list -> CSV-cell serialiser (JSON or plain text)
# ---------------------------------------------------------------------------
def to_cell(obj, fmt: str = "json") -> str:
    """Serialise `obj` for storage in a CSV cell.

    fmt='json' -> json.dumps(obj)            (machine-readable, recommended)
    fmt='text' -> human-readable bullet list (one item per line)
    """
    if fmt == "text":
        if isinstance(obj, list):
            # If list of dicts (reviewer results), render as nested text.
            if obj and isinstance(obj[0], dict):
                blocks = []
                for r in obj:
                    head = (
                        f"Reviewer {r.get('reviewer_id', '?')} "
                        f"[{r.get('focus', '')}]"
                    )
                    body = "\n".join(
                        f"  - {w}" for w in r.get("weaknesses", [])
                    )
                    blocks.append(f"{head}\n{body}")
                return "\n\n".join(blocks)
            # Flat list of strings -> one per line
            return "\n".join(f"- {x}" for x in obj)
        return str(obj)

    # default: JSON
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)