"""
reviewer.py
-----------
The actual "reviewer" that takes a paper text and produces a list of
limitations.

This is a faithful port of the `perform_review` function from
    https:/github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py
specialised to the limitation-generation task.  We keep the three pillars of
the AI-Scientist reviewer:

    1. Initial review using the NeurIPS-style review form (`prompts.neurips_form`).
    2. Self-reflection: the reviewer is asked to re-read its own JSON and
       refine it (`prompts.reviewer_reflection_prompt`), repeated
       `num_reflections - 1` times.
    3. Response ensembling: optionally sample `num_reviews_ensemble` reviews
       in parallel, then ask a meta-reviewer / area chair to merge them.

We do NOT score, decide, accept/reject, or evaluate anything else - we only
extract the "Limitations" field from the final JSON, because this code is
being used as a baseline strictly for the limitation-generation task.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm_client import LLMClient
from prompts import (
    get_meta_review_prompt,
    meta_reviewer_system_prompt,
    neurips_form,
    reviewer_reflection_prompt,
    reviewer_system_prompt_neg,
    reviewer_system_prompt_pos,
    template_instructions,
)
from utils import extract_json_between_markers, normalize_limitations

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _build_initial_prompt(paper_text: str, citation_block: str) -> str:
    """
    Compose the very first user message for the reviewer.  Matches the layout
    used by AI-Scientist:

        <NeurIPS form>
        <(optional) few-shot examples>      [we omit by default]
        <output template instructions>

        Here is the paper you are asked to review:
        ```
        <paper>
        ```
    """
    parts = [neurips_form, template_instructions]
    parts.append("\nHere is the paper you are asked to review:\n```\n")
    if citation_block:
        # Citation context goes right above the paper itself so the model
        # can use it as background reading without confusing it for the body.
        parts.append(citation_block + "\n\n")
    parts.append(paper_text.strip())
    parts.append("\n```")
    return "".join(parts)

def _safe_review(text: str) -> Dict[str, Any]:
    """Try to parse JSON; return an empty dict on failure."""
    parsed = extract_json_between_markers(text)
    return parsed if isinstance(parsed, dict) else {}

# ---------------------------------------------------------------------------
# Single-reviewer pass with reflection --------------------------------------
# ---------------------------------------------------------------------------
def _run_single_reviewer(
    client: LLMClient,
    paper_text: str,
    citation_block: str,
    system_prompt: str,
    num_reflections: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run one independent reviewer (initial review + N-1 reflection rounds).

    Returns the final review JSON plus the per-round JSON history (handy for
    debugging / for the agent-trace columns we save to the CSV).
    """
    initial_prompt = _build_initial_prompt(paper_text, citation_block)
    raw, history = client.get_response_from_llm(
        prompt=initial_prompt,
        system_message=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    review = _safe_review(raw)
    per_round: List[Dict[str, Any]] = [{"round": 0, "raw": raw, "json": review}]

    # Reflection loop ------------------------------------------------------
    for j in range(1, num_reflections):
        reflect_prompt = reviewer_reflection_prompt.format(
            current_round=j + 1,
            num_reflections=num_reflections,
        )
        raw, history = client.get_response_from_llm(
            prompt=reflect_prompt,
            system_message=system_prompt,
            msg_history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        new_review = _safe_review(raw)
        if new_review:
            review = new_review
        per_round.append({"round": j, "raw": raw, "json": new_review})
        # AI-Scientist short-circuits when the reviewer says "I am done".
        if "I am done" in raw:
            break

    return review, per_round

# ---------------------------------------------------------------------------
# Public entry point --------------------------------------------------------
# ---------------------------------------------------------------------------
def perform_limitation_review(
    client: LLMClient,
    paper_text: str,
    citation_block: str = "",
    num_reflections: int = 2,
    num_reviews_ensemble: int = 1,
    temperature: float = 0.75,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Generate limitations for one paper, faithful to AI-Scientist's pipeline.

    Returns a dict with:
      - "limitations":          str   (final, post-meta limitation bullets)
      - "agent_limitations":    list  (per-reviewer limitations, length = ensemble)
      - "agent_full_reviews":   list  (per-reviewer full review JSONs)
      - "meta_review":          dict  (the meta/area-chair JSON, or {} if N=1)
      - "reflection_traces":    list  (per-round raw text + parsed JSON per agent)
    """
    # ------------------------------------------------------------------
    # 1. Pick which reviewer prompts to use.  AI-Scientist runs an
    #    ensemble of reviewers with different "personalities" (negative
    #    and positive bias).  We alternate between them so a 2-reviewer
    #    ensemble has one strict + one lenient reviewer, just like the
    #    original code.
    # ------------------------------------------------------------------
    personality_prompts = [reviewer_system_prompt_neg, reviewer_system_prompt_pos]

    agent_full_reviews: List[Dict[str, Any]] = []
    agent_limitations: List[str] = []
    reflection_traces: List[List[Dict[str, Any]]] = []

    for i in range(num_reviews_ensemble):
        sys_prompt = personality_prompts[i % len(personality_prompts)]
        review, trace = _run_single_reviewer(
            client=client,
            paper_text=paper_text,
            citation_block=citation_block,
            system_prompt=sys_prompt,
            num_reflections=num_reflections,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        agent_full_reviews.append(review)
        agent_limitations.append(normalize_limitations(review.get("Limitations", "")))
        reflection_traces.append(trace)

    # ------------------------------------------------------------------
    # 2. Meta-review (only if we ran more than one reviewer).
    # ------------------------------------------------------------------
    meta_review: Dict[str, Any] = {}
    if num_reviews_ensemble > 1:
        # Stitch each reviewer's JSON into the meta prompt.
        review_strings = [
            json.dumps(r, indent=2, ensure_ascii=False) for r in agent_full_reviews
        ]
        meta_prompt = get_meta_review_prompt(review_strings)
        sys = meta_reviewer_system_prompt.format(reviewer_count=num_reviews_ensemble)
        raw_meta, _ = client.get_response_from_llm(
            prompt=meta_prompt,
            system_message=sys,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        meta_review = _safe_review(raw_meta)

    # ------------------------------------------------------------------
    # 3. Pick the final limitation string.  If we ensembled, prefer the
    #    meta-review; otherwise fall back to the single reviewer.
    # ------------------------------------------------------------------
    if meta_review:
        final_limitations = normalize_limitations(meta_review.get("Limitations", ""))
        # Some meta-reviews drop the field; in that case concat agent ones.
        if not final_limitations:
            final_limitations = "\n".join(x for x in agent_limitations if x)
    else:
        final_limitations = agent_limitations[0] if agent_limitations else ""

    return {
        "limitations": final_limitations,
        "agent_limitations": agent_limitations,
        "agent_full_reviews": agent_full_reviews,
        "meta_review": meta_review,
        "reflection_traces": reflection_traces,
    } 