"""
reviewer.py
-----------
The actual "reviewer" that takes a paper text and produces a list of
weaknesses / limitations.

Faithful port of `perform_review` from
    https:/github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py
We keep all three pillars of the AI-Scientist reviewer:

    1. Initial review using the NeurIPS-style review form.
    2. Self-reflection: re-read & refine, repeated `num_reflections - 1` times.
    3. Response ensembling: optionally sample `num_reviews_ensemble` reviewers
       in parallel and meta-review their JSONs.

Two output modes are supported:

    * mode="json"  -> AI-Scientist's full NeurIPS JSON review (recommended for
                      "we used this paper as baseline" claims).  We extract
                      the configured field (Limitations, Weaknesses, or both)
                      from the JSON.
    * mode="text"  -> the model returns a plain bulleted list of
                      weaknesses/limitations.  Useful for smaller open-weight
                      LLMs (Llama-3-8B, Mistral-7B, Qwen-2.5-3B) that
                      occasionally produce malformed JSON.

Sequential note: with local HF models we run reviewers SEQUENTIALLY
(one model per GPU); with OpenAI we still issue one call per reviewer.  This
matches AI-Scientist's behaviour when batching is unavailable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from llm_client import LLMClient
from prompts import (
    get_meta_review_prompt,
    get_text_meta_review_prompt,
    meta_reviewer_system_prompt,
    neurips_form,
    reviewer_reflection_prompt,
    reviewer_system_prompt_neg,
    reviewer_system_prompt_pos,
    template_instructions,
    text_reflection_prompt,
    text_template_instructions,
)
from utils import (
    extract_json_between_markers,
    extract_limitation_field,
    parse_bulleted_list,
)

# ---------------------------------------------------------------------------
# Prompt builders -----------------------------------------------------------
# ---------------------------------------------------------------------------
def _build_initial_prompt(paper_text: str, citation_block: str, mode: str) -> str:
    """
    Compose the very first user message for the reviewer.

    JSON mode mirrors AI-Scientist exactly: NeurIPS form + JSON template
    instructions + the paper.  Text mode keeps the NeurIPS framing for
    methodology fidelity but swaps the JSON instructions for a bulleted-list
    instruction (better for smaller LLMs).
    """
    parts: List[str] = [neurips_form]
    parts.append(template_instructions if mode == "json" else text_template_instructions)
    parts.append("\nHere is the paper you are asked to review:\n```\n")
    if citation_block:
        parts.append(citation_block + "\n\n")
    parts.append(paper_text.strip())
    parts.append("\n```")
    return "".join(parts)

# ---------------------------------------------------------------------------
# Single reviewer pass with reflection --------------------------------------
# ---------------------------------------------------------------------------
def _run_single_reviewer(
    client: LLMClient,
    paper_text: str,
    citation_block: str,
    system_prompt: str,
    num_reflections: int,
    temperature: float,
    max_tokens: int,
    mode: str,
    limitation_field: str,
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    """
    Run one independent reviewer (initial review + N-1 reflection rounds).

    Returns:
        review_obj   : the parsed review (dict for JSON mode; for text mode
                       it's a small dict {"raw": ..., "bullets": ...}).
        agent_output : the per-agent string we surface in the dataframe
                       (extracted Limitations/Weaknesses for JSON mode; the
                       cleaned bulleted list for text mode).
        per_round    : per-round trace for debugging.
    """
    initial_prompt = _build_initial_prompt(paper_text, citation_block, mode)
    raw, history = client.get_response_from_llm(
        prompt=initial_prompt,
        system_message=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Parse current attempt --------------------------------------------------
    review_obj: Dict[str, Any]
    if mode == "json":
        parsed = extract_json_between_markers(raw)
        review_obj = parsed if isinstance(parsed, dict) else {}
    else:
        review_obj = {"raw": raw, "bullets": parse_bulleted_list(raw)}
    per_round: List[Dict[str, Any]] = [{"round": 0, "raw": raw, "parsed": review_obj}]

    # Reflection loop --------------------------------------------------------
    reflect_template = reviewer_reflection_prompt if mode == "json" else text_reflection_prompt
    for j in range(1, num_reflections):
        reflect_prompt = reflect_template.format(
            current_round=j + 1, num_reflections=num_reflections
        )
        raw, history = client.get_response_from_llm(
            prompt=reflect_prompt,
            system_message=system_prompt,
            msg_history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if mode == "json":
            parsed = extract_json_between_markers(raw)
            if isinstance(parsed, dict) and parsed:
                review_obj = parsed
        else:
            review_obj = {"raw": raw, "bullets": parse_bulleted_list(raw)}
        per_round.append({"round": j, "raw": raw, "parsed": review_obj})
        if "I am done" in raw:
            break

    # Surface field ----------------------------------------------------------
    if mode == "json":
        agent_output = extract_limitation_field(review_obj, limitation_field)
        # Fallback: if JSON parsing produced no field text, use the raw bullets.
        if not agent_output:
            agent_output = parse_bulleted_list(raw)
    else:
        agent_output = review_obj.get("bullets", "") or raw.strip()

    return review_obj, agent_output, per_round

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
    mode: str = "json",
    limitation_field: str = "Limitations",
) -> Dict[str, Any]:
    """
    Generate weaknesses/limitations for one paper.

    Returns a dict with:
      - "limitations":          str   final, post-meta output
      - "agent_limitations":    list  per-reviewer outputs
      - "agent_full_reviews":   list  per-reviewer parsed objects
      - "meta_review":          dict  meta/area-chair object (or {} if N=1)
      - "reflection_traces":    list  per-round raw text + parsed object
    """
    if mode not in ("json", "text"):
        raise ValueError("mode must be 'json' or 'text'")

    # Alternate strict/lenient personalities, like AI-Scientist.
    personality_prompts = [reviewer_system_prompt_neg, reviewer_system_prompt_pos]

    agent_full_reviews: List[Dict[str, Any]] = []
    agent_limitations: List[str] = []
    reflection_traces: List[List[Dict[str, Any]]] = []

    for i in range(num_reviews_ensemble):
        sys_prompt = personality_prompts[i % len(personality_prompts)]
        review_obj, agent_output, trace = _run_single_reviewer(
            client=client,
            paper_text=paper_text,
            citation_block=citation_block,
            system_prompt=sys_prompt,
            num_reflections=num_reflections,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode,
            limitation_field=limitation_field,
        )
        agent_full_reviews.append(review_obj)
        agent_limitations.append(agent_output)
        reflection_traces.append(trace)

    # ------------------------------------------------------------------
    # Meta-review (only when ensembling more than one reviewer).
    # ------------------------------------------------------------------
    meta_review: Dict[str, Any] = {}
    final_limitations: str
    if num_reviews_ensemble > 1:
        sys = meta_reviewer_system_prompt.format(reviewer_count=num_reviews_ensemble)
        if mode == "json":
            review_strings = [json.dumps(r, indent=2, ensure_ascii=False) for r in agent_full_reviews]
            meta_prompt = get_meta_review_prompt(review_strings)
            raw_meta, _ = client.get_response_from_llm(
                prompt=meta_prompt,
                system_message=sys,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed_meta = extract_json_between_markers(raw_meta)
            meta_review = parsed_meta if isinstance(parsed_meta, dict) else {}
            final_limitations = extract_limitation_field(meta_review, limitation_field)
            if not final_limitations:
                final_limitations = "\n".join(x for x in agent_limitations if x)
        else:
            meta_prompt = get_text_meta_review_prompt(agent_limitations)
            raw_meta, _ = client.get_response_from_llm(
                prompt=meta_prompt,
                system_message=sys,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            final_limitations = parse_bulleted_list(raw_meta)
            meta_review = {"raw": raw_meta, "bullets": final_limitations}
    else:
        final_limitations = agent_limitations[0] if agent_limitations else ""

    return {
        "limitations": final_limitations,
        "agent_limitations": agent_limitations,
        "agent_full_reviews": agent_full_reviews,
        "meta_review": meta_review,
        "reflection_traces": reflection_traces,
    } 