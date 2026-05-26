"""
prompts.py
----------

Prompt templates for the limitation-generation pipeline.

Adapted from ReviewGrounder (Anonymous, ACL 2026), narrowed to limitations.

Two output formats supported:

  - "json": strict JSON. Reliable for capable models (gpt-4o-mini, gpt-4o, ...).
  - "text": structured plain text using `=== LIMITATION N ===` headers and
            `Field: value` lines. Far more reliable for smaller open-source
            instruct models (Llama-3-8B, Mistral-7B, Qwen-2.5-3B).

The downstream parser converts both formats into the same dict schema, so
agents.py and main.py don't care which format was used.

To switch globally, pass `--output-format json|text` to main.py. The default
is `text` for `--backend hf` and `json` for `--backend openai`.
"""

# ---------------------------------------------------------------------------
# Shared system message used by every agent.
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are an expert academic reviewer with deep knowledge across "
    "machine learning, NLP, computer vision, and adjacent fields. "
    "You write evidence-grounded, substantive critiques of research papers. "
    "Follow output formatting instructions exactly."
)

# ===========================================================================
# Agent prompt BODIES (format-agnostic).
# Each body uses Python `.format(...)` placeholders that the orchestrator
# fills in. The format-specific output instructions are appended later by
# `build_prompt()` below.
# ===========================================================================

DRAFTER_BODY = """You are reviewing a research paper. Read the paper content below and identify its **most plausible limitations**.

Consider all relevant axes: methodology, scope, assumptions, evaluation,
generalization, data quality, computational cost, ethical concerns, and
reproducibility.

Paper content:
{paper_text}

Provide a focused initial draft of limitations. Each limitation should be
specific, technically grounded in the paper, and actionable. Aim for 5-8
high-quality limitations.
"""

INSIGHT_MINER_BODY = """You will refine the *method and contribution* limitations of a candidate review using the paper as the source of truth.

SCOPE (strict):
- ONLY consider: core contributions, technical approach, model/algorithm
  design, mathematical formulation, assumptions, optimization/training,
  implementation details, and limitations of the *method itself*.
- DO NOT comment on experimental results or empirical claims.
- DO NOT do external related-work positioning.

Paper content:
{paper_text}

Candidate limitations draft:
{candidate_limitations}

Tasks:
1) Identify methodological/contribution limitations that are *grounded in the
   paper text*. For each, anchor evidence (section name/number, equation,
   algorithm step, or short snippet).
2) Be honest if a candidate-draft limitation is unsupported or contradicted
   by the paper -- in that case, set its status accordingly.
3) Add method-related limitations that the draft missed.
"""

RESULTS_ANALYZER_BODY = """You will refine the *experimental and results* limitations of a candidate review using the paper as the source of truth.

SCOPE (strict):
- ONLY consider: experimental setup, datasets, metrics, baselines, ablations,
  statistical significance, error analysis, and limitations of the empirical
  evaluation.
- DO NOT comment on the method itself.
- DO NOT comment on related-work positioning.

Paper content:
{paper_text}

Candidate limitations draft:
{candidate_limitations}

Tasks:
1) Identify experimental/results limitations grounded in the paper. For each,
   anchor evidence (table/figure number, section, dataset, metric, or short
   snippet).
2) Flag any draft limitations that mismatch the experimental setup.
3) Add experiment-related limitations that the draft missed (e.g. missing
   ablation, weak baseline, narrow benchmark, no significance test).
"""

RW_ANALYZER_BODY = """You will identify *related-work-grounded* limitations: limitations that emerge from comparing the paper with prior / cited work.

SCOPE (strict):
- ONLY consider: insufficient comparison with stronger baselines, missed
  prior approaches, mis-characterization of related work, novelty over-claim,
  or claims contradicted by prior work.
- DO NOT consider purely internal method or experimental issues.

Paper content:
{paper_text}

Candidate limitations draft:
{candidate_limitations}

Cited / related work context (extracted from the paper itself):
{cited_in_text}

Cited / related work context (retrieved abstracts from OpenAlex):
{cited_in_ret}

Tasks:
1) Using the related-work context above, identify limitations of the paper
   that arise from comparison with prior work.
2) For each, cite which related-work item supports the limitation.
3) Be conservative -- only flag a limitation if the related-work context
   *clearly* supports it. Do not invent prior work that isn't in the context.
4) If the related-work context is empty or insufficient to support any
   specific claim, output zero limitations.
"""

REFINER_BODY = """You are synthesizing a final, evidence-grounded list of LIMITATIONS for a research paper, using:

1) The paper text
2) An initial limitations draft
3) Method-grounded analysis (from the insight miner)
4) Results-grounded analysis (from the results analyzer)
5) Related-work-grounded analysis (may be empty)

Goals:
- Consolidate and deduplicate all candidate limitations.
- Keep ONLY limitations that are grounded in the paper text or supported by
  the related-work analysis.
- Drop unsupported / contradicted draft items.
- Each final limitation should be specific, actionable, and have concrete
  evidence (section/equation/table/figure or related-work source).
- Categorize each limitation.
- Aim for 5-10 high-quality limitations.

Paper text:
{paper_text}

Initial limitations draft:
{candidate_limitations}

Method-grounded analysis:
{insight_miner_json}

Results-grounded analysis:
{results_analyzer_json}

Related-work-grounded analysis:
{related_work_json}
"""

# ===========================================================================
# JSON-format output instructions.
# Used when output_format == "json".
# ===========================================================================

_JSON_DRAFTER = """You MUST respond with valid JSON only. Use this exact format:
{{
  "limitations": [
    {{
      "category": "<one of: method, experiments, scope, data, evaluation, generalization, reproducibility, ethics, computational, theoretical>",
      "description": "<the limitation, 1-3 sentences, specific>",
      "rationale": "<brief explanation of why this is a limitation>"
    }}
  ]
}}
Return JSON only, no extra text."""

_JSON_INSIGHT = """You MUST respond with valid JSON only. Use this exact format:
{{
  "method_limitations": [
    {{
      "description": "<specific method/contribution limitation>",
      "evidence": "<section / equation / snippet anchor from the paper>",
      "draft_status": "<one of: confirmed, missing_in_draft, contradicted_by_paper, unsupported_by_paper>"
    }}
  ],
  "draft_issues": [
    {{
      "draft_item": "<which draft limitation>",
      "issue": "<why it is wrong / unsupported>"
    }}
  ]
}}
Return JSON only, no extra text."""

_JSON_RESULTS = """You MUST respond with valid JSON only. Use this exact format:
{{
  "results_limitations": [
    {{
      "description": "<specific experimental/results limitation>",
      "evidence": "<table/figure/section anchor from the paper>",
      "draft_status": "<one of: confirmed, missing_in_draft, contradicted_by_paper, unsupported_by_paper>"
    }}
  ],
  "draft_issues": [
    {{
      "draft_item": "<which draft limitation>",
      "issue": "<why it is wrong / unsupported>"
    }}
  ]
}}
Return JSON only, no extra text."""

_JSON_RW = """You MUST respond with valid JSON only. Use this exact format:
{{
  "related_work_limitations": [
    {{
      "description": "<specific related-work-grounded limitation>",
      "evidence": "<which related-work item / snippet supports this>",
      "comparison_type": "<one of: missed_baseline, missed_method, novelty_overclaim, miscompared, contradicted_by_priorwork>"
    }}
  ]
}}
Return JSON only, no extra text."""

_JSON_REFINER = """You MUST respond with valid JSON only. Use this exact format:
{{
  "final_limitations": [
    {{
      "category": "<one of: method, experiments, scope, data, evaluation, generalization, reproducibility, ethics, computational, theoretical, related_work>",
      "description": "<final limitation, 1-3 sentences, specific>",
      "evidence": "<paper or related-work anchor>",
      "severity": "<one of: minor, moderate, major>"
    }}
  ],
  "summary": "<2-3 sentence summary of the most important limitations>"
}}
Return JSON only, no extra text."""

JSON_FORMAT = {
    "drafter": _JSON_DRAFTER,
    "insight_miner": _JSON_INSIGHT,
    "results_analyzer": _JSON_RESULTS,
    "rw_analyzer": _JSON_RW,
    "refiner": _JSON_REFINER,
}

# ===========================================================================
# Plain-text output instructions.
# Used when output_format == "text" (recommended for smaller HF models).
#
# Format conventions:
#   - Each item starts with `=== LIMITATION N ===` (or for refiner only,
#     the body may also include a single `SUMMARY: ...` line at the top).
#   - Within each item, fields are written as `FieldName: value`.
#   - Multi-line field values continue on subsequent lines until the next
#     `Field:` header or `=== LIMITATION` header.
#   - No JSON, no markdown fences.
# ===========================================================================

_TEXT_DRAFTER = """Output your answer in this EXACT plain-text format. Do NOT use JSON or markdown code blocks.

Use this template, repeating the LIMITATION block 5-8 times:

=== LIMITATION 1 ===
Category: <one of: method, experiments, scope, data, evaluation, generalization, reproducibility, ethics, computational, theoretical>
Description: <the limitation, 1-3 sentences, specific>
Rationale: <brief explanation of why this is a limitation>

=== LIMITATION 2 ===
Category: ...
Description: ...
Rationale: ...

Use this exact format. Do not add extra prose before, between, or after the LIMITATION blocks."""

_TEXT_INSIGHT = """Output your answer in this EXACT plain-text format. Do NOT use JSON or markdown code blocks.

For each method/contribution limitation you identify, write:

=== LIMITATION 1 ===
Description: <specific method/contribution limitation>
Evidence: <section / equation / snippet anchor from the paper>
Status: <one of: confirmed, missing_in_draft, contradicted_by_paper, unsupported_by_paper>

=== LIMITATION 2 ===
Description: ...
Evidence: ...
Status: ...

Use this exact format. Do not add extra prose before, between, or after the LIMITATION blocks. If you have no method-related limitations to report, output exactly: NO LIMITATIONS"""

_TEXT_RESULTS = """Output your answer in this EXACT plain-text format. Do NOT use JSON or markdown code blocks.

For each experimental/results limitation you identify, write:

=== LIMITATION 1 ===
Description: <specific experimental/results limitation>
Evidence: <table / figure / section anchor from the paper>
Status: <one of: confirmed, missing_in_draft, contradicted_by_paper, unsupported_by_paper>

=== LIMITATION 2 ===
Description: ...
Evidence: ...
Status: ...

Use this exact format. Do not add extra prose before, between, or after the LIMITATION blocks. If you have no results-related limitations to report, output exactly: NO LIMITATIONS"""

_TEXT_RW = """Output your answer in this EXACT plain-text format. Do NOT use JSON or markdown code blocks.

For each related-work-grounded limitation you identify, write:

=== LIMITATION 1 ===
Description: <specific related-work-grounded limitation>
Evidence: <which related-work item / snippet supports this>
Type: <one of: missed_baseline, missed_method, novelty_overclaim, miscompared, contradicted_by_priorwork>

=== LIMITATION 2 ===
Description: ...
Evidence: ...
Type: ...

Use this exact format. If the related-work context is empty or insufficient, output exactly: NO LIMITATIONS"""

_TEXT_REFINER = """Output your answer in this EXACT plain-text format. Do NOT use JSON or markdown code blocks.

Start with a SUMMARY line, then 5-10 LIMITATION blocks:

SUMMARY: <2-3 sentence summary of the most important limitations>

=== LIMITATION 1 ===
Category: <one of: method, experiments, scope, data, evaluation, generalization, reproducibility, ethics, computational, theoretical, related_work>
Description: <final limitation, 1-3 sentences>
Evidence: <paper or related-work anchor>
Severity: <minor | moderate | major>

=== LIMITATION 2 ===
Category: ...
Description: ...
Evidence: ...
Severity: ...

Use this exact format. Do not add extra prose before, between, or after the SUMMARY/LIMITATION blocks."""

TEXT_FORMAT = {
    "drafter": _TEXT_DRAFTER,
    "insight_miner": _TEXT_INSIGHT,
    "results_analyzer": _TEXT_RESULTS,
    "rw_analyzer": _TEXT_RW,
    "refiner": _TEXT_REFINER,
}

# ===========================================================================
# Body lookup + builder
# ===========================================================================

BODIES = {
    "drafter": DRAFTER_BODY,
    "insight_miner": INSIGHT_MINER_BODY,
    "results_analyzer": RESULTS_ANALYZER_BODY,
    "rw_analyzer": RW_ANALYZER_BODY,
    "refiner": REFINER_BODY,
}

def build_prompt(agent_name: str, output_format: str, **kwargs) -> str:
    """Compose `<body>\\n\\n<format-instructions>` for a given agent.

    Args:
        agent_name: one of BODIES keys.
        output_format: "json" or "text".
        **kwargs: format() placeholders (paper_text, candidate_limitations, ...).
    """
    if agent_name not in BODIES:
        raise KeyError(f"Unknown agent: {agent_name}")
    body = BODIES[agent_name].format(**kwargs)
    if output_format == "json":
        fmt = JSON_FORMAT[agent_name]
    elif output_format == "text":
        fmt = TEXT_FORMAT[agent_name]
    else:
        raise ValueError(
            f"Unknown output_format: {output_format!r} (expected 'json' or 'text')"
        )
    return body + "\n\n" + fmt 