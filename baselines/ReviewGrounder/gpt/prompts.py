"""
prompts.py
----------

Prompt templates for the limitation-generation pipeline.

Adapted from ReviewGrounder (Anonymous, ACL 2026), the multi-agent
rubric-guided / tool-integrated review framework. We narrow the scope to
*limitations only* and drop the rubric/scoring components.

Pipeline mapping to ReviewGrounder agents:
    DRAFTER_PROMPT          <- paper_reviewer (drafter)
    INSIGHT_MINER_PROMPT    <- paper_insight_miner   (method/contribution)
    RESULTS_ANALYZER_PROMPT <- paper_results_analyzer (experiments/results)
    RW_ANALYZER_PROMPT      <- related_work_searcher (related work positioning)
    REFINER_PROMPT          <- review_refiner

Every prompt asks for **JSON-only** output so we can parse and persist the
results into CSV columns as structured data.
"""

# ---------------------------------------------------------------------------
# Shared system message used by every agent. Keeps a consistent reviewer
# persona across the pipeline.
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are an expert academic reviewer with deep knowledge across "
    "machine learning, NLP, computer vision, and adjacent fields. "
    "You write evidence-grounded, substantive critiques of research papers. "
    "When asked for JSON, you respond with valid JSON only -- no prose, "
    "no markdown fences, no explanations."
)

# ---------------------------------------------------------------------------
# 1) DRAFTER -- initial limitations from the paper text alone, no tools.
# ---------------------------------------------------------------------------
DRAFTER_PROMPT = """You are reviewing a research paper. Read the paper content below and identify its **most plausible limitations**.

Consider all relevant axes: methodology, scope, assumptions, evaluation,
generalization, data quality, computational cost, ethical concerns, and
reproducibility.

Paper content:
{paper_text}

Provide a focused initial draft of limitations. Each limitation should be
specific, technically grounded in the paper, and actionable.

You MUST respond with valid JSON only. Use this exact format:
{{
  "limitations": [
    {{
      "category": "<one of: method, experiments, scope, data, evaluation, generalization, reproducibility, ethics, computational, theoretical>",
      "description": "<the limitation, 1-3 sentences, specific>",
      "rationale": "<brief explanation of why this is a limitation>"
    }}
  ]
}}

Aim for 5-8 high-quality limitations. Return JSON only, no extra text.
"""

# ---------------------------------------------------------------------------
# 2) INSIGHT MINER -- method/contribution-grounded limitations.
#    Mirrors ReviewGrounder's paper_insight_miner: strict scope, paper as
#    source of truth, anchor evidence to specific paper regions.
# ---------------------------------------------------------------------------
INSIGHT_MINER_PROMPT = """You will refine the *method and contribution* limitations of a candidate review using the paper as the source of truth.

SCOPE (strict):
- ONLY consider: core contributions, technical approach, model/algorithm
  design, mathematical formulation, assumptions, optimization/training,
  implementation details, and limitations of the *method itself*.
- DO NOT comment on experimental results or empirical claims (handled by
  another agent).
- DO NOT do external related-work positioning (handled by another agent).

Paper content:
{paper_text}

Candidate limitations draft:
{candidate_limitations}

Tasks:
1) Identify methodological/contribution limitations that are *grounded in the
   paper text*. For each, anchor evidence (section name/number, equation,
   algorithm step, or short snippet).
2) Flag any draft limitations that are unsupported or contradicted by the
   paper.
3) Suggest method-related limitations that the draft missed.

You MUST respond with valid JSON only. Use this exact format:
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

Return JSON only, no extra text.
"""

# ---------------------------------------------------------------------------
# 3) RESULTS ANALYZER -- experiment/results-grounded limitations.
# ---------------------------------------------------------------------------
RESULTS_ANALYZER_PROMPT = """You will refine the *experimental and results-related* limitations of a candidate review using the paper as the source of truth.

SCOPE (strict):
- ONLY consider: experimental setup, datasets, metrics, baselines, ablations,
  statistical significance, error analysis, and limitations of the empirical
  evaluation.
- DO NOT comment on the method itself (handled by another agent).
- DO NOT comment on related-work positioning (handled by another agent).

Paper content:
{paper_text}

Candidate limitations draft:
{candidate_limitations}

Tasks:
1) Identify experimental/results limitations grounded in the paper. For each,
   anchor evidence (table/figure number, section, dataset, metric, or short
   snippet).
2) Flag any draft limitations that mismatch the experimental setup as
   described.
3) Suggest experiment-related limitations that the draft missed (e.g. missing
   ablation, weak baseline, narrow benchmark, no significance test).

You MUST respond with valid JSON only. Use this exact format:
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

Return JSON only, no extra text.
"""

# ---------------------------------------------------------------------------
# 4) RELATED WORK ANALYZER -- limitations grounded in cited / retrieved work.
#    If both citation columns are empty, the orchestrator skips this agent;
#    here we still tell the model to be conservative when context is sparse.
# ---------------------------------------------------------------------------
RW_ANALYZER_PROMPT = """You will identify *related-work-grounded* limitations: limitations that emerge from comparing the paper with prior / cited work.

SCOPE (strict):
- ONLY consider: insufficient comparison with stronger baselines, missed
  prior approaches, mis-characterization of related work, novelty over-claim
  relative to existing methods, or claims contradicted by prior work.
- DO NOT consider purely internal method or experimental issues unrelated to
  comparison.

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
2) For each, cite which related work supports the limitation (quote a short
   anchor or name the source).
3) Be conservative -- only flag a limitation if the related-work context
   *clearly* supports it. Do not invent prior work that isn't in the context.
4) If the related-work context is empty, sparse, or insufficient to support
   any specific claim, return an empty list.

You MUST respond with valid JSON only. Use this exact format:
{{
  "related_work_limitations": [
    {{
      "description": "<specific related-work-grounded limitation>",
      "evidence": "<which related-work item / snippet supports this>",
      "comparison_type": "<one of: missed_baseline, missed_method, novelty_overclaim, miscompared, contradicted_by_priorwork>"
    }}
  ]
}}

Return JSON only, no extra text.
"""

# ---------------------------------------------------------------------------
# 5) REFINER -- consolidates the draft + 3 grounding analyses into a final
#    deduplicated list of evidence-grounded limitations. Mirrors
#    ReviewGrounder's review_refiner with placeholder substitution semantics.
# ---------------------------------------------------------------------------
REFINER_PROMPT = """You are synthesizing a final, evidence-grounded list of LIMITATIONS for a research paper, using:

1) The paper text
2) An initial limitations draft
3) Method-grounded analysis (from the insight miner)
4) Results-grounded analysis (from the results analyzer)
5) Related-work-grounded analysis (may be empty if no citations were available)

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

Method-grounded analysis JSON:
{insight_miner_json}

Results-grounded analysis JSON:
{results_analyzer_json}

Related-work-grounded analysis JSON:
{related_work_json}

You MUST respond with valid JSON only. Use this exact format:
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

Return JSON only, no extra text.
"""