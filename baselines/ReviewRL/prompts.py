# =============================================================================
# prompts.py
# -----------------------------------------------------------------------------
# Prompt templates for the ReviewRL-style baseline (LIMITATION GENERATION ONLY).
#
# All templates here are adapted from the prompts published in:
#   Zeng et al., 2025. "ReviewRL: Towards Automated Scientific Review with RL"
#   (EMNLP 2025).  See Tables 6, 7, 8, and 10 of the paper.
#
# ReviewRL produces a full review (Summary / Strengths / Weaknesses / Rating).
# In this baseline we keep only the components needed to generate the
# WEAKNESSES / LIMITATIONS section, since the downstream task we are
# benchmarking against is "limitation generation".
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Query-generation prompt  (Table 6 of the ReviewRL paper).
#    The original ReviewRL pipeline first generates three short questions
#    about the paper (novelty / methodology / prior-work relationship) and
#    then routes them to ArXiv-MCP for retrieval.
#
#    In this baseline the citations are already pre-retrieved (the columns
#    `cited_in_text` and `cited_in_ret` in the user's CSV), so the queries
#    are used only as a *re-ranking / filtering* signal over the cited
#    context. The prompt itself is preserved 1:1 from the paper so this
#    module remains a faithful re-implementation.
# -----------------------------------------------------------------------------
GENERATE_QUERIES_PROMPT = """You are now an academic paper review expert capable of conducting thorough analyses of research papers to provide the most reliable review results. You are now allowed to use the search tool to obtain background information on the paper—please provide three different questions. I will assist you with the search. Please present the three questions in the following format:
1. xxx
2. xxx
3. xxx
Do not include any additional content.

Here is a research paper:
{paper}
"""

# -----------------------------------------------------------------------------
# 2) Retrieval-system prompt  (Table 7 of the ReviewRL paper).
#    Used as a system message when re-ranking / cleaning the retrieved
#    context before feeding it to the policy model.
# -----------------------------------------------------------------------------
RETRIEVAL_SYSTEM_PROMPT = (
    "You are an academic expert who specializes in answering questions by "
    "retrieving information from arXiv."
)

# -----------------------------------------------------------------------------
# 3) Limitation-generation prompt  (adapted from Table 8 of the ReviewRL paper).
#
#    The original prompt asks for: Summary, Strengths, Weaknesses, Rating.
#    For our baseline we drop Summary / Strengths / Rating and keep only the
#    "<think> ... </think>" CoT block and the "## Weaknesses" section
#    (renamed "## Limitations" so downstream parsing matches our schema).
#
#    The {context} block is filled in from `cited_in_text` + `cited_in_ret`
#    (i.e. the same role that ArXiv-MCP retrieved excerpts play in ReviewRL).
# -----------------------------------------------------------------------------
LIMITATION_GENERATION_PROMPT = """You are a senior reviewer for top-tier AI conferences (NeurIPS/ICML/CVPR/ACL). You must be strict and professional enough.

Read the paper carefully:
- Analyze each paragraph of each section critically.
- Identify any logical flaws, technical inconsistencies, missing citations, or unclear explanations.

You may consult the retrieved related-work context (excerpts from cited / similar papers) when judging novelty, methodological choices, and comparison to prior work.

Use <think> </think> tags to document your detailed thought process during the review. Inside the <think> block you may discuss strengths, weaknesses, and overall impressions, but the FINAL output you produce after </think> MUST contain ONLY the limitations section using exactly the format below.

After your thinking, output the limitations of the paper using EXACTLY this format:

## Limitations
- [Major] <one-sentence limitation>
- [Major] <one-sentence limitation>
- [Minor] <one-sentence limitation>
- [Minor] <one-sentence limitation>

Each bullet must be a single self-contained sentence describing a concrete limitation (methodology flaw, experimental gap, presentation problem, missing comparison, scalability issue, etc.). Do NOT include strengths, summary, or a rating. Do NOT add any text after the bulleted list.

[Retrieved Related-Work Context]
{context}

[Paper]
{paper}
"""

# -----------------------------------------------------------------------------
# 4) GenRM (judge) prompt  (Table 10 of the ReviewRL paper).
#    Kept here for completeness so this module can be cited as a faithful
#    re-implementation of the ReviewRL training/eval recipe. We default to
#    NOT calling it during inference, but it is exposed for users who want
#    to reproduce the GenRM-based reward signal during RL training.
# -----------------------------------------------------------------------------
GENRM_PROMPT = """You are an expert academic peer reviewer. You will be shown the abstract/content of a research paper and two peer reviews for that paper. Your task is to determine which peer review is of higher quality based on the following criteria:

1. Factual Accuracy & Soundness
2. Completeness & Coverage
3. Level of Detail & Specificity
4. Comparison with Existing Work
5. Constructiveness
6. Clarity & Organization

Paper Context (Abstract/Content): {paper_context}

Review 1: {review1}

Review 2: {review2}

Which peer review is of higher quality based on the criteria above? Respond with EXACTLY one of these options:
- REVIEW_1_BETTER
- REVIEW_2_BETTER
YOU MUST CHOOSE A BETTER REVIEW. A TIE IS NOT ALLOWED.
"""

# -----------------------------------------------------------------------------
# 5) SFT instruction template (Section 3.3 of the ReviewRL paper).
#    Identical to LIMITATION_GENERATION_PROMPT but used as the *input* side
#    of supervised-fine-tuning examples; the *target* side is the gold
#    limitation paragraph from `input_text_without_lim` augmented with the
#    section header `## Limitations` and bullet formatting.
# -----------------------------------------------------------------------------
SFT_INPUT_TEMPLATE = LIMITATION_GENERATION_PROMPT  # alias for clarity