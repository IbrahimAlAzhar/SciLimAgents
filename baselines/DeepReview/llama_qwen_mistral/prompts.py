"""
prompts.py
==========
All prompts used in the DeepReview-baseline limitation-generation pipeline.

These prompts are written to faithfully mirror the four phases described in
the DeepReview paper (Zhu et al., ACL 2025):

    Stage 1: Novelty Verification (z1)        -> NOVELTY_QUESTION_PROMPT,
                                                 NOVELTY_ANALYSIS_PROMPT
    Stage 2: Multi-dimensional Review (z2)    -> REVIEWER_SYSTEM_PROMPT,
                                                 REVIEWER_USER_PROMPT
    Stage 3: Reliability Verification (z3)    -> VERIFICATION_PROMPT
    Final:   Meta-Review limitation synthesis -> META_LIMITATIONS_PROMPT

All four mirror DeepReview's `\\boxed_question{}`, `\\boxed_analysis{}`,
`\\boxed_simreviewers{}`, `\\boxed_verification{}`, `\\boxed_review{}` blocks
but are STRIPPED to keep ONLY the weakness / limitation chain — strengths,
ratings, soundness etc. are intentionally removed because the user does not
need them.
"""

# ---------------------------------------------------------------------------
# Stage 1 - Novelty Verification (question generation + literature analysis)
# ---------------------------------------------------------------------------
# DeepReview generates 3 questions about (a) research gaps, (b) innovative
# directions, (c) methodological breakthroughs.  We keep that 3-question
# format because it is what the trained DeepReviewer expects and what the
# paper's Stage-1 description says.
NOVELTY_QUESTION_PROMPT = """\
You are an expert academic reviewer following the DeepReview "Novelty
Verification" stage. Read the paper below and propose exactly THREE concise
research questions that would help you decide whether the paper genuinely
advances the state of the art.

Question 1 must focus on the *research gap* the paper claims to fill.
Question 2 must focus on the *innovative direction* / new contribution.
Question 3 must focus on the *methodological breakthrough* (or lack of one).

Output strictly in this format and nothing else:

<boxed_questions>
1. ...
2. ...
3. ...
</boxed_questions>

Paper:
\"\"\"
{paper}
\"\"\"
"""

# After we have the three questions and the (optional) cited-literature
# context, we ask the model to produce a structured novelty analysis that
# explicitly enumerates *gaps / weaknesses* relative to prior work.  This
# is the "z1" reasoning step the paper feeds into Stage 2.
NOVELTY_ANALYSIS_PROMPT = """\
You are an expert academic reviewer.  You have just generated three research
questions about a paper and we have retrieved related-work passages from the
literature.  Your job is to analyse the paper's novelty *against that
literature* and surface any limitations that arise from it.

Paper (truncated):
\"\"\"
{paper}
\"\"\"

Research questions:
{questions}

Retrieved related-work context (may be empty / partial):
\"\"\"
{citations}
\"\"\"

Write a focused novelty analysis (≤ 250 words) covering:
  * Does the paper actually fill the gap it claims?
  * What prior work is missing or under-cited?
  * What aspects of the methodology or evaluation feel under-novel
    compared with the retrieved literature?
  * Any over-claiming or scope overlap with prior work.

Output:

<boxed_analysis>
... your analysis here ...
</boxed_analysis>
"""

# ---------------------------------------------------------------------------
# Stage 2 - Multi-dimensional Review (R simulated reviewers, weaknesses only)
# ---------------------------------------------------------------------------
# DeepReview lets the trained 14B model emit `\boxed_simreviewers{...}` with
# `Reviewer 1 ... Reviewer N` sub-sections.  We do the same here, but ONLY
# ask for the Weaknesses field of each reviewer because we are after
# limitations.

# Each reviewer is given a different "expertise focus" so we get
# multi-dimensional coverage instead of N identical critiques.
REVIEWER_FOCUS = [
    "methodology and theoretical soundness",
    "experimental design, datasets, baselines and statistical rigor",
    "novelty, scope of contribution and positioning vs. prior work",
    "presentation, clarity, reproducibility and ethical considerations",
    "scalability, generalization and real-world applicability",
    "limitations the authors themselves acknowledge but under-discuss",
]

def reviewer_focus(idx: int) -> str:
    """Return a focus dimension; cycles if idx > len(REVIEWER_FOCUS)."""
    return REVIEWER_FOCUS[idx % len(REVIEWER_FOCUS)]

REVIEWER_SYSTEM_PROMPT = """\
You are Reviewer {reviewer_id} in a DeepReview-style multi-reviewer
simulation.  Your reviewing focus is: **{focus}**.

You must list the paper's weaknesses ONLY (no strengths, no ratings, no
suggestions, no questions).  Be specific, evidence-based, and try to surface
issues that another reviewer focused on a different dimension would miss.
Each weakness must be a concrete, falsifiable statement that points to a
real problem in the paper.
"""

REVIEWER_USER_PROMPT = """\
Below is the paper to review, followed by the novelty analysis produced in
Stage 1 (may be empty if Stage 1 was skipped).

Paper:
\"\"\"
{paper}
\"\"\"

Novelty analysis (Stage 1):
\"\"\"
{novelty_analysis}
\"\"\"

Produce 3-6 weaknesses, ordered from most to least important.

Output strictly in this format and nothing else:

<boxed_weaknesses>
1. <weakness statement>
2. <weakness statement>
3. <weakness statement>
...
</boxed_weaknesses>
"""

# ---------------------------------------------------------------------------
# Stage 3 - Reliability Verification
# ---------------------------------------------------------------------------
# DeepReview verifies every weakness statement by collecting evidence from
# the paper, analysing it, and assigning a confidence level
# ("methodology / experimental / comprehensive" verification chain).
# We compress that into a single per-weakness verification step.
VERIFICATION_PROMPT = """\
You are performing the DeepReview "Reliability Verification" step.  For each
candidate weakness below, decide whether it is genuinely supported by
evidence in the paper.

Paper:
\"\"\"
{paper}
\"\"\"

Candidate weaknesses (collected from {n_reviewers} simulated reviewers):
{weaknesses}

For every weakness, output a verification block in EXACTLY this format:

<boxed_verification>
Weakness <i>: <restate the weakness in one sentence>
Evidence: <quote or paraphrase the relevant passage of the paper, or write
"NO DIRECT EVIDENCE" if you cannot find any>
Analysis: <one sentence explaining whether the evidence supports the weakness>
Confidence: <High | Medium | Low>
Verdict: <KEEP | DROP | REVISE>
</boxed_verification>

Mark a weakness as KEEP only if it is well supported, REVISE if partially
supported (and rewrite it on the "Weakness" line), and DROP if it looks like
a hallucination.  Do NOT add any text outside the verification blocks.
"""

# ---------------------------------------------------------------------------
# Final stage - Meta-Reviewer limitation synthesis
# ---------------------------------------------------------------------------
# DeepReview's final step regenerates a Meta-Review with Summary / Strengths /
# Weaknesses / Suggestions / Soundness / Presentation / Contribution /
# Rating / Decision.  We strip this to ONLY the Weaknesses / Limitations
# block because that is the user's deliverable.
META_LIMITATIONS_PROMPT = """\
You are the DeepReview meta-reviewer. You have:

* a paper,
* a novelty analysis (may be empty),
* the verification report for several reviewer weaknesses.

Your job is to write the FINAL list of LIMITATIONS for this paper, suitable
for the "Limitations" section of an academic peer review.

Rules:
  * Use ONLY weaknesses whose verdict is KEEP or REVISE (apply the revision).
  * Discard duplicates and merge overlapping ones.
  * Order from most to least critical.
  * Each limitation must be a single, self-contained, evidence-grounded
    sentence (≤ 40 words).
  * Output exactly 3-7 limitations — quality over quantity.

Paper (truncated):
\"\"\"
{paper}
\"\"\"

Novelty analysis (may be empty):
\"\"\"
{novelty_analysis}
\"\"\"

Verification report:
\"\"\"
{verification}
\"\"\"

Output strictly in this format and nothing else:

<boxed_limitations>
1. <limitation>
2. <limitation>
3. <limitation>
...
</boxed_limitations>
""" 