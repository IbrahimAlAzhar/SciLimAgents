"""
prompts.py
----------
Reviewer prompts mirrored from "The AI Scientist" (Sakana AI, 2024).

Source for the original prompts:
    https:/github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py

We keep the *full* NeurIPS review form so that the limitation field is generated
in exactly the same context that the original paper uses (i.e. the limitation
section is conditioned on having reasoned about Summary / Strengths /
Weaknesses / Originality / Quality / Clarity / Significance / ...).  We only
*extract* the "Limitations" field from the resulting JSON afterwards, because
this work is being used as a baseline strictly for the
"limitation generation from a research paper" task.
"""

# ---------------------------------------------------------------------------
# System prompts ------------------------------------------------------------
# ---------------------------------------------------------------------------
# AI-Scientist uses two opposite-bias system prompts and ensembles them.
# We keep both available so that ensembling can be turned on through argparse.

reviewer_system_prompt_base = (
    "You are an AI researcher who is reviewing a paper that was submitted to a "
    "prestigious ML venue. "
)

reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + "Be critical and cautious in your decision."
)

reviewer_system_prompt_pos = (
    reviewer_system_prompt_base
    + "Be open-minded and willing to accept the paper if it is good."
)

# ---------------------------------------------------------------------------
# NeurIPS-style review form (verbatim from AI-Scientist) --------------------
# ---------------------------------------------------------------------------
neurips_form = (
    """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.

1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited
  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work
  - Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach
2. Questions: Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.
3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If not, please include constructive suggestions for improvement.
   In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.
4. Ethical Concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.
5. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
  4: excellent
  3: good
  2: fair
  1: poor
6. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
  4: excellent
  3: good
  2: fair
  1: poor
7. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
  4: excellent
  3: good
  2: fair
  1: poor
8. Overall: Please provide an "overall score" for this submission. Choices:
  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
  3: Reject: For example, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
  2: Strong Reject: For example, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
  1: Very Strong Reject: For example, a paper with trivial results or unaddressed ethical considerations
9. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
  5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
  4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
  3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

You must make sure that all sections are properly created: abstract, introduction, methods, results, and discussion. Points must be reduced from your scores if any of these are missing.
"""
)

# ---------------------------------------------------------------------------
# Output template instructions (mirrored from AI-Scientist) ------------------
# ---------------------------------------------------------------------------
template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
"Summary", "Strengths", "Weaknesses", "Originality", "Quality", "Clarity", "Significance", "Questions", "Limitations", "Ethical Concerns", "Soundness", "Presentation", "Contribution", "Overall", "Confidence", "Decision".

"Summary", "Strengths", "Weaknesses", "Questions", "Limitations" should be lists of strings.
"Originality", "Quality", "Clarity", "Significance", "Soundness", "Presentation", "Contribution" should be integers from 1 to 4.
"Overall" should be an integer from 1 to 10.
"Confidence" should be an integer from 1 to 5.
"Ethical Concerns" should be a boolean.
"Decision" should be one of "Accept" or "Reject".

This JSON will be automatically parsed, so ensure the format is precise.
"""

# ---------------------------------------------------------------------------
# Reflection prompt (verbatim from AI-Scientist) ----------------------------
# ---------------------------------------------------------------------------
reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
Include any other factors that you think are important in evaluating the paper.
Ensure the review is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your review.
Stick to the spirit of the original review unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
DO NOT INCLUDE "I am done" IF YOU ARE MAKING CHANGES."""

# ---------------------------------------------------------------------------
# Meta-review (ensembling) prompt -------------------------------------------
# ---------------------------------------------------------------------------
# AI-Scientist runs multiple reviewers in parallel and then has a "meta
# reviewer" / area-chair that aggregates their JSONs into one review.

meta_reviewer_system_prompt = (
    "You are an Area Chair at a machine learning conference. "
    "You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers. "
    "Your job is to aggregate the reviews into a single meta-review in the same format. "
    "Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."
)

def get_meta_review_prompt(reviews):
    """
    Build the meta-review prompt by stitching together all of the individual
    reviewer JSONs.  Mirrors AI-Scientist's `get_meta_review` helper.
    """
    parts = [
        "The following are reviews of the paper from {n} reviewers. "
        "Aggregate them into a single meta-review in the SAME JSON format.\n".format(
            n=len(reviews)
        )
    ]
    for i, r in enumerate(reviews):
        parts.append(f"\nReview {i + 1}/{len(reviews)}:\n```json\n{r}\n```\n")
    parts.append("\n" + template_instructions)
    return "".join(parts) 
