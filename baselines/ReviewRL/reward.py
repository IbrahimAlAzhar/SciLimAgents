# =============================================================================
# reward.py
# -----------------------------------------------------------------------------
# Format / structural reward function for the limitation-generation baseline.
#
# This module is a slimmed-down port of:
#     marti/verifiers/review_rl/review_eval.py  ->  EnhancedReviewRewardFunction
# from the official ReviewRL codebase
# (https://github.com/TsinghuaC3I/MARTI/tree/main/examples/reviewrl).
#
# Compared to the upstream class we drop:
#   * the rating-consistency reward    (no rating in our task)
#   * the GenRM-based judge reward     (out of scope; we only generate limitations)
#   * the depth/key-elements reward    (depended on a paper_data dict we don't have)
# and we keep:
#   * the "<think> ... </think>" check
#   * the "## Limitations" section check
#   * an additional "bullet-format" check that rewards the [Major]/[Minor] tags
#     used in our adapted prompt (see prompts.LIMITATION_GENERATION_PROMPT).
#
# The reward is INFORMATIONAL ONLY in this baseline (we don't run RL); it is
# stored alongside each generated row so it can be used for post-hoc filtering
# or, optionally, for an SFT/RL extension.
# =============================================================================

from __future__ import annotations

import re
from typing import Dict

# Regexes copied/adapted from review_eval.py
_THINK_REGEX = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")
# Note: we deliberately use `\Z` (end of string) instead of `$` here.
# The upstream ReviewRL regex uses `$`, which in MULTILINE mode matches end-of-line
# and therefore stops the section at the first newline. Since our prompt asks
# for "## Limitations" as the FINAL section, we want the section body to extend
# all the way to the end of the response (or to the next "## ..." header).
_LIM_SECTION_REGEX = re.compile(
    r"(?ms)^#{2,3}\s*(Limitations|Weaknesses)\s*\n(.*?)(?=\n#{2,}\s*\S+|\Z)"
)
_BULLET_TAG_REGEX = re.compile(r"^\s*[-*]\s*\[(Major|Minor)\]", re.IGNORECASE | re.MULTILINE)

def format_reward(response: str) -> Dict[str, float]:
    """
    Score a single model output for structural/format quality.

    Returns a dict with the same shape as the upstream EnhancedReviewRewardFunction:
        {
            'total_reward':   float in [0, 1],
            'format_reward':  float in [-1, 0]   (penalty if pieces missing),
            'section_reward': float in [0, 1],   (Limitations section present),
            'bullet_reward':  float in [0, 1],   (proper [Major]/[Minor] bullets),
        }

    The "format_reward" penalty mirrors the original ReviewRL formula:
        missing_penalty = -0.5 * missing_count   (but we have only 2 mandatory parts)
    """
    # 1. <think> block check
    think_match = bool(_THINK_REGEX.search(response))

    # 2. ## Limitations / ## Weaknesses section check
    section_match = _LIM_SECTION_REGEX.search(response)
    has_section = bool(section_match)

    # 3. Bullet [Major]/[Minor] format inside the section
    if has_section:
        section_text = section_match.group(2)
        bullets = _BULLET_TAG_REGEX.findall(section_text)
        n_bullets = len(bullets)
    else:
        n_bullets = 0
    bullet_reward = min(1.0, n_bullets / 4.0)  # full credit at 4+ tagged bullets

    # 4. Aggregate (same -0.25 per missing part as the upstream code)
    missing = 0
    if not think_match:
        missing += 1
    if not has_section:
        missing += 1
    fmt_penalty = -0.25 * missing  # in [-0.5, 0]

    # 5. Total reward in [0, 1]: section presence + bullet richness + (no penalty)
    total = 0.5 * float(has_section) + 0.5 * bullet_reward + fmt_penalty
    total = max(0.0, min(1.0, total))

    return {
        "total_reward": total,
        "format_reward": fmt_penalty,
        "section_reward": float(has_section),
        "bullet_reward": bullet_reward,
        "has_think": float(think_match),
        "n_bullets": float(n_bullets),
    }

def extract_limitations_section(response: str) -> str:
    """
    Return only the post-thinking '## Limitations' section.
    If parsing fails, fall back to the substring after '</think>' or the
    raw response itself - this guarantees the function ALWAYS returns something
    usable for downstream evaluation pipelines.
    """
    m = _LIM_SECTION_REGEX.search(response)
    if m:
        header = m.group(1)  # "Limitations" or "Weaknesses"
        body = m.group(2).strip()
        return f"## {header}\n{body}"

    # Fallback 1: strip the <think>...</think> block and return the rest.
    if "</think>" in response:
        tail = response.split("</think>", 1)[1].strip()
        if tail:
            return tail

    # Fallback 2: return the whole response.
    return response.strip()