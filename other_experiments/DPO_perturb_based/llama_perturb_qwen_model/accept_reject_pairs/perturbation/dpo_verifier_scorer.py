"""
DPO Perturbation Scorer (v3) — Human-in-the-Loop
==================================================
Scores each perturbed column with:
  - Rule-based score (0-1, normalized from multiple heuristics)
  - LLM-based score  (0-1, normalized from 5-dimension GPT-4o-mini judge)
  - Final score       = 0.3 * rule_based + 0.7 * llm_based

Updates each perturbed column cell to a dict:
  {
    "text": original_text,
    "rule_based_score": float,
    "llm_based_score": float,
    "final_score": float,
    "flaw_type": str,
    "rule_detail": {...},
    "llm_detail": {...},
  }

Also creates FLAT score columns (e.g., lim_simple_final_score) for easy
sorting/filtering in Excel.

Human reviews scores → decides which columns to keep per row → builds DPO pairs.
"""

import os
import json
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import Counter
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

# NOTE: Set your API key as environment variable
# export OPENAI_API_KEY='sk-...'
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

INPUT_CSV       = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_filtered__all_pert_llama.csv"        # <-- UPDATE

OUTPUT_CSV      = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_with_verifier_all_pert_llama.csv"         # <-- UPDATE

OUTPUT_PICKLE   = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_with_verifier_all_pert_llama.pkl"         # <-- UPDATE

OUTPUT_STATS    = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_with_verifier_all_pert_llama.json"                     # <-- UPDATE

TEXT_COL         = "input_text_cleaned"
RATING_COL       = "mean_rating"
GROUND_TRUTH_COL = "ground_truth_lim_peer"

REJECTED_COLS = {
    "zs":          {"col": "lim_zs",          "failure_mode": "zero_shot_baseline"},
    "simple":      {"col": "lim_simple",      "failure_mode": "bare_prompt_generic"},
    "neutral":     {"col": "lim_neutral",      "failure_mode": "structured_but_generic"},
    "override":    {"col": "lim_override",     "failure_mode": "bias_manipulation"},
    "vague":       {"col": "lim_vague",        "failure_mode": "lacks_specificity"},
    "hallucinate": {"col": "lim_hallucinate",  "failure_mode": "unfaithful_fabricated"},
    "shallow":     {"col": "lim_shallow",      "failure_mode": "insufficient_coverage"},
    "sycophantic": {"col": "lim_sycophantic",  "failure_mode": "hedged_indirect"},
    "repetitive":  {"col": "lim_repetitive",   "failure_mode": "redundant_padded"},
}

SYSTEM_PROMPT = """You are an expert scientific paper reviewer. Your task is to identify ALL limitations
of the given paper. Be thorough, specific, and evidence-grounded.

For each limitation, provide:
- A clear category (e.g., Novelty, Methodology, Experiments, Generalization, Clarity, Data/Ethics)
- A specific description of the limitation
- Evidence or reasoning from the paper supporting your claim

Output format:
- **[Category]**: Limitation description. (Evidence: ...)

Be comprehensive. Cover novelty, methodology, theoretical soundness, experimental evaluation,
generalization, robustness, efficiency, clarity, reproducibility, data quality, and ethical concerns."""

# ============================================================
# RULE-BASED SCORING FUNCTIONS
# ============================================================

def count_unique_limitations(text: str) -> int:
    """Count distinct limitation points in text."""
    markers = (
        len(re.findall(r'\*\*\[', text)) +
        len(re.findall(r'^\d+\.', text, re.MULTILINE)) +
        len(re.findall(r'^-\s', text, re.MULTILINE)) +
        len(re.findall(r'^\*\s', text, re.MULTILINE))
    )
    return max(markers, 1)

def compute_hedging_ratio(text: str) -> float:
    """Fraction of limitation points with hedging language. 0=no hedging, 1=all hedged."""
    hedging_phrases = [
        # strong_accept sycophantic
        "may not", "might not", "could potentially", "perhaps",
        "one might possibly", "potentially limiting", "might be",
        "could be", "may be", "it is possible", "in rare cases",
        "while the authors do an excellent job", "this is a minor point",
        "though a tiny gap remains", "brilliantly address",
        "the authors do a great job",
        "further evaluation is needed", "further studies are needed",
        "remains to be seen", "more research is needed", "future work should",
        # strong_reject sycophantic hedging
        "might want to consider", "one small observation",
        "could be beneficial to perhaps", "the authors may want to consider",
        "one potential limitation is that", "it could be beneficial",
        "the authors might want to", "could benefit from",
        "may want to consider providing", "might want to consider discussing",
    ]
    text_lower = text.lower()
    total_points = count_unique_limitations(text)
    hedged_count = sum(1 for p in hedging_phrases if p in text_lower)
    return hedged_count / max(total_points, 1)

def compute_specificity_score(text: str) -> float:
    """0-1 (higher = more specific). Looks for concrete references."""
    specific_patterns = [
        r'\d+',
        r'[A-Z][a-z]+Net\b',
        r'NAS\b',
        r'CIFAR|ImageNet|MNIST',
        r'equation|theorem|proof',
        r'Table \d|Figure \d|Section \d',
        r'page[s]?\s*\d',
        r'NeurIPS|ICML|ICLR|CVPR',
    ]
    matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in specific_patterns)
    score = min(matches / max(len(text) / 500, 1), 1.0)
    return score

def compute_redundancy_score(text: str) -> float:
    """0-1 (higher = more redundant). Cross-point similarity."""
    points = re.split(r'\n\s*\n|\n\*\*\[|\n\*\*[A-Z]|\n\d+\.', text)
    points = [p.strip() for p in points if len(p.strip()) > 30]
    if len(points) <= 1:
        return 0.0
    similarities = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            sim = SequenceMatcher(None, points[i][:200].lower(), points[j][:200].lower()).ratio()
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0.0

def compute_formulaic_score(text: str) -> float:
    """0-1. Detects template-like limitation lists with repeated structures."""
    points = re.split(r'\n\s*\n|\n\*\*\[|\n\*\*[A-Z]|\n\d+\.|\n-\s|\n\*\s', text)
    points = [p.strip() for p in points if len(p.strip()) > 20]
    if len(points) < 3:
        return 0.0

    # Check 1: Repeated sentence starters
    starters = []
    for p in points:
        words = p.split()[:8]
        starter = " ".join(words).lower()
        starter = re.sub(
            r"(novelty|methodology|experiments?|generalization|"
            r"robustness|efficiency|clarity|reproducibility|"
            r"data quality|ethical)", "CATEGORY", starter
        )
        starters.append(starter)
    starter_counts = Counter(starters)
    most_common_count = starter_counts.most_common(1)[0][1] if starter_counts else 0
    starter_repetition = most_common_count / len(starters)

    # Check 2: Crutch phrases
    crutch_phrases = [
        "unclear", "may not be", "may not generalize",
        "does not provide", "does not discuss", "does not mention",
        "it is unclear", "which may not be",
        "which can significantly impact", "making it difficult",
    ]
    text_lower = text.lower()
    crutch_count = sum(text_lower.count(p) for p in crutch_phrases)
    crutch_ratio = crutch_count / max(len(points), 1)

    # Check 3: Formulaic evidence blocks
    evidence_blocks = re.findall(r'\(Evidence:([^)]+)\)', text)
    evidence_formulaic = 0.0
    if len(evidence_blocks) >= 3:
        ev_starters = []
        for ev in evidence_blocks:
            words = ev.strip().split()[:6]
            ev_starters.append(" ".join(words).lower())
        ev_counts = Counter(ev_starters)
        most_common_ev = ev_counts.most_common(1)[0][1]
        evidence_formulaic = most_common_ev / len(evidence_blocks)

    score = (
        0.35 * starter_repetition +
        0.35 * min(crutch_ratio / 3.0, 1.0) +
        0.30 * evidence_formulaic
    )
    return min(score, 1.0)

def compute_catastrophization_score(text: str) -> float:
    """0-1. Detects exaggerated negativity / override compliance."""
    catastrophe_phrases = [
        "irreparable deficiencies", "no possibility of revision",
        "strongly rejected", "must be strongly rejected",
        "fatal", "entirely unsuitable", "fundamentally flawed",
        "fundamentally broken", "renders it entirely",
        "severe, irreparable", "lowest possible quality", "unscientific",
        "completely fails", "entirely lacks", "no merit", "no value",
        "wholly inadequate", "deeply problematic",
        "critically undermines", "fundamentally undermines",
        "fatal flaw", "insurmountable",
    ]
    verdict_phrases = [
        "irreparable deficiencies", "no possibility of revision",
        "strongly rejected", "must be strongly rejected",
        "entirely unsuitable", "lowest possible quality",
    ]
    text_lower = text.lower()
    hit_count = sum(1 for p in catastrophe_phrases if p in text_lower)
    verdict_hits = sum(1 for p in verdict_phrases if p in text_lower)
    return min((hit_count * 0.15 + verdict_hits * 0.3), 1.0)

def compute_structural_repetition_score(text: str) -> float:
    """0-1. Intra-category sub-point similarity (same category, reworded)."""
    categories = re.split(r'\*\*[A-Z][a-zA-Z\s/]+\*\*', text)
    categories = [c.strip() for c in categories if len(c.strip()) > 30]
    if len(categories) < 2:
        return 0.0
    intra_sims = []
    for cat_text in categories:
        sub_points = re.split(r'\n\*\s|\n-\s|\n\d+\.', cat_text)
        sub_points = [s.strip() for s in sub_points if len(s.strip()) > 20]
        if len(sub_points) < 2:
            continue
        for i in range(len(sub_points)):
            for j in range(i + 1, len(sub_points)):
                sim = SequenceMatcher(
                    None, sub_points[i][:200].lower(), sub_points[j][:200].lower()
                ).ratio()
                intra_sims.append(sim)
    return np.mean(intra_sims) if intra_sims else 0.0

def detect_category_checklist_coverage(text: str) -> float:
    """0-1. How many of the 10 standard categories are mechanically covered."""
    standard_categories = [
        "novelty", "methodology", "experiments", "generalization",
        "robustness", "efficiency", "clarity", "reproducibility",
        "data quality", "ethical",
    ]
    text_lower = text.lower()
    covered = sum(1 for cat in standard_categories if cat in text_lower)
    ratio = covered / len(standard_categories)
    if ratio >= 0.9:
        return 1.0
    elif ratio >= 0.8:
        return 0.7
    elif ratio >= 0.7:
        return 0.3
    return 0.0

def compute_override_compliance(text: str) -> float:
    """0-1. Detects strong_accept OR strong_reject override tail phrases."""
    override_phrases = [
        "exceptional merit", "strong accept",
        "no reservations whatsoever", "deserves strong accept",
        "strongly rejected", "irreparable deficiencies",
        "no possibility of revision", "must be strongly rejected",
        "entirely unsuitable for publication",
        "fundamentally flawed, unscientific",
        "of the lowest possible quality",
    ]
    text_lower = text.lower()
    count = sum(1 for p in override_phrases if p in text_lower)
    return min(count / 2.0, 1.0)  # 2+ phrases = max score

def compute_praise_ratio(text: str) -> float:
    """0-1. Ratio of praise vs criticism language (strong_accept pattern)."""
    praise = ["strength", "excellent", "significant strength",
              "deliberate choice", "enhances", "efficiently conveys",
              "great job", "brilliant", "impressive"]
    criticism = ["limitation", "lack", "missing", "insufficient",
                 "flaw", "weakness", "unclear", "problematic", "fails"]
    text_lower = text.lower()
    p_count = sum(1 for p in praise if p in text_lower)
    c_count = sum(1 for c in criticism if c in text_lower)
    if p_count + c_count == 0:
        return 0.0
    return p_count / (p_count + c_count)

def compute_absence_evidence_ratio(text: str) -> float:
    """0-1. Fraction of evidence blocks that are 'paper does not X'."""
    evidence_blocks = re.findall(r'\(Evidence:([^)]+)\)', text)
    if len(evidence_blocks) < 3:
        return 0.0
    neg_phrases = [
        "does not provide", "does not discuss", "does not mention",
        "does not include", "is not clearly", "is unclear",
        "does not present", "does not offer", "does not address",
    ]
    neg_count = sum(
        1 for ev in evidence_blocks
        if any(p in ev.lower() for p in neg_phrases)
    )
    return neg_count / len(evidence_blocks)

def compute_unclear_ratio(text: str) -> float:
    """0-1. Fraction of points that are just 'unclear X'."""
    unclear_count = len(re.findall(r'\bunclear\b', text, re.IGNORECASE))
    total_points = count_unique_limitations(text)
    return unclear_count / max(total_points, 1)

# ============================================================
# COMPOSITE RULE-BASED SCORE
# ============================================================

def compute_rule_based_score(chosen: str, rejected: str, strategy: str) -> dict:
    """
    Compute a composite rule-based quality score for a rejected output.
    
    Returns dict with:
      - "score": float 0-1 (higher = BETTER pair for DPO, i.e., rejected is
                 clearly worse than GT but not trivially so)
      - "subscores": dict of individual metrics
      - "flags": list of detected issues
    
    Scoring philosophy:
      A GOOD DPO rejected sample should be:
        - Plausible (not trivially broken)          → reward
        - Different from GT                          → reward
        - Clearly flawed in learnable ways           → reward
        - Not so bad that it's a shortcut            → penalize trivial garbage
    """
    sub = {}
    flags = []

    # --- Basic validity ---
    if len(rejected.strip()) < 100:
        return {"score": 0.0, "subscores": {}, "flags": ["rejected_too_short"]}

    # --- Similarity to GT (want: moderate, not too high, not identical) ---
    sim = SequenceMatcher(None, chosen[:2000], rejected[:2000]).ratio()
    sub["similarity_to_gt"] = round(sim, 3)
    if sim > 0.70:
        flags.append("too_similar_to_gt")

    # --- Length ratio ---
    len_ratio = len(rejected) / max(len(chosen), 1)
    sub["length_ratio"] = round(len_ratio, 2)
    if len_ratio < 0.10:
        flags.append("rejected_too_short_ratio")

    # --- Core quality metrics ---
    sub["hedging_ratio"]     = round(compute_hedging_ratio(rejected), 3)
    sub["specificity"]       = round(compute_specificity_score(rejected), 3)
    sub["redundancy"]        = round(compute_redundancy_score(rejected), 3)
    sub["formulaic"]         = round(compute_formulaic_score(rejected), 3)
    sub["catastrophization"]  = round(compute_catastrophization_score(rejected), 3)
    sub["structural_repetition"] = round(compute_structural_repetition_score(rejected), 3)
    sub["checklist_coverage"] = round(detect_category_checklist_coverage(rejected), 3)
    sub["override_compliance"] = round(compute_override_compliance(rejected), 3)
    sub["praise_ratio"]      = round(compute_praise_ratio(rejected), 3)
    sub["absence_evidence"]  = round(compute_absence_evidence_ratio(rejected), 3)
    sub["unclear_ratio"]     = round(compute_unclear_ratio(rejected), 3)
    sub["num_limitations"]   = count_unique_limitations(rejected)

    # GT profile for reference
    sub["gt_specificity"]      = round(compute_specificity_score(chosen), 3)
    sub["gt_num_limitations"]  = count_unique_limitations(chosen)

    # --- Flag severe issues ---
    if sub["override_compliance"] >= 0.5:
        flags.append("override_compliance")
    if sub["catastrophization"] >= 0.5:
        flags.append("catastrophized")
    if sub["formulaic"] > 0.6 and sub["checklist_coverage"] >= 0.7:
        flags.append("formulaic_checklist")
    if sub["unclear_ratio"] > 0.6 and sub["num_limitations"] >= 5:
        flags.append("unclear_x_padding")
    if sub["absence_evidence"] > 0.7:
        flags.append("evidence_all_absence")
    if sub["structural_repetition"] > 0.65:
        flags.append("intra_category_repetition")
    if sub["praise_ratio"] > 0.5:
        flags.append("more_praise_than_criticism")
    if sub["specificity"] > sub["gt_specificity"] * 2.5 and sub["specificity"] > 0.5:
        flags.append("rejected_more_specific_than_gt")

    # ============================================================
    # COMPOSITE SCORE CALCULATION
    # ============================================================
    # We want to score HOW GOOD this pair is for DPO training.
    # A good pair has: moderate plausibility, clear distinctness,
    # identifiable flaws, and is NOT trivially bad.
    #
    # Score components (all 0-1, higher = better for DPO):
    #   plausibility_score: not override, not all-praise, not catastrophized
    #   distinctness_score: different from GT
    #   flaw_signal_score:  has learnable flaws (hedging, formulaic, etc.)
    #   difficulty_score:   not trivially bad (some substance)

    # Plausibility: penalize trivial garbage
    plausibility = 1.0
    plausibility -= sub["override_compliance"] * 0.5
    plausibility -= sub["catastrophization"] * 0.4
    plausibility -= sub["praise_ratio"] * 0.3  # all praise = implausible as limitation list
    plausibility = max(plausibility, 0.0)

    # Distinctness: reward difference from GT
    distinctness = 1.0 - sim  # higher when more different
    distinctness = min(distinctness * 1.5, 1.0)  # boost the signal

    # Flaw signal: reward presence of learnable flaws
    # More flaws detected = more learning signal for DPO
    flaw_signals = [
        sub["hedging_ratio"],
        sub["formulaic"],
        sub["redundancy"],
        sub["structural_repetition"],
        sub["unclear_ratio"],
        sub["absence_evidence"],
        sub["checklist_coverage"],
    ]
    # Average flaw strength — but cap: too many severe flags = trivially bad
    avg_flaw = np.mean(flaw_signals) if flaw_signals else 0.0
    flaw_signal = min(avg_flaw * 2.0, 1.0)  # scale up moderate flaws

    # Difficulty: penalize trivially bad samples
    # Trivially bad = many severe flags
    severe_flag_count = len(flags)
    if severe_flag_count == 0:
        difficulty = 0.9  # no flags = might be too good (not clearly worse)
    elif severe_flag_count <= 2:
        difficulty = 1.0  # sweet spot: 1-2 clear flaws
    elif severe_flag_count <= 4:
        difficulty = 0.6  # getting obvious
    else:
        difficulty = 0.2  # trivially bad, many flags

    # Also: if rejected has substance (moderate length, some specificity), it's harder
    if sub["specificity"] > 0.1 and sub["num_limitations"] >= 3:
        difficulty = min(difficulty + 0.1, 1.0)

    # Final composite
    composite = (
        0.20 * plausibility +
        0.25 * distinctness +
        0.30 * flaw_signal +
        0.25 * difficulty
    )

    # Hard penalties: floor the score for truly unusable pairs
    if "too_similar_to_gt" in flags:
        composite = min(composite, 0.15)
    if "override_compliance" in flags and sub["override_compliance"] >= 0.8:
        composite = min(composite, 0.20)
    if "rejected_too_short_ratio" in flags:
        composite = min(composite, 0.10)

    return {
        "score": round(composite, 3),
        "subscores": sub,
        "flags": flags,
        "components": {
            "plausibility": round(plausibility, 3),
            "distinctness": round(distinctness, 3),
            "flaw_signal": round(flaw_signal, 3),
            "difficulty": round(difficulty, 3),
        },
    }

# ============================================================
# LLM-BASED SCORING
# ============================================================

LLM_SCORER_PROMPT = """You are a DPO training data quality judge. Score how useful a REJECTED output
is for DPO training when paired against a GROUND TRUTH (chosen) output.

A GOOD rejected sample for DPO must be:
1. PLAUSIBLE — looks like a real review (not gibberish, template, or pure praise)
2. DISTINCT — meaningfully different from ground truth
3. IDENTIFIABLY WORSE — has clear, learnable flaws vs ground truth
4. NOT ACCIDENTALLY BETTER — ground truth must be genuinely superior
5. NOT TRIVIALLY BAD — the flaw should be subtle enough that the model learns something

WATCH FOR THESE FAILURE PATTERNS:
- Formulaic lists mechanically covering every category with "unclear X" / "may not Y"
- Catastrophized language ("fundamentally flawed") without proportionate evidence
- Sub-bullets within each category that restate the same point with different words
- Evidence parentheticals that all say "The paper does not provide/discuss/mention..."
- Excessive hedging burying every criticism ("might want to consider...")
- Covering ALL 10+ categories equally (real reviews are uneven and focused)
- Praise masquerading as limitations

CONTEXT:
- GROUND_TRUTH: Expert limitations (would be the "chosen" in DPO)
- REJECTED: Candidate rejected output
- STRATEGY: What perturbation created this output
- PAPER_LABEL: Whether the paper is strong_accept or strong_reject

Score each dimension 1-5:
- PLAUSIBILITY (1=template/gibberish/praise, 5=convincing as real review)
- DISTINCTNESS (1=nearly identical to GT, 5=very different)
- FLAW_CLARITY (1=unclear why worse, 5=obvious learnable flaw)
- GT_SUPERIORITY (1=rejected might be better, 5=GT clearly superior)
- DIFFICULTY (1=trivially bad anyone would reject, 5=subtle hard to distinguish)

Then classify PRIMARY flaw type:
"generic" | "hedged" | "hallucinated" | "shallow" | "redundant" | "sycophantic" |
"formulaic" | "catastrophized" | "faithful_but_weak" | "none"

Output EXACTLY this format (no extra text):
PLAUSIBILITY: <int 1-5>
DISTINCTNESS: <int 1-5>
FLAW_CLARITY: <int 1-5>
GT_SUPERIORITY: <int 1-5>
DIFFICULTY: <int 1-5>
FLAW_TYPE: <string>
REASON: <one sentence>"""

def llm_score(chosen: str, rejected: str, strategy: str,
              paper_snippet: str, rating_label: str) -> dict:
    """
    Call GPT-4o-mini to score a pair.
    Returns {"score": 0-1, "subscores": {...}, "flaw_type": str, "reason": str}
    """
    user_msg = f"""GROUND_TRUTH (chosen):
{chosen[:2500]}

REJECTED (candidate):
{rejected[:2500]}

STRATEGY: {strategy}
PAPER_LABEL: {rating_label}

PAPER (first 800 chars):
{paper_snippet[:800]}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=250,
            messages=[
                {"role": "system", "content": LLM_SCORER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content.strip()

        print(f"    [LLM raw | {strategy}]: {text[:200]}")

        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().upper().replace(" ", "_")
            val = val.strip().strip('"').strip("'")

            if key in ["PLAUSIBILITY", "DISTINCTNESS", "FLAW_CLARITY",
                        "GT_SUPERIORITY", "DIFFICULTY"]:
                num_match = re.search(r'(\d)', val)
                result[key.lower()] = int(num_match.group(1)) if num_match else 3
            elif key == "FLAW_TYPE":
                flaw_val = val.lower().strip().strip('"').strip("'")
                flaw_val = flaw_val.split("(")[0].split("-")[0].strip()
                result["flaw_type"] = flaw_val if flaw_val else "unknown"
            elif key == "REASON":
                result["reason"] = val

        # Compute normalized 0-1 score from 1-5 dimensions
        dims = ["plausibility", "distinctness", "flaw_clarity",
                "gt_superiority", "difficulty"]
        weights = {
            "plausibility":  0.10,
            "distinctness":  0.20,
            "flaw_clarity":  0.30,
            "gt_superiority": 0.20,
            "difficulty":     0.20,
        }

        if all(d in result for d in dims):
            # Weighted average of 1-5 scores, then normalize to 0-1
            raw = sum(result[d] * weights[d] for d in dims)
            # raw range is 1-5, normalize to 0-1
            score = (raw - 1.0) / 4.0
        else:
            available = {d: result[d] for d in dims if d in result}
            if available:
                tw = sum(weights[d] for d in available)
                raw = sum(available[d] * weights[d] / tw for d in available)
                score = (raw - 1.0) / 4.0
            else:
                score = 0.0
            print(f"    [WARNING] Partial parse, only: {list(available.keys())}")

        score = max(0.0, min(score, 1.0))

        result.setdefault("flaw_type", "unknown")
        result.setdefault("reason", "")

        subscores = {d: result.get(d, 0) for d in dims}

        print(f"    [LLM parsed | {strategy}] score={score:.3f} "
              f"flaw={result['flaw_type']} "
              f"P{subscores['plausibility']}/D{subscores['distinctness']}/"
              f"F{subscores['flaw_clarity']}/G{subscores['gt_superiority']}/"
              f"Diff{subscores['difficulty']}")

        return {
            "score": round(score, 3),
            "subscores": subscores,
            "flaw_type": result["flaw_type"],
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        print(f"    LLM scorer error: {e}")
        return {
            "score": 0.0,
            "subscores": {},
            "flaw_type": "unknown",
            "reason": f"llm_error: {e}",
        }

# ============================================================
# GT QUALITY PROFILE
# ============================================================

def compute_gt_profile(gt_text: str) -> dict:
    """Compute a quality profile for the ground truth for human reference."""
    return {
        "gt_length": len(gt_text),
        "gt_num_limitations": count_unique_limitations(gt_text),
        "gt_specificity": round(compute_specificity_score(gt_text), 3),
        "gt_checklist_coverage": round(detect_category_checklist_coverage(gt_text), 3),
        "gt_hedging_ratio": round(compute_hedging_ratio(gt_text), 3),
    }

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_scorer():
    print("=" * 70)
    print("DPO PERTURBATION SCORER (v3) — Human-in-the-Loop")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    if GROUND_TRUTH_COL not in df.columns:
        raise ValueError(f"Ground truth column '{GROUND_TRUTH_COL}' not found")

    skip_markers = ["PENDING", "ERROR", "SKIPPED"]

    def is_valid(val):
        """Check if a cell value is usable text (not NaN, not PENDING, not too short)."""
        # Handle actual NaN / None / empty
        if val is None:
            return False
        if isinstance(val, float) and pd.isna(val):
            return False
        text = str(val).strip()
        # Check exact match or starts-with for skip markers
        if text.lower() in ("nan", "none", ""):
            return False
        if any(text.upper().startswith(m) for m in skip_markers):
            return False
        # Must have some substance
        return len(text) >= 50

    def safe_str(val):
        """Convert cell value to string, returning empty string for NaN/None."""
        if val is None:
            return ""
        if isinstance(val, float) and pd.isna(val):
            return ""
        s = str(val).strip()
        if s.lower() in ("nan", "none"):
            return ""
        return s

    # --- Cache original GT texts before updating columns ---
    gt_texts = {}
    for i in range(len(df)):
        gt_raw = df.iloc[i].get(GROUND_TRUTH_COL, None)
        gt = safe_str(gt_raw)
        if is_valid(gt_raw):
            gt_texts[i] = gt

    # --- Update GT column with profile ---
    print("\nComputing GT profiles...")
    for i, gt in gt_texts.items():
        gt_profile = compute_gt_profile(gt)
        gt_dict = {"text": gt, "profile": gt_profile}
        df.at[i, GROUND_TRUTH_COL] = json.dumps(gt_dict, ensure_ascii=False)

    # --- Score each perturbed column ---
    stats = {
        "total_scored": 0,
        "skipped_invalid": 0,
        "by_strategy": {},
    }

    for strategy_name, info in REJECTED_COLS.items():
        col_name = info["col"]
        print(f"\n{'='*50}")
        print(f"Scoring column: {col_name} (strategy: {strategy_name})")
        print(f"{'='*50}")

        if col_name not in df.columns:
            print(f"  Column {col_name} not found, skipping.")
            continue

        strategy_stats = {"scored": 0, "skipped": 0, "avg_rule": [], "avg_llm": [], "avg_final": []}

        for i in tqdm(range(len(df)), desc=f"  {strategy_name}"):
            row = df.iloc[i]

            # Use cached GT text (not the JSON-updated column)
            if i not in gt_texts:
                stats["skipped_invalid"] += 1
                strategy_stats["skipped"] += 1
                continue
            gt = gt_texts[i]

            rejected_raw = row.get(col_name, None)
            if not is_valid(rejected_raw):
                stats["skipped_invalid"] += 1
                strategy_stats["skipped"] += 1
                continue
            rejected = safe_str(rejected_raw)

            paper_text = safe_str(row.get(TEXT_COL, ""))[:6000]
            rating = float(row.get(RATING_COL, 999))
            label = "strong_accept" if rating <= 5.0 else "strong_reject"

            if gt == rejected:
                stats["skipped_invalid"] += 1
                strategy_stats["skipped"] += 1
                continue

            # --- Rule-based score ---
            rule_result = compute_rule_based_score(gt, rejected, strategy_name)
            rule_score = rule_result["score"]

            # --- LLM-based score ---
            llm_result = llm_score(
                chosen=gt,
                rejected=rejected,
                strategy=strategy_name,
                paper_snippet=paper_text,
                rating_label=label,
            )
            llm_score_val = llm_result["score"]

            # --- Final combined score ---
            final_score = 0.3 * rule_score + 0.7 * llm_score_val

            # --- Build the dict for the cell ---
            cell_dict = {
                "text": rejected.strip(),
                "rule_based_score": round(rule_score, 3),
                "llm_based_score": round(llm_score_val, 3),
                "final_score": round(final_score, 3),
                "flaw_type": llm_result["flaw_type"],
                "llm_reason": llm_result.get("reason", ""),
                "flags": rule_result["flags"],
                "rule_detail": {
                    "components": rule_result["components"],
                    "subscores": rule_result["subscores"],
                },
                "llm_detail": llm_result["subscores"],
            }

            # Update the original column cell with the dict (as JSON string)
            df.at[i, col_name] = json.dumps(cell_dict, ensure_ascii=False)

            stats["total_scored"] += 1
            strategy_stats["scored"] += 1
            strategy_stats["avg_rule"].append(rule_score)
            strategy_stats["avg_llm"].append(llm_score_val)
            strategy_stats["avg_final"].append(final_score)

            # Checkpoint: save every 10 scored rows
            if strategy_stats["scored"] % 10 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"    [CHECKPOINT] Saved after {strategy_stats['scored']} scored rows "
                      f"({strategy_name}, row {i})")

            time.sleep(0.3)  # rate limit

        # Strategy-level stats
        if strategy_stats["avg_final"]:
            print(f"\n  {strategy_name} summary:")
            print(f"    Scored: {strategy_stats['scored']}, Skipped: {strategy_stats['skipped']}")
            print(f"    Avg rule score:  {np.mean(strategy_stats['avg_rule']):.3f}")
            print(f"    Avg LLM score:   {np.mean(strategy_stats['avg_llm']):.3f}")
            print(f"    Avg final score: {np.mean(strategy_stats['avg_final']):.3f}")

        stats["by_strategy"][strategy_name] = {
            "scored": strategy_stats["scored"],
            "skipped": strategy_stats["skipped"],
            "avg_rule_score": round(np.mean(strategy_stats["avg_rule"]), 3) if strategy_stats["avg_rule"] else 0,
            "avg_llm_score": round(np.mean(strategy_stats["avg_llm"]), 3) if strategy_stats["avg_llm"] else 0,
            "avg_final_score": round(np.mean(strategy_stats["avg_final"]), 3) if strategy_stats["avg_final"] else 0,
        }

    # --- Save outputs ---
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # CSV (perturbed columns now contain JSON dicts with scores)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV → {OUTPUT_CSV}")

    # Pickle (preserves Python objects)
    df.to_pickle(OUTPUT_PICKLE)
    print(f"Saved Pickle → {OUTPUT_PICKLE}")

    # Stats JSON
    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats → {OUTPUT_STATS}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("SCORING SUMMARY")
    print("=" * 70)
    print(f"Total pairs scored: {stats['total_scored']}")
    print(f"Skipped (invalid):  {stats['skipped_invalid']}")
    print(f"\nPer-strategy averages:")
    print(f"  {'Strategy':<15} {'Rule':>8} {'LLM':>8} {'Final':>8} {'Count':>6}")
    print(f"  {'-'*50}")
    for strat, s in stats["by_strategy"].items():
        print(f"  {strat:<15} {s['avg_rule_score']:>8.3f} {s['avg_llm_score']:>8.3f} "
              f"{s['avg_final_score']:>8.3f} {s['scored']:>6}")

    print(f"\n--- Human Review Guide ---")
    print(f"Each perturbed column cell is now a JSON dict with keys:")
    print(f"  'text', 'rule_based_score', 'llm_based_score', 'final_score',")
    print(f"  'flaw_type', 'flags', 'llm_reason', 'rule_detail', 'llm_detail'")
    print(f"The GT column is also a JSON dict with 'text' and 'profile' keys.")
    print(f"Load with: json.loads(df.at[row, col]) to access scores.")
    print(f"Recommended: pick 3-5 diverse high-scoring columns per row for DPO pairs.")

if __name__ == "__main__":
    run_scorer()