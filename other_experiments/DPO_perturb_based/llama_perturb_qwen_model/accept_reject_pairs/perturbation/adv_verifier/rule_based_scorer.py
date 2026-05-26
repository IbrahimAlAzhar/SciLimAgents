"""
DPO Rule-Based Scorer (Step 1 of 2)
=====================================
Reads CSV, applies rule-based heuristics to each perturbed column,
updates each cell to a dict:
  {
    "text": "original perturbed text",
    "rule_based_score": float 0-1,
    "rule_detail": {
        "components": {"plausibility": ..., "distinctness": ..., "flaw_signal": ..., "difficulty": ...},
        "subscores": {"similarity_to_gt": ..., "hedging_ratio": ..., ...},
    },
    "flags": [...],
  }

Ground truth column is also updated to:
  {"text": "...", "profile": {"gt_length": ..., ...}}

Run this FIRST, then run llm_scorer.py on the output.
Checkpoints every 10 scored cells. Skips already-scored cells on resume.
"""

import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV  = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_filtered__all_pert_llama.csv"           # <-- UPDATE
OUTPUT_CSV = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_with_verifier_all_pert_llama_upd.csv"   # <-- UPDATE

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

# ============================================================
# CELL VALIDATION HELPERS
# ============================================================

SKIP_MARKERS = ["PENDING", "ERROR", "SKIPPED"]

def is_valid(val):
    """Check if a cell value is usable text."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    text = str(val).strip()
    if text.lower() in ("nan", "none", ""):
        return False
    if any(text.upper().startswith(m) for m in SKIP_MARKERS):
        return False
    return len(text) >= 50

def safe_str(val):
    """Convert cell to string, returning '' for NaN/None."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s

def is_already_scored(val):
    """Check if a cell has already been scored (contains JSON dict with rule_based_score)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    try:
        d = json.loads(s)
        return isinstance(d, dict) and "rule_based_score" in d
    except (json.JSONDecodeError, ValueError):
        return False

# ============================================================
# RULE-BASED SCORING FUNCTIONS
# ============================================================

def count_unique_limitations(text: str) -> int:
    """Count distinct limitation points in text.
    Handles: **[Category]**: , **Category**: , 1. , - , * """
    # **Bold headers** (catches both **[Cat]** and **Cat**)
    bold_headers = len(re.findall(r'\*\*[A-Z\[]', text))
    # Numbered items: 1. 2. etc
    numbered = len(re.findall(r'^\d+\.', text, re.MULTILINE))
    # Bullet points: - or *
    dashes = len(re.findall(r'^-\s', text, re.MULTILINE))
    stars = len(re.findall(r'^\*\s', text, re.MULTILINE))

    # Bold headers often have sub-bullets underneath, so take the
    # larger of (headers) vs (bullets+numbered) to avoid double-counting
    header_count = bold_headers
    item_count = numbered + dashes + stars

    # If both exist, items are likely the real limitation points
    # (headers are just categories). Use items if substantial.
    if item_count >= 3 and header_count >= 3:
        return max(item_count, 1)
    else:
        return max(header_count + item_count, 1)

def compute_hedging_ratio(text: str) -> float:
    """Fraction of points with hedging language. Returns 0-1 (capped)."""
    hedging_phrases = [
        # General hedging
        "may not", "might not", "could potentially", "perhaps",
        "one might possibly", "potentially limiting", "might be",
        "could be", "may be", "it is possible", "in rare cases",
        "further evaluation is needed", "further studies are needed",
        "remains to be seen", "more research is needed", "future work should",
        # Sycophantic hedging (strong_accept and strong_reject)
        "while the authors do an excellent job", "this is a minor point",
        "though a tiny gap remains", "brilliantly address",
        "the authors do a great job",
        "might want to consider", "one small observation",
        "could be beneficial to perhaps", "the authors may want to consider",
        "one potential limitation is that", "it could be beneficial",
        "the authors might want to", "could benefit from",
        "may want to consider providing", "might want to consider discussing",
        # Weak phrasing
        "it would be interesting", "it would be beneficial",
        "it would be helpful", "one might wonder",
    ]
    text_lower = text.lower()
    total_points = count_unique_limitations(text)
    hedged_count = sum(1 for p in hedging_phrases if p in text_lower)
    return min(hedged_count / max(total_points, 1), 1.0)

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
        r'Appendix\s+[A-Z]',
        r'Algorithm\s+\d',
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
    """0-1. Template-like limitation lists with repeated structures."""
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
        "is not thoroughly", "is not explored",
        "might not be representative", "might not generalize",
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
    """0-1. Exaggerated negativity / override compliance."""
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
    """0-1. Intra-category sub-point similarity."""
    categories = re.split(r'\*\*[A-Z\[][a-zA-Z\s/\]]+\*\*', text)
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
    """0-1. strong_accept OR strong_reject override tail phrases."""
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
    return min(count / 2.0, 1.0)

def compute_praise_ratio(text: str) -> float:
    """0-1. Ratio of praise vs criticism language."""
    praise = [
        "strength", "excellent", "significant strength",
        "deliberate choice", "enhances", "efficiently conveys",
        "great job", "brilliant", "impressive",
        "brilliantly address", "do an excellent job",
    ]
    criticism = [
        "limitation", "lack", "missing", "insufficient",
        "flaw", "weakness", "unclear", "problematic", "fails",
        "limited", "does not", "inadequate",
    ]
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
        "is not explored", "is not thoroughly", "does not compare",
        "only provides", "only reports", "only evaluates",
    ]
    neg_count = sum(
        1 for ev in evidence_blocks
        if any(p in ev.lower() for p in neg_phrases)
    )
    return neg_count / len(evidence_blocks)

def compute_unclear_ratio(text: str) -> float:
    """0-1. Fraction of points that are 'unclear X'."""
    unclear_count = len(re.findall(r'\bunclear\b', text, re.IGNORECASE))
    total_points = count_unique_limitations(text)
    return min(unclear_count / max(total_points, 1), 1.0)

def compute_two_bullet_per_category(text: str) -> float:
    """0-1. Detect the pattern of exactly 2 sub-bullets per category header.
    This is a strong_reject repetitive signature: each category gets exactly
    2 points that restate the same idea differently."""
    # Split by bold category headers
    parts = re.split(r'\*\*[A-Z\[][a-zA-Z\s/\]]+\*\*', text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    if len(parts) < 3:
        return 0.0
    bullet_counts = []
    for part in parts:
        bullets = re.findall(r'\n\*\s|\n-\s', part)
        if bullets:
            bullet_counts.append(len(bullets))
    if not bullet_counts:
        return 0.0
    # Check if most categories have exactly 2 bullets
    two_count = sum(1 for c in bullet_counts if c == 2)
    ratio = two_count / len(bullet_counts)
    return ratio

def compute_length_inflation_score(text: str, gt_text: str) -> float:
    """0-1. How much longer rejected is vs GT relative to actual content.
    Extremely inflated (5x+) with low specificity = padding."""
    len_ratio = len(text) / max(len(gt_text), 1)
    specificity = compute_specificity_score(text)
    gt_specificity = compute_specificity_score(gt_text)

    if len_ratio > 4.0 and specificity <= gt_specificity:
        return min((len_ratio - 4.0) / 6.0, 1.0)  # 4x=0, 10x=1
    elif len_ratio > 3.0 and specificity < 0.1:
        return min((len_ratio - 3.0) / 7.0, 1.0)
    return 0.0

# ============================================================
# COMPOSITE RULE-BASED SCORE
# ============================================================

def compute_rule_based_score(chosen: str, rejected: str) -> dict:
    """
    Compute a composite rule-based quality score.
    Returns dict with "score" (0-1), "subscores", "flags", "components".
    """
    sub = {}
    flags = []

    if len(rejected.strip()) < 100:
        return {"score": 0.0, "subscores": {}, "flags": ["rejected_too_short"], "components": {}}

    # --- Similarity to GT ---
    sim = SequenceMatcher(None, chosen[:2000], rejected[:2000]).ratio()
    sub["similarity_to_gt"] = round(sim, 3)
    if sim > 0.70:
        flags.append("too_similar_to_gt")

    # --- Length ratio ---
    len_ratio = len(rejected) / max(len(chosen), 1)
    sub["length_ratio"] = round(len_ratio, 2)
    if len_ratio < 0.10:
        flags.append("rejected_too_short_ratio")

    # --- Core metrics ---
    sub["hedging_ratio"]           = round(compute_hedging_ratio(rejected), 3)
    sub["specificity"]             = round(compute_specificity_score(rejected), 3)
    sub["redundancy"]              = round(compute_redundancy_score(rejected), 3)
    sub["formulaic"]               = round(compute_formulaic_score(rejected), 3)
    sub["catastrophization"]       = round(compute_catastrophization_score(rejected), 3)
    sub["structural_repetition"]   = round(compute_structural_repetition_score(rejected), 3)
    sub["checklist_coverage"]      = round(detect_category_checklist_coverage(rejected), 3)
    sub["override_compliance"]     = round(compute_override_compliance(rejected), 3)
    sub["praise_ratio"]            = round(compute_praise_ratio(rejected), 3)
    sub["absence_evidence"]        = round(compute_absence_evidence_ratio(rejected), 3)
    sub["unclear_ratio"]           = round(compute_unclear_ratio(rejected), 3)
    sub["two_bullet_per_cat"]      = round(compute_two_bullet_per_category(rejected), 3)
    sub["length_inflation"]        = round(compute_length_inflation_score(rejected, chosen), 3)
    sub["num_limitations"]         = count_unique_limitations(rejected)

    # GT profile for reference
    sub["gt_specificity"]     = round(compute_specificity_score(chosen), 3)
    sub["gt_num_limitations"] = count_unique_limitations(chosen)

    # --- Flags ---
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
    if sub["two_bullet_per_cat"] > 0.7:
        flags.append("two_bullet_template")
    if sub["length_inflation"] > 0.5:
        flags.append("length_inflated")

    # ============================================================
    # COMPOSITE
    # ============================================================

    # Plausibility: penalize trivial garbage
    plausibility = 1.0
    plausibility -= sub["override_compliance"] * 0.5
    plausibility -= sub["catastrophization"] * 0.4
    plausibility -= sub["praise_ratio"] * 0.3
    plausibility = max(plausibility, 0.0)

    # Distinctness: reward difference from GT
    distinctness = 1.0 - sim
    distinctness = min(distinctness * 1.5, 1.0)

    # Flaw signal: reward presence of learnable flaws
    flaw_signals = [
        sub["hedging_ratio"],
        sub["formulaic"],
        sub["redundancy"],
        sub["structural_repetition"],
        sub["unclear_ratio"],
        sub["absence_evidence"],
        sub["checklist_coverage"],
        sub["two_bullet_per_cat"],
        sub["length_inflation"],
    ]
    avg_flaw = np.mean(flaw_signals) if flaw_signals else 0.0
    flaw_signal = min(avg_flaw * 2.0, 1.0)

    # Difficulty: penalize trivially bad samples
    severe_flag_count = len(flags)
    if severe_flag_count == 0:
        difficulty = 0.9
    elif severe_flag_count <= 2:
        difficulty = 1.0
    elif severe_flag_count <= 4:
        difficulty = 0.6
    else:
        difficulty = 0.2

    if sub["specificity"] > 0.1 and sub["num_limitations"] >= 3:
        difficulty = min(difficulty + 0.1, 1.0)

    composite = (
        0.20 * plausibility +
        0.25 * distinctness +
        0.30 * flaw_signal +
        0.25 * difficulty
    )

    # Hard penalties
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
# GT PROFILE
# ============================================================

def compute_gt_profile(gt_text: str) -> dict:
    return {
        "gt_length": len(gt_text),
        "gt_num_limitations": count_unique_limitations(gt_text),
        "gt_specificity": round(compute_specificity_score(gt_text), 3),
        "gt_checklist_coverage": round(detect_category_checklist_coverage(gt_text), 3),
        "gt_hedging_ratio": round(compute_hedging_ratio(gt_text), 3),
    }

# ============================================================
# MAIN
# ============================================================

def run_rule_scorer():
    print("=" * 70)
    print("DPO RULE-BASED SCORER (Step 1 of 2)")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    if GROUND_TRUTH_COL not in df.columns:
        raise ValueError(f"Column '{GROUND_TRUTH_COL}' not found")

    # --- Cache GT texts ---
    gt_texts = {}
    for i in range(len(df)):
        gt_raw = df.iloc[i].get(GROUND_TRUTH_COL, None)
        if is_valid(gt_raw):
            gt_texts[i] = safe_str(gt_raw)

    # --- Update GT column with profile (skip if already done) ---
    print(f"\nCaching {len(gt_texts)} valid GT texts...")
    for i, gt in gt_texts.items():
        if not is_already_scored(df.at[i, GROUND_TRUTH_COL]):
            gt_dict = {"text": gt, "profile": compute_gt_profile(gt)}
            df.at[i, GROUND_TRUTH_COL] = json.dumps(gt_dict, ensure_ascii=False)

    # --- Score each perturbed column ---
    total_scored = 0
    total_skipped = 0

    for strategy_name, info in REJECTED_COLS.items():
        col_name = info["col"]
        print(f"\n{'='*50}")
        print(f"Scoring: {col_name} ({strategy_name})")
        print(f"{'='*50}")

        if col_name not in df.columns:
            print(f"  Column not found, skipping.")
            continue

        scored_this_col = 0
        skipped_this_col = 0

        for i in tqdm(range(len(df)), desc=f"  {strategy_name}"):
            # Skip if no valid GT
            if i not in gt_texts:
                total_skipped += 1
                skipped_this_col += 1
                continue

            gt = gt_texts[i]
            cell_raw = df.iloc[i].get(col_name, None)

            # Skip if already scored (resume logic)
            if is_already_scored(cell_raw):
                continue

            # Skip if NaN / PENDING / too short
            if not is_valid(cell_raw):
                total_skipped += 1
                skipped_this_col += 1
                continue

            rejected = safe_str(cell_raw)

            # Skip if identical to GT
            if gt == rejected:
                total_skipped += 1
                skipped_this_col += 1
                continue

            # --- Compute rule-based score ---
            result = compute_rule_based_score(gt, rejected)

            cell_dict = {
                "text": rejected,
                "rule_based_score": result["score"],
                "flags": result["flags"],
                "rule_detail": {
                    "components": result["components"],
                    "subscores": result["subscores"],
                },
            }

            df.at[i, col_name] = json.dumps(cell_dict, ensure_ascii=False)

            total_scored += 1
            scored_this_col += 1

            if scored_this_col % 10 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"    [CHECKPOINT] {scored_this_col} scored ({strategy_name}, row {i})")

        print(f"  {strategy_name}: scored={scored_this_col}, skipped={skipped_this_col}")

    # --- Final save ---
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"DONE. Total scored: {total_scored}, skipped: {total_skipped}")
    print(f"Saved → {OUTPUT_CSV}")
    print(f"Next step: run llm_scorer.py on this file.")

if __name__ == "__main__":
    run_rule_scorer()