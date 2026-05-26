"""
DPO Pair Builder (Step 3 of 3) — Unified Scoring
===================================================
Reads fully-scored CSV (output of rule_scorer + llm_scorer),
computes a UNIFIED score from all available sub-scores,
picks top 3-4 diverse columns per row, builds JSONL for DPO training.

UNIFIED SCORE PHILOSOPHY:
  A good DPO pair needs the rejected to be:
    (A) Plausible enough to not be a trivial shortcut
    (B) Clearly worse than GT in specific, identifiable ways
    (C) Moderately difficult — not obvious garbage, not accidentally good
  
  We combine rule sub-scores and LLM sub-scores into a single 0-1 number
  that captures all three properties.

Weights: 60% rule-derived, 25% LLM-derived, 15% structural penalties/bonuses
(LLM weight reduced because gpt-4o-mini gives flat scores across strategies)
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV       = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/df_with_verifier_all_pert_llama.csv"     # <-- UPDATE (output of scorer)
OUTPUT_JSONL    = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/dpo_training_pairs.jsonl"            # <-- UPDATE
OUTPUT_STATS    = "other_experiments/DPO_perturb_based/accept_reject_pairs/data/perturb/new/dpo_pair_builder_stats.json"         # <-- UPDATE

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

TOP_K_PER_ROW = 4

# Columns with ANY of these flags are auto-dropped (trivially bad for DPO)
DROP_FLAGS = [
    "override_compliance",
    "too_similar_to_gt",
    "rejected_too_short",
    "rejected_too_short_ratio",
]

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
# CELL HELPERS
# ============================================================

def safe_parse(val):
    """Parse JSON cell, return dict or None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) and "text" in d else None
    except (json.JSONDecodeError, ValueError):
        return None

def extract_gt_text(val):
    """Get GT text from raw or JSON cell."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    parsed = safe_parse(val)
    if parsed and "text" in parsed:
        return parsed["text"]
    if len(s) >= 50:
        return s
    return None

# ============================================================
# UNIFIED SCORE COMPUTATION
# ============================================================

def compute_unified_score(cell_dict: dict) -> dict:
    """
    Compute a unified DPO-utility score from all available sub-scores.
    
    Returns {"unified_score": float 0-1, "breakdown": {...}, "verdict": str}
    
    HIGH score (0.65+) = excellent pair
    MID score  (0.40-0.65) = usable pair
    LOW score  (0.0-0.40) = drop
    """
    rule_sub = cell_dict.get("rule_detail", {}).get("subscores", {})
    llm_sub = cell_dict.get("llm_detail", {})
    flags = cell_dict.get("flags", [])

    # =========================================================
    # COMPONENT 1: PLAUSIBILITY (0-1)
    # =========================================================
    plausibility = 1.0
    plausibility -= rule_sub.get("override_compliance", 0) * 0.6
    plausibility -= rule_sub.get("catastrophization", 0) * 0.5
    plausibility -= rule_sub.get("praise_ratio", 0) * 0.4
    plausibility = max(plausibility, 0.0)

    llm_plaus = (llm_sub.get("plausibility", 3) - 1) / 4.0
    plausibility = 0.6 * plausibility + 0.4 * llm_plaus

    # =========================================================
    # COMPONENT 2: DISTINCTNESS (0-1)
    # =========================================================
    sim = rule_sub.get("similarity_to_gt", 0)
    rule_distinct = min((1.0 - sim) * 1.3, 1.0)
    llm_distinct = (llm_sub.get("distinctness", 3) - 1) / 4.0
    distinctness = 0.6 * rule_distinct + 0.4 * llm_distinct

    # =========================================================
    # COMPONENT 3: FLAW SIGNAL (0-1)
    # "Does the rejected have clear, identifiable flaws?"
    # Each flaw metric contributes with its own weight
    # =========================================================
    flaw_metrics = [
        (rule_sub.get("hedging_ratio", 0),         0.15),
        (rule_sub.get("formulaic", 0),              0.15),
        (rule_sub.get("checklist_coverage", 0),     0.12),
        (rule_sub.get("absence_evidence", 0),       0.12),
        (rule_sub.get("redundancy", 0),             0.10),
        (rule_sub.get("structural_repetition", 0),  0.10),
        (rule_sub.get("unclear_ratio", 0),          0.08),
        (rule_sub.get("two_bullet_per_cat", 0),     0.08),
        (rule_sub.get("length_inflation", 0),       0.05),
        (rule_sub.get("praise_ratio", 0),           0.05),
    ]
    flaw_signal = sum(val * w for val, w in flaw_metrics)
    flaw_signal = min(flaw_signal * 2.5, 1.0)

    llm_flaw = (llm_sub.get("flaw_clarity", 3) - 1) / 4.0
    flaw_signal_final = 0.7 * flaw_signal + 0.3 * llm_flaw

    # =========================================================
    # COMPONENT 4: GT SUPERIORITY (0-1)
    # =========================================================
    gt_spec = rule_sub.get("gt_specificity", 0)
    rej_spec = rule_sub.get("specificity", 0)
    if gt_spec > 0:
        spec_gap = min(gt_spec / max(rej_spec, 0.01), 3.0) / 3.0
    else:
        spec_gap = 0.5

    gt_points = rule_sub.get("gt_num_limitations", 1)
    rej_points = rule_sub.get("num_limitations", 1)
    if rej_points > gt_points * 2:
        inflation_signal = 0.7
    elif rej_points > gt_points:
        inflation_signal = 0.5
    else:
        inflation_signal = 0.3

    llm_gt_sup = (llm_sub.get("gt_superiority", 3) - 1) / 4.0
    gt_superiority = 0.3 * spec_gap + 0.3 * inflation_signal + 0.4 * llm_gt_sup

    # =========================================================
    # COMPONENT 5: DIFFICULTY (0-1)
    # Sweet spot: 1-2 flags, moderate difficulty
    # =========================================================
    flag_count = len(flags)
    severe_flags = [f for f in flags if f in DROP_FLAGS]

    if severe_flags:
        difficulty = 0.0
    elif flag_count == 0:
        difficulty = 0.6
    elif flag_count <= 2:
        difficulty = 0.8
    elif flag_count <= 4:
        difficulty = 0.4
    else:
        difficulty = 0.1

    llm_diff = (llm_sub.get("difficulty", 3) - 1) / 4.0
    difficulty_final = 0.6 * difficulty + 0.4 * llm_diff

    # =========================================================
    # UNIFIED COMPOSITE
    # Flaw signal matters most — it's the learning signal for DPO
    # =========================================================
    unified = (
        0.15 * plausibility +
        0.20 * distinctness +
        0.30 * flaw_signal_final +
        0.20 * gt_superiority +
        0.15 * difficulty_final
    )

    # Hard penalties
    if any(f in DROP_FLAGS for f in flags):
        unified = min(unified, 0.10)
    if "more_praise_than_criticism" in flags:
        unified = min(unified, 0.35)
    if "rejected_more_specific_than_gt" in flags:
        unified = min(unified, 0.30)

    if unified >= 0.65:
        verdict = "excellent"
    elif unified >= 0.50:
        verdict = "good"
    elif unified >= 0.35:
        verdict = "marginal"
    else:
        verdict = "drop"

    return {
        "unified_score": round(unified, 3),
        "breakdown": {
            "plausibility":   round(plausibility, 3),
            "distinctness":   round(distinctness, 3),
            "flaw_signal":    round(flaw_signal_final, 3),
            "gt_superiority": round(gt_superiority, 3),
            "difficulty":     round(difficulty_final, 3),
        },
        "verdict": verdict,
    }

# ============================================================
# DIVERSITY-AWARE TOP-K SELECTION
# ============================================================

def select_top_k_diverse(candidates: list, k: int = TOP_K_PER_ROW) -> list:
    """Pick top-K with diverse flaw_types and strategies."""
    if len(candidates) <= k:
        return candidates

    by_flaw = {}
    for c in candidates:
        flaw = c.get("flaw_type", "unknown")
        by_flaw.setdefault(flaw, []).append(c)

    for flaw in by_flaw:
        by_flaw[flaw].sort(key=lambda x: x["unified_score"], reverse=True)

    best_per_flaw = [items[0] for items in by_flaw.values()]
    best_per_flaw.sort(key=lambda x: x["unified_score"], reverse=True)

    if len(best_per_flaw) >= k:
        return best_per_flaw[:k]

    selected = best_per_flaw.copy()
    used_strategies = {c["strategy"] for c in selected}

    remaining = sorted(
        [c for c in candidates if c["strategy"] not in used_strategies],
        key=lambda x: x["unified_score"], reverse=True,
    )
    for c in remaining:
        if len(selected) >= k:
            break
        selected.append(c)

    if len(selected) < k:
        rest = sorted(
            [c for c in candidates if c not in selected],
            key=lambda x: x["unified_score"], reverse=True,
        )
        for c in rest:
            if len(selected) >= k:
                break
            selected.append(c)

    return selected

# ============================================================
# MAIN
# ============================================================

def build_dpo_pairs():
    print("=" * 70)
    print("DPO PAIR BUILDER — Unified Scoring")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    stats = {
        "total_rows": len(df), "rows_with_gt": 0, "rows_with_pairs": 0,
        "total_candidates": 0, "dropped_unscored": 0,
        "dropped_flags": 0, "dropped_low_score": 0, "final_pairs": 0,
    }

    all_pairs = []

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        gt_text = extract_gt_text(row.get(GROUND_TRUTH_COL, None))
        if not gt_text:
            continue
        stats["rows_with_gt"] += 1

        paper_raw = row.get(TEXT_COL, "")
        paper_text = "" if (paper_raw is None or (isinstance(paper_raw, float) and pd.isna(paper_raw))) else str(paper_raw).strip()[:6000]
        try:
            rating = float(row.get(RATING_COL, 999))
        except (ValueError, TypeError):
            rating = 999.0
        rating_label = "strong_accept" if rating <= 5.0 else "strong_reject"

        candidates = []
        for strategy_name, info in REJECTED_COLS.items():
            col_name = info["col"]
            if col_name not in df.columns:
                continue

            cell_dict = safe_parse(row.get(col_name, None))
            if not cell_dict or "rule_based_score" not in cell_dict:
                stats["dropped_unscored"] += 1
                continue

            stats["total_candidates"] += 1
            flags = cell_dict.get("flags", [])

            if any(f in DROP_FLAGS for f in flags):
                stats["dropped_flags"] += 1
                continue

            unified = compute_unified_score(cell_dict)

            if unified["verdict"] == "drop":
                stats["dropped_low_score"] += 1
                continue

            candidates.append({
                "strategy":         strategy_name,
                "failure_mode":     info["failure_mode"],
                "text":             cell_dict["text"],
                "unified_score":    unified["unified_score"],
                "verdict":          unified["verdict"],
                "breakdown":        unified["breakdown"],
                "flaw_type":        cell_dict.get("flaw_type", "unknown"),
                "rule_based_score": cell_dict.get("rule_based_score", 0),
                "llm_based_score":  cell_dict.get("llm_based_score", 0),
                "flags":            flags,
                "llm_reason":       cell_dict.get("llm_reason", ""),
            })

        if not candidates:
            continue

        selected = select_top_k_diverse(candidates, k=TOP_K_PER_ROW)
        stats["rows_with_pairs"] += 1

        for cand in selected:
            prompt = (
                f"[SYSTEM] {SYSTEM_PROMPT}\n\n"
                f"[USER] Identify all limitations of the following scientific paper.\n\n"
                f"=== PAPER CONTENT ===\n{paper_text}\n\n"
                f"=== TASK ===\n"
                f"List all limitations covering: novelty, methodology, experiments, "
                f"generalization, robustness, efficiency, clarity, reproducibility, "
                f"data quality, and ethical concerns."
            )
            pair = {
                "prompt":   prompt,
                "chosen":   gt_text,
                "rejected": cand["text"],
                "metadata": {
                    "row_idx":          row_idx,
                    "strategy":         cand["strategy"],
                    "failure_mode":     cand["failure_mode"],
                    "flaw_type":        cand["flaw_type"],
                    "unified_score":    cand["unified_score"],
                    "verdict":          cand["verdict"],
                    "score_breakdown":  cand["breakdown"],
                    "rule_based_score": cand["rule_based_score"],
                    "llm_based_score":  cand["llm_based_score"],
                    "rating":           rating,
                    "rating_label":     rating_label,
                    "flags":            cand["flags"],
                    "llm_reason":       cand["llm_reason"],
                },
            }
            all_pairs.append(pair)

        if (row_idx + 1) % 50 == 0:
            print(f"  Processed {row_idx+1}/{len(df)}, {len(all_pairs)} pairs")

    stats["final_pairs"] = len(all_pairs)

    # Save
    with open(OUTPUT_JSONL, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_pairs)} pairs → {OUTPUT_JSONL}")

    if all_pairs:
        stats["avg_unified_score"] = round(np.mean([p["metadata"]["unified_score"] for p in all_pairs]), 3)
        stats["avg_pairs_per_row"] = round(len(all_pairs) / max(stats["rows_with_pairs"], 1), 2)
        stats["strategy_distribution"] = dict(Counter(p["metadata"]["strategy"] for p in all_pairs))
        stats["flaw_type_distribution"] = dict(Counter(p["metadata"]["flaw_type"] for p in all_pairs))
        stats["verdict_distribution"] = dict(Counter(p["metadata"]["verdict"] for p in all_pairs))

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Rows with GT:        {stats['rows_with_gt']}")
    print(f"Rows with pairs:     {stats['rows_with_pairs']}")
    print(f"Total candidates:    {stats['total_candidates']}")
    print(f"  Dropped (unscored):{stats['dropped_unscored']}")
    print(f"  Dropped (flags):   {stats['dropped_flags']}")
    print(f"  Dropped (score):   {stats['dropped_low_score']}")
    print(f"Final DPO pairs:     {stats['final_pairs']}")
    print(f"Avg unified score:   {stats.get('avg_unified_score', 0)}")

    if all_pairs:
        print(f"\nBy strategy:")
        for s, c in Counter(p["metadata"]["strategy"] for p in all_pairs).most_common():
            print(f"  {s:<15} {c:>5}")
        print(f"\nBy flaw type:")
        for ft, c in Counter(p["metadata"]["flaw_type"] for p in all_pairs).most_common():
            print(f"  {ft:<20} {c:>5}")
        scores = [p["metadata"]["unified_score"] for p in all_pairs]
        print(f"\nScore distribution: min={min(scores):.3f} median={np.median(scores):.3f} max={max(scores):.3f}")

if __name__ == "__main__":
    build_dpo_pairs()