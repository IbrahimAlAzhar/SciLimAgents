"""
DPO LLM-Based Scorer (Step 2 of 2)
=====================================
Reads the rule-scored CSV (output of rule_scorer.py), calls GPT-4o to
score each pair, and adds LLM keys to the existing dict in each cell:
  {
    "text": "...",
    "rule_based_score": ...,       ← already exists from step 1
    "flags": [...],                ← already exists from step 1
    "rule_detail": {...},          ← already exists from step 1
    "llm_based_score": float,      ← NEW (added by this script)
    "final_score": float,          ← NEW (0.3*rule + 0.7*llm)
    "flaw_type": str,              ← NEW
    "llm_reason": str,             ← NEW
    "llm_detail": {...},           ← NEW
  }

Checkpoints every 10 scored cells. Skips cells that already have llm_based_score.
"""

import os
import json
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

# This should be the OUTPUT of rule_scorer.py
INPUT_CSV  = "/path/to/df_rule_scored.csv"       # <-- UPDATE
OUTPUT_CSV = "/path/to/df_fully_scored.csv"       # <-- UPDATE

TEXT_COL         = "input_text_cleaned"
RATING_COL       = "mean_rating"
GROUND_TRUTH_COL = "ground_truth_lim_peer"

LLM_MODEL = "gpt-4o"  # Use gpt-4o for better differentiation (gpt-4o-mini gives flat scores)

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

RULE_WEIGHT = 0.3
LLM_WEIGHT  = 0.7

# ============================================================
# CELL HELPERS
# ============================================================

def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s

def parse_cell(val):
    """Parse a cell value. Returns dict if JSON, None otherwise."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "text" in d:
            return d
        return None
    except (json.JSONDecodeError, ValueError):
        return None

def extract_gt_text(val):
    """Get plain GT text from either raw string or JSON dict."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    parsed = parse_cell(val)
    if parsed and "text" in parsed:
        return parsed["text"]
    if len(s) >= 50:
        return s
    return None

def has_llm_score(val):
    """Check if cell already has LLM score (for resume)."""
    d = parse_cell(val)
    return d is not None and "llm_based_score" in d

# ============================================================
# LLM SCORER
# ============================================================

LLM_SCORER_PROMPT = """You are a DPO training data quality judge. You will compare a GROUND TRUTH
limitation review (the "chosen" output for DPO) against a REJECTED candidate
and score how useful this pair is for DPO training.

SCORING CRITERIA:
A GOOD rejected sample must be plausible enough to not be a trivial shortcut,
meaningfully different from GT, and clearly worse in identifiable ways.

IMPORTANT — COMPARE CAREFULLY:
- Read the GT first. Note its specific claims, evidence, and focus areas.
- Then read the REJECTED. Ask: does it identify the SAME real issues? Does it
  add substance, or just pad with generic observations?
- A rejected output that covers 10-15 categories with shallow "may not"
  observations is WORSE than a GT with 3 deep, evidence-grounded points.
- Pay attention to WHAT KIND of flaw the rejected has — each strategy
  produces different failure patterns.

FAILURE PATTERNS (look for these):
- "formulaic": mechanically covers every category with template sentences
- "generic": vague language, no paper-specific details or section references
- "hedged/sycophantic": buries criticism in praise ("do an excellent job, but...")
- "hallucinated": references specific paper elements but evidence is fabricated
- "shallow": many points but each is 1 sentence with no depth
- "redundant": 2 sub-bullets per category that restate the same idea
- "catastrophized": exaggerated negativity, verdict phrases
- "faithful_but_weak": real issues but less specific/deep than GT

Score each dimension 1-5:
- PLAUSIBILITY (1=template/gibberish/praise, 5=reads like real expert review)
- DISTINCTNESS (1=nearly identical to GT, 5=completely different content/structure)
- FLAW_CLARITY (1=unclear why it's worse, 5=obvious specific flaw pattern)
- GT_SUPERIORITY (1=rejected might actually be better, 5=GT clearly superior)
- DIFFICULTY (1=trivially bad, anyone would reject, 5=subtle, hard to distinguish)

Classify PRIMARY flaw type (pick ONE):
"generic" | "hedged" | "hallucinated" | "shallow" | "redundant" | "sycophantic" |
"formulaic" | "catastrophized" | "faithful_but_weak" | "none"

Output EXACTLY this format (no extra text):
PLAUSIBILITY: <int 1-5>
DISTINCTNESS: <int 1-5>
FLAW_CLARITY: <int 1-5>
GT_SUPERIORITY: <int 1-5>
DIFFICULTY: <int 1-5>
FLAW_TYPE: <string>
REASON: <one sentence explaining the key difference between GT and rejected>"""

def call_llm_scorer(chosen: str, rejected: str, strategy: str,
                    paper_snippet: str, rating_label: str,
                    rule_flags: list) -> dict:
    """
    Call LLM to score a pair.
    Returns {"score": 0-1, "subscores": {...}, "flaw_type": str, "reason": str}
    """
    # Include rule flags as hint so LLM can validate/refine
    flag_str = ", ".join(rule_flags) if rule_flags else "none"

    user_msg = f"""GROUND_TRUTH (chosen):
{chosen[:3000]}

REJECTED (candidate):
{rejected[:3000]}

STRATEGY that generated rejected: {strategy}
PAPER_LABEL: {rating_label}
RULE_ENGINE_FLAGS: {flag_str}

PAPER (first 1000 chars):
{paper_snippet[:1000]}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": LLM_SCORER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content.strip()

        print(f"    [LLM | {strategy}]: {text[:250]}")

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

        # Compute normalized 0-1 score
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
            raw = sum(result[d] * weights[d] for d in dims)
            score = (raw - 1.0) / 4.0
        else:
            available = {d: result[d] for d in dims if d in result}
            if available:
                tw = sum(weights[d] for d in available)
                raw = sum(available[d] * (weights[d] / tw) for d in available)
                score = (raw - 1.0) / 4.0
            else:
                score = 0.0
            print(f"    [WARNING] Partial parse: {list(available.keys())}")

        score = max(0.0, min(score, 1.0))

        result.setdefault("flaw_type", "unknown")
        result.setdefault("reason", "")

        subscores = {d: result.get(d, 0) for d in dims}

        print(f"    [LLM parsed | {strategy}] score={score:.3f} flaw={result['flaw_type']} "
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
        print(f"    LLM error: {e}")
        return {
            "score": 0.0,
            "subscores": {},
            "flaw_type": "unknown",
            "reason": f"llm_error: {e}",
        }

# ============================================================
# MAIN
# ============================================================

def run_llm_scorer():
    print("=" * 70)
    print("DPO LLM-BASED SCORER (Step 2 of 2)")
    print(f"Using model: {LLM_MODEL}")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    # --- Cache GT texts ---
    gt_texts = {}
    for i in range(len(df)):
        gt_text = extract_gt_text(df.iloc[i].get(GROUND_TRUTH_COL, None))
        if gt_text and len(gt_text) >= 50:
            gt_texts[i] = gt_text

    print(f"Valid GT rows: {len(gt_texts)}")

    # --- Score each perturbed column ---
    total_scored = 0
    total_skipped = 0

    for strategy_name, info in REJECTED_COLS.items():
        col_name = info["col"]
        print(f"\n{'='*50}")
        print(f"LLM Scoring: {col_name} ({strategy_name})")
        print(f"{'='*50}")

        if col_name not in df.columns:
            print(f"  Column not found, skipping.")
            continue

        scored_this_col = 0
        skipped_this_col = 0

        for i in tqdm(range(len(df)), desc=f"  {strategy_name}"):
            if i not in gt_texts:
                total_skipped += 1
                skipped_this_col += 1
                continue

            gt = gt_texts[i]
            cell_raw = df.iloc[i].get(col_name, None)

            # Skip if already has LLM score (resume)
            if has_llm_score(cell_raw):
                continue

            # Parse the rule-scored dict
            cell_dict = parse_cell(cell_raw)
            if cell_dict is None or "text" not in cell_dict:
                # Not rule-scored yet (NaN, PENDING, or raw text) — skip
                total_skipped += 1
                skipped_this_col += 1
                continue

            rejected = cell_dict["text"]
            rule_flags = cell_dict.get("flags", [])
            rule_score = cell_dict.get("rule_based_score", 0.0)

            paper_text = safe_str(df.iloc[i].get(TEXT_COL, ""))[:6000]
            try:
                rating = float(df.iloc[i].get(RATING_COL, 999))
            except (ValueError, TypeError):
                rating = 999.0
            label = "strong_accept" if rating <= 5.0 else "strong_reject"

            # --- Call LLM ---
            llm_result = call_llm_scorer(
                chosen=gt,
                rejected=rejected,
                strategy=strategy_name,
                paper_snippet=paper_text,
                rating_label=label,
                rule_flags=rule_flags,
            )

            # --- Add LLM scores to existing dict ---
            cell_dict["llm_based_score"] = llm_result["score"]
            cell_dict["final_score"]     = round(
                RULE_WEIGHT * rule_score + LLM_WEIGHT * llm_result["score"], 3
            )
            cell_dict["flaw_type"]  = llm_result["flaw_type"]
            cell_dict["llm_reason"] = llm_result.get("reason", "")
            cell_dict["llm_detail"] = llm_result["subscores"]

            # Write back
            df.at[i, col_name] = json.dumps(cell_dict, ensure_ascii=False)

            total_scored += 1
            scored_this_col += 1

            if scored_this_col % 10 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"    [CHECKPOINT] {scored_this_col} scored ({strategy_name}, row {i})")

            time.sleep(0.5)  # rate limit (gpt-4o is slower)

        print(f"  {strategy_name}: scored={scored_this_col}, skipped={skipped_this_col}")

    # --- Final save ---
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*70}")
    print(f"DONE. LLM scored: {total_scored}, skipped: {total_skipped}")
    print(f"Saved → {OUTPUT_CSV}")
    print(f"\nEach perturbed column cell now has keys:")
    print(f"  text, rule_based_score, llm_based_score, final_score,")
    print(f"  flaw_type, llm_reason, flags, rule_detail, llm_detail")
    print(f"\nNext: run dpo_pair_builder.py to select top-K per row.")

if __name__ == "__main__":
    run_llm_scorer()