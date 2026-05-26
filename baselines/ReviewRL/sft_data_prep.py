# =============================================================================
# sft_data_prep.py
# -----------------------------------------------------------------------------
# Optional SFT-data preparation script (mirrors Section 3.3 of ReviewRL).
#
# ReviewRL warms up the policy with supervised fine-tuning on long-CoT review
# data derived from DeepReview-13k. We reproduce the same recipe locally so
# we can claim "we used Review-RL as a baseline (including its SFT warm-up
# stage)" in the paper.
#
# Source CSV (chosen by the user):
#   data/
#       not_balanced_data/df_not_bal_final_strat_samp.csv
#
# Required columns:
#   --sft-input-column     (default: "input_text_without_lim")
#       Paper body WITHOUT the gold limitation paragraph.
#   --sft-target-column    (default: tries "limitations", "lim", "gold_limitations")
#       Gold limitations paragraph (the SFT target).  If not present we fall
#       back to extracting "## Limitations" from a "review" / "full_review"
#       column, mirroring how DeepReview-13k stores meta-reviews.
#
# Output: a JSONL file in HuggingFace SFTTrainer format:
#       { "messages": [
#             {"role": "user",      "content": <prompt>},
#             {"role": "assistant", "content": "<think>...</think>\n## Limitations\n..."}
#         ] }
#
# This file is then consumed by sft_train.py.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import pandas as pd

from prompts import LIMITATION_GENERATION_PROMPT
from retrieval import build_retrieval_context

# Pull out a "## Limitations" / "## Weaknesses" section from a free-form review.
_LIM_FALLBACK_RE = re.compile(
    r"(?ms)^#{2,3}\s*(Limitations|Weaknesses)\s*\n(.*?)(?=\n#{2,}\s*\S+|\Z)"
)

def _resolve_target(row, sft_target_column: Optional[str]) -> str:
    """
    Return the gold limitation text for one row, trying several columns.
    """
    candidates = []
    if sft_target_column and sft_target_column in row.index:
        candidates.append(row[sft_target_column])

    # Common fallbacks that may exist in the user's CSVs.
    for c in ("limitations", "lim", "gold_limitations", "weaknesses"):
        if c in row.index:
            candidates.append(row[c])

    # Last resort: parse a "## Limitations" block out of a full review column.
    for c in ("review", "full_review", "meta_review"):
        if c in row.index and isinstance(row[c], str):
            m = _LIM_FALLBACK_RE.search(row[c])
            if m:
                candidates.append(m.group(2).strip())

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ""

def _wrap_target_as_review(text: str, thinking: str = "") -> str:
    """
    Wrap a plain limitations paragraph in the same <think>+## Limitations
    structure that the policy is asked to produce, so the SFT target
    matches the inference-time format exactly.
    """
    if not thinking:
        thinking = (
            "I will critically read the paper section by section, look for "
            "methodological gaps, missing comparisons to prior work, "
            "experimental shortcuts, and writing issues, then summarise "
            "them as a list of [Major]/[Minor] limitations."
        )

    # Make sure the target already starts with the section header.
    text = text.strip()
    if not re.match(r"^#{2,3}\s*(Limitations|Weaknesses)", text):
        # Add a header and best-effort bullet formatting.
        lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
        bulleted = "\n".join(f"- [Major] {ln}" for ln in lines if ln)
        text = "## Limitations\n" + (bulleted or "- [Major] " + text)
    return f"<think>\n{thinking}\n</think>\n\n{text}"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SFT JSONL for the ReviewRL-style limitation baseline."
    )
    p.add_argument(
        "--sft-input-csv",
        type=str,
        default="data/not_balanced_data/df_not_bal_final_strat_samp.csv",
    )
    p.add_argument("--sft-input-column", type=str, default="input_text_without_lim")
    p.add_argument(
        "--sft-target-column",
        type=str,
        default=None,
        help="Column with the gold limitations paragraph (auto-detected if omitted).",
    )
    p.add_argument(
        "--sft-cited-in-text-column", type=str, default="cited_in_text"
    )
    p.add_argument(
        "--sft-cited-in-ret-column", type=str, default="cited_in_ret"
    )
    p.add_argument("--no-citations", action="store_true")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Optional number of rows to keep for SFT (default: all).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=str,
        default="review_rl/sft_data",
    )
    p.add_argument("--output-name", type=str, default="reviewrl_sft.jsonl")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.sft_input_csv)
    if args.num_rows is not None:
        end = min(args.start + args.num_rows, len(df))
        df = df.iloc[args.start:end].reset_index(drop=True)
    elif args.start:
        df = df.iloc[args.start:].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    print(f"[sft_data_prep] kept {len(df)} rows from {args.sft_input_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    n_kept, n_skipped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            paper = str(row.get(args.sft_input_column, "") or "").strip()
            if not paper:
                n_skipped += 1
                continue

            target = _resolve_target(row, args.sft_target_column)
            if not target:
                n_skipped += 1
                continue

            cited_in_text_val = (
                None if args.no_citations
                else row.get(args.sft_cited_in_text_column, None)
            )
            cited_in_ret_val = (
                None if args.no_citations
                else row.get(args.sft_cited_in_ret_column, None)
            )
            context = build_retrieval_context(
                cited_in_text=cited_in_text_val,
                cited_in_ret=cited_in_ret_val,
                use_citations=(not args.no_citations),
            )

            prompt = LIMITATION_GENERATION_PROMPT.format(
                paper=paper, context=context
            )
            assistant = _wrap_target_as_review(target)

            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[sft_data_prep] wrote {n_kept} examples (skipped {n_skipped}) -> {out_path}")

if __name__ == "__main__":
    main()