"""
main.py
-------
Entry point for the AI-Scientist limitation-generation baseline.

This script
  * loads a CSV of papers,
  * runs each paper through the AI-Scientist reviewer pipeline
    (initial review -> reflection -> optional ensembling),
  * extracts ONLY the "Limitations" field from each review,
  * appends new columns to the dataframe and saves it back to disk.

Example PBS invocation
----------------------
python main.py \\
    --model gpt-4o-mini \\
    --no-citations \\
    --input-csv /path/to/df_updated_with_retrieval.csv \\
    --text-column input_text_cleaned \\
    --output-dir /path/to/theAIScientists \\
    --start 0 \\
    --end 200 \\
    --save-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# Local imports (these files live next to main.py).
from llm_client import LLMClient
from reviewer import perform_limitation_review
from utils import build_citation_block

# ---------------------------------------------------------------------------
# Argument parsing ----------------------------------------------------------
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-Scientist limitation-generation baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model selection ------------------------------------------------
    p.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Either an OpenAI model id (e.g. 'gpt-4o-mini') or any string "
             "for a local Qwen run (paired with --qwen-model-path).",
    )
    p.add_argument(
        "--qwen-model-path",
        type=str,
        default="qwen2_5_3b_instruct",
        help="Local path to the Qwen weights (used when --model is not OpenAI).",
    )
    p.add_argument(
        "--qwen-cache-dir",
        type=str,
        default="qwen2_5_3b_instruct",
        help="HuggingFace cache directory for Qwen.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype for the local model.",
    )
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var).",
    )

    # ---- Data -----------------------------------------------------------
    p.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the input CSV.",
    )
    p.add_argument(
        "--text-column",
        type=str,
        default="input_text_cleaned",
        help="Column that holds the paper body text.",
    )
    p.add_argument(
        "--citation-text-column",
        type=str,
        default="cited_in_text",
        help="Column with citations from the paper itself (optional).",
    )
    p.add_argument(
        "--citation-retrieved-column",
        type=str,
        default="cited_in_ret",
        help="Column with retrieved abstracts of cited works (optional).",
    )
    p.add_argument(
        "--no-citations",
        action="store_true",
        help="Ignore citation columns even when they are present.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="First row index to process (inclusive).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index to process (exclusive).",
    )

    # ---- Output ---------------------------------------------------------
    p.add_argument(
        "--output-dir",
        type=str,
        default="theAIScientists",
        help="Directory to write the augmented CSV (and trace JSONs) to.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV filename (auto-generated from model+timestamp if omitted).",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the dataframe to disk every N processed rows.",
    )
    p.add_argument(
        "--save-traces",
        action="store_true",
        help="Also save per-row reviewer traces (large, JSON files).",
    )

    # ---- Reviewer hyper-parameters (mirroring AI-Scientist) -------------
    p.add_argument(
        "--num-reflections",
        type=int,
        default=2,
        help="How many self-reflection rounds (>=1, AI-Scientist default = 2).",
    )
    p.add_argument(
        "--num-reviews-ensemble",
        type=int,
        default=1,
        help="How many parallel reviewers to ensemble before the meta-review.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sampling temperature (AI-Scientist default = 0.75).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens per LLM call.",
    )
    p.add_argument(
        "--max-paper-chars",
        type=int,
        default=80_000,
        help="Truncate paper text past this many characters (defensive, GPT-4o-mini "
             "and Qwen 2.5 both accept long contexts but extreme outliers blow memory).",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _build_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv:
        csv_path = out_dir / args.output_csv
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = args.model.replace("/", "_")
        csv_path = out_dir / f"ai_scientist_limitations_{safe_model}_{args.start}_{args.end}_{ts}.csv"

    traces_dir = out_dir / "traces"
    if args.save_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)

    return csv_path, traces_dir

def _ensure_columns(df: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    """Add the result columns up-front so we don't get dtype-mismatch warnings."""
    new_cols = ["ai_scientist_limitations", "ai_scientist_meta_review_json", "ai_scientist_error"]
    for i in range(n_agents):
        new_cols.append(f"ai_scientist_agent{i+1}_limitations")
        new_cols.append(f"ai_scientist_agent{i+1}_review_json")
    for c in new_cols:
        if c not in df.columns:
            df[c] = ""
    return df

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    print("[main] args:")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")

    # Load CSV ------------------------------------------------------------
    df = pd.read_csv(args.input_csv)
    print(f"[main] loaded {len(df)} rows from {args.input_csv}")

    # Make sure the columns we need exist.
    if args.text_column not in df.columns:
        raise KeyError(f"text column '{args.text_column}' not in CSV; have {df.columns.tolist()}")

    df = _ensure_columns(df, args.num_reviews_ensemble)

    end = min(args.end, len(df))
    print(f"[main] processing rows [{args.start}, {end})")

    csv_path, traces_dir = _build_output_paths(args)
    print(f"[main] writing to {csv_path}")

    # Initialise LLM ------------------------------------------------------
    client = LLMClient(
        model=args.model,
        qwen_model_path=args.qwen_model_path,
        qwen_cache_dir=args.qwen_cache_dir,
        openai_api_key=args.openai_api_key,
        dtype=args.dtype,
    )

    # Main loop -----------------------------------------------------------
    processed = 0
    t0 = time.time()
    for idx in range(args.start, end):
        row = df.iloc[idx]
        paper_text = str(row.get(args.text_column, "") or "")
        if not paper_text.strip():
            print(f"[row {idx}] empty paper text, skipping.")
            df.at[idx, "ai_scientist_error"] = "empty_text"
            continue

        # Defensive truncation to avoid OOM on extreme outliers.
        if len(paper_text) > args.max_paper_chars:
            paper_text = paper_text[: args.max_paper_chars] + "\n... [truncated]"

        # Citations (optional) -------------------------------------------
        if args.no_citations:
            citation_block = ""
        else:
            citation_block = build_citation_block(
                cited_in_text=row.get(args.citation_text_column, ""),
                cited_in_ret=row.get(args.citation_retrieved_column, ""),
            )

        try:
            print(f"[row {idx}] running reviewer (paper={len(paper_text)} chars)...")
            t_row = time.time()
            result = perform_limitation_review(
                client=client,
                paper_text=paper_text,
                citation_block=citation_block,
                num_reflections=args.num_reflections,
                num_reviews_ensemble=args.num_reviews_ensemble,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            dt = time.time() - t_row
            print(f"[row {idx}] done in {dt:.1f}s; {len(result['limitations'])} chars of limitations.")

            # Write result columns -------------------------------------
            df.at[idx, "ai_scientist_limitations"] = result["limitations"]
            df.at[idx, "ai_scientist_meta_review_json"] = json.dumps(
                result["meta_review"], ensure_ascii=False
            )
            for i, (lim, full) in enumerate(
                zip(result["agent_limitations"], result["agent_full_reviews"])
            ):
                df.at[idx, f"ai_scientist_agent{i+1}_limitations"] = lim
                df.at[idx, f"ai_scientist_agent{i+1}_review_json"] = json.dumps(
                    full, ensure_ascii=False
                )

            # Optionally dump the full reflection trace --------------------
            if args.save_traces:
                trace_path = traces_dir / f"row_{idx}.json"
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "row_index": int(idx),
                            "limitations": result["limitations"],
                            "agent_limitations": result["agent_limitations"],
                            "agent_full_reviews": result["agent_full_reviews"],
                            "meta_review": result["meta_review"],
                            "reflection_traces": result["reflection_traces"],
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            print(f"[row {idx}] ERROR: {err}")
            traceback.print_exc()
            df.at[idx, "ai_scientist_error"] = err

        processed += 1

        # Periodic save ----------------------------------------------------
        if processed % args.save_every == 0:
            df.to_csv(csv_path, index=False)
            elapsed = time.time() - t0
            print(
                f"[main] saved checkpoint after {processed} rows "
                f"({elapsed:.1f}s elapsed) -> {csv_path}"
            )

    # Final save ----------------------------------------------------------
    df.to_csv(csv_path, index=False)
    print(f"[main] DONE. Final CSV at {csv_path}.  Total time: {time.time()-t0:.1f}s.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
    