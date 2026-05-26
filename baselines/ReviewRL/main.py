# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# CLI entry point for the ReviewRL-style limitation-generation baseline.
#
# Example invocation (from a .pbs file):
#
#   python review_rl/main.py \
#       --model qwen \
#       --model-id qwen2_5_3b_instruct \
#       --cache-dir qwen2_5_3b_instruct \
#       --input-csv data/balanced_data/df_updated_with_retrieval.csv \
#       --text-column input_text_cleaned \
#       --cited-in-text-column cited_in_text \
#       --cited-in-ret-column  cited_in_ret \
#       --output-dir review_rl \
#       --output-name reviewrl_qwen_limitations.csv \
#       --start 0 \
#       --end 200 \
#       --save-every 10
#
# To skip the 3-query generation step (faster) add --skip-query-step.
# To run the "w/o Retrieval" ablation add --no-citations.
# To use OpenAI instead of Qwen pass --model gpt-4o-mini (requires OPENAI_API_KEY).
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys

# We're in review_rl
# (or wherever this file lives), so make sure sibling files import cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_input_csv  # noqa: E402
from models import make_generator      # noqa: E402
from pipeline import run_pipeline      # noqa: E402

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReviewRL-style limitation-generation baseline"
    )

    # ---- Model selection -----------------------------------------------------
    p.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "gpt-4o-mini"],
        help="Which backend to use. 'qwen' runs locally; 'gpt-4o-mini' calls the OpenAI API.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="qwen2_5_3b_instruct",
        help="HuggingFace model id or local path (Qwen backend only).",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="qwen2_5_3b_instruct",
        help="HF cache directory (Qwen backend only).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the Qwen model.",
    )
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision for the Qwen model. bf16 fits in 40GB GPU comfortably.",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=24000,
        help="Hard cap on the user-prompt token count fed to Qwen.",
    ) 

    p.add_argument(
        "--lora-adapter",
        type=str,
        default="",
        help=(
            "Optional path to a LoRA adapter (the output of sft_train.py, "
            "typically <sft-output-dir>/final). When provided we load the "
            "base model first, attach the adapter, and merge_and_unload() "
            "so inference uses the SFT-tuned weights."
        ),
    )

    # ---- Data ----------------------------------------------------------------
    p.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the inference CSV (e.g. df_updated_with_retrieval.csv).",
    )
    p.add_argument(
        "--text-column",
        type=str,
        default="input_text_cleaned",
        help="Column with the paper body to review.",
    )
    p.add_argument(
        "--cited-in-text-column",
        type=str,
        default="cited_in_text",
        help="Column with in-text citation snippets.",
    )
    p.add_argument(
        "--cited-in-ret-column",
        type=str,
        default="cited_in_ret",
        help="Column with retrieved related-work text (e.g. from OpenAlex).",
    )
    p.add_argument(
        "--no-citations",
        action="store_true",
        help="Run the 'w/o Retrieval' ablation: ignore both citation columns.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="First row index (inclusive) to process.",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index (exclusive); default processes all rows.",
    )

    # ---- Output --------------------------------------------------------------
    p.add_argument(
        "--output-dir",
        type=str,
        default="review_rl",
        help="Directory where the result CSV (and checkpoints) are written.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="reviewrl_limitations.csv",
        help="Filename for the result CSV inside --output-dir.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the dataframe to disk every N processed rows.",
    )

    # ---- Generation hyper-params --------------------------------------------
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for both Qwen and OpenAI backends.",
    )
    p.add_argument(
        "--max-new-tokens-review",
        type=int,
        default=1024,
        help="Max new tokens for the limitation-generation step.",
    )
    p.add_argument(
        "--max-new-tokens-query",
        type=int,
        default=256,
        help="Max new tokens for the 3-query generation step.",
    )
    p.add_argument(
        "--skip-query-step",
        action="store_true",
        help="Skip the Section-3.2 'generate 3 queries' step (faster).",
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()

    # ---- Sanity print so the .pbs log shows the config we're running with ---
    print("=" * 78)
    print("ReviewRL-style limitation-generation baseline")
    print("=" * 78)
    for k, v in vars(args).items():
        print(f"  {k:>26s} : {v}")
    print("=" * 78, flush=True)

    # ---- Load CSV -----------------------------------------------------------
    df = load_input_csv(
        input_csv=args.input_csv,
        text_column=args.text_column,
        cited_in_text_column=None if args.no_citations else args.cited_in_text_column,
        cited_in_ret_column=None if args.no_citations else args.cited_in_ret_column,
        start=args.start,
        end=args.end,
    )
    print(f"[main] loaded {len(df)} rows from {args.input_csv} "
          f"(rows {args.start}..{args.end})", flush=True)

    # ---- Build generator ----------------------------------------------------
    print(f"[main] building generator backend = {args.model} ...", flush=True)
    generator = make_generator(args)
    print("[main] generator ready.", flush=True)

    # ---- Run pipeline -------------------------------------------------------
    df_out = run_pipeline(
        df=df,
        generator=generator,
        text_column=args.text_column,
        cited_in_text_column=None if args.no_citations else args.cited_in_text_column,
        cited_in_ret_column=None if args.no_citations else args.cited_in_ret_column,
        use_citations=(not args.no_citations),
        output_dir=args.output_dir,
        output_name=args.output_name,
        save_every=args.save_every,
        skip_query_step=args.skip_query_step,
        max_new_tokens_review=args.max_new_tokens_review,
        max_new_tokens_query=args.max_new_tokens_query,
        temperature=args.temperature,
    )

    print(f"[main] pipeline finished. final shape = {df_out.shape}")

if __name__ == "__main__":
    main()