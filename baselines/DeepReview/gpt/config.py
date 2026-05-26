"""
config.py
=========
Argument parser and global configuration for the DeepReview-baseline
limitation-generation pipeline.

The defaults are tuned so the whole pipeline (Qwen-2.5-3B-Instruct loaded
through vLLM) fits comfortably inside a single 40 GB GPU.

This script is intended to be imported by `main.py`.  All paths and knobs
that you would normally set inside a .pbs script are exposed here.
"""

import argparse
import os

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse.ArgumentParser used by `main.py`."""
    p = argparse.ArgumentParser(
        description=(
            "DeepReview baseline (paper review limitation generation only). "
            "Mimics the multi-stage 'Review-with-Thinking' framework "
            "(Novelty-Verification -> Multi-dim Reviewers -> Reliability "
            "Verification -> Meta-Review) but produces ONLY the final "
            "weaknesses / limitations."
        )
    )

    # ------------------------------------------------------------------
    # I/O paths
    # ------------------------------------------------------------------
    p.add_argument(
        "--input-csv",
        type=str,
        default=(
            ""
            "data/balanced_data/df_updated_with_retrieval_upd.csv"
        ),
        help="Path to the CSV file containing papers to process.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=(
            ""
            "deepreview"
        ),
        help="Directory where outputs (CSV checkpoints + raw JSON traces) are saved.",
    )
    p.add_argument(
        "--text-column",
        type=str,
        default="input_text_cleaned",
        help="Column in the CSV that contains the paper text used as input.",
    )
    p.add_argument(
        "--cited-text-column",
        type=str,
        default="cited_in_text_abs",
        help="Column with in-text citation abstracts (can be empty/NaN/'No citations found').",
    )
    p.add_argument(
        "--cited-retrieval-column",
        type=str,
        default="cited_in_ret",
        help="Column with retrieved citation passages from OpenAlex "
             "(can be empty/NaN/'No citations found').",
    )

    # ------------------------------------------------------------------
    # Row range
    # ------------------------------------------------------------------
    p.add_argument("--start", type=int, default=0,
                   help="First row index (inclusive) to process.")
    p.add_argument("--end", type=int, default=None,
                   help="Last row index (exclusive); default processes all rows.")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save checkpoint CSV every N processed rows.")

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    p.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "gpt-4o-mini"],
        help="Backbone LLM. 'qwen' = local Qwen-2.5-3B-Instruct via vLLM. "
             "'gpt-4o-mini' = OpenAI API (requires OPENAI_API_KEY env var).",
    )
    # Qwen specific
    p.add_argument(
        "--qwen-model-id",
        type=str,
        default="qwen2_5_3b_instruct",
        help="Local path / HF id for the Qwen model.",
    )
    p.add_argument(
        "--qwen-cache-dir",
        type=str,
        default="qwen2_5_3b_instruct",
        help="Cache dir for the Qwen tokenizer/model.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory vLLM is allowed to use (40 GB safe default).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=12288,
        help="Maximum model context length (input+output) for vLLM.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (1 fits in 40 GB).",
    )
    # OpenAI specific
    p.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name when --model=gpt-4o-mini.",
    )
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not set, falls back to OPENAI_API_KEY env var.",
    )

    # ------------------------------------------------------------------
    # Generation parameters (mirrors DeepReview defaults)
    # ------------------------------------------------------------------
    p.add_argument("--temperature", type=float, default=0.4,
                   help="Sampling temperature (DeepReview uses 0.4).")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens-stage", type=int, default=900,
                   help="Max new tokens for each intermediate stage call.")
    p.add_argument("--max-new-tokens-final", type=int, default=900,
                   help="Max new tokens for the final meta-review call.")

    # ------------------------------------------------------------------
    # Pipeline controls (mirror DeepReview's modes)
    # ------------------------------------------------------------------
    p.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["fast", "standard", "best"],
        help="Reasoning depth: fast=skip z1+z3, standard=z2+z3, "
             "best=z1+z2+z3 (uses citation columns).",
    )
    p.add_argument(
        "--reviewer-num",
        type=int,
        default=4,
        help="Number of simulated reviewers in Stage 2 (R in the paper).",
    )
    p.add_argument(
        "--no-citations",
        action="store_true",
        help="Force-disable citation use even in 'best' mode.",
    )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=18000,
        help="Truncate paper text to this many characters before prompting "
             "(prevents context blow-up when using a 12k-token Qwen model).",
    )
    p.add_argument(
        "--max-citation-chars",
        type=int,
        default=4000,
        help="Truncate concatenated citation context to this many characters.",
    )
    p.add_argument(
        "--save-raw-traces",
        action="store_true",
        help="If set, save per-row raw JSON traces under <output_dir>/traces/.",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix added to the output CSV filename.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for vLLM sampling.",
    )

    return p

def parse_args() -> argparse.Namespace:
    """Parse command line arguments and create the output directory."""
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_raw_traces:
        os.makedirs(os.path.join(args.output_dir, "traces"), exist_ok=True)
    return args 
