"""
config.py
=========
Argument parser and global configuration for the DeepReview-baseline
limitation-generation pipeline.

Open-weight backends supported (all fit on a single 40 GB GPU):
    --model qwen     -> data/.../qwen2_5_3b_instruct
    --model llama    -> meta-llama/Meta-Llama-3-8B-Instruct
    --model mistral  -> mistralai/Mistral-7B-Instruct-v0.3
Closed:
    --model gpt-4o-mini

Inference backends:
    --backend transformers  (default; no vLLM dependency)
    --backend vllm          (faster; requires `pip install vllm`)
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
        choices=["qwen", "llama", "mistral", "gpt-4o-mini"],
        help="Backbone LLM family.",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference engine for open-weight models. "
             "'transformers' has no vLLM dependency and is the safe default.",
    )

    # ----- per-family open-weight paths
    p.add_argument(
        "--qwen-model-id",
        type=str,
        default="qwen2_5_3b_instruct",
    )
    p.add_argument(
        "--qwen-cache-dir",
        type=str,
        default="qwen2_5_3b_instruct",
    )
    p.add_argument(
        "--llama-model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    p.add_argument(
        "--llama-cache-dir",
        type=str,
        default="llama3_8b_instruct",
    )
    p.add_argument(
        "--mistral-model-id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    p.add_argument(
        "--mistral-cache-dir",
        type=str,
        default="models/mistral_7b_v3_instruct",
    )

    # ----- shared open-weight knobs
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM only. Fraction of GPU memory vLLM may use.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context window (input + output).",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM only. Number of GPUs for tensor parallelism.",
    )
    p.add_argument(
        "--hf-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Weight dtype for the transformers backend.",
    )

    # ----- OpenAI specific
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
    p.add_argument("--max-new-tokens-stage", type=int, default=700,
                   help="Max new tokens for each intermediate stage call.")
    p.add_argument("--max-new-tokens-final", type=int, default=700,
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
    p.add_argument(
        "--store-format",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Format used to store list-typed outputs in CSV cells. "
             "'json' = json.dumps(list); 'text' = newline-separated bullet list.",
    )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=14000,
        help="Truncate paper text to this many characters before prompting.",
    )
    p.add_argument(
        "--max-citation-chars",
        type=int,
        default=3500,
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
        help="Random seed for sampling.",
    )

    return p

def parse_args() -> argparse.Namespace:
    """Parse command line arguments and create the output directory."""
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_raw_traces:
        os.makedirs(os.path.join(args.output_dir, "traces"), exist_ok=True)
    return args 