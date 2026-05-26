"""
main.py
=======
Entry point for the DeepReview-baseline limitation-generation pipeline.

It:
    1. Parses CLI arguments (see config.py).
    2. Loads the input CSV and slices rows [start:end].
    3. Builds the LLM backend (Qwen via vLLM OR gpt-4o-mini via OpenAI).
    4. Runs the 4-stage DeepReview pipeline on every row.
    5. Periodically checkpoints the augmented DataFrame to disk
       (every `--save-every` rows).

Example .pbs invocation:

    python \
deepreview/main.py \
        --model qwen \
        --input-csv df_updated_with_retrieval_upd.csv \
        --text-column input_text_cleaned \
        --cited-text-column cited_in_text_abs \
        --cited-retrieval-column cited_in_ret \
        --output-dir deepreview \
        --start 0 --end  \
        --save-every 10 \
        --mode best \
        --reviewer-num 4
"""

from __future__ import annotations

import os
import sys
import time
import json
import traceback
from datetime import datetime

# Make the package importable when this file is run as a script
# (i.e. `python .../deepreview_baseline/main.py`).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from deepreview.config import parse_args                        # noqa: E402
from deepreview.models import build_llm                         # noqa: E402
from deepreview.pipeline import DeepReviewLimitationPipeline    # noqa: E402
from deepreview.data_utils import (                             # noqa: E402
    load_papers,
    save_dataframe,
    ensure_output_columns,
    merge_citations,
    truncate_paper,
    NEW_COLUMNS,
)

def _output_filename(args) -> str:
    """Build the checkpoint CSV filename from the arguments."""
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    return (
        f"deepreview_limitations_{args.model}_{args.mode}_"
        f"r{args.reviewer_num}_s{args.start}_e{args.end}{suffix}.csv"
    )

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("=" * 78)
    print("DeepReview baseline -- limitation generation only")
    print(f"  model            : {args.model}")
    print(f"  mode             : {args.mode}")
    print(f"  reviewer_num     : {args.reviewer_num}")
    print(f"  rows             : [{args.start}, {args.end})")
    print(f"  save_every       : {args.save_every}")
    print(f"  no_citations     : {args.no_citations}")
    print(f"  input_csv        : {args.input_csv}")
    print(f"  output_dir       : {args.output_dir}")
    print("=" * 78)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 1. Load CSV slice
    # ------------------------------------------------------------------
    df = load_papers(args.input_csv, args.start, args.end, args.text_column)
    df = ensure_output_columns(df)
    print(f"[load] {len(df)} rows loaded "
          f"(columns: {list(df.columns)[:8]}...)")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 2. Build LLM (Qwen via vLLM is heavy; only do this once)
    # ------------------------------------------------------------------
    print(f"[llm] building backend '{args.model}' ...")
    t0 = time.time()
    llm = build_llm(args)
    print(f"[llm] backend ready in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 3. Build the pipeline
    # ------------------------------------------------------------------
    pipeline = DeepReviewLimitationPipeline(llm, args)

    # ------------------------------------------------------------------
    # 4. Iterate rows
    # ------------------------------------------------------------------
    output_path = os.path.join(args.output_dir, _output_filename(args))
    print(f"[run] writing checkpoints to {output_path}")
    sys.stdout.flush()

    use_citations = not args.no_citations and args.mode == "best"

    for local_i, (row_idx, row) in enumerate(df.iterrows(), start=1):
        global_idx = int(row.get("_orig_index", row_idx))
        t_row = time.time()

        # ----- inputs
        paper_text = truncate_paper(
            row[args.text_column], args.max_input_chars
        )
        if use_citations:
            citations = merge_citations(
                row.get(args.cited_text_column),
                row.get(args.cited_retrieval_column),
                args.max_citation_chars,
            )
        else:
            citations = ""

        # ----- run pipeline (never raises)
        try:
            result = pipeline.run(paper=paper_text, citations=citations)
        except Exception as e:                       # safety net
            traceback.print_exc()
            result = {col: "" for col in NEW_COLUMNS}
            result["deepreview_status"] = f"fatal: {type(e).__name__}: {e}"

        # ----- write back into the DataFrame
        for col in NEW_COLUMNS:
            df.at[row_idx, col] = result.get(col, "")

        # ----- (optional) per-row trace
        if args.save_raw_traces:
            trace_path = os.path.join(
                args.output_dir, "traces", f"row_{global_idx}.json"
            )
            with open(trace_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {"global_index": global_idx, **result},
                    fp, ensure_ascii=False, indent=2,
                )

        elapsed = time.time() - t_row
        print(
            f"[run] {local_i}/{len(df)} (orig idx={global_idx}) "
            f"status={result['deepreview_status']} "
            f"in {elapsed:.1f}s"
        )
        sys.stdout.flush()

        # ----- checkpoint every `save_every` rows
        if local_i % args.save_every == 0:
            save_dataframe(df, args.output_dir, _output_filename(args))
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[ckpt] {ts}  saved checkpoint after {local_i} rows")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # 5. Final save
    # ------------------------------------------------------------------
    final_path = save_dataframe(df, args.output_dir, _output_filename(args))
    print(f"[done] wrote final CSV -> {final_path}")

if __name__ == "__main__":
    main() 