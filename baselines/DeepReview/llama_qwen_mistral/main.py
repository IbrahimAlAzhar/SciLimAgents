"""
main.py
=======
Entry point for the DeepReview-baseline limitation-generation pipeline.

Runs the four DeepReview-style stages SEQUENTIALLY on every row of a CSV
and checkpoints the augmented DataFrame every `--save-every` rows.

Compatible with TWO layouts:
    A) Packaged layout :  /.../deepreview/deepreview_baseline/{config,...}.py
    B) Flat layout     :  /.../deepreview/{config,...}.py
                          (this is what triggered the user's earlier
                           ModuleNotFoundError -- handled below)
"""

from __future__ import annotations

import os
import sys
import time
import json
import traceback
from datetime import datetime

# ---- Path bootstrap ------------------------------------------------------
# Make sure both `deepreview_baseline` (package) AND its individual modules
# (flat) are importable regardless of where the user dropped the files.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for p in (_PARENT_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- Tolerant imports ----------------------------------------------------
try:
    from deepreview_baseline.config import parse_args
    from deepreview_baseline.models import build_llm
    from deepreview_baseline.pipeline import DeepReviewLimitationPipeline
    from deepreview_baseline.data_utils import (
        load_papers, save_dataframe, ensure_output_columns,
        merge_citations, truncate_paper, NEW_COLUMNS,
    )
except ImportError:                                 # flat layout fallback
    from config import parse_args                                       # type: ignore
    from models import build_llm                                        # type: ignore
    from pipeline import DeepReviewLimitationPipeline                   # type: ignore
    from data_utils import (                                            # type: ignore
        load_papers, save_dataframe, ensure_output_columns,
        merge_citations, truncate_paper, NEW_COLUMNS,
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
    print(f"  backend          : {args.backend}")
    print(f"  mode             : {args.mode}")
    print(f"  reviewer_num     : {args.reviewer_num}")
    print(f"  rows             : [{args.start}, {args.end})")
    print(f"  save_every       : {args.save_every}")
    print(f"  no_citations     : {args.no_citations}")
    print(f"  store_format     : {args.store_format}")
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
    # 2. Build LLM (heavy; do this once)
    # ------------------------------------------------------------------
    print(f"[llm] building backend '{args.model}/{args.backend}' ...")
    t0 = time.time()
    llm = build_llm(args)
    print(f"[llm] backend ready in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 3. Build the pipeline
    # ------------------------------------------------------------------
    pipeline = DeepReviewLimitationPipeline(llm, args)

    # ------------------------------------------------------------------
    # 4. Iterate rows (sequential — DeepReview's pipeline is sequential)
    # ------------------------------------------------------------------
    output_path = os.path.join(args.output_dir, _output_filename(args))
    print(f"[run] writing checkpoints to {output_path}")
    sys.stdout.flush()

    use_citations = (not args.no_citations) and (args.mode == "best")

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