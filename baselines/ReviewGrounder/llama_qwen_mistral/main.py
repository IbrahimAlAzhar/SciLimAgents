"""
main.py
-------

Limitation-generation pipeline for a CSV of papers.

Adapts the ReviewGrounder multi-agent grounding architecture
(Anonymous, ACL 2026) to focus *only* on limitation generation:

   Drafter --> {InsightMiner, ResultsAnalyzer, RelatedWorkAnalyzer} (parallel)
                                                                |
                                                                v
                                                            Refiner

For each paper row in the CSV:
  - read paper text from --text-column
  - read citation context from --cited-text-column and --retrieval-column
  - skip the related-work agent if --no-citations is set OR both citation
    columns are missing/empty/'No citations found'
  - write the JSON output of every agent into new columns
  - save the dataframe to disk every --save-every rows (also at the end)

Example invocation (.pbs / shell):

    python main.py \\
        --model gpt-4o-mini \\
        --no-citations \\
        --input-csv /path/to/df_updated_with_retrieval.csv \\
        --text-column input_text_cleaned \\
        --output-dir /path/to/agentreview \\
        --start 0 \\
        --end 200 \\
        --save-every 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

# agents.py / llm_client.py / prompts.py are expected to sit next to main.py.
# We add this directory to sys.path so the script works no matter where it's
# launched from (e.g. from an HPC scheduler).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agents import (  # noqa: E402
    DrafterAgent,
    InsightMinerAgent,
    ResultsAnalyzerAgent,
    RelatedWorkAnalyzerAgent,
    RefinerAgent,
)
from llm_client import make_client  # noqa: E402

# ===========================================================================
# Output column names. We persist every agent's JSON output so the pipeline
# is fully traceable / auditable from the CSV alone.
# ===========================================================================
COL_DRAFT_JSON      = "lim_initial_draft_json"
COL_DRAFT_RAW       = "lim_initial_draft_raw"
COL_METHOD_JSON     = "lim_method_grounding_json"
COL_RESULTS_JSON    = "lim_results_grounding_json"
COL_RW_JSON         = "lim_related_work_grounding_json"
COL_FINAL_JSON      = "lim_final_json"
COL_FINAL_TEXT      = "lim_final_text"      # human-readable rendering
COL_STATUS          = "lim_status"          # success / error / skipped_*
COL_ERROR           = "lim_error"
COL_PROCESSED_AT    = "lim_processed_at"

NEW_COLS = [
    COL_DRAFT_JSON, COL_DRAFT_RAW, COL_METHOD_JSON, COL_RESULTS_JSON,
    COL_RW_JSON, COL_FINAL_JSON, COL_FINAL_TEXT,
    COL_STATUS, COL_ERROR, COL_PROCESSED_AT,
]

# ===========================================================================
# Argparse
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paper limitations using a ReviewGrounder-style "
                    "multi-agent pipeline (drafter + 3 grounding agents + refiner).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- IO ----
    p.add_argument("--input-csv", required=True, type=str,
                   help="Input CSV path.")
    p.add_argument("--output-dir", required=True, type=str,
                   help="Directory to write outputs into (created if missing).")
    p.add_argument("--output-filename", type=str,
                   default="limitations_generated.csv",
                   help="Filename inside --output-dir.")

    # ---- Columns ----
    p.add_argument("--text-column", type=str, default="input_text_cleaned",
                   help="Column with the paper text.")
    p.add_argument("--cited-text-column", type=str, default="cited_in_text",
                   help="Column with citation context extracted from the paper.")
    p.add_argument("--retrieval-column", type=str, default="cited_in_ret",
                   help="Column with retrieved related-work text from OpenAlex.")

    # ---- Row range ----
    p.add_argument("--start", type=int, default=0,
                   help="Start row index (inclusive).")
    p.add_argument("--end", type=int, default=None,
                   help="End row index (exclusive).")

    # ---- Save / resume ----
    p.add_argument("--save-every", type=int, default=10,
                   help="Save the dataframe to disk after every N processed rows.")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="If output CSV exists, skip rows whose lim_status == 'success'.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Disable resume; reprocess every row in the range.")

    # ---- Citations ----
    p.add_argument("--no-citations", action="store_true",
                   help="Globally skip the related-work agent (do not use citation columns).")

    # ---- LLM backend selection ----
    p.add_argument("--backend", type=str, default="openai",
                   choices=["openai", "hf"],
                   help="Which LLM backend to use. "
                        "'openai' = OpenAI Chat Completions API. "
                        "'hf' = local HuggingFace transformers (Llama / Mistral / Qwen).")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                   help="Backend-specific model identifier. "
                        "For openai: 'gpt-4o-mini', 'gpt-4o', etc. "
                        "For hf: HF repo id ('meta-llama/Meta-Llama-3-8B-Instruct') "
                        "or a local path to a downloaded model.")
    p.add_argument("--output-format", type=str, default=None,
                   choices=["json", "text", "auto"],
                   help="Format the agents demand from the LLM. "
                        "'auto' (default) = json for openai, text for hf. "
                        "'text' = structured plain text -- use this for smaller "
                        "open-source instruct models which are unreliable at JSON.")

    # ---- Sampling ----
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max output tokens per LLM call.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Temperature for the drafter and grounding agents.")
    p.add_argument("--refiner-temperature", type=float, default=0.0,
                   help="Temperature for the refiner (low for consistency).")

    # ---- OpenAI-specific ----
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                   help="(openai backend) Env var holding the OpenAI API key.")
    p.add_argument("--llm-timeout", type=int, default=120,
                   help="(openai backend) HTTP timeout per call (seconds).")
    p.add_argument("--llm-max-retries", type=int, default=4,
                   help="(openai backend) Max retries on transient errors.")

    # ---- HF-specific ----
    p.add_argument("--cache-dir", type=str, default=None,
                   help="(hf backend) HuggingFace cache directory. "
                        "Equivalent to setting HF_HOME / TRANSFORMERS_CACHE.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="(hf backend) torch dtype. bfloat16 is best on A100/H100.")
    p.add_argument("--device", type=str, default="cuda",
                   help="(hf backend) Device for inference (cuda / cpu).")
    p.add_argument("--device-map", type=str, default="auto",
                   help="(hf backend) accelerate device_map (auto / cuda:0 / ...).")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="(hf backend) Pass trust_remote_code=True to HF loaders.")
    p.add_argument("--hf-token-env", type=str, default="HUGGING_FACE_HUB_TOKEN",
                   help="(hf backend) Env var holding the HF token (for gated repos).")
    p.add_argument("--attn-implementation", type=str, default=None,
                   help="(hf backend) Optional attn impl ('flash_attention_2', 'sdpa').")

    # ---- Prompt sizing ----
    p.add_argument("--max-content-chars", type=int, default=12000,
                   help="Truncate paper text to this many characters before the LLM call.")
    p.add_argument("--max-citation-chars", type=int, default=6000,
                   help="Truncate each citation column to this many characters.")

    # ---- Concurrency ----
    p.add_argument("--parallel-grounding", dest="parallel_grounding",
                   action="store_true", default=True,
                   help="Run the 3 grounding agents in parallel (default).")
    p.add_argument("--no-parallel-grounding", dest="parallel_grounding",
                   action="store_false",
                   help="Run grounding agents sequentially (easier to debug).")

    # ---- Logging ----
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=str, default=None,
                   help="Optional path to also write logs to a file.")

    return p.parse_args()

# ===========================================================================
# Helpers
# ===========================================================================

# Sentinel strings that should be treated the same as a missing value, in
# addition to actual NaN/None/empty. Easy to extend.
MISSING_SENTINELS = {
    "no citations found",
    "no_citations_found",
    "none found",
    "n/a",
    "na",
}

def is_missing(val: Any) -> bool:
    """True if `val` should be treated as missing (NaN, None, empty, or a
    user-defined sentinel string)."""
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    s = str(val).strip()
    if not s:
        return True
    low = s.lower()
    if low in {"nan", "none", "null"}:
        return True
    if low in MISSING_SENTINELS:
        return True
    return False

def truncate(text: str, max_chars: int) -> str:
    """Hard character cap. We keep the head of the text -- that's where the
    title/abstract/intro live in this dataset."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def setup_logging(level: str, log_file: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Quiet noisy libs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def render_draft_for_prompt(draft_parsed: Dict[str, Any]) -> str:
    """Convert the drafter's JSON output into a compact text block to use as
    `candidate_limitations` for downstream agents."""
    items = (draft_parsed or {}).get("limitations", []) or []
    if not items:
        return "(empty draft)"
    parts = []
    for i, it in enumerate(items, 1):
        cat = it.get("category", "uncategorized")
        desc = it.get("description", "")
        rationale = it.get("rationale", "")
        parts.append(f"{i}. [{cat}] {desc}\n   rationale: {rationale}")
    return "\n".join(parts)

def render_final_text(final_parsed: Dict[str, Any]) -> str:
    """Render final JSON to plain-text for easy human consumption from CSV."""
    items = (final_parsed or {}).get("final_limitations", []) or []
    if not items:
        return ""
    lines = []
    summary = final_parsed.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
        lines.append("")
    for i, it in enumerate(items, 1):
        cat = it.get("category", "uncategorized")
        sev = it.get("severity", "")
        desc = it.get("description", "")
        ev = it.get("evidence", "")
        head = f"{i}. [{cat}" + (f" / {sev}" if sev else "") + f"] {desc}"
        lines.append(head)
        if ev:
            lines.append(f"   evidence: {ev}")
    return "\n".join(lines)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add output columns if missing. Existing values are preserved (this is
    what makes --resume work)."""
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = ""
    return df

# ===========================================================================
# Per-row pipeline
# ===========================================================================

def process_row(
    args: argparse.Namespace,
    row: pd.Series,
    drafter: DrafterAgent,
    insight: InsightMinerAgent,
    results: ResultsAnalyzerAgent,
    rw: RelatedWorkAnalyzerAgent,
    refiner: RefinerAgent,
) -> Dict[str, Any]:
    """Run the full pipeline on a single row.

    Returns a dict of {column_name: value} that the caller writes back to df.
    Never raises; catastrophic errors are encoded into the status/error cols.
    """
    out: Dict[str, Any] = {c: "" for c in NEW_COLS}
    out[COL_PROCESSED_AT] = datetime.now(timezone.utc).isoformat()

    # -- 0) Validate paper text --------------------------------------------
    paper_text_raw = row.get(args.text_column)
    if is_missing(paper_text_raw):
        out[COL_STATUS] = "skipped_no_paper_text"
        out[COL_ERROR] = f"Column '{args.text_column}' is empty or missing."
        return out
    paper_text = truncate(str(paper_text_raw), args.max_content_chars)

    # -- Resolve citation context ------------------------------------------
    cited_in_text_raw = row.get(args.cited_text_column)
    cited_in_ret_raw = row.get(args.retrieval_column)
    citations_available = (
        (not is_missing(cited_in_text_raw))
        or (not is_missing(cited_in_ret_raw))
    )
    use_citations = (not args.no_citations) and citations_available

    cited_in_text = (
        truncate(str(cited_in_text_raw), args.max_citation_chars)
        if not is_missing(cited_in_text_raw) else ""
    )
    cited_in_ret = (
        truncate(str(cited_in_ret_raw), args.max_citation_chars)
        if not is_missing(cited_in_ret_raw) else ""
    )

    # -- 1) Drafter --------------------------------------------------------
    try:
        draft = drafter.generate(paper_text=paper_text)
    except Exception as e:  # noqa: BLE001
        out[COL_STATUS] = "error"
        out[COL_ERROR] = f"drafter failed: {e}"
        return out
    out[COL_DRAFT_JSON] = json.dumps(draft.get("parsed", {}), ensure_ascii=False)
    out[COL_DRAFT_RAW] = draft.get("raw", "")
    candidate_text = render_draft_for_prompt(draft.get("parsed", {}))

    # -- 2/3/4) Three grounding agents (parallel by default) ---------------
    method_out  = {"parsed": {}, "raw": "", "ok": False}
    results_out = {"parsed": {}, "raw": "", "ok": False}
    rw_out      = {"parsed": {}, "raw": "", "ok": False, "skipped": False}
    grounding_errors: list[str] = []

    def run_method() -> Dict[str, Any]:
        return insight.generate(paper_text=paper_text,
                                candidate_limitations=candidate_text)

    def run_results() -> Dict[str, Any]:
        return results.generate(paper_text=paper_text,
                                candidate_limitations=candidate_text)

    def run_rw() -> Dict[str, Any]:
        # Skip this agent entirely if we don't have any usable citation
        # context (or the user passed --no-citations). Returning a
        # well-formed empty payload keeps downstream code simple.
        if not use_citations:
            return {
                "parsed": {"related_work_limitations": []},
                "raw": "",
                "ok": True,
                "skipped": True,
            }
        r = rw.generate(
            paper_text=paper_text,
            candidate_limitations=candidate_text,
            cited_in_text=cited_in_text,
            cited_in_ret=cited_in_ret,
        )
        r["skipped"] = False
        return r

    if args.parallel_grounding:
        # The 3 agents are independent -> run concurrently for speed.
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(run_method): "method",
                ex.submit(run_results): "results",
                ex.submit(run_rw): "rw",
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    res = fut.result()
                    if name == "method":
                        method_out = res
                    elif name == "results":
                        results_out = res
                    else:
                        rw_out = res
                except Exception as e:  # noqa: BLE001
                    grounding_errors.append(f"{name}: {e}")
    else:
        # Sequential mode -- handy when debugging.
        try:
            method_out = run_method()
        except Exception as e:  # noqa: BLE001
            grounding_errors.append(f"method: {e}")
        try:
            results_out = run_results()
        except Exception as e:  # noqa: BLE001
            grounding_errors.append(f"results: {e}")
        try:
            rw_out = run_rw()
        except Exception as e:  # noqa: BLE001
            grounding_errors.append(f"rw: {e}")

    out[COL_METHOD_JSON]  = json.dumps(method_out.get("parsed", {}),  ensure_ascii=False)
    out[COL_RESULTS_JSON] = json.dumps(results_out.get("parsed", {}), ensure_ascii=False)
    out[COL_RW_JSON]      = json.dumps(rw_out.get("parsed", {}),      ensure_ascii=False)

    # Note skipped-RW in the error column (status will still be 'success' if
    # everything else went fine; this is just informative).
    notes: list[str] = []
    if rw_out.get("skipped"):
        notes.append("rw_skipped_flag" if args.no_citations else "rw_skipped_no_citations")
    if grounding_errors:
        notes.append("grounding_errors=" + " | ".join(grounding_errors))

    # -- 5) Refiner --------------------------------------------------------
    try:
        final = refiner.generate(
            paper_text=paper_text,
            candidate_limitations=candidate_text,
            insight_miner_json=out[COL_METHOD_JSON],
            results_analyzer_json=out[COL_RESULTS_JSON],
            related_work_json=out[COL_RW_JSON],
        )
    except Exception as e:  # noqa: BLE001
        out[COL_STATUS] = "error"
        out[COL_ERROR] = "; ".join(notes + [f"refiner failed: {e}"])
        return out

    out[COL_FINAL_JSON] = json.dumps(final.get("parsed", {}), ensure_ascii=False)
    out[COL_FINAL_TEXT] = render_final_text(final.get("parsed", {}))
    out[COL_STATUS] = "success" if not grounding_errors else "partial_success"
    out[COL_ERROR] = "; ".join(notes)
    return out

# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    log = logging.getLogger("limagents")

    # ---- Output dir + path ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_filename

    # ---- Load input CSV ----
    log.info("Reading input CSV: %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    log.info("Loaded %d rows; total columns: %d", len(df), len(df.columns))

    # Required column.
    if args.text_column not in df.columns:
        log.error("--text-column '%s' not found in input.", args.text_column)
        sys.exit(1)
    # Optional citation columns -- create empty if missing so downstream code
    # can read them without conditional logic.
    if args.cited_text_column not in df.columns:
        log.warning("'%s' not found; will treat as missing for every row.",
                    args.cited_text_column)
        df[args.cited_text_column] = ""
    if args.retrieval_column not in df.columns:
        log.warning("'%s' not found; will treat as missing for every row.",
                    args.retrieval_column)
        df[args.retrieval_column] = ""

    # ---- Resume: prefer existing output if present ----
    if args.resume and out_path.exists():
        log.info("--resume enabled and %s exists; loading prior outputs.", out_path)
        try:
            prior = pd.read_csv(out_path)
            if len(prior) == len(df):
                df = prior
            else:
                log.warning(
                    "Prior output has %d rows, input has %d; merging by index.",
                    len(prior), len(df),
                )
                for c in NEW_COLS:
                    if c in prior.columns:
                        df[c] = prior[c].reindex(df.index)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load prior output (%s); starting fresh.", e)

    df = ensure_columns(df)

    # ---- Resolve output format ('auto' picks json for openai, text for hf)
    output_format = args.output_format
    if output_format in (None, "auto"):
        output_format = "json" if args.backend == "openai" else "text"
    log.info("Resolved output_format=%s for backend=%s", output_format, args.backend)

    # ---- Auto-disable parallel grounding for HF backend ------------------
    # A single GPU + single model instance can't actually run 3 generate()
    # calls in parallel; ThreadPoolExecutor would just serialize them and
    # is not safe with HF model state. So force-sequential for backend=hf.
    if args.backend == "hf" and args.parallel_grounding:
        log.info("backend=hf -- forcing --no-parallel-grounding (sequential agents).")
        args.parallel_grounding = False

    # ---- Build LLM client ------------------------------------------------
    if args.backend == "openai":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            log.error(
                "Env var %s is not set. Export it (e.g. `export OPENAI_API_KEY=sk-...`) "
                "before running, or pass --api-key-env to point at a different var.",
                args.api_key_env,
            )
            sys.exit(2)
        client = make_client(
            backend="openai",
            model=args.model,
            api_key=api_key,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
    else:  # backend == "hf"
        hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
        # If --cache-dir is set, also propagate to env so child loaders agree.
        if args.cache_dir:
            os.environ.setdefault("HF_HOME", args.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
        client = make_client(
            backend="hf",
            model=args.model,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            hf_token=hf_token,
            attn_implementation=args.attn_implementation,
        )

    # ---- Build agents (all share the same client + output_format) --------
    agent_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_format=output_format,
    )
    drafter = DrafterAgent(client, **agent_kwargs)
    insight = InsightMinerAgent(client, **agent_kwargs)
    results = ResultsAnalyzerAgent(client, **agent_kwargs)
    rw      = RelatedWorkAnalyzerAgent(client, **agent_kwargs)
    refiner = RefinerAgent(
        client,
        max_tokens=args.max_tokens,
        temperature=args.refiner_temperature,
        output_format=output_format,
    )

    # ---- Row range ----
    n = len(df)
    start = max(0, args.start)
    end = min(n, args.end if args.end is not None else n)
    if end <= start:
        log.error("No rows to process (start=%d, end=%d, n=%d).", start, end, n)
        sys.exit(0)

    log.info(
        "Processing rows [%d, %d) | backend=%s | model=%s | output_format=%s | "
        "parallel_grounding=%s | no_citations=%s | save_every=%d | resume=%s",
        start, end, args.backend, args.model, output_format,
        args.parallel_grounding, args.no_citations, args.save_every, args.resume,
    )

    # ---- Main loop ----
    processed_since_save = 0
    success_count = 0
    error_count = 0
    skipped_count = 0

    pbar = tqdm(range(start, end), desc="rows", unit="row")
    for idx in pbar:
        row = df.iloc[idx]

        # Resume: skip rows we already finished successfully.
        if args.resume and str(row.get(COL_STATUS, "")).strip() == "success":
            skipped_count += 1
            continue

        t0 = time.time()
        try:
            out = process_row(args, row, drafter, insight, results, rw, refiner)
        except Exception as e:  # noqa: BLE001
            # process_row itself handles errors, but be defensive.
            log.exception("Row %d failed catastrophically.", idx)
            out = {c: "" for c in NEW_COLS}
            out[COL_STATUS] = "error"
            out[COL_ERROR] = f"unhandled: {e}"
            out[COL_PROCESSED_AT] = datetime.now(timezone.utc).isoformat()

        # Write back into the dataframe.
        for col, val in out.items():
            df.at[idx, col] = val

        if out[COL_STATUS] == "success":
            success_count += 1
        elif out[COL_STATUS] in ("partial_success",):
            success_count += 1  # still produced final limitations
        else:
            error_count += 1
        processed_since_save += 1

        pbar.set_postfix({
            "ok":   success_count,
            "err":  error_count,
            "skip": skipped_count,
            "row_s": f"{time.time() - t0:.1f}",
        })

        # Periodic save -- protects against long-run failures on the cluster.
        if args.save_every > 0 and processed_since_save >= args.save_every:
            df.to_csv(out_path, index=False)
            log.info("Saved partial results -> %s (after %d rows)",
                     out_path, processed_since_save)
            processed_since_save = 0

    # ---- Final save ----
    df.to_csv(out_path, index=False)
    log.info(
        "Done. Final save -> %s. ok=%d, err=%d, resumed_skip=%d.",
        out_path, success_count, error_count, skipped_count,
    )

if __name__ == "__main__":
    main() 