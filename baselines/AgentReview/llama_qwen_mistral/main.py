"""
main.py
=======
Entry point for AgentReview-Limitations.

Three backends are supported via --provider:

  --provider openai       (default; uses OPENAI_API_KEY)
  --provider azure        (uses AZURE_ENDPOINT / AZURE_OPENAI_KEY)
  --provider hf           (local Hugging Face model via transformers)

Examples
--------

  # OpenAI (unchanged)
  python main.py --model gpt-4o-mini

  # Llama 3 8B locally
  python main.py \
      --provider hf \
      --hf-model-id meta-llama/Meta-Llama-3-8B-Instruct \
      --hf-cache-dirllama3_8b_instruct \
      --hf-dtype bf16 \
      --max-paper-chars 12000 --max-tokens 1024

  # Mistral 7B v0.3 locally
  python main.py \
      --provider hf \
      --hf-model-id mistralai/Mistral-7B-Instruct-v0.3 \
      --hf-cache-dir mistral_7b_v3_instruct \
      --hf-dtype bf16

  # Qwen 2.5 3B from a local checkpoint
  python main.py \
      --provider hf \
      --hf-model-id qwen2_5_3b_instruct \
      --hf-cache-dir qwen2_5_3b_instruct \
      --hf-dtype bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from typing import List

from config import (
    ExperimentConfig, LLMConfig,
    ReviewerProfile, AreaChairProfile,
)
from data_loader import CSVDataLoader
from pipeline import AgentReviewLimitationPipeline

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agentreview_limitations")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AgentReview-Limitations: multi-agent LLM pipeline that "
                    "produces a list of limitations for each scientific paper."
    )
    # ---- Dataset / I/O ----
    p.add_argument(
        "--input-csv",
        default=ExperimentConfig.__dataclass_fields__["input_csv"].default,
        help="Path to the input CSV.",
    )
    p.add_argument("--text-column", default="input_text_cleaned")
    p.add_argument("--citations-column", default="cited_in_text")
    p.add_argument("--id-column", default=None)
    p.add_argument("--output-dir", default="./outputs/limitations_run")

    # ---- Range / checkpointing ----
    p.add_argument("--start", type=int, default=0,
                   help="Index of first row to process.")
    p.add_argument("--end", type=int, default=None,
                   help="Index *after* the last row to process.")
    p.add_argument("--save-every", type=int, default=5)

    # ---- Pipeline behaviour ----
    p.add_argument("--num-reviewers", type=int, default=3)
    p.add_argument("--no-rebuttal", action="store_true",
                   help="Disable Phases II & III.")
    p.add_argument("--no-citations", action="store_true",
                   help="Ignore the cited_in_text column.")
    p.add_argument("--max-paper-chars", type=int, default=18000,
                   help="Truncation budget for paper text. Use ~12000 for "
                        "Llama-3-8B (8K context).")
    p.add_argument("--max-citation-chars", type=int, default=4000)

    # ---- Reviewer profile axes ----
    p.add_argument("--reviewer-knowledge", nargs="+",
                   default=["knowledgeable", "knowledgeable", "knowledgeable"],
                   choices=["knowledgeable", "unknowledgeable"])
    p.add_argument("--reviewer-commit", nargs="+",
                   default=["responsible", "responsible", "responsible"],
                   choices=["responsible", "irresponsible"])
    p.add_argument("--reviewer-intent", nargs="+",
                   default=["benign", "benign", "benign"],
                   choices=["benign", "malicious"])
    p.add_argument("--ac-style",
                   default="inclusive",
                   choices=["inclusive", "authoritarian", "conformist"])

    # ---- Backend selection ----
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"),
                   choices=["openai", "azure", "hf"])
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o"),
                   help="OpenAI/Azure model name (ignored when --provider hf).")

    # ---- Hugging Face local ----
    p.add_argument("--hf-model-id", default=os.getenv("HF_MODEL_ID", ""),
                   help="HF hub id (e.g. meta-llama/Meta-Llama-3-8B-Instruct) "
                        "or a local checkpoint path. Required if --provider hf.")
    p.add_argument("--hf-cache-dir", default=os.getenv("HF_CACHE_DIR", ""),
                   help="HF cache directory.")
    p.add_argument("--hf-dtype", default=os.getenv("HF_DTYPE", "bf16"),
                   choices=["bf16", "fp16", "fp32"],
                   help="Model precision (bf16 recommended for A100/H100).")
    p.add_argument("--hf-device", default=os.getenv("HF_DEVICE", "auto"),
                   help='"auto" | "cuda" | "cuda:0" | "cpu".')

    # ---- Generation params ----
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=1500,
                   help="OpenAI: max_tokens. HF: max_new_tokens.")
    p.add_argument("--top-p", type=float, default=1.0)
    return p

def _build_configs(ns: argparse.Namespace):
    """Translate parsed CLI args into LLMConfig + ExperimentConfig."""
    n = ns.num_reviewers
    for axis_name, axis_vals in (
        ("--reviewer-knowledge", ns.reviewer_knowledge),
        ("--reviewer-commit",    ns.reviewer_commit),
        ("--reviewer-intent",    ns.reviewer_intent),
    ):
        if len(axis_vals) < n:
            raise ValueError(
                f"{axis_name} expects at least {n} values "
                f"(one per reviewer), got {len(axis_vals)}."
            )

    reviewers: List[ReviewerProfile] = [
        ReviewerProfile(
            name=f"R{i + 1}",
            knowledgeability=ns.reviewer_knowledge[i],
            commitment=ns.reviewer_commit[i],
            intention=ns.reviewer_intent[i],
        )
        for i in range(n)
    ]

    exp_cfg = ExperimentConfig(
        input_csv=ns.input_csv,
        text_column=ns.text_column,
        citations_column=ns.citations_column,
        id_column=ns.id_column,
        output_dir=ns.output_dir,
        num_reviewers=n,
        enable_rebuttal=not ns.no_rebuttal,
        use_citation_context=not ns.no_citations,
        max_paper_chars=ns.max_paper_chars,
        max_citation_chars=ns.max_citation_chars,
        reviewers=reviewers,
        area_chair=AreaChairProfile(style=ns.ac_style),
        start_index=ns.start,
        end_index=ns.end,
        save_every=ns.save_every,
        verbose=True,
    )

    # When using HF, --model is mostly cosmetic; --hf-model-id is the source of truth.
    if ns.provider == "hf" and not ns.hf_model_id:
        raise ValueError(
            "--provider hf requires --hf-model-id (HF hub id or local path)."
        )

    llm_cfg = LLMConfig(
        provider=ns.provider,
        model=ns.model,
        temperature=ns.temperature,
        max_tokens=ns.max_tokens,
        top_p=ns.top_p,
        hf_model_id=ns.hf_model_id,
        hf_cache_dir=ns.hf_cache_dir,
        hf_dtype=ns.hf_dtype,
        hf_device=ns.hf_device,
    )
    return exp_cfg, llm_cfg

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main() -> None:
    args = _build_argparser().parse_args()
    exp_cfg, llm_cfg = _build_configs(args)

    os.makedirs(exp_cfg.output_dir, exist_ok=True)

    # Persist run config for reproducibility
    cfg_dump_path = os.path.join(exp_cfg.output_dir, "run_config.json")
    with open(cfg_dump_path, "w", encoding="utf-8") as f:
        json.dump(
            {"experiment": _safe_asdict(exp_cfg),
             "llm": _safe_asdict(
                 llm_cfg,
                 drop_keys=("openai_api_key", "azure_api_key"),
             )},
            f, indent=2, default=str,
        )
    logger.info("Wrote run config -> %s", cfg_dump_path)

    loader = CSVDataLoader(exp_cfg)
    pipeline = AgentReviewLimitationPipeline(exp_cfg=exp_cfg, llm_cfg=llm_cfg)

    results_path = os.path.join(exp_cfg.output_dir, "results.jsonl")
    summary_path = os.path.join(exp_cfg.output_dir, "summary.csv")

    n_total = len(loader)
    logger.info("Will process %d paper(s).", n_total)

    summary_rows = []
    with open(results_path, "a", encoding="utf-8") as out_f:
        for i, record in enumerate(loader.iter_records(), start=1):
            logger.info("[%d/%d] Processing paper %s ...", i, n_total,
                        record.paper_id)
            result = pipeline.run_one(
                paper_id=record.paper_id,
                paper_text=record.text,
                citation_text=record.citations,
            )

            row = {
                "paper_id": record.paper_id,
                "n_final_limitations": len(result.final_limitations),
                "final_limitations_text": _flatten_limitations(result.final_limitations),
                "initial_reviewer_limitations": json.dumps(result.initial_reviewer_limitations),
                "author_rebuttal": result.author_rebuttal,
                "final_reviewer_limitations": json.dumps(result.final_reviewer_limitations),
                **(record.extra or {}),
            }
            summary_rows.append(row)

            out_f.write(json.dumps(result.to_dict(), default=str) + "\n")
            out_f.flush()

            if i % exp_cfg.save_every == 0:
                _write_summary(summary_path, summary_rows)
                logger.info("  Checkpoint summary saved (%d rows).", len(summary_rows))

    _write_summary(summary_path, summary_rows)
    logger.info("Done. Detailed results -> %s", results_path)
    logger.info("       Summary CSV    -> %s", summary_path)

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def _flatten_limitations(items: list) -> str:
    if not items:
        return ""
    parts = []
    for j, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        cat = it.get("category", "Other")
        text = it.get("limitation", "").strip()
        if text:
            parts.append(f"{j}. [{cat}] {text}")
    return "\n".join(parts)

def _write_summary(path: str, rows: list) -> None:
    if not rows:
        return
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)

def _safe_asdict(obj, drop_keys: tuple = ()) -> dict:
    d = asdict(obj)
    for k in drop_keys:
        d.pop(k, None)
    if "reviewers" in d:
        d["reviewers"] = [asdict_or_str(p) for p in d["reviewers"]]
    return d

def asdict_or_str(p):
    try:
        return asdict(p)
    except TypeError:
        return str(p)

if __name__ == "__main__":
    main()