"""
MAMORX-adapted Baseline for Limitation Generation  (Llama-3-8B-Instruct version)
================================================================================
Adapts the MAMORX multi-agent peer-review framework (Taechoyotin et al., 2024)
for limitation generation, running locally on Llama-3-8B-Instruct.

Architecture (per paper):
    - Clarity    Agent  -> clarity / reproducibility limitations
    - Impact     Agent  -> novelty / impact limitations
                           (gets `cited_in_ret` as novelty-tool analog)
    - Experiment Agent  -> methodology / experimental limitations
                           (gets `cited_papers_context` as domain-knowledge analog)
    - Leader     Agent  -> synthesizes into final consolidated limitations

Input CSV columns used:
    - input_text_cleaned    : cleaned paper text              (all agents)
    - cited_papers_context  : text of works the paper cites    (domain knowledge)
    - cited_in_ret          : retrieved relevant literature    (novelty signal)

Output CSV: per-agent limitations + synthesized final limitations.

Run:
    python mamorx_limitation_generation_llama.py \
        --input_csv  data/.../df_with_cited_in_processed.csv \
        --output_csv ./mamorx_limitations_llama_output.csv \
        --max_papers -1
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------------------------------------------------------- #
#  CONFIG  (matches the snippet you provided)
# --------------------------------------------------------------------------- #
MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR  = "llama3_8b_instruct"

MAX_NEW_TOKENS      = 768
MAX_CONTEXT_TOKENS  = 8000
TEMPERATURE         = 0.3

# Token budgets for the CONTENT inside prompts (conservative for 8k context)
PAPER_TOKEN_BUDGET    = 5200   # used by the three expert agents
CITATION_TOKEN_BUDGET = 1200   # per-agent citation context (cited_papers or cited_in_ret)

# Smaller paper budget for the leader so 3 expert outputs + paper excerpt fit
LEADER_PAPER_BUDGET = 3500

# --------------------------------------------------------------------------- #
#  LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("mamorx-llama")

# --------------------------------------------------------------------------- #
#  AGENT SYSTEM PROMPTS  (adapted from MAMORX appendix for LIMITATION generation)
# --------------------------------------------------------------------------- #
CLARITY_AGENT_SYSTEM = """You are part of a group of agents analyzing a scientific paper to \
identify its LIMITATIONS. You are highly curious and have incredible attention to detail. Your \
focus is on clarity, organization, and reproducibility of the paper.

Identify limitations related to:
- Unclear or under-specified methodology
- Missing hyperparameters, implementation details, or equipment/material specs
- Ambiguous statements or undefined terms
- Poor organization that impedes understanding
- Background concepts assumed but not explained
- Missing information required to reproduce the results

Do not invent limitations that the paper already addresses. Be specific, constructive, and \
explain WHY each point is a limitation. Prefer a few concrete, important limitations over many \
vague ones. Respond with a numbered list and nothing else."""

IMPACT_AGENT_SYSTEM = """You are part of a group of agents analyzing a scientific paper to \
identify its LIMITATIONS. You are curious and skeptical, focusing on novelty, significance, and \
impact. You will also be given EXTERNAL RETRIEVED LITERATURE (recent related work) to help \
assess novelty.

Identify limitations related to:
- Weakly justified or overstated motivations
- Overclaimed contributions relative to evidence
- Narrow scope or limited generalizability
- Missing positioning against closely related prior work (use the retrieved literature)
- Hidden assumptions that undermine claimed significance
- Lack of discussion of broader / real-world implications

Do not fabricate concerns the paper already addresses. Each limitation must include a short \
rationale explaining why it matters. Respond with a numbered list and nothing else."""

EXPERIMENT_AGENT_SYSTEM = """You are part of a group of agents analyzing a scientific paper to \
identify its LIMITATIONS. You are an expert scientist who designs high-quality experiments, \
ablations, methodology, and analyses. You will also be given context from CITED PAPERS for \
domain-specific reference.

Identify limitations related to:
- Missing or weak baselines (name specific missing baselines where possible)
- Insufficient datasets, domains, or evaluation settings
- Inadequate metrics or measurement techniques
- Missing ablations needed to isolate contributions
- Statistical issues (no error bars / seeds, small sample, no significance tests)
- Flawed or misleading experimental setup
- Results that do not fully support the paper's claims

Do not suggest things the paper already includes. Be specific. Respond with a numbered list and \
nothing else."""

LEADER_AGENT_SYSTEM = """You are the review leader. Three expert agents (clarity, impact, \
experiments) each produced a list of limitations for a scientific paper. Your job is to \
synthesize their outputs into a single, coherent, well-prioritized list of LIMITATIONS.

Rules:
1. Merge duplicates and near-duplicates across agents.
2. Prioritize MAJOR limitations (things that affect overall impact) over minor ones.
3. Each limitation must be SPECIFIC: name the concrete issue, not a generic complaint.
4. Each limitation must include a brief rationale for why it matters.
5. Cover all three aspects (clarity, impact, experiments) when warranted.
6. Do NOT add limitations not supported by the expert inputs or the paper itself.

Output: a numbered list of the final limitations. No preamble, no postscript."""

# --------------------------------------------------------------------------- #
#  LOAD MODEL  (as in your snippet)
# --------------------------------------------------------------------------- #
log.info("Loading tokenizer and model: %s", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

# Llama tokenizers commonly ship without a pad token — use eos as pad for safety
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    return_full_text=False,
)
log.info("Model loaded.")

# --------------------------------------------------------------------------- #
#  TOKEN-LEVEL UTILITIES
# --------------------------------------------------------------------------- #
def count_tokens(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_by_tokens(text: Optional[str], max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` tokens using the model tokenizer."""
    if not isinstance(text, str) or not text.strip() or max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def call_llama(
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Call Llama-3-Instruct via the HF pipeline, using the official chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Llama 3 ends generations on <|eot_id|>; pass it as a terminator
    terminators = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        terminators.append(eot)

    out = llm_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
    )
    return out[0]["generated_text"].strip()

# --------------------------------------------------------------------------- #
#  AGENT CALLS
# --------------------------------------------------------------------------- #
def run_clarity_agent(paper_text: str) -> str:
    paper = truncate_by_tokens(paper_text, PAPER_TOKEN_BUDGET)
    user_msg = (
        "Below is the full text of a scientific paper. Identify its LIMITATIONS from the "
        "CLARITY / REPRODUCIBILITY perspective.\n\n"
        f"=== PAPER TEXT ===\n{paper}\n=== END PAPER TEXT ===\n\n"
        "Now list the clarity/reproducibility limitations."
    )
    return call_llama(CLARITY_AGENT_SYSTEM, user_msg)

def run_impact_agent(paper_text: str, retrieved_lit: str) -> str:
    # Impact agent gets retrieved related literature (novelty-tool analog)
    paper     = truncate_by_tokens(paper_text,    PAPER_TOKEN_BUDGET)
    retrieved = truncate_by_tokens(retrieved_lit, CITATION_TOKEN_BUDGET)
    retrieved_block = retrieved if retrieved else "(no retrieved literature provided)"
    user_msg = (
        "Below is a scientific paper and a block of RETRIEVED related literature from recent "
        "external sources. Identify the paper's LIMITATIONS from the NOVELTY / IMPACT "
        "perspective, using the retrieved literature to judge positioning and novelty.\n\n"
        f"=== PAPER TEXT ===\n{paper}\n=== END PAPER TEXT ===\n\n"
        f"=== RETRIEVED RELATED LITERATURE ===\n{retrieved_block}\n=== END RETRIEVED ===\n\n"
        "Now list the novelty/impact limitations."
    )
    return call_llama(IMPACT_AGENT_SYSTEM, user_msg)

def run_experiment_agent(paper_text: str, cited_papers: str) -> str:
    # Experiment agent gets cited-papers context (domain-knowledge analog)
    paper = truncate_by_tokens(paper_text,   PAPER_TOKEN_BUDGET)
    cited = truncate_by_tokens(cited_papers, CITATION_TOKEN_BUDGET)
    cited_block = cited if cited else "(no cited-papers context provided)"
    user_msg = (
        "Below is a scientific paper and a block of context from papers it CITES (domain "
        "knowledge). Identify the paper's LIMITATIONS from the EXPERIMENTS / METHODOLOGY "
        "perspective, using the cited context when it helps.\n\n"
        f"=== PAPER TEXT ===\n{paper}\n=== END PAPER TEXT ===\n\n"
        f"=== CITED PAPERS CONTEXT ===\n{cited_block}\n=== END CITED ===\n\n"
        "Now list the experimental/methodological limitations."
    )
    return call_llama(EXPERIMENT_AGENT_SYSTEM, user_msg)

def run_leader_agent(
    clarity_out: str,
    impact_out: str,
    experiment_out: str,
    paper_text: str,
) -> str:
    """Leader synthesizes three expert outputs into the final limitations list."""
    paper = truncate_by_tokens(paper_text, LEADER_PAPER_BUDGET)
    user_msg = (
        "Three expert agents have identified limitations of the same paper. Synthesize their "
        "outputs into ONE prioritized numbered list of the paper's LIMITATIONS, following your "
        "rules.\n\n"
        f"=== CLARITY AGENT ===\n{clarity_out}\n\n"
        f"=== IMPACT AGENT ===\n{impact_out}\n\n"
        f"=== EXPERIMENTS AGENT ===\n{experiment_out}\n\n"
        "For reference only (do not introduce new limitations not raised above):\n"
        f"=== PAPER EXCERPT ===\n{paper}\n=== END PAPER EXCERPT ==="
    )
    return call_llama(LEADER_AGENT_SYSTEM, user_msg, temperature=0.2)

def generate_limitations(paper_text: str, cited_papers: str, retrieved_lit: str) -> dict:
    """Run the full MAMORX-style pipeline for one paper."""
    clarity_out    = run_clarity_agent(paper_text)
    impact_out     = run_impact_agent(paper_text, retrieved_lit)
    experiment_out = run_experiment_agent(paper_text, cited_papers)
    final_out      = run_leader_agent(clarity_out, impact_out, experiment_out, paper_text)
    return {
        "clarity_limitations":    clarity_out,
        "impact_limitations":     impact_out,
        "experiment_limitations": experiment_out,
        "final_limitations":      final_out,
    }

# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAMORX-style limitation generation (Llama 3 8B).")
    p.add_argument(
        "--input_csv",
        type=str,
        default=""
                "data/balanced_data/df_with_cited_in_processed.csv",
    )
    p.add_argument("--output_csv",        type=str, default="baselines/MAMORX/llama/mamorx_limitations_llama_output.csv")
    p.add_argument("--paper_col",         type=str, default="input_text_cleaned")
    p.add_argument("--cited_papers_col",  type=str, default="cited_papers_context")
    p.add_argument("--retrieved_col",     type=str, default="cited_in_ret")
    p.add_argument("--max_papers",        type=int, default=-1)
    p.add_argument("--checkpoint_every",  type=int, default=5)
    p.add_argument("--resume",            action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------- #
    #  LOAD DATA
    # ------------------------------------------------------------------- #
    log.info("Reading input CSV: %s", args.input_csv)
    df = pd.read_csv(args.input_csv) 
    for col in (args.paper_col, args.cited_papers_col, args.retrieved_col):
        if col not in df.columns:
            sys.exit(f"Missing expected column '{col}'. Columns in CSV: {list(df.columns)}")
    log.info("Loaded %d papers.", len(df))

    if args.max_papers > 0:
        df = df.head(args.max_papers).copy()
        log.info("Limiting to first %d papers.", len(df))

    # ------------------------------------------------------------------- #
    #  ADD NEW COLUMNS TO THE EXISTING DATAFRAME
    # ------------------------------------------------------------------- #
    new_cols = [
        "clarity_limitations",
        "impact_limitations",
        "experiment_limitations",
        "final_limitations",
        "mamorx_error",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # ------------------------------------------------------------------- #
    #  PREPARE OUTPUT PATH (+ optional resume)
    # ------------------------------------------------------------------- #
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        shared = prev.index.intersection(df.index)
        for c in new_cols:
            if c in prev.columns:
                df.loc[shared, c] = prev.loc[shared, c]
        log.info("Resumed from existing output (%s).", out_path)

    # ------------------------------------------------------------------- #
    #  MAIN LOOP -> write results INTO the existing dataframe
    # ------------------------------------------------------------------- #
    processed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MAMORX-Llama"):
        # Skip rows already completed (resume-safe)
        existing = df.at[idx, "final_limitations"]
        if isinstance(existing, str) and existing.strip():
            continue

        paper_text   = row.get(args.paper_col, "")        or ""
        cited_papers = row.get(args.cited_papers_col, "") or ""
        retrieved    = row.get(args.retrieved_col, "")    or ""

        if not isinstance(paper_text, str) or not paper_text.strip():
            log.warning("Row %s has empty paper text. Skipping.", idx)
            df.at[idx, "mamorx_error"] = "empty_input"
            continue

        try:
            outputs = generate_limitations(paper_text, str(cited_papers), str(retrieved))
            df.at[idx, "clarity_limitations"]    = outputs["clarity_limitations"]
            df.at[idx, "impact_limitations"]     = outputs["impact_limitations"]
            df.at[idx, "experiment_limitations"] = outputs["experiment_limitations"]
            df.at[idx, "final_limitations"]      = outputs["final_limitations"]
            df.at[idx, "mamorx_error"]           = None
        except torch.cuda.OutOfMemoryError:
            log.exception("CUDA OOM on row %s — clearing cache and recording error.", idx)
            torch.cuda.empty_cache()
            df.at[idx, "mamorx_error"] = "cuda_oom"
        except Exception as e:
            log.exception("Failed on row %s: %s", idx, e)
            df.at[idx, "mamorx_error"] = str(e)

        processed += 1
        if processed % args.checkpoint_every == 0:
            df.to_csv(out_path, index=False)
            log.info("Checkpoint saved (%d processed) -> %s", processed, out_path)

    df.to_csv(out_path, index=False)
    log.info("Done. Wrote full dataframe with new columns to %s", out_path)

if __name__ == "__main__":
    main()