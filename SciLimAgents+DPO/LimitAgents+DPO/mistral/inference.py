#!/usr/bin/env python
"""
inference.py — Multi-adapter inference pipeline for limitation extraction.

Works with any HuggingFace causal LM (Qwen, Llama-3, Mistral, ...).
All paths / hyperparameters are configurable via CLI args, so a single
script can be driven by different PBS jobs.

Usage (Llama-3):
    python inference.py \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --cache-dir  /path/to/llama3_8b_instruct \
        --worker-dpo-adapter /path/to/worker_dpo/final \
        --leader-sft-adapter /path/to/leader_sft/final \
        --master-sft-adapter /path/to/master_sft/final \
        --input-csv  /path/to/input.csv \
        --output-dir /path/to/out \
        --output-suffix llama3_8b
"""

import argparse
import ast
import gc
import os
import re
import time

import pandas as pd
import tiktoken
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Multi-adapter inference pipeline.")

    # ---- Model ----
    p.add_argument("--base-model", required=True,
                   help="HF model id or local path "
                        "(e.g. meta-llama/Meta-Llama-3-8B-Instruct, "
                        "mistralai/Mistral-7B-Instruct-v0.3, or a local dir).")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache directory (where weights are downloaded to).")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16"],
                   help="Compute dtype for the base model.")
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    # ---- LoRA adapters ----
    p.add_argument("--worker-dpo-adapter", required=True,
                   help="Path to worker DPO LoRA adapter (used by all 7 specialists).")
    p.add_argument("--leader-sft-adapter", required=True,
                   help="Path to leader SFT LoRA adapter.")
    p.add_argument("--master-sft-adapter", required=True,
                   help="Path to master SFT LoRA adapter.")

    # ---- Input / output ----
    p.add_argument("--input-csv", required=True)
    p.add_argument("--text-column", default="input_text_cleaned",
                   help="Column with the paper text.")
    p.add_argument("--cited-column", default="cited_in",
                   help="Column with the cited-papers dict (string-encoded).") 
    p.add_argument("--cited-ret-column", default="cited_in_ret",
               help="Additional citation/retrieval text column to concatenate.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-suffix", default="run",
                   help="Suffix for the output csv filename.")
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1,
                   help="-1 means run to end of file.")

    # ---- Truncation budgets (in tokens, tiktoken-approx) ----
    p.add_argument("--max-paper-tokens",         type=int, default=20000)
    p.add_argument("--max-citation-tokens",      type=int, default=5000)
    p.add_argument("--max-leader-input-tokens",  type=int, default=12000)
    p.add_argument("--max-master-input-tokens",  type=int, default=12000)
    p.add_argument("--context-window",           type=int, default=30000,
                   help="Hard cap on input ids passed to the model.")

    # ---- Generation ----
    p.add_argument("--max-new-tokens-worker", type=int, default=2048)
    p.add_argument("--max-new-tokens-leader", type=int, default=2048)
    p.add_argument("--max-new-tokens-master", type=int, default=3000)
    p.add_argument("--temperature",        type=float, default=0.2)
    p.add_argument("--top-p",              type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--do-sample", action="store_true", default=True)

    return p.parse_args()

# =============================================================================
# Multi-adapter model (load base once, attach all adapters, switch instantly)
# =============================================================================

class MultiAdapterModel:
    """
    Loads the base model once and attaches all three LoRA adapters
    (worker_dpo, leader_sft, master_sft) simultaneously.
    Switching is O(1) via set_adapter() — no reload, no fragmentation.
    """

    def __init__(self, args):
        self.args = args
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

        print("=" * 60)
        print("Loading base model and all LoRA adapters")
        print(f"  base_model = {args.base_model}")
        print(f"  cache_dir  = {args.cache_dir}")
        print(f"  dtype      = {args.dtype}")
        print("=" * 60)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        # Most Llama / Mistral models don't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Base
        print("[1/4] Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

        # Adapter 1: worker_dpo (this creates the PeftModel)
        print(f"[2/4] Attaching worker_dpo adapter from {args.worker_dpo_adapter}")
        self.model = PeftModel.from_pretrained(
            base_model,
            args.worker_dpo_adapter,
            adapter_name="worker_dpo",
            dtype=dtype,
        )

        # Adapter 2: leader_sft
        print(f"[3/4] Attaching leader_sft adapter from {args.leader_sft_adapter}")
        self.model.load_adapter(args.leader_sft_adapter, adapter_name="leader_sft")

        # Adapter 3: master_sft
        print(f"[4/4] Attaching master_sft adapter from {args.master_sft_adapter}")
        self.model.load_adapter(args.master_sft_adapter, adapter_name="master_sft")

        self.model.eval()
        self.model.set_adapter("worker_dpo")
        self.active_adapter = "worker_dpo"

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nGPU memory after load — allocated: {allocated:.2f} GB, "
                  f"reserved: {reserved:.2f} GB\n")

    def switch_adapter(self, adapter_name: str):
        if self.active_adapter != adapter_name:
            self.model.set_adapter(adapter_name)
            self.active_adapter = adapter_name

    def generate(self, system_prompt: str, user_message: str,
                 max_new_tokens: int = 2048) -> str:
        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_message},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            print(f"    [{self.active_adapter}] Input tokens: {input_len}")

            ctx = self.args.context_window
            if input_len > ctx:
                print(f"    WARNING: Truncating {input_len} -> {ctx} tokens")
                inputs["input_ids"] = inputs["input_ids"][:, :ctx]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :ctx]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    do_sample=self.args.do_sample,
                    repetition_penalty=self.args.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            del inputs, outputs, generated
            torch.cuda.empty_cache()
            return response

        except torch.cuda.OutOfMemoryError:
            print("    CUDA OOM! Clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            return "ERROR: CUDA out of memory"
        except Exception as e:
            print(f"    LLM call error: {e}")
            return f"ERROR: {e}"

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

# =============================================================================
# Prompts (kept identical to the original Qwen pipeline)
# =============================================================================

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is exclusively in **novelty and significance**.

Your task: Analyze the paper below and identify all limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.

Look for issues such as:
- Rebranding existing ideas without substantial improvement
- Lack of clear differentiation from prior work
- Exaggerated claims of breakthrough
- Narrow scope that limits broader significance
- Failure to articulate why the work matters beyond a niche setting
- Unaddressed alternatives or ignored related problems that diminish perceived impact

OUTPUT FORMAT:
Provide a concise bullet list of novelty- and significance-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent** in a multi-agent system for identifying limitations in scientific papers.

Your task: Compare the main article against the cited papers' information below. Identify limitations related to how the paper uses, interprets, or fails to engage with its citations.

Specifically check:
- Did the article fail to address key insights from its citations?
- Does the paper misinterpret or selectively cite prior work to make its own contribution look stronger?
- Are important related works missing from the citation landscape?

OUTPUT FORMAT:
Provide a bullet list of citation-related limitations. Each bullet should follow this format:
- [Limitation]: Explanation (Ref: Paper X)

=== MAIN ARTICLE ===
{paper_content}

=== CITED PAPERS INFO ===
{citation_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **theoretical and methodological soundness**, including ablations and component analysis.

Your task: Analyze the paper below and identify all limitations related to theoretical foundations, methodology, and component analysis.

Look for issues such as:
- Unstated or overly strong assumptions
- Incomplete theoretical analysis or errors in derivations
- Methods that only work under restricted conditions not clearly acknowledged
- Missing ablations or lack of isolation of individual contributions
- Ablations that do not convincingly attribute performance gains
- Logical gaps between claims and supporting evidence

OUTPUT FORMAT:
Provide a concise bullet list of theoretical, methodological, and ablation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **experimental evaluation**, including validation, rigor, comparisons, baselines, and metrics.

Your task: Analyze the paper below and identify all limitations in the empirical evaluation.

Look for issues such as:
- Insufficient runs or lack of statistical significance testing
- Cherry-picked results or narrow experimental conditions
- Outdated, weak, or missing baselines and key competitors
- Unfair hyperparameter tuning or comparison setups
- Missing error bars, confidence intervals, or variance reporting
- Reliance on misleading metrics or missing standard metrics
- Overemphasis on minor gains without practical or statistical significance

OUTPUT FORMAT:
Provide a concise bullet list of experimental evaluation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise covers **generalization, robustness, computational efficiency, and real-world applicability**.

Your task: Analyze the paper below and identify all limitations related to how well the method generalizes, how robust it is, how efficient it is, and how applicable it is to real-world scenarios.

Look for issues such as:
- Overfitting to benchmarks or lack of out-of-distribution testing
- Sensitivity to hyperparameters or poor performance under distribution shifts
- Excessive training/inference demands or high resource requirements
- Reliance on synthetic data without real-world validation
- Ignoring deployment constraints like cost, latency, or hardware limitations
- Missing user studies or field tests

OUTPUT FORMAT:
Provide a concise bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **clarity, interpretability, and reproducibility**.

Your task: Analyze the paper below and identify all limitations related to how clearly the work is presented, how interpretable the method/results are, and whether the work can be reproduced.

Look for issues such as:
- Unclear explanations of methods, settings, or key concepts
- Ambiguities or unstated assumptions that hinder comprehension
- Black-box behavior without explanations or mechanistic understanding
- Missing code, data, or hyperparameter details needed for replication
- Unreported random seeds or ambiguous experimental procedures

OUTPUT FORMAT:
Provide a concise bullet list of clarity-, interpretability-, and reproducibility-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **data integrity, bias, fairness, and ethical considerations**.

Your task: Analyze the paper below and identify all limitations related to data quality, potential biases, fairness issues, and ethical concerns.

Look for issues such as:
- Small or non-diverse datasets, labeling errors, or undocumented preprocessing
- Data leakage or reliance on flawed/biased datasets without validation
- Biased outcomes that could lead to discrimination
- Lack of fairness metrics or unreported subgroup performance
- Privacy risks, dual-use concerns, or failure to discuss misuse potential

OUTPUT FORMAT:
Provide a concise bullet list of data integrity-, bias-, fairness-, and ethics-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Leader Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

CONTEXT: Multiple specialist worker agents have independently analyzed the paper below, each focusing on a different aspect (novelty, methodology, experiments, generalization, clarity, data/ethics, citations). Their outputs have been collected and will be provided to you.

YOUR TASK:
1. Read all specialist outputs carefully.
2. Review each for quality, relevance, and completeness.
3. Flag any weak or unsupported claims that lack evidence from the paper.
4. Note redundancies across specialists (similar limitations raised by multiple agents).
5. Organize all limitations into a well-structured consolidated summary, grouped by category.
6. Preserve the specificity and evidence from the original analyses.

OUTPUT FORMAT:
Produce a structured summary organized by category (e.g., Novelty & Significance, Methodology, Experimental Evaluation, etc.). For each category:
- List the relevant limitations with their supporting evidence
- Mark any limitations that appear across multiple specialists as [CROSS-VALIDATED]
- Mark any limitations that seem weakly supported as [NEEDS EVIDENCE]

This consolidated summary will be passed to the Master Agent for final synthesis.

PAPER CONTENT (for reference):
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

CONTEXT: The Leader Agent has reviewed and consolidated limitation analyses from multiple specialist workers. The Leader Agent's consolidated summary will be provided to you.

YOUR TASK:
- Carefully read and integrate the Leader Agent's consolidated summary.
- Remove redundancies by merging similar limitations into single, well-stated points.
- Prioritize the most severe and well-justified limitations.
- Preserve specificity and evidence from the original analyses.
- Organize the final list logically, grouped by category.
- Ensure each limitation is clearly stated, concise, and grounded in the paper.
- Do NOT introduce new limitations not raised by the specialists.
- Aim for 10-20 strong limitations (adjust based on paper quality).

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then provide a bulleted list:
- **Category:** Specific limitation statement (with brief explanation and evidence reference if it adds value).

If no major limitations were found, state: "The paper appears methodologically sound with only minor limitations: [list them]."

PAPER CONTENT (for reference):
{paper_content}"""

# =============================================================================
# Helpers
# =============================================================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry):
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    out = []
    for idx, (pid, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t = clean_text_detailed(data.get("title", ""))
        a = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i = clean_text_detailed(intro)
        if t or a or i:
            out.append(
                f"'Paper{idx}_Title: {t}', 'Paper{idx}_Abstract': '{a}', "
                f"'Paper{idx}_Introduction': '{i}'."
            )
    return "\n".join(out)

_ENC = None
def _get_encoding():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    enc = _get_encoding()
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    print(f"  Warning: Truncating input: {len(toks)} -> {max_tokens} tokens.")
    return enc.decode(toks[:max_tokens]) + "... [TRUNCATED]"

def log_gpu_memory(label=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        print(f"  [GPU {label}] allocated: {a:.2f} GB, reserved: {r:.2f} GB")

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(args):
    print("=" * 60)
    print("INFERENCE PIPELINE: DPO Worker + SFT Leader + SFT Master")
    print(f"Model: {args.base_model}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(
        args.output_dir,
        f"df_inference_{args.output_suffix}.csv",
    )

    # ---- Load data ----
    print(f"\nLoading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows total.")

    # Slice [start:end]
    end = args.end if args.end >= 0 else len(df)
    df = df.iloc[args.start:end].reset_index(drop=True)
    print(f"Working on slice [{args.start}:{end}] -> {len(df)} rows.")

    # ---- Resume from checkpoint if exists ----
    agent_columns = [
        "lim_novelty_significance",
        "lim_theoretical_methodological",
        "lim_experimental_evaluation",
        "lim_generalization_robustness_efficiency",
        "lim_clarity_interpretability_reproducibility",
        "lim_data_ethics",
        "lim_citation",
        "leader_consolidated",
        "final_limitations_master",
    ]

    start_idx = 0
    if os.path.exists(output_csv):
        print(f"Found existing checkpoint at {output_csv}, loading...")
        ckpt = pd.read_csv(output_csv)
        if "final_limitations_master" in ckpt.columns and len(ckpt) == len(df):
            for col in agent_columns:
                if col in ckpt.columns:
                    df[col] = ckpt[col]
            for idx in range(len(df)):
                v = str(df.iloc[idx].get("final_limitations_master", "PENDING"))
                if v in ("PENDING", "nan", "", "None"):
                    start_idx = idx
                    break
            else:
                start_idx = len(df)
        del ckpt
        print(f"Resuming from row {start_idx}")

    for col in agent_columns:
        if col not in df.columns:
            df[col] = "PENDING"

    if start_idx >= len(df):
        print("All rows already processed!")
        return

    # ---- Load model ----
    multi_model = MultiAdapterModel(args)
    log_gpu_memory("after loading all adapters")

    specialist_config = [
        ("lim_novelty_significance",                      get_novelty_significance_prompt),
        ("lim_theoretical_methodological",                get_theoretical_methodological_prompt),
        ("lim_experimental_evaluation",                   get_experimental_evaluation_prompt),
        ("lim_generalization_robustness_efficiency",      get_generalization_robustness_efficiency_prompt),
        ("lim_clarity_interpretability_reproducibility",  get_clarity_interpretability_reproducibility_prompt),
        ("lim_data_ethics",                               get_data_ethics_prompt),
    ]

    worker_user_msg = (
        "Analyze the paper provided in your instructions. "
        "Identify all limitations in your domain of expertise. "
        "Provide a concise bullet list of limitations with explanations and evidence."
    )

    total_rows = len(df)
    print(f"\nProcessing rows {start_idx} to {total_rows - 1}...")
    t0 = time.time()

    for i in tqdm(range(start_idx, total_rows), initial=start_idx, total=total_rows):
        row = df.iloc[i]

        cur = str(row.get("final_limitations_master", "PENDING"))
        if cur not in ("PENDING", "nan", "", "None"):
            continue

        paper_text    = str(row.get(args.text_column, ""))
        # citation_text = extract_intro_and_abstract(row.get(args.cited_column, "")) 
        citation_value = row.get(args.cited_column, "")
        citation_ret_value = row.get(args.cited_ret_column, "") 
        citation_parts = []
        if not pd.isna(citation_value) and str(citation_value).strip():
            citation_parts.append(f"=== {args.cited_column} ===\n{citation_value}")
        if not pd.isna(citation_ret_value) and str(citation_ret_value).strip():
            citation_parts.append(f"=== {args.cited_ret_column} ===\n{citation_ret_value}")
        citation_text = "\n\n".join(citation_parts)

        paper_text    = truncate_text_to_tokens(paper_text,    args.max_paper_tokens)
        citation_text = truncate_text_to_tokens(citation_text, args.max_citation_tokens)

        if len(paper_text) < 100:
            for col in agent_columns:
                df.iat[i, df.columns.get_loc(col)] = "SKIPPED_SHORT_TEXT"
            continue

        row_t0 = time.time()
        print(f"\n{'='*40} Row {i}/{total_rows-1} {'='*40}")

        try:
            # ===== Stage 1: Worker DPO — 7 specialists =====
            multi_model.switch_adapter("worker_dpo")
            specialist_outputs = []

            for col_name, prompt_func in specialist_config:
                label = col_name.replace("lim_", "").replace("_", " ").title()
                print(f"  [Worker] {label}...")
                sys_prompt = prompt_func(paper_text)
                out = multi_model.generate(sys_prompt, worker_user_msg,
                                            max_new_tokens=args.max_new_tokens_worker)
                df.iat[i, df.columns.get_loc(col_name)] = out
                specialist_outputs.append(f"=== {label} ===\n{out}")

            # Citation agent (also worker_dpo)
            print("  [Worker] Citation Agent...")
            cite_sys = get_citation_agent_prompt(paper_text, citation_text)
            cite_out = multi_model.generate(
                cite_sys,
                "Analyze citation-related limitations as described in your instructions.",
                max_new_tokens=args.max_new_tokens_worker,
            )
            df.iat[i, df.columns.get_loc("lim_citation")] = cite_out
            specialist_outputs.append(f"=== Citation Agent ===\n{cite_out}")

            # ===== Stage 2: Leader SFT =====
            multi_model.switch_adapter("leader_sft")
            combined = "\n\n".join(specialist_outputs)
            combined = truncate_text_to_tokens(combined, args.max_leader_input_tokens)

            print("  [Leader] Consolidating...")
            leader_sys = get_leader_agent_prompt(paper_text)
            leader_in  = (
                "Here are the limitation analyses from the specialist worker agents:\n\n"
                f"{combined}\n\n"
                "Please review, organize, flag any weak claims, and produce a consolidated "
                "summary of all limitations grouped by category for the Master Agent."
            )
            leader_out = multi_model.generate(leader_sys, leader_in,
                                               max_new_tokens=args.max_new_tokens_leader)
            df.iat[i, df.columns.get_loc("leader_consolidated")] = leader_out

            # ===== Stage 3: Master SFT =====
            multi_model.switch_adapter("master_sft")
            leader_trunc = truncate_text_to_tokens(leader_out, args.max_master_input_tokens)

            print("  [Master] Final synthesis...")
            master_sys = get_master_agent_prompt(paper_text)
            master_in  = (
                "The Leader Agent has reviewed and consolidated the specialist analyses. "
                "Here is the Leader Agent's consolidated summary:\n\n"
                f"{leader_trunc}\n\n"
                "Please synthesize this into a single, final, high-quality, non-redundant "
                "list of limitations, grouped by category."
            )
            final_out = multi_model.generate(master_sys, master_in,
                                              max_new_tokens=args.max_new_tokens_master)

            if len(final_out.strip()) < 50 or final_out.startswith("ERROR"):
                final_out = "NO_OUTPUT_FROM_MASTER"

            df.iat[i, df.columns.get_loc("final_limitations_master")] = final_out
            print(f"  Row {i} done in {time.time() - row_t0:.1f}s")

        except Exception as e:
            print(f"  ERROR on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_limitations_master")] = f"ERROR: {e}"
            torch.cuda.empty_cache()

        # ---- Checkpoint ----
        if (i + 1) % args.checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            elapsed = time.time() - t0
            done = i - start_idx + 1
            rate = elapsed / max(done, 1)
            eta = (total_rows - i - 1) * rate
            print(f"  >>> Checkpoint at row {i} | "
                  f"{done} rows in {elapsed/60:.1f}min | "
                  f"~{rate:.1f}s/row | ETA: {eta/3600:.1f}h")

        time.sleep(0.2)

    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved to: {output_csv}")
    print(f"Total time: {(time.time() - t0)/3600:.2f} hours")
    multi_model.cleanup()

if __name__ == "__main__":
    run_pipeline(parse_args())
