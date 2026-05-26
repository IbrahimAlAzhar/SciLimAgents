#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA 3 8B Instruct — Sequential Novelty & Significance Limitation Analysis
vLLM backend (offline, fully local) — fixes HF 429 rate-limit and is ~5-10x faster.

Pipeline, prompts, agents, columns, and CSV I/O are IDENTICAL to the original.
Only the model-loading and generation calls have been swapped to vLLM, and we
load from a LOCAL DIRECTORY so no HuggingFace API call is ever made.
"""

import os
# ----- Force fully offline BEFORE any HF import -----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import time
import ast
import re
import json
import pandas as pd
from tqdm import tqdm

import torch
import tiktoken
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ============================================================
# 0) PATHS & CONFIGURATION
# ============================================================
INPUT_CSV = "llm_agents/llama_3_8B_new/novagents/input/df_with_retrieved_sections.csv"
OUTPUT_CSV = "llm_agents/llama_3_8B_new/novagents/output/df_llm_novagents_llama3_8b_output.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


TEXT_COL_MAIN = "input_text_cleaned"
RELATED_COL = "relevant_papers_list"

# *** Point this at the LOCAL directory you pre-downloaded with huggingface-cli ***
# huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
#   --local-dir llama3_8b_instruct/Meta-Llama-3-8B-Instruct
MODEL_PATH = "llama3_8b_instruct/Meta-Llama-3-8B-Instruct"

# Token budgets
PAPER_TOK_BUDGET           = 5_500
CITATION_TOK_BUDGET        = 1_200
PER_RETR_PAPER_INPUT_TOK   = 2_500
PER_RETR_PAPER_SUMMARY_TOK = 600
MAX_NEW_TOKENS_SPECIALIST  = 1_500
MAX_NEW_TOKENS_MASTER      = 2_500
MAX_NEW_TOKENS_SUMMARY     = 700
MODEL_MAX_INPUT_TOKENS     = 7_500

TEMPERATURE = 0.2

# tiktoken for budget accounting only
try:
    _tiktoken_enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")

# ============================================================
# 1) MODEL LOADING (vLLM, OFFLINE)
# ============================================================
print(f"Loading model from local path: {MODEL_PATH}")
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"MODEL_PATH does not exist: {MODEL_PATH}\n"
        "Run on a login node first:\n"
        "  huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct "
        f"--local-dir {MODEL_PATH} --local-dir-use-symlinks False"
    )

# We still use the HF tokenizer for chat-template formatting; loaded from local files only.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# vLLM engine
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enforce_eager=False,        # use CUDA graphs for max speed
    disable_log_stats=True,
)
print("vLLM engine ready ✓")

# ============================================================
# 2) GLOBAL GUARDRAILS & WORKER APPENDAGES (UNCHANGED)
# ============================================================
HARSH_REVIEWER_POLICY = """
You are an extremely strict, harsh peer reviewer.

STRICT REVIEW MODE:
- Default assumption: the paper has limitations unless it provides clear, concrete evidence.
- Prefer flagging possible weaknesses rather than giving benefit of the doubt.
- Focus on limitations that undermine novelty, significance, credibility, rigor, or generalizability.

WHAT TO PRODUCE:
- Only limitations-focused content (no scores, no ratings).
- Evidence pointers from Paper A / Paper B excerpts when possible.
""".strip()

WORKER_FEEDBACK_INSTRUCTION = """
Provide detailed critique. Deliver a bullet list of limitations (tailored to your domain) with supporting evidence from Paper A and Paper B.
""".strip()

# ============================================================
# 3) WORKER PROMPTS (UNCHANGED)
# ============================================================
PROMPT_1_NOVELTY_TECH = """
Prompt 1: Technical Contributions & Incremental Nature
Identify limitations in the paper's technical contributions that undermine claims of novelty or significance.
Focus on whether ideas are rebranded existing methods, minor tweaks, combinations of known components, or
lack substantive advancement beyond prior work.
Output format (STRICT):
Limitations in Technical Contributions (A vs B):
- <bullet 1 limitation with explanation; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_2_EXPERIMENTS = """
Prompt 2: Experimental Validation & Comparative Analysis
Identify limitations in experimental design, benchmarking, or comparative analysis that weaken the paper's
novelty or claimed significance (e.g., missing strong baselines, inadequate datasets, no ablation, overstated
 improvements).

Output format (STRICT):
Limitations in Experimental Validation (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_3_LIT_REVIEW = """
Prompt 3: Literature Review & Contextualization
Identify limitations in the literature review or positioning that undermine perceived novelty or significance
(overlooking key prior work, vague differentiation, failure to explain why the gap matters).
Output format (STRICT):
Limitations in Literature Review & Contextualization (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_4_SCOPE_GENERALIZABILITY = """
Prompt 4: Scope of Analysis & Generalizability
Identify limitations in scope, datasets, tasks, or discussed implications that restrict the work's broader
significance or generalizability (narrow domain, toy settings, ignored real-world constraints).
Output format (STRICT):
Limitations in Scope & Generalizability (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_5_CLAIMS_OVERCLAIMING = """
Prompt 5: Claim Accuracy & Overclaiming
Identify limitations stemming from overstated novelty, impact, effectiveness, or importance claims that lack
supporting evidence or ignore caveats.
Output format (STRICT):
Limitations in Claims & Overclaiming (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_6_METHOD_CLARITY_RIGOR = """
Prompt 6: Methodological Clarity & Rigor
Identify limitations in methodological description, reproducibility, or rigor that erode confidence in the
claimed novelty or significance (missing details, ambiguous setups, unverifiable experiments).

Output format (STRICT):
Limitations in Methodological Clarity & Rigor (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

SPECIALIST_AGENTS = [
    ("Literature_Review_and_Data_Analysis_Agent",
     PROMPT_3_LIT_REVIEW + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Hypothesis_Refinement_and_Critical_Reflection_Agent",
     PROMPT_1_NOVELTY_TECH + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Methodological_Novelty_Agent",
     PROMPT_6_METHOD_CLARITY_RIGOR + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Experimental_Novelty_Agent",
     PROMPT_2_EXPERIMENTS + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Problem_Formulation_Novelty_Agent",
     PROMPT_4_SCOPE_GENERALIZABILITY + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Writing_Claim_Novelty_Agent",
     PROMPT_5_CLAIMS_OVERCLAIMING + "\n" + WORKER_FEEDBACK_INSTRUCTION),
]

# ============================================================
# 4) MASTER AGENT PROMPT (UNCHANGED)
# ============================================================
MASTER_AGENT_PROMPT = """
You are the Master Agent.
Synthesize an overall limitations summary focused on the novelty & significance of the evaluated paper.

CRITICAL FINAL REPORT RULES:
1. The final report MUST state the limitations directly regarding the evaluated paper only.
2. You are STRICTLY FORBIDDEN from using the phrases "Paper A", "Paper B", "the main paper", or "the retrieved papers".
3. Transform comparative statements from the specialists into objective weaknesses.
   - INSTEAD OF: "Paper A lacks robust baselines compared to Paper B."
   - WRITE: "The experimental validation lacks robust baselines, failing to account for contemporary state-of-the-art standards."
4. Remove redundancies and ensure the tone is professional and objective.

OUTPUT FORMAT (STRICT):
**Technical Contributions:**
- <bullet 1 limitation>
- <bullet 2 limitation>

**Experimental Validation:**
- <bullet 1 limitation>

**Literature Review & Contextualization:**
- <bullet 1 limitation>

**Scope & Generalizability:**
- <bullet 1 limitation>

**Claims & Overclaiming:**
- <bullet 1 limitation>

**Methodological Clarity & Rigor:**
- <bullet 1 limitation>
""".strip()

# ============================================================
# 5) HELPERS
# ============================================================

def tok_len(text: str) -> int:
    return len(_tiktoken_enc.encode(text or ""))

def truncate_to_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    if not text:
        return ""
    ids = _tiktoken_enc.encode(text or "", disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
    return _tiktoken_enc.decode(ids)

def normalize_any_to_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x)

def parse_relevant_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x.strip())
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [x.strip()]
    return [str(x)]

def build_dual_paper_input(main_text: str, b_summaries_text: str) -> str:
    main_text = truncate_to_tokens(main_text, PAPER_TOK_BUDGET, keep="head")
    b_summaries_text = truncate_to_tokens(b_summaries_text, CITATION_TOK_BUDGET, keep="head")
    return (
        f"=== MAIN PAPER (A) ===\n{main_text}\n\n"
        f"=== RELEVANT PAPERS (B) ===\n{b_summaries_text}"
    ).strip()

# ============================================================
# 6) LOCAL LLM CALL — vLLM backend
# ============================================================

def call_llm(system_prompt: str, user_message: str,
             max_new_tokens: int = 1500) -> str:
    """Run a single inference call through the local vLLM engine."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Hard guard on prompt length (truncate the user portion if needed)
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_ids) > MODEL_MAX_INPUT_TOKENS:
            print(f"    ⚠ Input too long ({len(input_ids)} tok) — truncating to {MODEL_MAX_INPUT_TOKENS}")
            input_ids = input_ids[:MODEL_MAX_INPUT_TOKENS]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Pick LLaMA-3 stop tokens to avoid runaway generation
        stop_token_ids = []
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            stop_token_ids.append(eot)
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)

        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repetition_penalty=1.1,
            stop_token_ids=stop_token_ids or None,
        )

        outputs = llm.generate([input_text], sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return "ERROR: empty generation"
        return outputs[0].outputs[0].text.strip()

    except torch.cuda.OutOfMemoryError:
        print("    ✗ CUDA OOM — clearing cache")
        torch.cuda.empty_cache()
        return "ERROR: CUDA out of memory"
    except Exception as e:
        print(f"    ✗ LLM error: {e}")
        return f"ERROR: {e}"

def summarise_paper_b(raw_text: str) -> str:
    """Summarise a retrieved paper using the local model."""
    raw_text = truncate_to_tokens(raw_text, PER_RETR_PAPER_INPUT_TOK)
    system = "You are a concise academic summariser. Output a short summary suitable for limitations comparison."
    user = f"Summarize the following paper:\n{raw_text}"
    return call_llm(system, user, max_new_tokens=MAX_NEW_TOKENS_SUMMARY)

# ============================================================
# 7) BUILD PAPER-B SUMMARIES
# ============================================================

def build_b_summaries_from_list(lst, k=3):
    summaries = []
    for idx in range(k):
        if idx < len(lst):
            item_text = normalize_any_to_text(lst[idx])
            print(f"    Summarising retrieved paper #{idx+1} …")
            summaries.append(summarise_paper_b(item_text))
        else:
            summaries.append("Paper B Summary:\n(Missing item.)")
    combined = "\n\n".join(
        [f"--- Retrieved Paper #{i+1} ---\n{summaries[i]}" for i in range(k)]
    ).strip()
    return combined, summaries

# ============================================================
# 8) MAIN PIPELINE — SEQUENTIAL
# ============================================================

def run_pipeline():
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.reset_index(drop=True, inplace=True)
    print(f"Processing {len(df)} rows")

    needed_cols = [
        "relevant_paper_sum1", "relevant_paper_sum2", "relevant_paper_sum3",
        "novelty_report", "full_chat_history",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = "PENDING"

    agent_col_names = [name for name, _ in SPECIALIST_AGENTS]
    for agent_name in agent_col_names:
        col = f"{agent_name}_limitations"
        if col not in df.columns:
            df[col] = "PENDING"

    for i in tqdm(range(len(df)), desc="LLaMA-3-8B Novelty Analysis"):
        row = df.iloc[i]
        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, ""))
        rel_list = parse_relevant_list(row.get(RELATED_COL, ""))

        if len(main_text_raw) < 200:
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        print(f"\n  Row {i}: building Paper-B summaries …")
        b_combined, b_summaries = build_b_summaries_from_list(rel_list, k=3)
        combined_input = build_dual_paper_input(main_text_raw, b_combined)

        df.at[df.index[i], "relevant_paper_sum1"] = b_summaries[0]
        df.at[df.index[i], "relevant_paper_sum2"] = b_summaries[1] if len(b_summaries) > 1 else ""
        df.at[df.index[i], "relevant_paper_sum3"] = b_summaries[2] if len(b_summaries) > 2 else ""

        chat_history = {}

        try:
            # ---- PHASE 1: specialists ----
            specialist_outputs = []
            for agent_name, agent_prompt in SPECIALIST_AGENTS:
                print(f"    Running {agent_name} …")
                system = HARSH_REVIEWER_POLICY + "\n\n" + agent_prompt
                user_msg = (
                    f"Analyze the following paper for limitations in your domain. "
                    f"Follow the output format strictly.\n\n{combined_input}"
                )
                output = call_llm(system, user_msg, max_new_tokens=MAX_NEW_TOKENS_SPECIALIST)
                specialist_outputs.append((agent_name, output))
                chat_history[agent_name] = output
                df.at[df.index[i], f"{agent_name}_limitations"] = output

            # ---- PHASE 2: master ----
            print("    Running Master_Agent …")
            handoff_parts = [f"[{name}]\n{out}" for name, out in specialist_outputs]
            handoff_block = (
                "=== MASTER_HANDOFF_START ===\n"
                + "\n\n".join(handoff_parts)
                + "\n=== MASTER_HANDOFF_END ==="
            )
            handoff_block = truncate_to_tokens(handoff_block, 10_000)

            master_user_msg = (
                f"Here are the specialist limitation analyses:\n\n"
                f"{handoff_block}\n\n"
                f"Produce the final consolidated novelty & significance limitations report."
            )
            final_output = call_llm(
                MASTER_AGENT_PROMPT, master_user_msg,
                max_new_tokens=MAX_NEW_TOKENS_MASTER,
            )
            chat_history["Master_Agent"] = final_output

            final_output = final_output.replace("TERMINATE", "").strip()
            if len(final_output) < 50 or final_output.startswith("ERROR"):
                final_output = "NO_REPORT_GENERATED"

            df.at[df.index[i], "novelty_report"] = final_output
            df.at[df.index[i], "full_chat_history"] = str(chat_history)

        except Exception as e:
            print(f"  ✗ Error on row {i}: {e}")
            df.at[df.index[i], "novelty_report"] = f"ERROR: {repr(e)}"

        if i % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  💾 Checkpoint saved at row {i}")

        time.sleep(0.1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done — saved to {OUTPUT_CSV}")

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    run_pipeline()