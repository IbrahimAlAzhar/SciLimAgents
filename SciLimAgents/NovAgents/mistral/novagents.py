#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 2.5-3B Instruct — Sequential Novelty & Significance Limitation Analysis
Adapted from the GPT-4o-mini AutoGen group-chat version.

Changes vs. original:
  - Model loaded locally via transformers (no OpenAI API, no vLLM server).
  - AutoGen group chat replaced by sequential calls (3B models can't
    reliably do speaker selection or verifier-based convergence).
  - Verifier convergence loop replaced by a single-pass + one refinement
    round per specialist (practical ceiling for a 3B model).
  - All specialist prompts, master prompt, and output schema are UNCHANGED.
  - Paper-B summarisation now uses the same local Qwen model.
"""

import os
import sys
import time
import ast
import re
import json
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tiktoken

# ============================================================
# 0) PATHS & CONFIGURATION
# ============================================================
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR  = "models/mistral_7b_v3_instruct"

INPUT_CSV  = "llm_agents/gpt/novagents/data/df_with_retrieved_sections.csv"
OUTPUT_DIR = "llm_agents/mistral/novagents"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "df_mistral_7b_novelty_agents_output.csv")


TEXT_COL_MAIN = "input_text_cleaned"
RELATED_COL   = "relevant_papers_list"

# Token budgets (conservative for 3B / 32 K context)
PAPER_TOK_BUDGET          = 18_000
CITATION_TOK_BUDGET       = 5_000
PER_RETR_PAPER_INPUT_TOK  = 2_500
PER_RETR_PAPER_SUMMARY_TOK = 600
MAX_NEW_TOKENS_SPECIALIST  = 1_500
MAX_NEW_TOKENS_MASTER      = 2_500
MAX_NEW_TOKENS_SUMMARY     = 700
MODEL_MAX_INPUT_TOKENS     = 28_000  # hard guard

TEMPERATURE = 0.2

# tiktoken for counting (approx; real tokeniser loaded later)
try:
    _tiktoken_enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")

# ============================================================
# 1) MODEL LOADING
# ============================================================
print("Loading Mistral-7B-Instruct-v0.3 model …")
qwen_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
qwen_model.eval()
print("Model loaded ✓")

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

# Agent name → (prompt, display label)
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
    # ids = _tiktoken_enc.encode(text) 
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
# 6) LOCAL LLM CALL
# ============================================================

def call_llm(system_prompt: str, user_message: str,
             max_new_tokens: int = 1500) -> str:
    """Run a single inference call on the locally-loaded Mistral model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    try:
        input_text = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_tokenizer(input_text, return_tensors="pt").to(qwen_model.device)
        input_len = inputs["input_ids"].shape[1]

        # Hard guard on context length
        if input_len > MODEL_MAX_INPUT_TOKENS:
            print(f"    ⚠ Input too long ({input_len} tok) — truncating to {MODEL_MAX_INPUT_TOKENS}")
            inputs["input_ids"] = inputs["input_ids"][:, :MODEL_MAX_INPUT_TOKENS]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, :MODEL_MAX_INPUT_TOKENS]
            input_len = MODEL_MAX_INPUT_TOKENS

        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        generated = outputs[0][input_len:]
        return qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    except torch.cuda.OutOfMemoryError:
        print("    ✗ CUDA OOM — clearing cache")
        torch.cuda.empty_cache()
        return "ERROR: CUDA out of memory"
    except Exception as e:
        print(f"    ✗ LLM error: {e}")
        return f"ERROR: {e}"

def summarise_paper_b(raw_text: str) -> str:
    """Summarise a retrieved paper using the local model (replaces OpenAI call)."""
    raw_text = truncate_to_tokens(raw_text, PER_RETR_PAPER_INPUT_TOK)
    system = "You are a concise academic summariser. Output a short summary suitable for limitations comparison."
    user = f"Summarize the following paper:\n{raw_text}"
    result = call_llm(system, user, max_new_tokens=MAX_NEW_TOKENS_SUMMARY)
    torch.cuda.empty_cache()
    return result

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
    print(f"Processing {len(df)} rows")

    needed_cols = [
        "relevant_paper_sum1", "relevant_paper_sum2", "relevant_paper_sum3",
        "novelty_report", "full_chat_history",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = "PENDING"

    # Per-agent columns for individual specialist limitations
    agent_col_names = [name for name, _ in SPECIALIST_AGENTS]
    for agent_name in agent_col_names:
        col = f"{agent_name}_limitations"
        if col not in df.columns:
            df[col] = "PENDING"

    for i in tqdm(range(len(df)), desc="Mistral-7B Novelty Analysis"):
        row = df.iloc[i]
        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, ""))
        rel_list = parse_relevant_list(row.get(RELATED_COL, ""))

        if len(main_text_raw) < 200:
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        # ---- Paper-B summaries ----
        print(f"\n  Row {i}: building Paper-B summaries …")
        b_combined, b_summaries = build_b_summaries_from_list(rel_list, k=3)
        combined_input = build_dual_paper_input(main_text_raw, b_combined)

        df.at[df.index[i], "relevant_paper_sum1"] = b_summaries[0]
        df.at[df.index[i], "relevant_paper_sum2"] = b_summaries[1] if len(b_summaries) > 1 else ""
        df.at[df.index[i], "relevant_paper_sum3"] = b_summaries[2] if len(b_summaries) > 2 else ""

        chat_history = {}

        try:
            # ---- PHASE 1: Run each specialist sequentially ----
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

                # Save individual agent output to its dedicated column
                df.at[df.index[i], f"{agent_name}_limitations"] = output

                torch.cuda.empty_cache()

            # ---- PHASE 2: Master Agent synthesises ----
            print("    Running Master_Agent …")

            # Build the handoff block the master expects
            handoff_parts = []
            for agent_name, output in specialist_outputs:
                handoff_parts.append(f"[{agent_name}]\n{output}")
            handoff_block = (
                "=== MASTER_HANDOFF_START ===\n"
                + "\n\n".join(handoff_parts)
                + "\n=== MASTER_HANDOFF_END ==="
            )

            # Truncate if the combined specialist outputs are too long
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
            torch.cuda.empty_cache()

            # Clean up TERMINATE if the model appended it
            final_output = final_output.replace("TERMINATE", "").strip()

            if len(final_output) < 50 or final_output.startswith("ERROR"):
                final_output = "NO_REPORT_GENERATED"

            df.at[df.index[i], "novelty_report"] = final_output
            df.at[df.index[i], "full_chat_history"] = str(chat_history)

        except Exception as e:
            print(f"  ✗ Error on row {i}: {e}")
            df.at[df.index[i], "novelty_report"] = f"ERROR: {repr(e)}"
            torch.cuda.empty_cache()

        if i % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  💾 Checkpoint saved at row {i}")

        time.sleep(0.3)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done — saved to {OUTPUT_CSV}")

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    run_pipeline()