"""
Ablation study: leave-one-agent-out master synthesis.

For each row in the input CSV (which already contains the 7 specialist agent
outputs), we run ONLY the master agent seven times, each time excluding one
specialist's output. The resulting consolidated limitations list is saved into
a new column named  final_merged_without_<agent_name>.

Model: local Llama-3-8B-Instruct via vLLM.
"""

import os
import sys
import time
import ast
import re
import signal
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ============================================================
# 1) CONFIG
# ============================================================

INPUT_CSV = (
    ""
    "llm_agents/llama_3_8B_new/Limagents/limagents/"
    "df_llama3_8b_7_agents_novelty_output.csv"
)

OUTPUT_DIR = (
    "llm_agents/llama_3_8B_new/Limagents/limagents/ablation"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ablation_leave_one_out.csv")

# Local model files
MODEL_PATH = "llama3_8b_instruct/Meta-Llama-3-8B-Instruct"

# Generation
TEMPERATURE = 0.3
MAX_NEW_TOKENS_MASTER = 1500
MODEL_MAX_LEN = 8192
# Leave room for generation + chat-template overhead
MODEL_MAX_INPUT_TOKENS = MODEL_MAX_LEN - MAX_NEW_TOKENS_MASTER - 256  # = 6436

# Save cadence
SAVE_EVERY = 10

# Optional row slice (set to None to process all rows in the input CSV)
START_ROW: Optional[int] = None
END_ROW: Optional[int] = None

# Paper-content column to use as the "context" inside the master prompt
# (same column the original pipeline fed into the master agent)
NOVELTY_INPUT_COL = "input_text_for_novelty"
RELEVANT_SUM_COL = "relevant_papers_sum"

# Token budgets for the paper-context portion of the master prompt
PAPER_TOKEN_BUDGET = 3500
RELEVANT_TOKEN_BUDGET = 1000

# The seven specialist columns and the readable label used inside the prompt.
SPECIALIST_COLUMNS = [
    ("Novelty_Significance_Agent",                    "Novelty & Significance"),
    ("Citation_Agent",                                 "Citation Analysis"),
    ("Theoretical_Methodological_Agent",               "Theoretical & Methodological"),
    ("Experimental_Evaluation_Agent",                  "Experimental Evaluation"),
    ("Generalization_Robustness_Efficiency_Agent",     "Generalization / Robustness / Efficiency"),
    ("Clarity_Interpretability_Reproducibility_Agent", "Clarity / Interpretability / Reproducibility"),
    ("Data_Ethics_Agent",                              "Data / Ethics"),
]

# ============================================================
# 2) MODEL + TOKENIZER (vLLM, local files only)
# ============================================================

print("Loading tokenizer (local files only)...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading vLLM engine...")
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    max_model_len=MODEL_MAX_LEN,
    enforce_eager=False,
    disable_log_stats=True,
)
print("vLLM engine ready ✓")

# ============================================================
# 3) HELPERS
# ============================================================

def clean_text_detailed(text) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "... [TRUNCATED]"

def call_llm(system_prompt: str, user_message: str,
             max_new_tokens: int = MAX_NEW_TOKENS_MASTER) -> str:
    """Single inference call through local vLLM engine."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Hard guard on prompt length
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_ids) > MODEL_MAX_INPUT_TOKENS:
            print(f"    ⚠ Input too long ({len(input_ids)} tok) — truncating to {MODEL_MAX_INPUT_TOKENS}")
            input_ids = input_ids[:MODEL_MAX_INPUT_TOKENS]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # LLaMA-3 stop tokens
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

# ============================================================
# 4) MASTER PROMPT (leave-one-out variant)
# ============================================================

MASTER_SYSTEM_PROMPT = "You are the Master Agent."

def build_master_prompt(paper_content: str,
                        included_sections: List[tuple]) -> str:
    """
    included_sections: list of (title, content) tuples — already excludes the
    ablated agent. Falsy contents are skipped (so a row with a missing
    specialist output still produces a clean prompt).
    """
    joined = []
    for title, content in included_sections:
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        content = content.strip()
        if not content:
            continue
        joined.append(f"=== {title} ===\n{content}")
    all_reports = "\n\n".join(joined)

    return f"""You are the **Master Agent**. Your role is to receive limitation analyses from multiple specialist agents and produce a single, final, high-quality, consolidated list of limitations for the scientific paper.

    TASK:
    - Carefully read and integrate all provided specialist outputs below.
    - Remove redundancies (merge similar limitations).
    - Prioritize the most severe and well-justified limitations.
    - Preserve specificity and evidence from the original analyses.
    - Organize the final list logically by category.
    - Ensure each limitation is clearly stated, concise, and grounded in the paper.
    - Avoid introducing new limitations not raised by the specialists.

    OUTPUT FORMAT:
    Start with: "Here is the consolidated list of key limitations identified in the paper:"
    Then bullets:
    - **Category:** Specific limitation statement (brief explanation / evidence if useful)
    If specialists found only minor issues, say so and list them.

    PAPER CONTENT (context):
    {paper_content}

    SPECIALIST OUTPUTS (ONLY use these; do not invent new limitations):
    {all_reports}
    """

# ============================================================
# 5) GRACEFUL EXIT
# ============================================================

global_df: Optional[pd.DataFrame] = None
global_current_row = -1

def signal_handler(signum, frame):
    print(f"\n⚠️  Received signal {signum}. Saving progress...")
    if global_df is not None:
        save_path = os.path.join(OUTPUT_DIR, f"emergency_save_row_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================
# 6) MAIN
# ============================================================

def build_paper_context(row: pd.Series) -> str:
    """Reconstruct the same paper-context block the original pipeline fed to
    the master agent: truncated main paper + summarized relevant papers."""
    main_text = str(row.get(NOVELTY_INPUT_COL, "") or "")
    rel_sum   = str(row.get(RELEVANT_SUM_COL, "") or "")

    main_text_tr = truncate_to_tokens(main_text, PAPER_TOKEN_BUDGET)
    rel_sum_tr   = truncate_to_tokens(rel_sum,   RELEVANT_TOKEN_BUDGET)

    return (
        "=== INPUT PAPER ===\n"
        f"{main_text_tr}\n\n"
        "=== RELEVANT PAPERS (SUMMARIZED) ===\n"
        f"{rel_sum_tr}"
    ).strip()

def run_pipeline():
    global global_df, global_current_row

    print(f"Loading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")

    if START_ROW is not None or END_ROW is not None:
        s = START_ROW or 0
        e = END_ROW or len(df)
        df = df.iloc[s:e].copy().reset_index(drop=False).rename(columns={"index": "orig_index"})
        print(f"Sliced to rows {s}:{e} → {len(df)} rows.")
    else:
        df = df.copy()

    # Make sure every ablation column exists up-front
    ablation_cols = [f"final_merged_without_{col}" for col, _ in SPECIALIST_COLUMNS]
    for c in ablation_cols:
        if c not in df.columns:
            df[c] = ""

    global_df = df

    print(f"Running leave-one-out master ablation for {len(df)} rows × "
          f"{len(SPECIALIST_COLUMNS)} agents each...")

    for r in tqdm(range(len(df))):
        global_current_row = r
        row = df.iloc[r]

        paper_context = build_paper_context(row)

        # Gather the seven specialist outputs for this row
        specialist_outputs = []
        for col, title in SPECIALIST_COLUMNS:
            specialist_outputs.append((col, title, row.get(col, "")))

        # Run the master agent 7 times — once per ablated specialist
        for ablated_col, ablated_title, _ablated_content in specialist_outputs:
            included = [
                (title, content)
                for col, title, content in specialist_outputs
                if col != ablated_col
            ]

            try:
                prompt = build_master_prompt(paper_context, included)
                out = call_llm(MASTER_SYSTEM_PROMPT, prompt,
                               max_new_tokens=MAX_NEW_TOKENS_MASTER)
            except Exception as e:
                out = f"ERROR: {e}"
                print(f"Row {r} | ablating {ablated_col} | error: {e}")

            df.at[df.index[r], f"final_merged_without_{ablated_col}"] = out

        # Save every SAVE_EVERY rows
        if (r + 1) % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"  💾 saved progress at row {r + 1}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()