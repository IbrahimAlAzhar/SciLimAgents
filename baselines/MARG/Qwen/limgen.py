"""
MARG Pipeline — Qwen 2.5 3B Instruct (Local HuggingFace)
Replaces Mistral 7B Instruct v0.3 with Qwen2.5-3B-Instruct.
All prompts are kept identical to the original.
"""

import os
import ast
import re
import time
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Column names
TEXT_COL = "input_text_cleaned"
CITED_COL = "cited_in"

# Model
MODEL_ID     = "qwen2_5_3b_instruct"
CACHE_DIR    = "qwen2_5_3b_instruct"

# Generation / context
MAX_NEW_TOKENS      = 768
MAX_CONTEXT_TOKENS  = 8000
TEMPERATURE         = 0.3

# Token budgets for paper/citations inside prompts (conservative for 8k context)
PAPER_TOKEN_BUDGET    = 5200
CITATION_TOKEN_BUDGET = 1200

# Paths
INPUT_CSV    = "data/balanced_data/df_updated_with_retrieval.csv"
OUTPUT_SLICE = "MARG/Qwen/df_marg_qwen2_5_3b_output.csv"

SAVE_EVERY = 5
SLEEP_SEC  = 0.2

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Build a text-generation pipeline (handles chat formatting internally)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    return_full_text=False,   # return only the generated continuation
)

print("Model loaded.\n")

# ==========================================
# 3. MARG PROMPT DEFINITIONS  (unchanged)
# ==========================================

def get_marg_impact_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Impact & Novelty Expert** (MARG-Impact).
ROLE: You are highly skeptical of the paper's claims. Your goal is to identify limitations regarding the **significance, novelty, and hidden assumptions** of the work.

STRATEGY:
1. **Analyze Motivation:** Does the paper clearly justify its goals? Are the motivating problems real or contrived?
2. **Check Assumptions:** Identify "hidden assumptions" (e.g., assuming a robot is omnidirectional, assuming clean data). If the paper fails to justify these, it is a significant limitation.
3. **Verify Novelty:** Compare the paper's contribution against the provided 'CITED PAPERS INFO'. Does it merely rebrand existing work? Is the "gap" in literature real?
4. **Scope:** Does the method only work in narrow settings?

OUTPUT INSTRUCTIONS:
- Provide a bullet list of **Implicit Limitations** related to scope, novelty, and assumptions.
- Be specific. Do not say "The scope is limited." Say "The method is limited because it assumes X, which is rare in real-world settings."

=== MAIN PAPER CONTENT ===
{paper_content}

=== CITED PAPERS INFO ===
{citation_content}"""

def get_marg_experiments_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Methodology & Experiments Expert** (MARG-Experiments).
ROLE: You are an expert scientist who designs high-quality evaluations. Your goal is to identify **methodological flaws and missing experiments**.

STRATEGY:
1. **Hypothesize Ideal Experiments:** Before judging, imagine what experiments *should* be run to rigorously prove the paper's claims (e.g., specific baselines, ablations, statistical tests).
2. **Gap Analysis:** Compare your "Ideal Experiments" to the "Actual Experiments" in the text.
   - Missing Baselines?
   - Missing Ablation Studies (component analysis)?
   - Weak Metrics?
3. **Verify Support:** Do the results actually support the strong claims made? Look for over-claiming based on weak evidence.

OUTPUT INSTRUCTIONS:
- List **Methodological Limitations**.
- Example: "The study is limited by the lack of an ablation study on component X, making it impossible to attribute the performance gains."

PAPER CONTENT:
{paper_content}"""

def get_marg_clarity_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Clarity & Reproducibility Expert** (MARG-Clarity).
ROLE: You have extreme attention to detail. Your goal is to ensure the paper is reproducible and unambiguous.

STRATEGY:
1. **Reproducibility Check:** Look for missing implementation details: hyperparameters, seed numbers, hardware specs, or data filtering steps.
2. **Concept Definitions:** Identify vague terms or "black box" explanations. If a term is used without definition, it is a clarity limitation.
3. **Inconsistencies:** Check for contradictions in the text (e.g., Figure 1 shows X, but text says Y).

OUTPUT INSTRUCTIONS:
- List **Reproducibility & Clarity Limitations**.
- Example: "The method is not reproducible because the authors fail to specify the hyperparameters for the baseline models."

PAPER CONTENT:
{paper_content}"""

def get_marg_master_refinement_prompt(
    paper_content: str,
    impact_output: str,
    experiments_output: str,
    clarity_output: str,
) -> str:
    return f"""You are the **Master Agent** (Refinement Stage).
ROLE: Your job is to take the raw limitations identified by the team and **refine** them into a final, high-quality list.

TASK:
1. **Prune Invalid Comments:** Remove any limitation that is factually incorrect or trivial (e.g., grammar issues).
2. **Categorize:** Group the limitations into two categories:
   - **Explicit Limitations:** Weaknesses explicitly admitted by the authors (e.g., in the Discussion/Conclusion).
   - **Implicit/Methodological Limitations:** Flaws identified by the agents that the authors did not admit (e.g., missing comparisons, hidden assumptions).
3. **Sharpen Specificity:** Ensure every bullet point is detailed.
   - Bad: "The evaluation is weak."
   - Good: "The evaluation is limited by the exclusion of standard datasets (e.g., ImageNet), testing only on synthetic data."

=== RAW AGENT OUTPUTS ===

[Impact_Agent]:
{impact_output}

[Experiments_Agent]:
{experiments_output}

[Clarity_Agent]:
{clarity_output}

=== MAIN PAPER CONTENT ===
{paper_content}

OUTPUT FORMAT:
Start with: "Here is the consolidated, refined list of research limitations:"
Then provide a bulleted list categorized by Explicit vs Implicit limitations."""

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry):
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except:
        return ""
    if not isinstance(parsed, dict):
        return ""

    processed = []
    for idx, (pid, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)
        if t_clean or a_clean or i_clean:
            processed.append(
                f"'Paper{idx}_Title: {t_clean}', 'Paper{idx}_Abstract': '{a_clean}', "
                f"'Paper{idx}_Introduction': '{i_clean}'."
            )
    return "\n".join(processed)

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a token budget using the loaded tokenizer."""
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    print(f"  ⚠️  Truncating: {len(ids)} → {max_tokens} tokens")
    truncated_ids = ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True) + "... [TRUNCATED]"

def call_agent(system_prompt: str) -> str:
    """
    Format a Qwen-Instruct chat message and run inference.
    Returns the model's response as a plain string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Please provide your analysis now."},
    ]

    # Apply the Qwen instruct chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm_pipeline(formatted)
    response = outputs[0]["generated_text"].strip()
    return response

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df_full = pd.read_csv(INPUT_CSV)

        print(f"Loaded {len(df)} rows.\n")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Initialise output columns
    for col in ("final_merged_limitations", "impact_output",
                "experiments_output", "clarity_output", "full_chat_history"):
        if col not in df.columns:
            df[col] = "PENDING"

    for i in tqdm(range(len(df)), desc="MARG rows"):
        row = df.iloc[i]

        # ── Data prep ──────────────────────────────────────────────
        paper_text   = str(row.get(TEXT_COL, ""))
        citation_text = extract_intro_and_abstract(row.get(CITED_COL, ""))

        paper_text    = truncate_text_to_tokens(paper_text,    PAPER_TOKEN_BUDGET)
        citation_text = truncate_text_to_tokens(citation_text, CITATION_TOKEN_BUDGET)

        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue

        # ── Agent calls (sequential, same logic as AutoGen rounds) ─
        try:
            # 1. Impact Agent
            print(f"  [Row {i}] → Impact_Agent")
            impact_out = call_agent(
                get_marg_impact_agent_prompt(paper_text, citation_text)
            )

            # 2. Experiments Agent
            print(f"  [Row {i}] → Experiments_Agent")
            experiments_out = call_agent(
                get_marg_experiments_agent_prompt(paper_text)
            )

            # 3. Clarity Agent
            print(f"  [Row {i}] → Clarity_Agent")
            clarity_out = call_agent(
                get_marg_clarity_agent_prompt(paper_text)
            )

            # 4. Master Agent — synthesise all three outputs
            print(f"  [Row {i}] → Master_Agent (refinement)")
            master_out = call_agent(
                get_marg_master_refinement_prompt(
                    paper_text, impact_out, experiments_out, clarity_out
                )
            )

            # ── Store results ──────────────────────────────────────
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = master_out
            df.iat[i, df.columns.get_loc("impact_output")]            = impact_out
            df.iat[i, df.columns.get_loc("experiments_output")]       = experiments_out
            df.iat[i, df.columns.get_loc("clarity_output")]           = clarity_out

            # Lightweight chat-history record (mirrors original field)
            chat_log = (
                f"[Impact_Agent]\n{impact_out}\n\n"
                f"[Experiments_Agent]\n{experiments_out}\n\n"
                f"[Clarity_Agent]\n{clarity_out}\n\n"
                f"[Master_Agent]\n{master_out}"
            )
            df.iat[i, df.columns.get_loc("full_chat_history")] = chat_log

        except Exception as e:
            print(f"  ✗ Error on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = f"ERROR: {e}"

        # ── Periodic save ──────────────────────────────────────────
        if i % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_SLICE, index=False)
            print(f"  💾 Checkpoint saved at row {i}")

        time.sleep(SLEEP_SEC)

    # Final save
    df.to_csv(OUTPUT_SLICE, index=False)
    print(f"\n✅ Done. Output saved to: {OUTPUT_SLICE}")

if __name__ == "__main__":
    run_pipeline() 
