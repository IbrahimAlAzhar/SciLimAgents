import os
import sys
import time
import ast
import re
import signal
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random

# ============================================================
# 1) CONFIG
# ============================================================

# Input/Output
INPUT_CSV = "data/balanced_data/df_with_retrieved_sections.csv" 
OUTPUT_DIR = "llm_agents/llama_3_8B_inst/new/limagents_part1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df_llama3_8b_7_agents_novelty_output.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Row slice (Python slicing: [START:END] => END is exclusive)
START_ROW = 0
END_ROW = 100 # 300

# Column names
TEXT_COL = "input_text_cleaned"
CITED_COL = "cited_in"

# Novelty columns (same as Mistral code)
NOVELTY_INPUT_COL = "input_text_for_novelty"
RELEVANT_LIST_COL = "relevant_papers_list"
RELEVANT_SUM_COL = "relevant_papers_sum"

# LLaMA 3 8B model
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "llama3_8b_instruct"

# Sampling
TEMPERATURE = 0.3
DO_SAMPLE = True

# ── Context budget (LLaMA 3 8B: 8k context window) ──────────────────────────
# Keep a safety buffer so we never overflow the rope embeddings.
MAX_CONTEXT_TOKENS = 8000
MAX_NEW_TOKENS = 700          # output tokens per agent call
# Prompt token budget: context - generation - template overhead
MAX_PROMPT_TOKENS = MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS - 512   # = 6788

# Token budgets for paper / citations inside prompts (conservative for 8k)
PAPER_TOKEN_BUDGET = 3500     # input_text_for_novelty (main paper for novelty agents)
CITATION_TOKEN_BUDGET = 1000  # relevant_papers_sum  +  extracted citations

# Save frequency
SAVE_EVERY = 5
SLEEP_SEC = 0.5

# ============================================================
# 2) LOAD LLAMA 3 8B (4-bit quantization to fit GPU memory)
# ============================================================

print("Loading LLaMA 3 8B model and tokenizer...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3) SAFE TEXT HELPERS
# ============================================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_intro_and_abstract(cited_entry):
    """
    cited_entry expected to be dict-like (or str repr) with paper info:
    title / abstractText or abstract / sections...
    """
    if pd.isna(cited_entry) or cited_entry is None:
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""

    processed = []
    for idx, (_, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue

        intro = ""
        for sec in data.get("sections", []):
            if not isinstance(sec, dict):
                continue
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break

        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)

        if t_clean or a_clean or i_clean:
            processed.append(
                f"'Paper{idx}_Title: {t_clean}', "
                f"'Paper{idx}_Abstract': '{a_clean}', "
                f"'Paper{idx}_Introduction': '{i_clean}'."
            )
    return "\n".join(processed)

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate text to at most max_tokens tokens."""
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "... [TRUNCATED]"

def _messages_token_len(messages: List[Dict[str, str]]) -> int:
    """Measure actual tokenized length of a messages list via the chat template."""
    tmp = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return int(tmp.shape[-1])

def truncate_user_prompt_for_context(system_prompt: Optional[str], user_prompt: str) -> str:
    """
    Binary-search truncation: shrinks the user prompt until the full
    chat-templated messages fit within MAX_PROMPT_TOKENS.
    Mirrors the Mistral approach exactly, adapted for LLaMA 3's 8k window.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})

    cur_len = _messages_token_len(msgs)
    if cur_len <= MAX_PROMPT_TOKENS:
        return user_prompt  # already fits

    user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    low, high = 512, len(user_ids)
    best = low

    while low <= high:
        mid = (low + high) / 2
        trial_prompt = tokenizer.decode(user_ids[:mid], skip_special_tokens=True)
        trial_msgs = []
        if system_prompt:
            trial_msgs.append({"role": "system", "content": system_prompt})
        trial_msgs.append({"role": "user", "content": trial_prompt})

        if _messages_token_len(trial_msgs) <= MAX_PROMPT_TOKENS:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    truncated = tokenizer.decode(user_ids[:best], skip_special_tokens=True)
    return truncated + "\n\n... [TRUNCATED TO FIT CONTEXT]"

# ============================================================
# 4) RELEVANT PAPERS: PARSE + SUMMARIZE  (from Mistral code)
# ============================================================

def parse_relevant_papers_list(x) -> List[str]:
    """Convert a cell (stringified list / list / other) into list[str]."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i) for i in parsed if str(i).strip()]
            return [str(parsed)]
        except Exception:
            return [s]
    return [str(x)]

def summarization_prompt_for_relevant_paper(item_text: str, idx: int) -> str:
    return f"""You will summarize one relevant paper for novelty comparison. Produce a compact, structured summary focusing ONLY on the following dimensions:

1) Literature analysis
2) Data analysis
3) Hypothesis refinement and critical reflection
4) Methodological novelty
5) Experimental novelty
6) Problem formulation novelty
7) Writing/claim novelty

Rules:
- Use plain text only.
- Keep it concise and information-dense.
- If a dimension is not stated in the text, write "Not stated".
- Do not invent details.

Relevant paper #{idx} text:
{item_text}

Output format (STRICT):
Relevant Paper #{idx} Summary:
- Literature analysis: ...
- Data analysis: ...
- Hypothesis refinement and critical reflection: ...
- Methodological novelty: ...
- Experimental novelty: ...
- Problem formulation novelty: ...
- Writing/claim novelty: ...
""".strip()

def summarize_relevant_papers_list(relevant_list: List[str], max_items: int = 3) -> str:
    """
    Summarize each item from relevant_papers_list using LLaMA,
    then combine into one string stored in RELEVANT_SUM_COL.
    Uses a smaller token budget suitable for the 8k context window.
    """
    if not relevant_list:
        return ""

    take = relevant_list[:max_items]
    summaries = []

    for j, item in enumerate(take, start=1):
        item_clean = clean_text_detailed(item)
        # Tighter input budget for summarization calls (8k window constraint)
        item_tr = truncate_to_tokens(item_clean, max_tokens=2500)
        sp = summarization_prompt_for_relevant_paper(item_tr, j)
        # Smaller generation for summaries to leave room for the main agents
        out = llama_generate(sp, system_prompt="You are an expert scientific summarizer.", max_new_tokens=500)
        summaries.append(out.strip())

    return "\n\n".join(summaries).strip()

# ============================================================
# 5) PROMPTS  (identical to Mistral code)
# ============================================================

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
The review_leader will ask for your feedback; respond thoroughly and ask clarifying questions if needed. When finished, inform the review_leader and provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
Identify issues like unstated or overly strong assumptions, incomplete theoretical analysis, errors in derivations, methods that only work under restricted conditions not clearly acknowledged, missing ablations, lack of isolation of individual contributions, or ablations that do not convincingly attribute performance gains.
The review_leader will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the review_leader and deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
Highlight issues like small-scale experiments, missing error bars or confidence intervals, unreported failed experiments, outdated or weak baselines, missing key competitors, unfair hyperparameter tuning, reliance on misleading metrics, missing standard metrics, or overemphasis on minor gains without practical or statistical significance.
The review_leader will interact with you; respond critically and seek clarification if needed. When finished, inform the review_leader and provide a bullet list of experimental evaluation-related limitations, including validation, comparisons, baselines, and metrics.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings (e.g., different datasets, domains, noise, adversarial conditions), is practical in terms of resources (time, memory, hardware, scalability), and addresses genuine deployment needs without ignoring real-world constraints.
Point out limitations like overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, poor performance under shifts, excessive training/inference demands, high resource needs restricting deployment, reliance on synthetic data, ignoring constraints like cost or latency, lack of user studies or field tests, or over-optimistic assumptions about environments.
The review_leader will seek your input; respond thoroughly and clarify ambiguities. When finished, inform the review_leader and provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations of methods, settings, concepts, or organization hindering understanding; lack of explainability or insights into decisions; and insufficient details for replication, such as code, data, hyperparameters, or protocols.
Identify issues like ambiguities, unstated assumptions, vague terms undermining comprehension, black-box behavior without explanations, missing feature importance or mechanistic understanding, poorly organized sections, missing code/data release, unreported seeds, ambiguous procedures, or lack of open science practices.
The review_leader will ask questions; respond and ask follow-up questions if needed. When done, inform the review_leader and provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, including suggestions for improvement where relevant.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for issues in collection, labeling, cleaning, representativeness, or documentation; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Point out limitations such as small or non-diverse data, labeling errors, undocumented preprocessing, data leakage, reliance on flawed datasets without validation, biased outcomes leading to discrimination, lack of fairness metrics, unreported subgroup performance, ethical oversights, or failure to discuss misuse potential.
The review_leader will consult you; provide evidence-based critique and ask clarifying questions. When done, inform the review_leader and provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_master_synthesis_prompt(paper_content: str, specialist_outputs: dict) -> str:
    sections = [
        ("Novelty & Significance",                   specialist_outputs.get("Novelty_Significance_Agent", "")),
        ("Citation Analysis",                         specialist_outputs.get("Citation_Agent", "")),
        ("Theoretical & Methodological",              specialist_outputs.get("Theoretical_Methodological_Agent", "")),
        ("Experimental Evaluation",                   specialist_outputs.get("Experimental_Evaluation_Agent", "")),
        ("Generalization / Robustness / Efficiency",  specialist_outputs.get("Generalization_Robustness_Efficiency_Agent", "")),
        ("Clarity / Interpretability / Reproducibility", specialist_outputs.get("Clarity_Interpretability_Reproducibility_Agent", "")),
        ("Data / Ethics",                             specialist_outputs.get("Data_Ethics_Agent", "")),
    ]

    joined = [f"=== {title} ===\n{content}".strip() for title, content in sections]
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
# 6) GRACEFUL EXIT (emergency save)
# ============================================================

global_df = None
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
# 7) LLAMA GENERATION  (mirrors mistral_generate with 8k budget)
# ============================================================

def llama_generate(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate text using LLaMA 3's chat template.
    Binary-search truncation (same as Mistral) keeps prompts inside the 8k window.
    """
    # Truncate to fit context before building final messages
    user_prompt = truncate_user_prompt_for_context(system_prompt, user_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return text.strip()

# ============================================================
# 8) MAIN PIPELINE  (sequential 7 agents + master)
# ============================================================

def run_pipeline():
    global global_df, global_current_row

    print("Loading CSV file...")
    df_all = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_all)} rows total.")

    df_slice = df_all.iloc[START_ROW:END_ROW].copy()
    df_slice = df_slice.reset_index(drop=False).rename(columns={"index": "orig_index"})
    global_df = df_slice

    # Output columns
    cols = [
        "Novelty_Significance_Agent",
        "Citation_Agent",
        "Theoretical_Methodological_Agent",
        "Experimental_Evaluation_Agent",
        "Generalization_Robustness_Efficiency_Agent",
        "Clarity_Interpretability_Reproducibility_Agent",
        "Data_Ethics_Agent",
        "final_merged_limitations",
    ]
    for c in cols:
        if c not in df_slice.columns:
            df_slice[c] = ""

    # Ensure relevant_papers_sum column exists
    if RELEVANT_SUM_COL not in df_slice.columns:
        df_slice[RELEVANT_SUM_COL] = ""

    print(
        f"Starting sequential 7-agent generation (LLaMA 3 8B) "
        f"for rows {START_ROW}:{END_ROW} ({len(df_slice)} rows)..."
    )

    for r in tqdm(range(len(df_slice))):
        global_current_row = r
        row = df_slice.iloc[r]

        # ── Novelty inputs (mirrors Mistral exactly) ──────────────────────
        main_text   = str(row.get(NOVELTY_INPUT_COL, "") or "")
        relevant_raw = row.get(RELEVANT_LIST_COL, "")

        # Citation pipeline (keep original columns)
        paper_text_for_citation = str(row.get(TEXT_COL, "") or "")
        cited_raw    = row.get(CITED_COL, "")
        citation_text = extract_intro_and_abstract(cited_raw)

        # ── Parse & summarize relevant_papers_list ────────────────────────
        rel_list = parse_relevant_papers_list(relevant_raw)
        rel_sum  = summarize_relevant_papers_list(rel_list, max_items=3)
        df_slice.at[r, RELEVANT_SUM_COL] = rel_sum

        # ── Build combined_for_novelty (main paper + summarized relevant papers)
        # Truncated to 8k-safe budgets
        main_text_tr = truncate_to_tokens(main_text, PAPER_TOKEN_BUDGET)
        rel_sum_tr   = truncate_to_tokens(rel_sum,   CITATION_TOKEN_BUDGET)

        combined_for_novelty = (
            "=== INPUT PAPER ===\n"
            f"{main_text_tr}\n\n"
            "=== RELEVANT PAPERS (SUMMARIZED) ===\n"
            f"{rel_sum_tr}"
        ).strip()

        # Citation agent uses the original paper text + extracted citations
        citation_text_tr = truncate_to_tokens(citation_text, CITATION_TOKEN_BUDGET)

        outputs = {}

        try:
            # 1) Novelty / Significance  ← combined_for_novelty
            p   = get_novelty_significance_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert reviewer.")
            outputs["Novelty_Significance_Agent"] = out
            df_slice.at[r, "Novelty_Significance_Agent"] = out

            # 2) Citation Agent  ← paper_text + extracted citations
            p   = get_citation_agent_prompt(paper_text_for_citation, citation_text_tr)
            out = llama_generate(p, system_prompt="You are an expert citation analyst.")
            outputs["Citation_Agent"] = out
            df_slice.at[r, "Citation_Agent"] = out

            # 3) Theoretical / Methodological  ← combined_for_novelty
            p   = get_theoretical_methodological_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert methodologist.")
            outputs["Theoretical_Methodological_Agent"] = out
            df_slice.at[r, "Theoretical_Methodological_Agent"] = out

            # 4) Experimental Evaluation  ← combined_for_novelty
            p   = get_experimental_evaluation_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert experimentalist.")
            outputs["Experimental_Evaluation_Agent"] = out
            df_slice.at[r, "Experimental_Evaluation_Agent"] = out

            # 5) Generalization / Robustness / Efficiency  ← combined_for_novelty
            p   = get_generalization_robustness_efficiency_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert on robustness and efficiency.")
            outputs["Generalization_Robustness_Efficiency_Agent"] = out
            df_slice.at[r, "Generalization_Robustness_Efficiency_Agent"] = out

            # 6) Clarity / Interpretability / Reproducibility  ← combined_for_novelty
            p   = get_clarity_interpretability_reproducibility_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert on clarity and reproducibility.")
            outputs["Clarity_Interpretability_Reproducibility_Agent"] = out
            df_slice.at[r, "Clarity_Interpretability_Reproducibility_Agent"] = out

            # 7) Data / Ethics  ← combined_for_novelty
            p   = get_data_ethics_prompt(combined_for_novelty)
            out = llama_generate(p, system_prompt="You are an expert on data and ethics.")
            outputs["Data_Ethics_Agent"] = out
            df_slice.at[r, "Data_Ethics_Agent"] = out

            # ── Master synthesis ──────────────────────────────────────────
            master_prompt = get_master_synthesis_prompt(combined_for_novelty, outputs)
            final_out = llama_generate(master_prompt, system_prompt="You are the Master Agent.")
            df_slice.at[r, "final_merged_limitations"] = final_out

        except Exception as e:
            df_slice.at[r, "final_merged_limitations"] = f"ERROR: {e}"
            print(f"Error on row r={r} (orig_index={row.get('orig_index')}): {e}")

        # Periodic saves
        if r % SAVE_EVERY == 0:
            df_slice.to_csv(OUTPUT_FILE, index=False)

        time.sleep(SLEEP_SEC)

    df_slice.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()