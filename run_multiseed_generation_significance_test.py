import os
import sys
import gc
import time
import ast
import re
import signal
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ============================================================
# 1) CONFIG
# ============================================================


# =============================================================================
# ENVIRONMENT CONFIGURATION
# All input/output locations are supplied at run time. No paths, dataset sizes
# or credentials are stored in this file.
# =============================================================================
def _require_env(name, hint=""):
    """Return a mandatory environment variable, or exit with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Required environment variable {name} is not set."
            + (f"  Expected: {hint}" if hint else "")
        )
    return value


def _optional_int(name):
    """Return an int env var, or None when unset/blank (meaning 'no limit')."""
    raw = os.environ.get(name, "").strip()
    if raw in ("", "none", "None", "null"):
        return None
    return int(raw)


INPUT_CSV  = _require_env("INPUT_CSV", "source CSV of papers")
OUTPUT_DIR = _require_env("OUTPUT_DIR", "directory to write multi-seed generations into")

OTHER_MODEL_NAME = os.environ.get("OTHER_MODEL_NAME", "model_b")
THIS_MODEL_NAME  = os.environ.get("THIS_MODEL_NAME", "model_a")

OUTPUT_FILE = os.environ.get(
    "OUTPUT_CSV", os.path.join(OUTPUT_DIR, f"df_{THIS_MODEL_NAME}_multiseed.csv"))

# The counterpart model's output CSV (used for the cross-model significance
# test). Required only when the comparison stage is run.
OTHER_MODEL_CSV = os.environ.get("OTHER_MODEL_CSV", "")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional slice of the input file. Both unset => process the whole file.
START_ROW = _optional_int("START_ROW") or 0
END_ROW = _optional_int("END_ROW")

# Multiple seeds for significance testing.
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",") if x.strip() != ""]

# Columns
TEXT_COL = "input_text_cleaned"
CITED_COL = "cited_in"
NOVELTY_INPUT_COL = "input_text_for_novelty"
RELEVANT_LIST_COL = "relevant_papers_list"
RELEVANT_SUM_COL = "relevant_papers_sum"
GT_COL = os.environ.get("GT_COL", "ground_truth_lim_peer")  # reference for scoring

# Model
MODEL_PATH = _require_env("MODEL", "checkpoint directory for the model under test")
CACHE_DIR = MODEL_PATH

# Metric models (set to local dirs if offline).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
BERT_MODEL = os.environ.get("BERT_MODEL", "roberta-large")

# Sampling
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Context budget (Qwen 2.5 3B: 32k window)
SAFE_INPUT_LIMIT = 28000
MAX_NEW_TOKENS = 700
MAX_PROMPT_TOKENS = SAFE_INPUT_LIMIT - MAX_NEW_TOKENS - 512
MODEL_MAX_INPUT_TOKENS = MAX_PROMPT_TOKENS
PAPER_TOKEN_BUDGET = 16000
CITATION_TOKEN_BUDGET = 4000

# vLLM
VLLM_MAX_MODEL_LEN = SAFE_INPUT_LIMIT + MAX_NEW_TOKENS + 256
VLLM_GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.85))
VLLM_DTYPE = os.environ.get("DTYPE", "float16")
VLLM_TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))
VLLM_QUANTIZATION = os.environ.get("QUANTIZATION", "")  # empty for 3B (unquantized)

# This model is Qwen2.5 (no thinking-mode toggle in the chat template).
IS_QWEN3 = False

SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 5))
SLEEP_SEC = 0.2

# ============================================================
# 2) LOAD MODEL VIA vLLM
# ============================================================
print(f"Loading {THIS_MODEL_NAME} tokenizer + vLLM engine from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

_llm_kwargs = dict(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    trust_remote_code=True,
    dtype=VLLM_DTYPE,
    max_model_len=VLLM_MAX_MODEL_LEN,
    gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL,
    enforce_eager=False,
)
if VLLM_QUANTIZATION:
    _llm_kwargs["quantization"] = VLLM_QUANTIZATION
llm = LLM(**_llm_kwargs)

_STOP_TOKEN_IDS: List[int] = []
_imend = tokenizer.convert_tokens_to_ids("<|im_end|>")
if _imend is not None and _imend != tokenizer.unk_token_id:
    _STOP_TOKEN_IDS.append(_imend)
if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in _STOP_TOKEN_IDS:
    _STOP_TOKEN_IDS.append(tokenizer.eos_token_id)


# ============================================================
# 3) TEXT HELPERS
# ============================================================
def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_intro_and_abstract(cited_entry):
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
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "... [TRUNCATED]"


def _apply_template(messages, tokenize=False, return_tensors=None):
    kwargs = dict(add_generation_prompt=True, tokenize=tokenize)
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if IS_QWEN3:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def _messages_token_len(messages: List[Dict[str, str]]) -> int:
    tmp = _apply_template(messages, tokenize=True, return_tensors="pt")
    return int(tmp.shape[-1])


def truncate_user_prompt_for_context(system_prompt: Optional[str], user_prompt: str) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    if _messages_token_len(msgs) <= MAX_PROMPT_TOKENS:
        return user_prompt
    user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    low, high, best = 512, len(user_ids), 512
    while low <= high:
        mid = (low + high) // 2
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
    return tokenizer.decode(user_ids[:best], skip_special_tokens=True) + "\n\n... [TRUNCATED TO FIT CONTEXT]"


def to_text(val):
    if pd.isna(val):
        return ""
    if isinstance(val, list):
        items = val
    else:
        s = str(val)
        try:
            parsed = ast.literal_eval(s)
            items = parsed if isinstance(parsed, list) else [s]
        except (ValueError, SyntaxError):
            items = [s]
    return " ".join(str(i).strip() for i in items).strip()


# ============================================================
# 4) RELEVANT PAPERS PARSE + SUMMARIZE
# ============================================================
def parse_relevant_papers_list(x) -> List[str]:
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
    if not relevant_list:
        return ""
    take = relevant_list[:max_items]
    summaries = []
    for j, item in enumerate(take, start=1):
        item_clean = clean_text_detailed(item)
        item_tr = truncate_to_tokens(item_clean, max_tokens=8000)
        sp = summarization_prompt_for_relevant_paper(item_tr, j)
        out = qwen_generate(sp, system_prompt="You are an expert scientific summarizer.",
                            max_new_tokens=500, seed=SEEDS[0])
        summaries.append(out.strip())
    return "\n\n".join(summaries).strip()


# ============================================================
# 5) PROMPTS
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
        ("Novelty & Significance", specialist_outputs.get("Novelty_Significance_Agent", "")),
        ("Citation Analysis", specialist_outputs.get("Citation_Agent", "")),
        ("Theoretical & Methodological", specialist_outputs.get("Theoretical_Methodological_Agent", "")),
        ("Experimental Evaluation", specialist_outputs.get("Experimental_Evaluation_Agent", "")),
        ("Generalization / Robustness / Efficiency", specialist_outputs.get("Generalization_Robustness_Efficiency_Agent", "")),
        ("Clarity / Interpretability / Reproducibility", specialist_outputs.get("Clarity_Interpretability_Reproducibility_Agent", "")),
        ("Data / Ethics", specialist_outputs.get("Data_Ethics_Agent", "")),
    ]
    all_reports = "\n\n".join(f"=== {t} ===\n{c}".strip() for t, c in sections)
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
# 6) GRACEFUL EXIT
# ============================================================
global_df = None
global_current_row = -1


def signal_handler(signum, frame):
    print(f"\n[signal {signum}] saving progress...")
    if global_df is not None:
        p = os.path.join(OUTPUT_DIR, f"emergency_save_row_{global_current_row}.csv")
        global_df.to_csv(p, index=False)
        print(f"Saved to: {p}")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# 7) GENERATION
# ============================================================
def call_llm(system_prompt: str, user_message: str,
             max_new_tokens: int = MAX_NEW_TOKENS, seed: Optional[int] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    try:
        input_text = _apply_template(messages, tokenize=False)
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_ids) > MODEL_MAX_INPUT_TOKENS:
            input_ids = input_ids[:MODEL_MAX_INPUT_TOKENS]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        sampling = SamplingParams(
            temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_new_tokens,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=_STOP_TOKEN_IDS or None, seed=seed,
        )
        outputs = llm.generate([input_text], sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return "ERROR: empty generation"
        return outputs[0].outputs[0].text.strip()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "ERROR: CUDA out of memory"
    except Exception as e:
        return f"ERROR: {e}"


def qwen_generate(user_prompt: str, system_prompt: Optional[str] = None,
                  max_new_tokens: int = MAX_NEW_TOKENS, seed: Optional[int] = None) -> str:
    user_prompt = truncate_user_prompt_for_context(system_prompt, user_prompt)
    return call_llm(system_prompt or "", user_prompt, max_new_tokens=max_new_tokens, seed=seed)


def run_seven_agents(combined_for_novelty: str, paper_text_for_citation: str,
                     citation_text_tr: str, seed: int) -> Dict[str, str]:
    outputs = {}
    outputs["Novelty_Significance_Agent"] = qwen_generate(
        get_novelty_significance_prompt(combined_for_novelty),
        "You are an expert reviewer.", seed=seed)
    outputs["Citation_Agent"] = qwen_generate(
        get_citation_agent_prompt(paper_text_for_citation, citation_text_tr),
        "You are an expert citation analyst.", seed=seed)
    outputs["Theoretical_Methodological_Agent"] = qwen_generate(
        get_theoretical_methodological_prompt(combined_for_novelty),
        "You are an expert methodologist.", seed=seed)
    outputs["Experimental_Evaluation_Agent"] = qwen_generate(
        get_experimental_evaluation_prompt(combined_for_novelty),
        "You are an expert experimentalist.", seed=seed)
    outputs["Generalization_Robustness_Efficiency_Agent"] = qwen_generate(
        get_generalization_robustness_efficiency_prompt(combined_for_novelty),
        "You are an expert on robustness and efficiency.", seed=seed)
    outputs["Clarity_Interpretability_Reproducibility_Agent"] = qwen_generate(
        get_clarity_interpretability_reproducibility_prompt(combined_for_novelty),
        "You are an expert on clarity and reproducibility.", seed=seed)
    outputs["Data_Ethics_Agent"] = qwen_generate(
        get_data_ethics_prompt(combined_for_novelty),
        "You are an expert on data and ethics.", seed=seed)
    outputs["final"] = qwen_generate(
        get_master_synthesis_prompt(combined_for_novelty, outputs),
        "You are the Master Agent.", seed=seed)
    return outputs


# ============================================================
# 8) MAIN PIPELINE (multi-seed)
# ============================================================
def run_pipeline():
    global global_df, global_current_row
    print("Loading CSV...")
    df_all = pd.read_csv(INPUT_CSV)
    df_slice = df_all.iloc[START_ROW:END_ROW].copy()
    df_slice = df_slice.reset_index(drop=False).rename(columns={"index": "orig_index"})
    global_df = df_slice

    agent_cols = [
        "Novelty_Significance_Agent", "Citation_Agent",
        "Theoretical_Methodological_Agent", "Experimental_Evaluation_Agent",
        "Generalization_Robustness_Efficiency_Agent",
        "Clarity_Interpretability_Reproducibility_Agent", "Data_Ethics_Agent",
    ]
    for c in agent_cols + [RELEVANT_SUM_COL]:
        if c not in df_slice.columns:
            df_slice[c] = ""
    for s in SEEDS:
        col = f"final_merged_limitations_seed{s}"
        if col not in df_slice.columns:
            df_slice[col] = ""

    print(f"Generating over the configured slice with seeds={SEEDS}...")
    for r in tqdm(range(len(df_slice))):
        global_current_row = r
        row = df_slice.iloc[r]

        main_text = str(row.get(NOVELTY_INPUT_COL, "") or "")
        relevant_raw = row.get(RELEVANT_LIST_COL, "")
        paper_text_for_citation = str(row.get(TEXT_COL, "") or "")
        citation_text = extract_intro_and_abstract(row.get(CITED_COL, ""))

        # Preprocessing computed ONCE (shared across seeds).
        rel_list = parse_relevant_papers_list(relevant_raw)
        rel_sum = summarize_relevant_papers_list(rel_list, max_items=3)
        df_slice.at[r, RELEVANT_SUM_COL] = rel_sum

        main_text_tr = truncate_to_tokens(main_text, PAPER_TOKEN_BUDGET)
        rel_sum_tr = truncate_to_tokens(rel_sum, CITATION_TOKEN_BUDGET)
        combined_for_novelty = (
            "=== INPUT PAPER ===\n" f"{main_text_tr}\n\n"
            "=== RELEVANT PAPERS (SUMMARIZED) ===\n" f"{rel_sum_tr}"
        ).strip()
        citation_text_tr = truncate_to_tokens(citation_text, CITATION_TOKEN_BUDGET)

        for si, seed in enumerate(SEEDS):
            try:
                outs = run_seven_agents(combined_for_novelty, paper_text_for_citation,
                                        citation_text_tr, seed)
                df_slice.at[r, f"final_merged_limitations_seed{seed}"] = outs["final"]
                if si == 0:  # keep one representative set of agent outputs
                    for c in agent_cols:
                        df_slice.at[r, c] = outs[c]
            except Exception as e:
                df_slice.at[r, f"final_merged_limitations_seed{seed}"] = f"ERROR: {e}"
                print(f"Error row {r} seed {seed}: {e}")

        if r % SAVE_EVERY == 0:
            df_slice.to_csv(OUTPUT_FILE, index=False)
        time.sleep(SLEEP_SEC)

    df_slice.to_csv(OUTPUT_FILE, index=False)
    print(f"Generation done -> {OUTPUT_FILE}")
    return df_slice


# ============================================================
# 9) SCORING (per seed vs ground truth) + SEED VARIANCE
# ============================================================
def free_vllm_engine():
    """Release the vLLM engine so metric models have VRAM."""
    global llm
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


METRICS = ["cosine", "bertscore_f1", "rougeL_f1"]


def score_generations(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer, util
    from rouge_score import rouge_scorer
    from bert_score import score as bertscore_score

    refs = [to_text(v) for v in df[GT_COL]]
    embedder = SentenceTransformer(EMBED_MODEL)
    ref_emb = embedder.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for s in SEEDS:
        cands = [to_text(v) for v in df[f"final_merged_limitations_seed{s}"]]
        cand_emb = embedder.encode(cands, convert_to_tensor=True, show_progress_bar=False)
        df[f"cosine_seed{s}"] = [float(util.cos_sim(cand_emb[i], ref_emb[i]).item()) for i in range(len(df))]
        _, _, F1 = bertscore_score(cands, refs, model_type=BERT_MODEL, lang="en", verbose=False)
        df[f"bertscore_f1_seed{s}"] = [float(x) for x in F1]
        df[f"rougeL_f1_seed{s}"] = [rscorer.score(refs[i], cands[i])["rougeL"].fmeasure for i in range(len(df))]

    for m in METRICS:
        cols = [f"{m}_seed{s}" for s in SEEDS]
        df[f"{m}_mean"] = df[cols].mean(axis=1)
        df[f"{m}_std"] = df[cols].std(axis=1)
    return df


def report_seed_variance(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"[{THIS_MODEL_NAME}] SEED VARIANCE (first {len(df)} rows, seeds={SEEDS})")
    print("=" * 60)
    for m in METRICS:
        per_seed_corpus_means = [df[f"{m}_seed{s}"].mean() for s in SEEDS]
        mu = np.mean(per_seed_corpus_means)
        sd = np.std(per_seed_corpus_means, ddof=1) if len(SEEDS) > 1 else 0.0
        print(f"{m:14s}: corpus-mean over seeds = {mu:.4f} ± {sd:.4f}  "
              f"(per-seed means: {[round(x,4) for x in per_seed_corpus_means]})")
        # within-paper variability averaged across corpus
        print(f"{'':14s}  avg within-paper std across seeds = {df[f'{m}_std'].mean():.4f}")
    print("=" * 60)


# ============================================================
# 10) CROSS-MODEL PAIRED SIGNIFICANCE TEST
# ============================================================
def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cross_model_significance(this_df: pd.DataFrame):
    if not os.path.exists(OTHER_MODEL_CSV):
        print(f"\n[significance] counterpart CSV not found yet: {OTHER_MODEL_CSV}")
        print("[significance] run the other model, then re-run this (or run both, this step "
              "auto-executes when both exist).")
        return
    try:
        from scipy.stats import wilcoxon
    except Exception:
        print("[significance] scipy not available; skipping paired test.")
        return

    other = pd.read_csv(OTHER_MODEL_CSV)
    # Ensure the other df has *_mean columns; if not, it wasn't scored.
    need = [f"{m}_mean" for m in METRICS]
    if not all(c in other.columns for c in need):
        print("[significance] counterpart CSV lacks scored *_mean columns; skipping.")
        return

    merged = this_df.merge(other, on="orig_index", suffixes=(f"_{THIS_MODEL_NAME}", f"_{OTHER_MODEL_NAME}"))
    print("\n" + "=" * 60)
    print(f"CROSS-MODEL PAIRED SIGNIFICANCE: {THIS_MODEL_NAME} vs {OTHER_MODEL_NAME}")
    print(f"(paired over {len(merged)} papers; scores = mean over seeds)")
    print("=" * 60)
    for m in METRICS:
        a = merged[f"{m}_mean_{THIS_MODEL_NAME}"].astype(float).to_numpy()
        b = merged[f"{m}_mean_{OTHER_MODEL_NAME}"].astype(float).to_numpy()
        diff = a - b
        try:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        lo, hi = paired_bootstrap_ci(diff)
        print(f"{m:14s}: {THIS_MODEL_NAME}={a.mean():.4f}  {OTHER_MODEL_NAME}={b.mean():.4f}  "
              f"Δ={diff.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  Wilcoxon p={p:.4g}")
    print("=" * 60)


# ============================================================
if __name__ == "__main__":
    df_out = run_pipeline()
    free_vllm_engine()
    df_out = score_generations(df_out)
    df_out.to_csv(OUTPUT_FILE, index=False)
    report_seed_variance(df_out)
    cross_model_significance(df_out)
    print(f"\nAll done -> {OUTPUT_FILE}")