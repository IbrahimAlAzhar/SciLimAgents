"""
run_multi_agent_inference_shared_memory.py — inference.py + a SHARED MEMORY block of retrieved
cited-paper evidence available to every agent.

Difference from inference.py
----------------------------
Each row's `retrieved_abs_int` and `ret_abs_int_cit` columns are parsed once into
a single evidence block and appended to the system prompt of whichever roles are
listed in SHARED_MEMORY_ROLES (default: all three). Workers, leader and master
therefore reason over the same external evidence.

    === SHARED MEMORY: RETRIEVED / CITED EVIDENCE ===
    Paper1_Title: ...
    Paper1_Abstract: ...
    Paper1_Introduction: ...

TWO CORRECTNESS NOTES
---------------------
1. TRAIN/TEST MISMATCH. The SFT and DPO prompts contained the paper ONLY —
   retrieval context was used during retrieval_heavy rollout generation but
   stripped from the canonical training prompts. Injecting it here is a change of
   input distribution. Run both conditions and compare:
       SHARED_MEMORY=0   reproduces plain inference.py
       SHARED_MEMORY=1   (default here) adds the block
2. PROMPT TRUNCATION. inference.py truncated the tokenized prompt with
   `input_ids[:, :MAX_INPUT_TOKENS]`, which keeps the HEAD and drops the TAIL —
   i.e. it silently deletes the user instruction and the assistant generation
   header when a prompt is long, so the model is asked to continue the paper
   rather than answer. This version instead shrinks the PAPER text until the full
   prompt fits, leaving the instruction and generation header intact.

Everything else — adapters, stacking, generation settings, checkpointing,
resume — is unchanged.

Requires build_retrieval_context_block.py next to this file.
"""

import gc
import inspect
import os
import re
import time

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from retrieval_context import build_shared_memory
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(
        "build_retrieval_context_block.py must sit next to this script "
        f"({os.path.dirname(os.path.abspath(__file__))}).\n{exc}"
    )


# =============================================================================
# 1. CONFIG
# =============================================================================



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


BASE_MODEL_PATH = _require_env("STUDENT_BASE", "base instruct checkpoint directory")

_ADAPTERS = _require_env("ADAPTER_DIR", "directory holding the trained LoRA adapters")
WORKER_SFT_ADAPTER = os.environ.get("WORKER_SFT_ADAPTER", f"{_ADAPTERS}/worker_sft/final")
# augmented (bottom-25% synthetic rejections) is the variant being evaluated
WORKER_DPO_ADAPTER = os.environ.get("WORKER_DPO_ADAPTER", f"{_ADAPTERS}/worker_dpo_augmented/final")
LEADER_SFT_ADAPTER = os.environ.get("LEADER_SFT_ADAPTER", f"{_ADAPTERS}/leader_sft/final")
MASTER_SFT_ADAPTER = os.environ.get("MASTER_SFT_ADAPTER", f"{_ADAPTERS}/master_sft/final")

WORKER_DPO_MODE = os.environ.get("WORKER_DPO_MODE", "stack").strip().lower()
if WORKER_DPO_MODE not in {"stack", "dpo_only", "disabled"}:
    raise ValueError("WORKER_DPO_MODE must be one of: stack, dpo_only, disabled")

WORKER_STACK_STRATEGY = os.environ.get("WORKER_STACK_STRATEGY", "merge").strip().lower()
if WORKER_STACK_STRATEGY not in {"merge", "runtime"}:
    raise ValueError("WORKER_STACK_STRATEGY must be one of: merge, runtime")

WORKER_SFT_WEIGHT = float(os.environ.get("WORKER_SFT_WEIGHT", "1.0"))
WORKER_DPO_WEIGHT = float(os.environ.get("WORKER_DPO_WEIGHT", "1.0"))
STACKED_ADAPTER_NAME = "worker_stacked"

INPUT_CSV = _require_env("INPUT_CSV", "evaluation CSV with the paper and retrieval columns")
INPUT_COL = os.environ.get("INPUT_COL", "input_text_cleaned")

# ---- shared memory ----
SHARED_MEMORY = os.environ.get("SHARED_MEMORY", "1") != "0"
SHARED_MEMORY_ROLES = {
    r.strip().lower()
    for r in os.environ.get("SHARED_MEMORY_ROLES", "worker,leader,master").split(",")
    if r.strip()
}
# parsed = structured Paper1_Title/Abstract/Introduction layout
# raw    = the two columns verbatim (lightly de-punctuated) — immune to schema
#          surprises, costs more tokens
# auto   = parsed when it yields records, otherwise raw   [default]
SHARED_MEMORY_MODE = os.environ.get("SHARED_MEMORY_MODE", "auto").strip().lower()
if SHARED_MEMORY_MODE not in {"auto", "parsed", "raw"}:
    raise ValueError("SHARED_MEMORY_MODE must be one of: auto, parsed, raw")
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
# Token budget for the evidence block. Kept modest because the worker prompt
# already carries the paper; the block competes with it for context.
RETRIEVAL_MAX_TOKENS = int(os.environ.get("RETRIEVAL_MAX_TOKENS", "2500"))
SHARED_MEMORY_HEADER = "\n\n=== SHARED MEMORY: RETRIEVED / CITED EVIDENCE ===\n"

# Role-specific instructions. A single shared note would put the master in
# conflict with its own system prompt ("Do NOT introduce new limitations not
# raised by specialists") by inviting it to mine the evidence for fresh points,
# and would tell the leader to generate limitations when its job is to critique.
SHARED_MEMORY_NOTES = {
    "worker": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. It is NOT part of the paper being "
        "reviewed and its contents are not limitations of that paper.\n"
        "Use it to:\n"
        "- test novelty claims against what prior work already did;\n"
        "- check whether the baselines compared against are the right ones;\n"
        "- spot claims the paper asserts but prior work contradicts or already established.\n"
        "When a limitation rests on this evidence, cite it as (Ref: PaperN). Do not "
        "raise a limitation about a cited paper itself.\n"),
    "leader": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. Use it only to JUDGE the workers' "
        "submissions — whether a novelty or baseline claim they make is actually "
        "supported by, or contradicted by, this prior work. Do not generate "
        "limitations yourself; your role remains to critique and forward.\n"),
    "master": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. Use it ONLY to verify and merge the "
        "limitations the specialists already raised — for example to confirm a "
        "novelty claim or attach a (Ref: PaperN) citation to an existing point. "
        "Do NOT introduce any new limitation derived from this evidence; your "
        "output must stay grounded in what the specialists reported.\n"),
}

# Optional slice of the input file. Both unset => process the whole file.
START_ROW = _optional_int("START_ROW") or 0
END_ROW = _optional_int("END_ROW")

OUTPUT_DIR = _require_env("INFER_OUT_DIR", "directory to write inference results into")
_tag = "sharedmem" if SHARED_MEMORY else "nomem"
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    os.path.join(OUTPUT_DIR, f"df_inference_{_tag}.csv"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "10"))
PAPER_CHARS = int(os.environ.get("PAPER_CHARS", "20000"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "12000"))

GEN_KWARGS = {
    "temperature": float(os.environ.get("GEN_TEMPERATURE", "0.2")),
    "top_p": float(os.environ.get("GEN_TOP_P", "0.9")),
    "do_sample": os.environ.get("GEN_DO_SAMPLE", "1") != "0",
    "repetition_penalty": float(os.environ.get("GEN_REPETITION_PENALTY", "1.1")),
}
WORKER_MAX_NEW = int(os.environ.get("WORKER_MAX_NEW", "1400"))
LEADER_MAX_NEW = int(os.environ.get("LEADER_MAX_NEW", "1800"))
MASTER_MAX_NEW = int(os.environ.get("MASTER_MAX_NEW", "1800"))
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# =============================================================================
# 2. PROMPTS  (identical to inference.py)
# =============================================================================

def get_novelty_significance_prompt(paper_content):
    return f"""You are a highly skeptical expert focused on limitations related to novelty and significance. Scrutinize whether contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for: rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters.
Provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""


def get_theoretical_methodological_prompt(paper_content):
    return f"""You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws: unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, missing ablations, or ablations that do not convincingly attribute performance gains.
Provide a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""


def get_experimental_evaluation_prompt(paper_content):
    return f"""You specialize in experimental evaluation: validation, rigor, comparisons, baselines, and metrics. Find weaknesses: insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate or outdated baselines, incomplete comparisons, misleading metrics, missing error bars, or overemphasis on minor gains.
Provide a bullet list of experimental evaluation-related limitations.
PAPER CONTENT:
{paper_content}"""


def get_generalization_robustness_efficiency_prompt(paper_content):
    return f"""Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings, is practical in terms of resources, and addresses genuine deployment needs.
Point out: overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, excessive training/inference demands, reliance on synthetic data, ignoring cost/latency constraints, or lack of user studies.
Provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""


def get_clarity_interpretability_reproducibility_prompt(paper_content):
    return f"""You focus on clarity, interpretability, and reproducibility. Scrutinize for: unclear explanations, lack of explainability or insights into decisions, and insufficient details for replication (code, data, hyperparameters, protocols, seeds).
Provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, with suggestions where relevant.
PAPER CONTENT:
{paper_content}"""


def get_data_ethics_prompt(paper_content):
    return f"""You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for: collection, labeling, cleaning, representativeness, or documentation issues; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""


WORKER_CONFIGS = [
    ("lim_novelty_significance", get_novelty_significance_prompt, "novelty and significance"),
    ("lim_theoretical_methodological", get_theoretical_methodological_prompt,
     "theoretical and methodological soundness (including ablations)"),
    ("lim_experimental_evaluation", get_experimental_evaluation_prompt,
     "experimental evaluation, baselines, and metrics"),
    ("lim_generalization_robustness_efficiency", get_generalization_robustness_efficiency_prompt,
     "generalization, robustness, efficiency, and applicability"),
    ("lim_clarity_interpretability_reproducibility", get_clarity_interpretability_reproducibility_prompt,
     "clarity, interpretability, and reproducibility"),
    ("lim_data_ethics", get_data_ethics_prompt, "data integrity, bias, fairness, and ethics"),
]


def get_leader_system_prompt():
    return """You are the Leader Agent coordinating a team of 6 specialist agents to identify limitations in a scientific paper.

Your job has TWO modes:

MODE A - Providing FEEDBACK to a worker:
When a worker has just submitted their initial bullet list of limitations, give targeted feedback in ONE message:
- Identify vague statements and demand specificity.
- Flag limitations that lack evidence from the paper.
- Point out missing angles in the worker's specialty area.
- If a limitation is generic (could apply to any paper), say so.
- Be strict but constructive. Keep feedback focused - 3 to 6 concrete points.
- End with: "Please revise and send your updated bullet list."

MODE B - Handing off to the Master Agent:
Once you have collected revised outputs from all 6 workers, summarize and forward them. Format:
"Master Agent, here are the limitation analyses from the team:
[Worker name]: [bullet list]
[Worker name]: [bullet list]
...
Please synthesize them into a single, consolidated, non-redundant, high-quality list of limitations, grouped by category."

Do NOT generate limitations yourself. Only orchestrate, critique, and forward."""


def get_master_system_prompt():
    return """You are the Master Agent. You receive limitation analyses from 6 specialist workers (via the Leader Agent) and produce ONE final consolidated list.

TASK:
- Integrate all specialist outputs.
- Remove redundancies (merge similar limitations).
- Prioritize the most severe and well-justified ones.
- Preserve specificity and evidence from the originals.
- Organize by category (Novelty & Significance, Theoretical, Experimental, Generalization, Clarity, Data & Ethics).
- Do NOT introduce new limitations not raised by specialists.
- Aim for 10-20 strong limitations.

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then a bulleted list:
- **Category:** Specific limitation (with brief explanation and evidence)."""


MASTER_TOT_SUFFIX = (
    "\n\n[TREE-OF-THOUGHT + SELF-REPAIR] Internally explore a few distinct "
    "consolidation paths (different groupings / prioritizations), pick the "
    "strongest, then self-critique it once and repair any weak, redundant, or "
    "poorly-grounded points before returning the FINAL list only.")

WORKER_USER = ("Identify limitations focused on {specialty}. "
               "Return only an evidence-grounded bullet list.")


# =============================================================================
# 3. MODEL LOADING / ADAPTER HANDLING  (identical to inference.py)
# =============================================================================

def _require_path(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _input_device(model):
    return next(model.parameters()).device


def _set_adapter(model, adapter):
    if isinstance(adapter, (list, tuple)):
        names = list(adapter)
        if len(names) == 1:
            model.set_adapter(names[0])
            return
        try:
            model.base_model.set_adapter(names)
            model.active_adapter = names[0]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "This PEFT version could not activate multiple worker adapters "
                f"({names}). Try WORKER_STACK_STRATEGY=merge, or "
                "WORKER_DPO_MODE=dpo_only if the DPO checkpoint already "
                "contains the SFT weights.") from exc
    else:
        model.set_adapter(adapter)


def _try_merge_worker_adapters(model):
    if not hasattr(model, "add_weighted_adapter"):
        print("[stack] PEFT has no add_weighted_adapter; falling back to runtime stacking.")
        return None
    try:
        kwargs = {
            "adapters": ["worker_sft", "worker_dpo"],
            "weights": [WORKER_SFT_WEIGHT, WORKER_DPO_WEIGHT],
            "adapter_name": STACKED_ADAPTER_NAME,
        }
        sig = inspect.signature(model.add_weighted_adapter)
        if "combination_type" in sig.parameters:
            kwargs["combination_type"] = "cat"
        model.add_weighted_adapter(**kwargs)
        print(f"[stack] Merged worker_sft ({WORKER_SFT_WEIGHT}) + "
              f"worker_dpo ({WORKER_DPO_WEIGHT}) -> '{STACKED_ADAPTER_NAME}'.")
        return STACKED_ADAPTER_NAME
    except Exception as exc:  # noqa: BLE001
        print(f"[stack] add_weighted_adapter failed ({exc}); falling back to runtime stacking.")
        return None


def _load_base_model(dtype):
    common = dict(device_map="auto", trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=dtype, **common)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=dtype, **common)


def load_model():
    _require_path(BASE_MODEL_PATH, "Base model")
    _require_path(WORKER_SFT_ADAPTER, "Worker SFT adapter")
    _require_path(LEADER_SFT_ADAPTER, "Leader SFT adapter")
    _require_path(MASTER_SFT_ADAPTER, "Master SFT adapter")
    if WORKER_DPO_MODE != "disabled":
        _require_path(WORKER_DPO_ADAPTER, "Worker DPO adapter")

    print("=== Model paths ===")
    print(f"Base        : {BASE_MODEL_PATH}")
    print(f"Worker SFT  : {WORKER_SFT_ADAPTER}")
    print(f"Worker DPO  : {WORKER_DPO_ADAPTER} | mode={WORKER_DPO_MODE}")
    print(f"Leader SFT  : {LEADER_SFT_ADAPTER}")
    print(f"Master SFT  : {MASTER_SFT_ADAPTER}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = _load_base_model(dtype)

    model = PeftModel.from_pretrained(base, WORKER_SFT_ADAPTER, adapter_name="worker_sft")
    if WORKER_DPO_MODE != "disabled":
        model.load_adapter(WORKER_DPO_ADAPTER, adapter_name="worker_dpo")
    model.load_adapter(LEADER_SFT_ADAPTER, adapter_name="leader_sft")
    model.load_adapter(MASTER_SFT_ADAPTER, adapter_name="master_sft")
    model.eval()

    if WORKER_DPO_MODE == "dpo_only":
        worker_adapter = "worker_dpo"
    elif WORKER_DPO_MODE == "disabled":
        worker_adapter = "worker_sft"
    else:
        worker_adapter = None
        if WORKER_STACK_STRATEGY == "merge":
            worker_adapter = _try_merge_worker_adapters(model)
        if worker_adapter is None:
            worker_adapter = ["worker_sft", "worker_dpo"]

    print(f"[stack] Worker adapter(s): {worker_adapter}")
    _set_adapter(model, worker_adapter)

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return tokenizer, model, worker_adapter


# =============================================================================
# 3b. PROMPT ASSEMBLY — shared memory + safe truncation
# =============================================================================

MERGE_SYSTEM = os.environ.get("MERGE_SYSTEM", "0") == "1"


def _merge_system(messages):
    """Fold a system turn into the first user turn.

    Mistral-7B-Instruct v0.x templates define no system role and raise on one;
    Qwen3 accepts it. Every prompt here is [system, user], so without this the
    Mistral runs die on the first paper.
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_txt, rest = messages[0]["content"], messages[1:]
    if rest and rest[0].get("role") == "user":
        return ([{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}]
                + rest[1:])
    return [{"role": "user", "content": sys_txt}] + rest


def _render_chat(tokenizer, system_prompt, user_msg):
    """Render the chat, tolerating templates without system-role or thinking
    support. Falls back automatically, so one file serves Qwen and Mistral."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}]
    first = _merge_system(messages) if MERGE_SYSTEM else messages
    for attempt in (first, _merge_system(messages)):
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    attempt, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                continue
    raise RuntimeError("apply_chat_template failed for every fallback; check the "
                       "tokenizer's chat template")


def _n_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_worker_system(tokenizer, prompt_fn, paper, memory_block):
    """System prompt that fits MAX_INPUT_TOKENS by shrinking the PAPER only.

    inference.py truncated the tokenized prompt from the right, which deletes the
    user instruction and the assistant generation header. Here the evidence block
    and the instructions are preserved and the paper is trimmed instead.
    """
    paper_text = paper
    for _ in range(6):
        system_prompt = prompt_fn(paper_text) + memory_block
        rendered = _render_chat(tokenizer, system_prompt, "x")
        n = _n_tokens(tokenizer, rendered)
        if n <= MAX_INPUT_TOKENS - 200:      # 200 = headroom for the real user msg
            return system_prompt
        over = n - (MAX_INPUT_TOKENS - 200)
        # ~4 chars/token, trim 10% extra so we converge in one or two passes
        cut = max(500, int(over * 4 * 1.1))
        if cut >= len(paper_text):
            paper_text = paper_text[: max(1000, len(paper_text) // 2)]
        else:
            paper_text = paper_text[: len(paper_text) - cut]
    return prompt_fn(paper_text) + memory_block


@torch.no_grad()
def generate(model, tokenizer, adapter, system_prompt, user_msg, max_new_tokens):
    _set_adapter(model, adapter)
    text = _render_chat(tokenizer, system_prompt, user_msg)

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > MAX_INPUT_TOKENS:
        # Should be rare now that the paper is pre-trimmed. Keep the HEAD of the
        # system prompt and the ENTIRE TAIL (instruction + generation header),
        # rather than blindly dropping the tail as the original did.
        keep_tail = 600
        head = MAX_INPUT_TOKENS - keep_tail
        inputs["input_ids"] = torch.cat(
            [input_ids[:, :head], input_ids[:, -keep_tail:]], dim=1)
        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [am[:, :head], am[:, -keep_tail:]], dim=1)

    device = _input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id, **GEN_KWARGS)
        response = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return THINK_RE.sub(" ", response).strip()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return "ERROR: CUDA OOM"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


# =============================================================================
# 4. PIPELINE
# =============================================================================

# Text columns, initialized to "PENDING".
AGENT_COLUMNS = [c for c, _, _ in WORKER_CONFIGS] + [
    "leader_consolidated", "final_limitations_master"]
# Numeric bookkeeping column — kept OUT of AGENT_COLUMNS. Initializing it to
# "PENDING" would make pandas infer a string dtype, and writing an int into a
# string-backed (pyarrow) column raises TypeError.
MEMORY_COL = "shared_memory_chars"
ALL_COLUMNS = AGENT_COLUMNS + [MEMORY_COL]


def _pending(value):
    return str(value) in {"PENDING", "nan", "", "None"}


def _load_checkpoint(df):
    if not os.path.exists(OUTPUT_CSV):
        return df, 0
    checkpoint = pd.read_csv(OUTPUT_CSV)
    rows_to_copy = min(len(df), len(checkpoint))
    for col in ALL_COLUMNS:
        if col in checkpoint.columns:
            df.loc[: rows_to_copy - 1, col] = checkpoint.loc[: rows_to_copy - 1, col].values
    start_idx = 0
    if "final_limitations_master" in df.columns:
        for i in range(len(df)):
            if _pending(df.iloc[i].get("final_limitations_master", "PENDING")):
                start_idx = i
                break
        else:
            start_idx = len(df)
    print(f"Resuming from row index {start_idx} inside this slice.")
    return df, start_idx


def run_pipeline():
    print("=== Loading CSV ===")
    print(f"Input CSV     : {INPUT_CSV}")
    print(f"Rows          : {START_ROW}:{END_ROW} (end-exclusive)")
    print(f"Output CSV    : {OUTPUT_CSV}")
    print(f"Shared memory : {'ON' if SHARED_MEMORY else 'OFF'}"
          f"{f' mode={SHARED_MEMORY_MODE}' if SHARED_MEMORY else ''}"
          f"{' roles=' + ','.join(sorted(SHARED_MEMORY_ROLES)) if SHARED_MEMORY else ''}"
          f"{f' top_k={RETRIEVAL_TOP_K} max_tokens={RETRIEVAL_MAX_TOKENS}' if SHARED_MEMORY else ''}")

    df = pd.read_csv(INPUT_CSV).iloc[START_ROW:END_ROW].reset_index(drop=True)
    print(f"Loaded {len(df)} rows.")

    if SHARED_MEMORY:
        from retrieval_context import DEFAULT_COLS
        have = [c for c in DEFAULT_COLS if c in df.columns]
        missing = [c for c in DEFAULT_COLS if c not in df.columns]
        print(f"Evidence cols : present={have} missing={missing}")
        if not have:
            print("[WARN] no evidence columns found — shared memory will be empty, "
                  "which makes this run identical to plain inference.")

    for col in AGENT_COLUMNS:
        if col not in df.columns:
            df[col] = "PENDING"
    if MEMORY_COL not in df.columns:
        df[MEMORY_COL] = 0
    df[MEMORY_COL] = pd.to_numeric(df[MEMORY_COL], errors="coerce").fillna(0).astype(int)

    df, start_idx = _load_checkpoint(df)
    if start_idx >= len(df):
        print("All rows already processed.")
        return

    tokenizer, model, worker_adapter = load_model()
    leader_sys_base = get_leader_system_prompt()
    master_sys_base = get_master_system_prompt() + MASTER_TOT_SUFFIX
    start_time = time.time()
    mem_sizes = []
    mode_counts = {}

    for i in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df)):
        if not _pending(df.iloc[i].get("final_limitations_master", "PENDING")):
            continue

        row = df.iloc[i]
        paper = str(row.get(INPUT_COL, ""))[:PAPER_CHARS]
        if len(paper) < 100:
            for col in AGENT_COLUMNS:          # text columns only
                df.iat[i, df.columns.get_loc(col)] = "SKIPPED_SHORT_TEXT"
            df.iat[i, df.columns.get_loc(MEMORY_COL)] = 0
            continue

        # ---- build the shared memory block once per paper ----
        memory_block = ""
        if SHARED_MEMORY:
            body, used_mode = build_shared_memory(
                row, mode=SHARED_MEMORY_MODE, top_k=RETRIEVAL_TOP_K,
                tokenizer=tokenizer, max_tokens=RETRIEVAL_MAX_TOKENS)
            mode_counts[used_mode] = mode_counts.get(used_mode, 0) + 1
            if body:
                # strip the generic header; the role-specific framing is added
                # per agent below, since each role must use the evidence differently
                body = SHARED_MEMORY_HEADER + body.split("===\n", 1)[-1]
                memory_block = body
        df.iat[i, df.columns.get_loc(MEMORY_COL)] = int(len(memory_block))
        mem_sizes.append(len(memory_block))

        def role_mem(role):
            if not memory_block or role not in SHARED_MEMORY_ROLES:
                return ""
            return memory_block + SHARED_MEMORY_NOTES[role]

        w_mem, l_mem, m_mem = role_mem("worker"), role_mem("leader"), role_mem("master")

        try:
            worker_outputs = {}
            for col_name, prompt_fn, specialty in WORKER_CONFIGS:
                system_prompt = build_worker_system(tokenizer, prompt_fn, paper, w_mem)
                out = generate(model, tokenizer, worker_adapter, system_prompt,
                               WORKER_USER.format(specialty=specialty), WORKER_MAX_NEW)
                df.iat[i, df.columns.get_loc(col_name)] = out
                name = col_name.replace("lim_", "").replace("_", " ").title().replace(" ", "_")
                worker_outputs[f"{name}_Agent"] = out

            leader_user = (
                "You have collected revised outputs from all 6 specialist workers. Produce the "
                "MODE B handoff to the Master Agent. Here are the 6 outputs:\n\n"
                + "\n\n".join(f"### {w}:\n{o}" for w, o in worker_outputs.items()))
            leader_out = generate(model, tokenizer, "leader_sft",
                                  leader_sys_base + l_mem, leader_user, LEADER_MAX_NEW)
            df.iat[i, df.columns.get_loc("leader_consolidated")] = leader_out

            final_out = generate(model, tokenizer, "master_sft",
                                 master_sys_base + m_mem, leader_out, MASTER_MAX_NEW)
            if len(final_out.strip()) < 30 or final_out.startswith("ERROR"):
                final_out = "NO_OUTPUT_FROM_MASTER"
            df.iat[i, df.columns.get_loc("final_limitations_master")] = final_out

        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] row {i}: {exc}")
            df.iat[i, df.columns.get_loc("final_limitations_master")] = f"ERROR: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            done = i - start_idx + 1
            spr = (time.time() - start_time) / max(done, 1)
            eta = spr * (len(df) - i - 1) / 3600
            nz = [m for m in mem_sizes if m]
            print(f">>> checkpoint at slice row {i} | {spr:.1f}s/row | ETA {eta:.2f}h"
                  + (f" | shared memory: {len(nz)}/{len(mem_sizes)} rows, "
                     f"mean {sum(nz)//max(len(nz),1)} chars" if SHARED_MEMORY else ""))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to: {OUTPUT_CSV}")
    if SHARED_MEMORY and mem_sizes:
        nz = [m for m in mem_sizes if m]
        print(f"Shared memory attached on {len(nz)}/{len(mem_sizes)} rows "
              f"(mean {sum(nz)//max(len(nz),1)} chars, max {max(mem_sizes)}).")
        print(f"  block source: {mode_counts}")
        if not nz:
            print("[WARN] every block was empty — inspect the evidence columns:")
            print("       python build_retrieval_context_block.py \"$INPUT_CSV\" 0 --scan")


if __name__ == "__main__":
    run_pipeline()