import os
import ast
import json
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ==========================================
# 1. CONFIGURATION & LOCAL MODEL SETUP (vLLM + AWQ)
# ==========================================


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


MODEL_PATH = _require_env("MODEL", "AWQ-quantized classifier checkpoint directory")

# Fits a single 40 GB GPU: ~16-18 GB weights (AWQ 4-bit) + KV cache.
# If you OOM, lower MAX_MODEL_LEN or gpu_memory_utilization.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 16384))
GPU_MEM_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.90))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 1))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq_marlin")  # use "awq" if marlin unsupported
DTYPE = os.environ.get("DTYPE", "float16")

print("Loading the AWQ classifier checkpoint via vLLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(
    model=MODEL_PATH,
    quantization=QUANTIZATION,
    dtype=DTYPE,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=400,
)

# ==========================================
# LU CLASSIFICATION TAXONOMY
# ==========================================
CATEGORY_LABELS = {
    1: "valid_grounded_missing",
    2: "valid_excessively_broad",
    3: "partially_supported",
    4: "unsupported_hallucinated",
    5: "irrelevant_offtarget",
    6: "redundant",
}
VALID_CATEGORIES = {1, 2}


# ==========================================
# 2. PARSE llm_evaluation_results -> list
# ==========================================
def parse_eval_results(cell):
    if isinstance(cell, list):
        return cell
    val = cell
    for _ in range(2):
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                break
        else:
            break
    return val if isinstance(val, list) else []


def extract_unmatched_llm_limitations(parsed):
    """{llm_id: llm_limitation} for UNMATCHED generated LUs (every pair is 'No')."""
    info = {}
    for pair in parsed:
        verdict = llm_id = llm_lim = None
        for field in pair:
            if not isinstance(field, str):
                continue
            if field.startswith("Pair"):
                if ":" in field:
                    verdict = field.split(":", 1)[1].strip().lower()
            elif field.startswith("llm_id:"):
                llm_id = field[len("llm_id:"):].strip()
            elif field.startswith("llm_limitation:"):
                llm_lim = field[len("llm_limitation:"):].strip()
        if llm_id is None:
            continue
        entry = info.setdefault(llm_id, {"lim": None, "verdicts": set()})
        if llm_lim is not None and entry["lim"] is None:
            entry["lim"] = llm_lim
        if verdict is not None:
            entry["verdicts"].add(verdict)
    return {lid: d["lim"] for lid, d in info.items() if "yes" not in d["verdicts"]}


# ==========================================
# 3. PROMPT BUILDING + PARSING
# ==========================================
def build_prompt(llm_limitation, input_text_cleaned):
    system_prompt = "You are a careful scientific reviewer. Respond ONLY with a valid JSON object."
    user_prompt = f"""You are given the FULL TEXT of a paper (MANUSCRIPT) and a single generated
LIMITATION UNIT (LU) that was NOT matched to any reference limitation.

Classify the LU into EXACTLY ONE of these categories:
1 = valid and grounded but missing from the reference (clearly supported by the manuscript)
2 = valid but excessively broad (defensible, but too generic / not specific to this paper)
3 = partially supported (some grounding in the manuscript, but incomplete or overstated)
4 = unsupported or hallucinated (not supported by the manuscript; fabricated)
5 = irrelevant / off-target (does not pertain to this manuscript's content)
6 = redundant with another limitation (essentially restates a point already covered)

Return ONLY a JSON object with EXACTLY these keys:
- "category": integer 1-6
- "valid": "Yes" if category is 1 or 2, otherwise "No"
- "evidence": a direct quote or specific reference from the MANUSCRIPT supporting the LU
             (REQUIRED and non-empty when category is 1 or 2; empty string otherwise)
- "reasoning": one brief sentence justifying the category

LIMITATION UNIT (LU):
{llm_limitation}

MANUSCRIPT (input_text_cleaned):
{input_text_cleaned}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # enable_thinking=False -> Qwen3 skips the <think> block and returns clean JSON.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def parse_response(response):
    response = response.strip()
    if response.startswith("```"):
        lines = response.splitlines()
        response = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if not m:
            return {"category": None, "category_label": "ParseError",
                    "valid": "", "evidence": "", "reasoning": response[:500]}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"category": None, "category_label": "ParseError",
                    "valid": "", "evidence": "", "reasoning": response[:500]}
    try:
        cat = int(data.get("category"))
    except (TypeError, ValueError):
        cat = None
    return {
        "category": cat,
        "category_label": CATEGORY_LABELS.get(cat, "Unknown"),
        "valid": "Yes" if cat in VALID_CATEGORIES else "No",
        "evidence": data.get("evidence", ""),
        "reasoning": data.get("reasoning", ""),
    }


# ==========================================
# 4. MAIN LOOP (batched vLLM, checkpoint by tasks)
# ==========================================
INPUT_CSV  = _require_env("INPUT_CSV", "evaluation CSV containing the unmatched-LU columns")
OUTPUT_DIR = _require_env("OUTPUT_DIR", "directory to write classification results into")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", os.path.join(OUTPUT_DIR, "llm_lu_classification.csv"))
POOLED_CSV = os.environ.get("POOLED_CSV", os.path.join(OUTPUT_DIR, "pooled_valid_unmatched_lus.csv"))
SYSTEM_NAME = os.environ.get("SYSTEM_NAME", "gpt_7_agents")
# Optional: process only the first N rows (leave unset to run all).
HEAD_N = os.environ.get("HEAD_N", "")


def process_dataframe(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build all classification tasks first (one per unmatched LU).
    tasks = []
    for row_idx, row in df.iterrows():
        parsed = parse_eval_results(row["llm_evaluation_results"])
        limitations = extract_unmatched_llm_limitations(parsed)
        input_text = row["input_text_cleaned"]
        for llm_id, llm_lim in limitations.items():
            tasks.append(
                {"row_index": row_idx, "llm_id": llm_id,
                 "llm_limitation": llm_lim, "input_text": input_text}
            )

    records = []
    for start in tqdm(range(0, len(tasks), BATCH_SIZE), desc="Batches"):
        batch = tasks[start:start + BATCH_SIZE]
        prompts = [build_prompt(t["llm_limitation"], t["input_text"]) for t in batch]
        outputs = llm.generate(prompts, sampling_params)  # vLLM batched inference

        for t, out in zip(batch, outputs):
            res = parse_response(out.outputs[0].text)
            records.append(
                {
                    "system": SYSTEM_NAME,
                    "row_index": t["row_index"],
                    "llm_id": t["llm_id"],
                    "llm_limitation": t["llm_limitation"],
                    "category": res["category"],
                    "category_label": res["category_label"],
                    "valid": res["valid"],
                    "evidence": res["evidence"],
                    "reasoning": res["reasoning"],
                }
            )
        pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)  # checkpoint per batch

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False)
    return results_df


# ==========================================
# 5. RATES + POOLED AUGMENTED REFERENCE
# ==========================================
def report_rates(results_df):
    valid_df = results_df[results_df["category"].notna()].copy()
    valid_df["category"] = valid_df["category"].astype(int)

    print("\n================ LU CLASSIFICATION RATES ================")
    for system, g in valid_df.groupby("system"):
        n = len(g)
        if n == 0:
            continue
        c = g["category"].value_counts().to_dict()
        rates = {
            "valid_unmatched_rate": (c.get(1, 0) + c.get(2, 0)) / n,
            "partially_supported_rate": c.get(3, 0) / n,
            "unsupported_hallucination_rate": c.get(4, 0) / n,
            "off_target_rate": c.get(5, 0) / n,
            "redundancy_rate": c.get(6, 0) / n,
        }
        print(f"\nSystem: {system}  (unmatched LUs = {n})")
        for k, v in rates.items():
            print(f"  {k:32s}: {v:.3f}")
        print("  per-category counts:",
              {CATEGORY_LABELS[k]: c.get(k, 0) for k in sorted(CATEGORY_LABELS)})
    print("========================================================\n")


def build_pooled_augmented_reference(results_df):
    valid_lus = results_df[results_df["valid"] == "Yes"].copy()

    def norm(s):
        return re.sub(r"\s+", " ", str(s).strip().lower())

    valid_lus["_key"] = valid_lus["llm_limitation"].map(norm)
    pooled = valid_lus.drop_duplicates(subset="_key").drop(columns="_key")
    pooled.to_csv(POOLED_CSV, index=False)
    print(f"Pooled valid-unmatched LUs (deduped): {len(pooled)} -> {POOLED_CSV}")
    return pooled


if __name__ == "__main__":
    df_scilimagents = pd.read_csv(INPUT_CSV)
    if HEAD_N.strip():
        df_scilimagents = df_scilimagents.head(int(HEAD_N))

    results_df = process_dataframe(df_scilimagents)
    report_rates(results_df)
    build_pooled_augmented_reference(results_df)

    print(f"Saved classifications to: {OUTPUT_CSV}")
    print("Classification results written.")