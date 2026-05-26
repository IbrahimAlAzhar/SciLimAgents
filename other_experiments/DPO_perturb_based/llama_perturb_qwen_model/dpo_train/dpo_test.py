import os
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =============================================================================
# Paths — UPDATE: point to the chat-template trained model
# =============================================================================
BASE_MODEL_PATH = "qwen2_5_3b_instruct"
ADAPTER_PATH = "other_experiments/DPO_perturb_based/dpo_train/qwen25_3b_dpo_output_chat/final_model"

TEST_CSV = "data/not_balanced_data/df_not_bal_final_strat_samp.csv"
OUTPUT_DIR = "other_experiments/DPO_perturb_based/dpo_train/qwen25_3b_dpo_output_chat"
PREDICTION_CSV = os.path.join(OUTPUT_DIR, "test_predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Generation settings
# =============================================================================
MAX_PROMPT_TOKENS = 4096
PAPER_TOKEN_BUDGET = 3200   # tokens reserved for paper content
MAX_NEW_TOKENS = 1024
NUM_TEST_SAMPLES = None     # int to limit, None for all

# =============================================================================
# System / user message content (same text as training)
# =============================================================================
SYSTEM_MSG = (
    "You are an expert scientific paper reviewer. Your task is to identify ALL limitations\n"
    "of the given paper. Be thorough, specific, and evidence-grounded.\n"
    "\n"
    "For each limitation, provide:\n"
    "- A clear category (e.g., Novelty, Methodology, Experiments, Generalization, Clarity, Data/Ethics)\n"
    "- A specific description of the limitation\n"
    "- Evidence or reasoning from the paper supporting your claim\n"
    "\n"
    "Output format:\n"
    "- **[Category]**: Limitation description. (Evidence: ...)\n"
    "\n"
    "Be comprehensive. Cover novelty, methodology, theoretical soundness, experimental evaluation,\n"
    "generalization, robustness, efficiency, clarity, reproducibility, data quality, and ethical concerns."
)

def make_user_msg(paper_text: str) -> str:
    return (
        "Identify all limitations of the following scientific paper.\n"
        "\n"
        "=== PAPER CONTENT ===\n"
        f"{paper_text}\n"
        "\n"
        "=== TASK ===\n"
        "List all limitations covering: novelty, methodology, experiments, generalization, robustness,\n"
        "efficiency, clarity, reproducibility, data quality, and ethical concerns."
    )

# =============================================================================
# Load tokenizer
# =============================================================================
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Pre-compute instruction overhead to know how much room is left for paper text
_overhead_prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_MSG},
     {"role": "user",   "content": make_user_msg("")}],
    tokenize=True, add_generation_prompt=True,
)
instruction_overhead = len(_overhead_prompt) + 10
effective_paper_budget = min(PAPER_TOKEN_BUDGET, MAX_PROMPT_TOKENS - instruction_overhead)
print(f"Instruction overhead: ~{instruction_overhead} tokens")
print(f"Paper token budget:   ~{effective_paper_budget} tokens")

# =============================================================================
# Load base model + LoRA adapter
# =============================================================================
print("Loading base model …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter …")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model ready.\n")

# =============================================================================
# Load test data
# =============================================================================

df = pd.read_csv(TEST_CSV)
if NUM_TEST_SAMPLES is not None:
    df = df.head(NUM_TEST_SAMPLES)

if "input_text_without_lim" not in df.columns:
    raise ValueError("Test CSV must contain column: input_text_without_lim")
if "ground_truth_lim_peer" not in df.columns:
    raise ValueError("Test CSV must contain column: ground_truth_lim_peer")

print(f"Test samples: {len(df)}\n")

# =============================================================================
# Helpers
# =============================================================================
def truncate_paper(paper_text: str, max_tokens: int) -> str:
    """Truncate paper content to fit token budget (instructions stay intact)."""
    ids = tokenizer.encode(paper_text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return paper_text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def clean_prediction(raw: str) -> str:
    """Extract limitation list, remove leading junk and repetition loops."""
    if "**[" in raw:
        raw = raw[raw.index("**["):]
    elif "- **" in raw:
        raw = raw[raw.index("- **"):]

    # De-duplicate repeated limitation blocks
    lines = raw.split("\n")
    seen = set()
    deduped = []
    for line in lines:
        s = line.strip()
        if s and s in seen and (s.startswith("**[") or s.startswith("- **[")):
            continue
        seen.add(s)
        deduped.append(line)
    return "\n".join(deduped).strip()

# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def generate_limitations(paper_text: str) -> str:
    paper_truncated = truncate_paper(paper_text, effective_paper_budget)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": make_user_msg(paper_truncated)},
    ]

    # apply_chat_template produces:
    #   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=False,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][prompt_len:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return clean_prediction(raw)

# =============================================================================
# Run
# =============================================================================
predictions, references, inputs_used = [], [], []

for idx, row in df.iterrows():
    paper_text = str(row["input_text_without_lim"])
    ground_truth = str(row["ground_truth_lim_peer"])

    pred = generate_limitations(paper_text)

    inputs_used.append(paper_text)
    references.append(ground_truth)
    predictions.append(pred)

    if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
        pd.DataFrame({
            "input_text_without_lim": inputs_used,
            "ground_truth_lim_peer": references,
            "prediction": predictions,
        }).to_csv(PREDICTION_CSV, index=False)
        print(f"[{idx + 1}/{len(df)}]  saved → {PREDICTION_CSV}")

    if idx < 5:
        print(f"\n{'='*60}")
        print(f"Sample {idx + 1} — prediction:")
        print(pred[:800])
        print(f"\n--- Ground truth ---")
        print(ground_truth[:500])
        print(f"{'='*60}\n")

# Final save
pd.DataFrame({
    "input_text_without_lim": inputs_used,
    "ground_truth_lim_peer": references,
    "prediction": predictions,
}).to_csv(PREDICTION_CSV, index=False)
print(f"\nDone. Predictions saved to: {PREDICTION_CSV}")