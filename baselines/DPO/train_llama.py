import pandas as pd
import os
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# ==========================================
# 0. Paths & basic config
# ==========================================

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "llama3_8b_instruct"
OUTPUT_DIR = "DPO_training_mist_gen_text/output"

MAX_CONTEXT_TOKENS = 8000
MAX_NEW_TOKENS = 256

# Column where we'll store the full Master prompt
PROMPT_COL = "master_prompt"

# ==========================================
# 1. Load your dataframe
# ==========================================

CSV_PATH = "evaluations_for_mistral/output/df_mistral_text_nlp_metrics.csv"
df = pd.read_csv(CSV_PATH)

print("Columns in df:", df.columns.tolist())

# ---- Build the Master prompt from 4 agent outputs ----
# (using your actual column names)

def get_merger_prompt(extractor_out, analyzer_out, reviewer_out, citation_out):
    return f"""You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by four specialized agents: 

**Agents:**
1. **Extractor** (explicit limitations from the article).
2. **Analyzer** (inferred limitations from critical analysis).
3. **Reviewer** (limitations from an open review perspective).
4. **Citation** (limitations inferred from cited papers context).

**Goals**:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Format the final list in a clear, concise, and professional manner.

**Output Format**:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived], [Citation-context]).

Extractor Agent Analysis:
{extractor_out}

Analyzer Agent Analysis:
{analyzer_out}

Reviewer Agent Analysis:
{reviewer_out}

Citation Agent Analysis:
{citation_out}

Please merge these four different perspectives into a comprehensive, well-organized analysis."""
    

# IMPORTANT: use your actual columns: mistral_extractor, mistral_analyzer, ...
df[PROMPT_COL] = df.apply(
    lambda row: get_merger_prompt(
        row["mistral_extractor"],
        row["mistral_analyzer"],
        row["mistral_reviewer"],
        row["mistral_citation"],
    ),
    axis=1,
)

# ---- Sanity check that required columns EXIST ----
required_cols = [
    PROMPT_COL,
    "mistral_master_0.4",
    "mistral_master_0.6",
    "mistral_master_0.8",
    "mistral_master_1.0",
    "mistral_master_04_nlp_score",
    "mistral_master_06_nlp_score",
    "mistral_master_08_nlp_score",
    "mistral_master_1_nlp_score",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in df: {missing}")

# ==========================================
# ==========================================

score_mapping = {
    "0.4": "mistral_master_04_nlp_score",
    "0.6": "mistral_master_06_nlp_score",
    "0.8": "mistral_master_08_nlp_score",
    "1.0": "mistral_master_1_nlp_score",
}

def compute_margin_info(row):
    # collect non-NaN scores
    scores = {}
    for temp, col in score_mapping.items():
        val = row[col]
        if not pd.isna(val):
            scores[temp] = float(val)

    # need at least 2 scores for a margin
    if len(scores) < 2:
        return pd.Series(
            {
                "best_temp": None,
                "best_score": None,
                "second_best_score": None,
                "margin": None,
            }
        )

    # sort temps by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (best_temp, best_score), (second_temp, second_score) = sorted_items[0], sorted_items[1]
    margin = best_score - second_score

    return pd.Series(
        {
            "best_temp": best_temp,
            "best_score": best_score,
            "second_best_score": second_score,
            "margin": margin,
        }
    )

df_margin = df.apply(compute_margin_info, axis=1)
df = pd.concat([df, df_margin], axis=1)

# Keep rows where we actually have a margin
df_valid = df.dropna(subset=["margin"])

df_selected = df_valid.sort_values("margin", ascending=False).reset_index(drop=True)
print("Number of rows selected for DPO (by margin):", len(df_selected))

# ==========================================
# 3. Convert df_selected → DPO preference dataset
# ==========================================

def row_to_pairs(row):
    """
    Turn one row into multiple (prompt, chosen, rejected) pairs
    using the best_temp and the 3 other temps as rejected.
    """
    mapping = {
        "0.4": ("mistral_master_0.4", "mistral_master_04_nlp_score"),
        "0.6": ("mistral_master_0.6", "mistral_master_06_nlp_score"),
        "0.8": ("mistral_master_0.8", "mistral_master_08_nlp_score"),
        "1.0": ("mistral_master_1.0", "mistral_master_1_nlp_score"),
    }

    # gather candidate texts and scores (in case some are NaN)
    candidates = []
    for temp, (text_col, score_col) in mapping.items():
        text = row[text_col]
        score = row[score_col]
        if pd.isna(text) or pd.isna(score):
            continue
        candidates.append((temp, str(text), float(score)))

    if len(candidates) < 2:
        return []

    # use best_temp from the margin computation
    best_temp = row["best_temp"]
    if pd.isna(best_temp) or best_temp not in [c[0] for c in candidates]:
        # fallback: recompute best from candidates
        best_temp, best_text, best_score = max(candidates, key=lambda x: x[2])
    else:
        # find the corresponding text for best_temp
        for temp, text, score in candidates:
            if temp == best_temp:
                best_text = text
                best_score = score
                break

    prompt = str(row[PROMPT_COL])

    pairs = []
    for temp, text, score in candidates:
        if temp == best_temp:
            continue  # this is chosen, not rejected

        pairs.append(
            {
                "prompt": prompt,
                "chosen": best_text,  # best score = accepted
                "rejected": text,     # other temps = rejected
            }
        )

    return pairs

all_pairs = []
for _, r in df_selected.iterrows():
    all_pairs.extend(row_to_pairs(r))

if not all_pairs:
    raise ValueError("No preference pairs were generated; check your scores/columns/top-50 selection.")

dpo_dataset = Dataset.from_list(all_pairs)
print("Number of preference pairs:", len(dpo_dataset))
print("Example pair:", dpo_dataset[0])

# optional: train/val split
train_test = dpo_dataset.train_test_split(test_size=0.05, seed=42)

# ==========================================
# 4. Load Llama 3 8B in 4-bit (your code)
# ==========================================

print("Loading Llama 3 8B model and tokenizer with 4-bit Quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# disable cache for training
model.config.use_cache = False

# ==========================================
# 5. Define LoRA (QLoRA) config for DPO
# ==========================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ==========================================
# 6. DPO training config
# ==========================================

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    num_train_epochs=1,               # start small
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    bf16=torch.cuda.is_available(),   # on A100 40GB this is fine
    remove_unused_columns=False,
    max_length=1024,                  # total prompt+response length
    max_prompt_length=512,            # prompt truncation length
    beta=0.1,                         # <-- move beta here
)

# optional: train/val split (if not already done above)
train_test = dpo_dataset.train_test_split(test_size=0.05, seed=42)

# ==========================================
# 7. Create DPOTrainer (PEFT + 4-bit) & train
# ==========================================

trainer = DPOTrainer(
    model=model,
    ref_model=None,               # reference-free DPO; or load a ref model if you want
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    processing_class=tokenizer,   # <- for newer TRL: replaces tokenizer=tokenizer
    peft_config=peft_config,      # QLoRA adapters
    # no beta here anymore
)

trainer.train()

# Save LoRA adapter (and tokenizer) into OUTPUT_DIR
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("DPO fine-tuning complete. Adapter + tokenizer saved to:", OUTPUT_DIR)

