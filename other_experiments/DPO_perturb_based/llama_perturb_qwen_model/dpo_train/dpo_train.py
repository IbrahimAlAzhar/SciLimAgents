import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

# -----------------------------
# Paths  — UPDATE: uses the chat-template JSONL
# -----------------------------
MODEL_PATH = "qwen2_5_3b_instruct"
TRAIN_JSONL = "other_experiments/DPO_perturb_based/dpo_train/final_dpo_pairs/dpo_training_pairs_qwen_chat.jsonl"

OUTPUT_DIR = "other_experiments/DPO_perturb_based/dpo_train/qwen25_3b_dpo_output_chat"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# -----------------------------
# Quantization config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

# -----------------------------
# Load base model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA config
# -----------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -----------------------------
# Load DPO dataset (chat-template format)
# -----------------------------
raw_dataset = load_dataset("json", data_files=TRAIN_JSONL, split="train")

print("Columns:", raw_dataset.column_names)
print("Dataset size:", len(raw_dataset))
print("First prompt (200 chars):", raw_dataset[0]["prompt"][:200])

# Validate
required_cols = {"prompt", "chosen", "rejected"}
missing = required_cols - set(raw_dataset.column_names)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

def clean_row(example):
    return {
        "prompt":   str(example["prompt"]).strip(),
        "chosen":   str(example["chosen"]).strip(),
        "rejected": str(example["rejected"]).strip(),
    }

dataset = raw_dataset.map(clean_row, remove_columns=raw_dataset.column_names)
dataset = dataset.filter(
    lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
)

print("Cleaned dataset size:", len(dataset))

# Train / validation split
split_dataset = dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))

# -----------------------------
# DPO training config
# -----------------------------
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
    max_prompt_length=4096,
    max_length=6144,
    beta=0.1,
    label_smoothing=0.0,
    gradient_checkpointing=True,
    seed=SEED,
)

# -----------------------------
# DPO trainer
# -----------------------------
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save
trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nTraining complete. Saved to: {FINAL_MODEL_DIR}")