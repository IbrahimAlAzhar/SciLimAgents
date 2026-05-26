"""
train_worker_dpo.py — DPO training for the Worker agent.

Run this AFTER train_worker_sft.py finishes.
Loads the SFT LoRA checkpoint and applies DPO on top.

Usage:
  python train_worker_dpo.py
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL      = "qwen2_5_3b_instruct"
SFT_CHECKPOINT  = "other_experiments/dpo/train/worker_sft/final"
DPO_DATA        = "other_experiments/dpo/sft_and_dpo_pairs/dpo_dataset_worker.json"
OUTPUT_DIR      = "other_experiments/dpo/train/worker_dpo"

# DPO hyperparameters
NUM_EPOCHS       = 2
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 16    # effective batch = 1 * 16 = 16
LEARNING_RATE    = 5e-7   # DPO needs much lower LR than SFT
MAX_SEQ_LENGTH   = 4096
MAX_PROMPT_LENGTH = 3072
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
BETA             = 0.1    # DPO beta — controls deviation from reference
LOGGING_STEPS    = 5
SAVE_STEPS       = 50
EVAL_SPLIT       = 0.05

# LoRA config for DPO (new adapter on top of merged SFT)
LORA_R       = 16         # smaller rank for DPO fine-tuning
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

USE_4BIT = False

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dpo_data(path: str, tokenizer) -> Dataset:
    """
    Load DPO JSON and convert to HuggingFace Dataset.
    DPO records have:
      - input_messages: [{"role": "system", ...}, {"role": "user", ...}]
      - chosen: str (good output)
      - rejected: str (bad output)

    TRL DPOTrainer expects: prompt, chosen, rejected
    where prompt/chosen/rejected are chat-formatted strings.
    """
    with open(path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} DPO records from {path}")

    processed = []
    for r in records:
        input_msgs = r.get("input_messages", [])
        chosen_text = r.get("chosen", "")
        rejected_text = r.get("rejected", "")

        if not input_msgs or not chosen_text or not rejected_text:
            continue

        # Build prompt, chosen, rejected as full message lists
        prompt_messages = input_msgs  # system + user messages

        chosen_messages = prompt_messages + [
            {"role": "assistant", "content": chosen_text}
        ]
        rejected_messages = prompt_messages + [
            {"role": "assistant", "content": rejected_text}
        ]

        processed.append({
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
        })

    print(f"After filtering: {len(processed)} valid DPO pairs")

    if len(processed) == 0:
        raise ValueError(
            f"No valid DPO pairs found in {path}. "
            "Check that dpo_dataset_worker.json has entries with "
            "input_messages, chosen, and rejected fields."
        )

    return Dataset.from_list(processed)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("WORKER DPO TRAINING")
    print("=" * 60)
    print(f"Base model:     {BASE_MODEL}")
    print(f"SFT checkpoint: {SFT_CHECKPOINT}")
    print(f"DPO data:       {DPO_DATA}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"DPO beta:       {BETA}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Verify SFT checkpoint exists ----
    if not os.path.exists(SFT_CHECKPOINT):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {SFT_CHECKPOINT}. "
            "Run train_worker_sft.py first."
        )

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Load base model + merge SFT LoRA ----
    print("Loading base model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    print("Loading and merging SFT LoRA adapter...")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    print("SFT adapter merged into base model.")

    model.config.use_cache = False

    # ---- Reference model (same as merged SFT — DPOTrainer handles this) ----
    # DPOTrainer creates its own ref model copy internally.
    # We apply a NEW LoRA for DPO training.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # ---- Load data ----
    dataset = load_dpo_data(DPO_DATA, tokenizer)

    # Train/eval split
    if EVAL_SPLIT > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Train: {len(train_dataset)}, Eval: None")

    # ---- DPO Training config ----
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=SAVE_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ---- Train ----
    print("\nStarting DPO training...")
    trainer.train()

    # ---- Save final ----
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nWorker DPO model saved to: {final_dir}")
    print("Done.")

if __name__ == "__main__":
    main()