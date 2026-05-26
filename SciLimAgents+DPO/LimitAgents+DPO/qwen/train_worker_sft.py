"""
train_worker_sft.py — SFT training for the Worker agent (LoRA on Qwen 2.5 3B).

Run this FIRST (or in parallel with leader/master SFT).
The resulting checkpoint will be used as the base for DPO training.

Usage:
  python train_worker_sft.py
  python train_worker_sft.py --resume   # to resume from latest checkpoint
"""

import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL   = "qwen2_5_3b_instruct"
SFT_DATA     = "other_experiments/dpo/sft_and_dpo_pairs/sft_dataset_worker.json"
OUTPUT_DIR   = "other_experiments/dpo/train/worker_sft"

# Training hyperparameters
NUM_EPOCHS       = 3
BATCH_SIZE       = 2
EVAL_BATCH_SIZE  = 1      # <-- IMPORTANT: keep eval batch small to avoid OOM
GRAD_ACCUM_STEPS = 8      # effective batch = 2 * 8 = 16
LEARNING_RATE    = 2e-5
MAX_SEQ_LENGTH   = 4096
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
LR_SCHEDULER     = "cosine"
LOGGING_STEPS    = 10
SAVE_STEPS       = 100
EVAL_SPLIT       = 0.05   # 5% held out for eval

# LoRA config
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Set to True if you want 4-bit quantization (saves VRAM)
USE_4BIT = False

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sft_data(path: str) -> Dataset:
    """
    Load SFT JSON and convert to HuggingFace Dataset.
    The JSON has a 'messages' field already in chat format:
      [{"role": "system", "content": ...},
       {"role": "user", "content": ...},
       {"role": "assistant", "content": ...}]
    """
    with open(path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} SFT records from {path}")

    # Keep only the messages field for training
    processed = []
    for r in records:
        messages = r.get("messages", [])
        if not messages:
            continue
        # Validate: must contain at least one assistant turn
        roles = [m["role"] for m in messages]
        if "assistant" not in roles:
            continue
        processed.append({"messages": messages})

    print(f"After filtering: {len(processed)} valid samples")
    return Dataset.from_list(processed)

def find_latest_checkpoint(output_dir: str):
    """Return the path to the most recent checkpoint-* dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, ckpts[-1])

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Simple flag: pass --resume to resume from latest checkpoint
    resume = "--resume" in sys.argv

    print("=" * 60)
    print("WORKER SFT TRAINING")
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL}")
    print(f"SFT data:    {SFT_DATA}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"LoRA rank:   {LORA_R}")
    print(f"4-bit quant: {USE_4BIT}")
    print(f"Resume:      {resume}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Load model ----
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,        # renamed from torch_dtype
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
    model.config.use_cache = False

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Load data ----
    dataset = load_sft_data(SFT_DATA)

    # Train/eval split
    if EVAL_SPLIT > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Train: {len(train_dataset)}, Eval: None (too few samples)")

    # ---- Training arguments (TRL 0.27.x) ----
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,   # <-- FIX: explicit & small
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_accumulation_steps=4,                    # <-- FIX: stream logits to CPU
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=SAVE_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # SFT-specific:
        max_length=MAX_SEQ_LENGTH,      # renamed from max_seq_length
        completion_only_loss=True,      # train loss only on assistant turns
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ---- Train ----
    resume_path = None
    if resume:
        resume_path = find_latest_checkpoint(OUTPUT_DIR)
        if resume_path:
            print(f"\nResuming from checkpoint: {resume_path}")
        else:
            print("\nNo checkpoint found; starting from scratch.")

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_path)

    # ---- Save final ----
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nWorker SFT model saved to: {final_dir}")
    print("Done.")

if __name__ == "__main__":
    main()