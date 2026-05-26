"""
train_master_sft.py — SFT (LoRA) training for the Master agent.

Can run in PARALLEL with train_worker_sft.py and train_leader_sft.py.

Works with any HuggingFace causal LM — pass --base-model on the CLI:
  - Llama-3:  meta-llama/Meta-Llama-3-8B-Instruct
  - Mistral:  mistralai/Mistral-7B-Instruct-v0.3

NOTE: Master samples may be fewer than worker/leader since there is only
1 master turn per rollout (vs 12 worker + 7 leader turns). Consider
increasing --num-epochs to 5-8 if your master dataset is small.

Usage:
  python train_master_sft.py \
      --base-model meta-llama/Meta-Llama-3-8B-Instruct \
      --cache-dir  data/.../llama3_8b_instruct \
      --data-path  data/.../sft_dataset_master_long.json \
      --output-dir data/.../train/llama/master_sft
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Master SFT training.")

    p.add_argument("--base-model", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--use-4bit", action="store_true")

    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--num-epochs",       type=int,   default=3)
    p.add_argument("--batch-size",       type=int,   default=2)
    p.add_argument("--eval-batch-size",  type=int,   default=1)
    p.add_argument("--grad-accum-steps", type=int,   default=8)
    p.add_argument("--learning-rate",    type=float, default=2e-5)
    p.add_argument("--max-seq-length",   type=int,   default=4096)
    p.add_argument("--warmup-ratio",     type=float, default=0.05)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--lr-scheduler",     default="cosine")
    p.add_argument("--logging-steps",    type=int,   default=10)
    p.add_argument("--save-steps",       type=int,   default=100)
    p.add_argument("--eval-split",       type=float, default=0.05)

    p.add_argument("--lora-r",       type=int,   default=32)
    p.add_argument("--lora-alpha",   type=int,   default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    return p.parse_args()

# =============================================================================
# Data
# =============================================================================

def load_sft_data(path: str) -> Dataset:
    with open(path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} SFT records from {path}")

    processed = []
    for r in records:
        msgs = r.get("messages", [])
        if not msgs:
            continue
        if "assistant" not in [m["role"] for m in msgs]:
            continue
        processed.append({"messages": msgs})

    print(f"After filtering: {len(processed)} valid samples")
    if len(processed) < 50:
        print(f"[WARN] Only {len(processed)} master samples. "
              "Consider increasing --num-epochs to 5-8 for better convergence.")
    return Dataset.from_list(processed)

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 60)
    print("MASTER SFT TRAINING")
    print("=" * 60)
    print(f"Base model:  {args.base_model}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Data:        {args.data_path}")
    print(f"Output dir:  {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "device_map": "auto",
        "cache_dir": args.cache_dir,
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_data(args.data_path)
    if args.eval_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    else:
        train_ds, eval_ds = dataset, None
        print(f"Train: {len(train_ds)}, Eval: None")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss" if eval_ds else None,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nMaster SFT model saved to: {final_dir}")
    print("Done.")

if __name__ == "__main__":
    main()