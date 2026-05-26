#!/usr/bin/env python
"""
Worker DPO trainer with safer defaults for Llama.

Key fixes:
- DPO chosen/rejected are assistant completions only
- tokenizer padding_side="left" as required by TRL DPOTrainer
- logs train/eval loss and DPO reward metrics
- uses low LR and small LoRA rank by default to avoid preference overfitting
"""

import argparse
import inspect
import os

import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from training_utils import (
    LossHistoryCallback,
    load_dpo_messages,
    plot_history,
    set_seed,
    split_dataset,
)

def build_dpo_config(**kwargs):
    supported = set(inspect.signature(DPOConfig).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - supported)
    if ignored:
        print(f"[WARN] Ignoring DPOConfig args unsupported by this TRL version: {ignored}")
    return DPOConfig(**filtered)

def parse_args():
    p = argparse.ArgumentParser(description="Worker DPO training.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--sft-checkpoint", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-7)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--max-prompt-length", type=int, default=6144)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-split", type=float, default=0.05)
    p.add_argument("--eval-on-start", action="store_true")

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if not os.path.exists(args.sft_checkpoint):
        raise FileNotFoundError(f"SFT checkpoint not found: {args.sft_checkpoint}")

    print("=" * 60)
    print("WORKER DPO TRAINING")
    print("=" * 60)
    print(f"Base model:     {args.base_model}")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print(f"Data:           {args.data_path}")
    print(f"Output:         {args.output_dir}")
    print(f"LR:             {args.learning_rate}")
    print(f"Beta:           {args.beta}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Merging SFT adapter before DPO...")
    model = PeftModel.from_pretrained(base_model, args.sft_checkpoint)
    model = model.merge_and_unload()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    dataset = load_dpo_messages(args.data_path)
    train_ds, eval_ds = split_dataset(dataset, args.eval_split, args.seed)

    training_args = build_dpo_config(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch_fused",
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        truncation_mode="keep_end",
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        tf32=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        eval_on_start=args.eval_on_start,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = [LossHistoryCallback(args.output_dir)]
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    plot_history(args.output_dir)
    print(f"Saved final DPO adapter to {final_dir}")

if __name__ == "__main__":
    main()
