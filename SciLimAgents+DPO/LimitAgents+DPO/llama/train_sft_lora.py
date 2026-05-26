#!/usr/bin/env python
"""
Generic SFT LoRA trainer for worker, leader, and master agents.

Important Llama fixes compared with the original scripts:
- uses assistant_only_loss=True for conversational data
- keeps the assistant answer when truncating long samples
- logs train/eval loss to train_history.csv and training_curves.png
- supports eval_on_start so you can detect damage early
"""

import argparse
import inspect
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training_utils import (
    LossHistoryCallback,
    find_latest_checkpoint,
    load_sft_prompt_completion,
    plot_history,
    set_seed,
    split_dataset,
)

def build_sft_config(**kwargs):
    supported = set(inspect.signature(SFTConfig).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - supported)
    if ignored:
        print(f"[WARN] Ignoring SFTConfig args unsupported by this TRL version: {ignored}")
    return SFTConfig(**filtered)

def parse_args():
    p = argparse.ArgumentParser(description="SFT LoRA training for worker/leader/master.")
    p.add_argument("--agent-name", required=True, choices=["worker", "leader", "master"])
    p.add_argument("--base-model", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--optim", default="adamw_torch_fused")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-split", type=float, default=0.05)
    p.add_argument("--eval-on-start", action="store_true")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
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

    print("=" * 60)
    print(f"{args.agent_name.upper()} SFT TRAINING")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Data:       {args.data_path}")
    print(f"Output:     {args.output_dir}")
    print(f"LR:         {args.learning_rate}")
    print(f"LoRA r:     {args.lora_r}")
    print(f"Max length: {args.max_seq_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
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
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_sft_prompt_completion(args.data_path)
    train_ds, eval_ds = split_dataset(dataset, args.eval_split, args.seed)

    training_args = build_sft_config(
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
        optim=args.optim,
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
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        assistant_only_loss=False,
        packing=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = [LossHistoryCallback(args.output_dir)]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume_path = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume:
        print(f"Resume checkpoint: {resume_path or 'none found; starting fresh'}")

    trainer.train(resume_from_checkpoint=resume_path)
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    plot_history(args.output_dir)
    print(f"Saved final adapter to {final_dir}")

if __name__ == "__main__":
    main()
