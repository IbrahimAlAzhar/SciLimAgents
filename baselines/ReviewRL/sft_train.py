# =============================================================================
# sft_train.py
# -----------------------------------------------------------------------------
# Optional supervised fine-tuning script that mirrors Section 3.3 of ReviewRL
# ("Supervised Finetuning"), but trims the recipe so it fits in a single
# 40 GB GPU.
#
# What ReviewRL does upstream (paper + Appendix C.2):
#     - Backbone: Qwen2.5-7B-Instruct
#     - DeepSpeed ZeRO-3, batch size 8, lr 5e-6, 2 epochs
#     - Training data: ICLR 2024 split of DeepReview-13k, with novelty
#       verification queries embedded in the input and the meta-review
#       used as the target.
#
# What this script does (single-GPU friendly):
#     - Backbone: Qwen2.5-3B-Instruct (the user's local checkpoint)
#     - LoRA (PEFT) so we don't need 80GB+ of activation memory
#     - bf16, gradient checkpointing
#     - SFTTrainer (TRL) consuming the JSONL produced by sft_data_prep.py
#
# This is provided so the user can claim "we replicated the SFT warm-up step
# of ReviewRL on our limitation-only data".  Inference still works without
# running this file (inference uses the bare base model).
# =============================================================================

from __future__ import annotations

import argparse
import os

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReviewRL-style SFT (LoRA, single GPU)")
    p.add_argument(
        "--sft-jsonl",
        type=str,
        required=True,
        help="Path to the JSONL produced by sft_data_prep.py.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="qwen2_5_3b_instruct",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="qwen2_5_3b_instruct",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="review_rl/sft_ckpt",
    )
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Lazy imports so users without TRL/PEFT can still run inference.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Tokenizer + model -------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=args.cache_dir, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ---- 2. LoRA config (keeps memory < 40GB) ---------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- 3. Dataset (JSONL with "messages" -> chat-style SFT) -----------
    ds = load_dataset("json", data_files=args.sft_jsonl, split="train")
    print(f"[sft_train] loaded {len(ds)} SFT examples from {args.sft_jsonl}")

    # ---- 4. SFT config --------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_seq_len,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        packing=False,
    )

    # ---- 5. Train -------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_config,
        peft_config=peft_config,
    )
    trainer.train()

    # ---- 6. Persist final adapter --------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[sft_train] saved final LoRA adapter to {final_dir}")

if __name__ == "__main__":
    main()