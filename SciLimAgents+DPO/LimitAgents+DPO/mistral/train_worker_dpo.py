"""
train_worker_dpo.py - DPO (LoRA) training for the Worker agent.

Run this AFTER train_worker_sft.py finishes. Loads the base model, merges the
SFT LoRA into it, then trains a NEW LoRA via DPO.

This version is intentionally defensive around chat data because Mistral's chat
template accepts only user/assistant messages, plus one optional initial system
message. Unsupported roles in the DPO prompt are folded into user-visible text.
"""

import argparse
import json
import os
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Worker DPO training.")

    # Model
    p.add_argument("--base-model", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--use-4bit", action="store_true")

    # Adapters / data
    p.add_argument("--sft-checkpoint", required=True,
                   help="SFT LoRA adapter to merge into the base model.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)

    # Training
    p.add_argument("--num-epochs",        type=int,   default=2)
    p.add_argument("--batch-size",        type=int,   default=1)
    p.add_argument("--grad-accum-steps",  type=int,   default=16)
    p.add_argument("--learning-rate",     type=float, default=5e-7)
    p.add_argument("--max-seq-length",    type=int,   default=4096)
    p.add_argument("--max-prompt-length", type=int,   default=3072)
    p.add_argument("--warmup-ratio",      type=float, default=0.1)
    p.add_argument("--weight-decay",      type=float, default=0.01)
    p.add_argument("--beta",              type=float, default=0.1,
                   help="DPO beta - controls deviation from reference.")
    p.add_argument("--logging-steps",     type=int,   default=5)
    p.add_argument("--save-steps",        type=int,   default=50)
    p.add_argument("--eval-split",        type=float, default=0.05)

    # LoRA (smaller rank for DPO)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    return p.parse_args()

# =============================================================================
# Data
# =============================================================================

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
}

def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()

def _append_message(messages: list[dict[str, str]], role: str, content: str) -> None:
    if not content:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += "\n\n" + content
    else:
        messages.append({"role": role, "content": content})

def normalize_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Convert arbitrary training records into a Mistral-compatible conversation.

    Mistral-v0.3 allows an optional initial system message, then strictly
    alternating user/assistant messages. Tool/developer/function/etc. messages
    are preserved as labeled text under the user role.
    """
    normalized: list[dict[str, str]] = []

    for i, msg in enumerate(messages):
        raw_role = str(msg.get("role", "user")).lower().strip()
        content = _content_to_text(msg.get("content"))
        if not content:
            continue

        if raw_role == "system" and not normalized:
            _append_message(normalized, "system", content)
            continue

        role = ROLE_ALIASES.get(raw_role)
        if role is None:
            role = "user"
            content = f"[{raw_role}]\n{content}"
        elif raw_role == "system":
            role = "user"
            content = f"[system]\n{content}"

        if normalized and normalized[-1]["role"] == "system" and role == "assistant":
            _append_message(normalized, "user", "[context]")
        elif normalized and normalized[-1]["role"] == role and role != "user":
            _append_message(normalized, "user", "[continue]")

        _append_message(normalized, role, content)

    if not normalized:
        return []

    if normalized[-1]["role"] == "assistant":
        # DPO completions are assistant messages, so the prompt must end in user.
        normalized.append({"role": "user", "content": "Continue."})

    # Final pass: make sure role alternation is valid after an optional system.
    fixed: list[dict[str, str]] = []
    start = 0
    if normalized[0]["role"] == "system":
        fixed.append(normalized[0])
        start = 1

    expected = "user"
    for msg in normalized[start:]:
        role = msg["role"]
        content = msg["content"]
        if role != expected:
            if expected == "user":
                content = f"[{role}]\n{content}"
                role = "user"
            else:
                fixed.append({"role": expected, "content": "[continue]"})
        fixed.append({"role": role, "content": content})
        expected = "assistant" if role == "user" else "user"

    if fixed and fixed[-1]["role"] == "assistant":
        fixed.append({"role": "user", "content": "Continue."})

    return fixed

def load_dpo_data(path: str) -> Dataset:
    """
    DPO records have:
      - input_messages: list[ {role, content} ]
      - chosen:   str
      - rejected: str

    TRL's explicit conversational preference format is:
      - prompt:   conversation up to the response
      - chosen:   assistant completion only
      - rejected: assistant completion only
    """
    with open(path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} DPO records from {path}")

    processed = []
    dropped = 0
    for r in records:
        msgs = normalize_prompt_messages(r.get("input_messages", []))
        chosen = _content_to_text(r.get("chosen"))
        rejected = _content_to_text(r.get("rejected"))
        if not msgs or not chosen or not rejected:
            dropped += 1
            continue

        processed.append({
            "prompt": msgs,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })

    print(f"After filtering: {len(processed)} valid DPO pairs")
    if dropped:
        print(f"Dropped {dropped} invalid DPO records")
    if not processed:
        raise ValueError(
            f"No valid DPO pairs in {path}. "
            "Check that the JSON has input_messages / chosen / rejected fields."
        )
    return Dataset.from_list(processed)

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=" * 60)
    print("WORKER DPO TRAINING")
    print("=" * 60)
    print(f"Base model:     {args.base_model}")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print(f"Data:           {args.data_path}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Beta:           {args.beta}")
    print(f"LR:             {args.learning_rate}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.sft_checkpoint):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {args.sft_checkpoint}. "
            "Run train_worker_sft.py first."
        )

    tokenizer_source = args.sft_checkpoint if os.path.exists(
        os.path.join(args.sft_checkpoint, "tokenizer_config.json")
    ) else args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, cache_dir=args.cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
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

    print("Loading + merging SFT LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.sft_checkpoint)
    model = model.merge_and_unload()
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    print("SFT adapter merged into base model.")

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    dataset = load_dpo_data(args.data_path)
    if args.eval_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    else:
        train_ds, eval_ds = dataset, None
        print(f"Train: {len(train_ds)}, Eval: None")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nStarting DPO training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nWorker DPO model saved to: {final_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
