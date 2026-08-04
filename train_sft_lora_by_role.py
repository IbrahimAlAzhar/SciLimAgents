"""
train_sft_lora_by_role.py — role-parameterized SFT (LoRA) for the **Qwen3-4B** student.

One script for all three agents; pick the role with env vars:

    BASE_MODEL=$BASE_MODEL SELECT_DIR=$SELECT_DIR TRAIN_ROOT=$TRAIN_ROOT \
    ROLE=worker python train_sft_lora_by_role.py

Set ROLE to worker, leader or master. SFT_DATA and OUT_DIR default to
role-derived names inside SELECT_DIR / TRAIN_ROOT, and can be overridden.

Why prompt/completion instead of `messages`
-------------------------------------------
Qwen3's chat template injects thinking scaffolding (`<think> ... </think>`)
depending on `enable_thinking`. Rollouts were generated with
`enable_thinking=false`, so we pre-render every example with the SAME setting
and hand TRL a plain {"prompt", "completion"} pair. Training-time formatting
then matches vLLM inference-time formatting exactly, and completion-only loss
is unambiguous. Set PRERENDER=0 to fall back to raw `messages`.

Output: a LoRA adapter at $OUT_DIR/final that vLLM can serve with
    --enable-lora --lora-modules <role>=$OUT_DIR/final
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# PEFT's LoRA dispatcher tries dispatch_awq for EVERY model — it checks whether
# autoawq is INSTALLED, not whether it IMPORTS. So a broken AutoAWQ (0.2.x against
# transformers >= ~4.52, missing PytorchGELUTanh) makes get_peft_model fail on a
# plain bf16 model that has nothing to do with AWQ. Patch the removed activation
# symbols before any PEFT call. No-op when autoawq is absent or already working.
try:
    import awq_compat
    awq_compat.patch_activations(verbose=True)
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

ROLE       = os.environ.get("ROLE", "worker")


# =============================================================================
# ENVIRONMENT CONFIGURATION
# All input/output locations are supplied at run time. No paths, dataset sizes
# or credentials are stored in this file.
# =============================================================================
def _require_env(name, hint=""):
    """Return a mandatory environment variable, or exit with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Required environment variable {name} is not set."
            + (f"  Expected: {hint}" if hint else "")
        )
    return value


def _optional_int(name):
    """Return an int env var, or None when unset/blank (meaning 'no limit')."""
    raw = os.environ.get(name, "").strip()
    if raw in ("", "none", "None", "null"):
        return None
    return int(raw)


BASE_MODEL = _require_env("BASE_MODEL", "base student checkpoint directory")
DATA_DIR   = _require_env("SELECT_DIR", "directory holding the selected training datasets")
TRAIN_ROOT = _require_env("TRAIN_ROOT", "directory to write trained adapters into")

SFT_DATA   = os.environ.get("SFT_DATA", os.path.join(DATA_DIR, f"sft_dataset_{ROLE}.json"))
OUTPUT_DIR = os.environ.get("OUT_DIR", os.path.join(TRAIN_ROOT, f"{ROLE}_sft"))
FINAL_DIR  = os.path.join(OUTPUT_DIR, "final")

# ---- hyperparameters (env-overridable) ----
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS", 3))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 1))
EVAL_BATCH_SIZE  = int(os.environ.get("EVAL_BATCH_SIZE", 1))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM", 16))
LEARNING_RATE    = float(os.environ.get("LR", 2e-5))
MAX_SEQ_LENGTH   = int(os.environ.get("MAX_SEQ_LEN", 8192))
WARMUP_RATIO     = float(os.environ.get("WARMUP_RATIO", 0.05))
WEIGHT_DECAY     = 0.01
LR_SCHEDULER     = "cosine"
LOGGING_STEPS    = 10
SAVE_STEPS       = int(os.environ.get("SAVE_STEPS", 100))
EVAL_SPLIT       = float(os.environ.get("EVAL_SPLIT", 0.05))

# LoRA — r=32 must match vLLM's --max-lora-rank when serving.
LORA_R       = int(os.environ.get("LORA_R", 32))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", 64))
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

USE_4BIT  = os.environ.get("USE_4BIT", "0") == "1"
PRERENDER = os.environ.get("PRERENDER", "1") == "1"

# Small roles (leader/master: 1 sample per paper) need more passes.
if ROLE in ("leader", "master") and "NUM_EPOCHS" not in os.environ:
    NUM_EPOCHS = 5


# =============================================================================
# DATA
# =============================================================================

MERGE_SYSTEM = os.environ.get("MERGE_SYSTEM", "0") == "1"


def merge_system(messages):
    """Fold a system turn into the first user turn.

    Mistral-7B-Instruct v0.x templates define no system role and raise on one,
    while every canonical prompt here is [system, user].
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_txt, rest = messages[0]["content"], messages[1:]
    if rest and rest[0].get("role") == "user":
        return ([{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}]
                + rest[1:])
    return [{"role": "user", "content": sys_txt}] + rest


def apply_template(tokenizer, messages):
    """Render a prompt, tolerating templates without system-role or thinking support."""
    msgs = merge_system(messages) if MERGE_SYSTEM else messages
    for attempt_msgs in (msgs, merge_system(messages)):
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    attempt_msgs, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                continue
    raise RuntimeError("apply_chat_template failed for every fallback; check the "
                       "tokenizer's chat template")


def _render(tokenizer, messages):
    """Split a chat into (prompt_text, completion_text) with thinking disabled."""
    assistant_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prefix = messages[:assistant_idx]
    answer = messages[assistant_idx]["content"]
    prompt_text = apply_template(tokenizer, prefix)
    eos = tokenizer.eos_token or "<|im_end|>"
    return prompt_text, answer.strip() + eos


def load_sft_data(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} SFT records from {path}")

    processed, skipped = [], 0
    for r in records:
        messages = r.get("messages", [])
        if not messages or not any(m["role"] == "assistant" for m in messages):
            skipped += 1
            continue
        if not str(messages[-1].get("content", "")).strip():
            skipped += 1
            continue
        if PRERENDER:
            prompt_text, completion_text = _render(tokenizer, messages)
            processed.append({"prompt": prompt_text, "completion": completion_text})
        else:
            processed.append({"messages": messages})

    print(f"After filtering: {len(processed)} valid samples ({skipped} skipped)")
    if not processed:
        raise ValueError(f"No usable SFT samples in {path}")
    if len(processed) < 50:
        print(f"[WARN] Only {len(processed)} {ROLE} samples — consider more epochs.")

    # length report (helps you catch silent truncation)
    if PRERENDER:
        lens = [len(tokenizer(p["prompt"] + p["completion"],
                              add_special_tokens=False)["input_ids"])
                for p in processed[:64]]
        over = sum(1 for l in lens if l > MAX_SEQ_LENGTH)
        print(f"Token length (first {len(lens)}): mean={sum(lens)//len(lens)} "
              f"max={max(lens)}  >{MAX_SEQ_LENGTH}: {over}")

    return Dataset.from_list(processed)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"{ROLE.upper()} SFT TRAINING — Qwen3-4B + LoRA")
    print("=" * 60)
    print(f"base model: {BASE_MODEL}")
    print(f"sft data:   {SFT_DATA}")
    print(f"output dir: {OUTPUT_DIR}")
    print(f"epochs={NUM_EPOCHS} bs={BATCH_SIZE} accum={GRAD_ACCUM_STEPS} "
          f"lr={LEARNING_RATE} max_len={MAX_SEQ_LENGTH} lora_r={LORA_R}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16, "device_map": "auto"}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    ))
    model.enable_input_require_grads()   # gradient checkpointing + LoRA
    model.print_trainable_parameters()

    dataset = load_sft_data(SFT_DATA, tokenizer)

    # How many optimizer steps will this actually run?
    eff_batch = max(1, BATCH_SIZE * GRAD_ACCUM_STEPS)
    steps_per_epoch = max(1, -(-len(dataset) // eff_batch))      # ceil
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"Planned: {steps_per_epoch} steps/epoch x {NUM_EPOCHS} epochs "
          f"= {total_steps} optimizer steps (effective batch {eff_batch})")

    # leader/master have ~1 sample per paper, so a 5% eval split costs real data
    # AND never runs: with ~47 total steps and save/eval every 100, no evaluation
    # or checkpoint ever fires and load_best_model_at_end silently does nothing.
    # Below that threshold, train on everything and checkpoint per epoch instead.
    if EVAL_SPLIT > 0 and len(dataset) > 20 and total_steps >= 2 * SAVE_STEPS:
        split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
        strategy, step_interval = "steps", SAVE_STEPS
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)} "
              f"(eval/save every {SAVE_STEPS} steps)")
    else:
        train_dataset, eval_dataset = dataset, None
        strategy, step_interval = "epoch", None
        if EVAL_SPLIT > 0:
            print(f"Train: {len(train_dataset)}, Eval: None — only {total_steps} "
                  f"steps, too few for eval every {SAVE_STEPS}; using all data "
                  f"and checkpointing per epoch")
        else:
            print(f"Train: {len(train_dataset)}, Eval: None")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=LOGGING_STEPS,
        # save/eval strategies must match for load_best_model_at_end
        save_strategy=strategy,
        save_steps=step_interval if strategy == "steps" else 500,
        save_total_limit=2,
        eval_strategy=strategy if eval_dataset else "no",
        eval_steps=step_interval if (eval_dataset and strategy == "steps") else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n{ROLE} SFT adapter saved to: {FINAL_DIR}\nDone.")


if __name__ == "__main__":
    main()