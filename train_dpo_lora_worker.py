"""
train_dpo_lora_worker.py — worker DPO (LoRA) on top of the Qwen3-4B worker SFT adapter.

Pipeline position:
    train_sft_lora_by_role.py (ROLE=worker)  ->  worker_sft/final
    train_dpo_lora_worker.py                 ->  worker_dpo/final  <-- serve this

Preference data comes from `select_data_stepwise_process_reward.py`
(dpo_dataset_worker.json: best vs worst candidate per paper/worker group,
scored by the stepwise process reward + the GPT-4o-mini alignment judge).

Uses the DPO path that actually works with this TRL/transformers/peft combo:
  * inject model.warnings_issued (TRL expects it)
  * merge the SFT adapter, then apply the DPO LoRA ourselves with get_peft_model
    and DO NOT pass peft_config to DPOTrainer (avoids the double-wrap error)
  * enable_input_require_grads() for gradient checkpointing + LoRA
  * the reference policy is the adapter-disabled model, i.e. base+SFT — no
    second model copy on the GPU

Prompts/completions are pre-rendered through the Qwen3 chat template with
`enable_thinking=False`, matching both the rollouts and train_sft_lora_by_role.py.

Usage:
    BASE_MODEL=$BASE_MODEL SELECT_DIR=$SELECT_DIR TRAIN_ROOT=$TRAIN_ROOT \
    ROLE=worker python train_dpo_lora_worker.py

PREV_ADAPTER, DPO_DATA and OUT_DIR default to role-derived names inside
SELECT_DIR / TRAIN_ROOT, and can each be overridden.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# PEFT's LoRA dispatcher tries dispatch_awq for EVERY model — it checks whether
# autoawq is INSTALLED, not whether it IMPORTS. So a broken AutoAWQ (0.2.x against
# transformers >= ~4.52, missing PytorchGELUTanh) makes get_peft_model AND
# PeftModel.from_pretrained fail on a plain bf16 model that has nothing to do with
# AWQ. Patch the removed activation symbols before any PEFT call. No-op when
# autoawq is absent or already working.
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

DPO_DATA     = os.environ.get("DPO_DATA", os.path.join(DATA_DIR, f"dpo_dataset_{ROLE}.json"))
PREV_ADAPTER = os.environ.get("PREV_ADAPTER", os.path.join(TRAIN_ROOT, f"{ROLE}_sft", "final"))
OUTPUT_DIR   = os.environ.get("OUT_DIR", os.path.join(TRAIN_ROOT, f"{ROLE}_dpo"))
FINAL_DIR    = os.path.join(OUTPUT_DIR, "final")

DPO_EPOCHS       = int(os.environ.get("DPO_EPOCHS", 2))
DPO_BATCH_SIZE   = int(os.environ.get("DPO_BATCH_SIZE", 1))
DPO_GRAD_ACCUM   = int(os.environ.get("DPO_GRAD_ACCUM", 16))
# 5e-7 is the FULL-FINETUNE convention and is ~10x too low for LoRA adapters:
# with a few hundred pairs it yields ~20 optimizer steps that barely move the
# policy off the SFT checkpoint, and you measure "no difference between reward
# models" when in fact nothing trained. 5e-6 is the LoRA-appropriate value.
DPO_LR           = float(os.environ.get("DPO_LR", 5e-6))
DPO_MAX_SEQ_LEN  = int(os.environ.get("DPO_MAX_SEQ_LEN", 8192))
DPO_MAX_PROMPT   = int(os.environ.get("DPO_MAX_PROMPT", 6656))
DPO_WARMUP_RATIO = 0.1
DPO_BETA         = float(os.environ.get("DPO_BETA", 0.1))
DPO_LORA_R       = int(os.environ.get("DPO_LORA_R", 16))
DPO_LORA_ALPHA   = int(os.environ.get("DPO_LORA_ALPHA", 32))
# DPO eval casts full logits to fp32 (~OOM on 40GB). Off by default.
DPO_EVAL_SPLIT   = float(os.environ.get("DPO_EVAL_SPLIT", 0.0))

LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
WEIGHT_DECAY = 0.01
USE_4BIT     = os.environ.get("USE_4BIT", "0") == "1"


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


def _render_prompt(tokenizer, messages):
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


def load_dpo_data(path: str, tokenizer) -> Dataset:
    """Plain-text preference format: {"prompt", "chosen", "rejected"}."""
    with open(path) as f:
        records = json.load(f)
    print(f"[DPO] Loaded {len(records)} records from {path}")

    eos = tokenizer.eos_token or "<|im_end|>"
    processed = []
    for r in records:
        prompt_msgs   = r.get("prompt") or r.get("input_messages") or []
        chosen_text   = (r.get("chosen") or "").strip()
        rejected_text = (r.get("rejected") or "").strip()
        if not prompt_msgs or not chosen_text or not rejected_text:
            continue
        if chosen_text == rejected_text:
            continue
        processed.append({
            "prompt":   _render_prompt(tokenizer, prompt_msgs),
            "chosen":   chosen_text + eos,
            "rejected": rejected_text + eos,
        })

    print(f"[DPO] After filtering: {len(processed)} valid pairs")
    if not processed:
        raise ValueError(f"No valid DPO pairs in {path}.")

    lens = [len(tokenizer(p["prompt"], add_special_tokens=False)["input_ids"])
            for p in processed[:64]]
    print(f"[DPO] Prompt tokens (first {len(lens)}): mean={sum(lens)//len(lens)} "
          f"max={max(lens)}  >{DPO_MAX_PROMPT}: {sum(1 for l in lens if l > DPO_MAX_PROMPT)}")
    return Dataset.from_list(processed)


def _split(dataset, eval_split):
    if eval_split > 0 and len(dataset) > 20:
        s = dataset.train_test_split(test_size=eval_split, seed=42)
        print(f"Train: {len(s['train'])}, Eval: {len(s['test'])}")
        return s["train"], s["test"]
    print(f"Train: {len(dataset)}, Eval: None")
    return dataset, None


def _base_model_kwargs():
    kw = {"trust_remote_code": True, "dtype": torch.bfloat16, "device_map": "auto"}
    if USE_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    return kw


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"{ROLE.upper()} DPO TRAINING — Qwen3-4B + LoRA")
    print("=" * 60)
    print(f"base model : {BASE_MODEL}")
    print(f"sft ckpt   : {PREV_ADAPTER}")
    print(f"dpo data   : {DPO_DATA}")
    print(f"output dir : {OUTPUT_DIR}")
    print(f"beta={DPO_BETA} lr={DPO_LR} epochs={DPO_EPOCHS} "
          f"max_len={DPO_MAX_SEQ_LEN} lora_r={DPO_LORA_R}\n")

    if not os.path.exists(PREV_ADAPTER):
        raise FileNotFoundError(f"SFT checkpoint missing: {PREV_ADAPTER}. Run train_sft_lora_by_role.py first.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base + merging SFT adapter...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **_base_model_kwargs())
    model = PeftModel.from_pretrained(model, PREV_ADAPTER)
    model = model.merge_and_unload()      # clean CausalLM with the SFT policy folded in
    model.config.use_cache = False

    # FIX 1: TRL expects model.warnings_issued (missing in some transformers versions)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # FIX 2: apply the DPO LoRA ourselves; do NOT pass peft_config to DPOTrainer
    model = get_peft_model(model, LoraConfig(
        r=DPO_LORA_R, lora_alpha=DPO_LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    ))
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds, eval_ds = _split(load_dpo_data(DPO_DATA, tokenizer), DPO_EVAL_SPLIT)

    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRAD_ACCUM,
        learning_rate=DPO_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=DPO_WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        beta=DPO_BETA,
        max_length=DPO_MAX_SEQ_LEN,
        max_prompt_length=DPO_MAX_PROMPT,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50 if eval_ds else None,
        load_best_model_at_end=bool(eval_ds),
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting DPO...")
    trainer.train()

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n{ROLE} DPO adapter saved to: {FINAL_DIR}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()