"""
STAGE 2 — Supervised Fine-Tuning (QLoRA) for Qwen 2.5 3B Instruct
===================================================================
Adapted from the Llama 3 8B SFT script.

Key changes from Llama version:
  - Model: Qwen 2.5 3B Instruct (ChatML template)
  - Response template: <|im_start|>assistant  (for completion-only loss)
  - Pad token: uses <|endoftext|> (Qwen's native EOS)
  - Lower memory footprint (~10-14 GB with QLoRA)
  - Data paths: filtered SFT dataset from the verifier pipeline

Tested with: TRL 0.27.0, transformers, peft, bitsandbytes
"""

import os
import json
import logging
import importlib
import traceback
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Silence deprecation warnings ─────────────────────────────────────────────
os.environ.setdefault("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", ""))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_alloc = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
if _alloc:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", _alloc)

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# ── Version-safe TRL imports ─────────────────────────────────────────────────
def _import_collator():
    for module_path in ("trl", "trl.trainer", "trl.trainer.utils"):
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, "DataCollatorForCompletionOnlyLM")
        except (ImportError, AttributeError):
            continue
    return None

DataCollatorForCompletionOnlyLM = _import_collator()

try:
    from trl import SFTTrainer
except ImportError:
    from trl.trainer import SFTTrainer

try:
    from trl import SFTConfig
    USE_SFT_CONFIG = True
    log.info("Using TRL SFTConfig (TRL >= 0.9)")
except ImportError:
    USE_SFT_CONFIG = False
    log.info("Using TrainingArguments (TRL < 0.9)")

# ---------------------------------------------------------------------------
# CONFIG — Qwen 2.5 3B Instruct
# ---------------------------------------------------------------------------
MODEL_DIR    = "qwen2_5_3b_instruct"
TRAIN_JSONL  = "other_experiments/sft/output/sft_filtered_train.jsonl"
VAL_JSONL    = "other_experiments/sft/output/sft_filtered_val.jsonl"
OUTPUT_DIR   = "other_experiments/sft/sft_qwen25_3b_model"

# LoRA config — same rank works well for 3B; all linear projections targeted
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LEN  = 3500 # 4096 ER
BATCH_SIZE   = 1       # 4, 3B model → can afford larger batch than 8B
GRAD_ACCUM   = 16       #4,  effective batch = 4 * 4 = 16
LR           = 2e-5    # slightly lower LR for 3B (less capacity, easier to destabilise)
EPOCHS       = 3
WARMUP_RATIO = 0.05
SAVE_STEPS   = 50
LOG_STEPS    = 5

# packing=False → DataCollatorForCompletionOnlyLM (loss on assistant turns only)
USE_PACKING  = False

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

# Metadata columns from the verifier pipeline that must be stripped before training
METADATA_COLS = [
    "composite_score", "f1_score", "semantic_sim", "grounding_score",
    "gt_concept_coverage", "judge_overall", "template_rule_flag",
    "template_sim_flag", "generic_phrase_count", "cross_paper_sim",
    "difficulty", "weakness_types", "row_idx", "ground_truth",
]

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Keep only the "messages" field; drop scoring metadata
                if "messages" in rec:
                    records.append({"messages": rec["messages"]})
                else:
                    log.warning(f"Skipping record without 'messages' key in {path}")
    log.info(f"Loaded {len(records)} records from {path}")
    if not records:
        raise ValueError(f"{path} is empty — cannot train/evaluate with 0 samples.")
    return Dataset.from_list(records)

def format_for_qwen(example, tokenizer):
    """
    Apply the Qwen ChatML template to produce the training text.

    Qwen 2.5 ChatML format:
      <|im_start|>system
      You are ...
      <|im_end|>
      <|im_start|>user
      ...
      <|im_end|>
      <|im_start|>assistant
      ...
      <|im_end|>
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Qwen 2.5 was trained with bf16
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,  # Qwen models may need this
    )

    # Qwen 2.5 Instruct has <|endoftext|> as EOS but may not set pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    log.info(f"Tokenizer loaded: vocab_size={tok.vocab_size}, "
             f"eos='{tok.eos_token}' (id={tok.eos_token_id}), "
             f"pad='{tok.pad_token}' (id={tok.pad_token_id})")

    # Verify chat template works
    test_msgs = [{"role": "user", "content": "hello"}]
    test_text = tok.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
    log.info(f"Chat template test: {repr(test_text[:200])}")

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )

    mdl = prepare_model_for_kbit_training(mdl, use_gradient_checkpointing=True)
    return mdl, tok

def apply_lora(model):
    cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading Qwen 2.5 3B Instruct ...")
    model, tokenizer = load_base_model()
    model = apply_lora(model)

    # Debug dtype info
    log.info(f"First parameter dtype: {next(model.parameters()).dtype}")
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    log.info(f"Trainable parameter dtypes: {trainable_dtypes}")

    # ── Load & format datasets ────────────────────────────────────────────────
    def _fmt(ex):
        return format_for_qwen(ex, tokenizer)

    train_raw = load_jsonl(TRAIN_JSONL)
    val_raw = load_jsonl(VAL_JSONL)

    # Remove original columns — keep only the new "text" column
    train_remove = [c for c in train_raw.column_names if c != "text"]
    val_remove = [c for c in val_raw.column_names if c != "text"]

    train_ds = train_raw.map(_fmt, remove_columns=train_remove)
    val_ds = val_raw.map(_fmt, remove_columns=val_remove)

    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    log.info(f"Train columns: {train_ds.column_names}")
    log.info(f"Sample text (first 500 chars):\n{train_ds[0]['text'][:500]}")

    # ── Verify token lengths ──────────────────────────────────────────────────
    sample_tokens = tokenizer(train_ds[0]["text"], return_length=True)
    log.info(f"Sample token count: {sample_tokens['length'][0]}")

    lengths = []
    for i in range(min(50, len(train_ds))):
        toks = tokenizer(train_ds[i]["text"], return_length=True)
        lengths.append(toks["length"][0])
    import numpy as np
    log.info(f"Token length stats (first 50): "
             f"mean={np.mean(lengths):.0f}, max={max(lengths)}, "
             f"min={min(lengths)}, median={np.median(lengths):.0f}")
    if max(lengths) > MAX_SEQ_LEN:
        log.warning(f"Some samples exceed MAX_SEQ_LEN={MAX_SEQ_LEN} — they will be truncated")

    # ── Data collator (ONLY when packing is OFF) ─────────────────────────────
    #
    # Qwen 2.5 ChatML uses:  <|im_start|>assistant\n  as the assistant header
    # We use this as the response template so loss is computed ONLY on
    # assistant completions, not on system/user prompts.
    #
    data_collator = None
    if not USE_PACKING and DataCollatorForCompletionOnlyLM is not None:
        # Qwen ChatML response header
        response_template = "<|im_start|>assistant\n"

        # Tokenize to verify it produces stable token IDs
        response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        log.info(f"Response template: {repr(response_template)}")
        log.info(f"Response template token IDs: {response_token_ids}")

        # Verify the template appears in actual formatted text
        sample_text = train_ds[0]["text"]
        if response_template in sample_text:
            log.info("✓ Response template found in formatted sample text")
        else:
            log.warning(f"✗ Response template NOT found in sample text! "
                        f"Check chat template format. First 800 chars:\n{sample_text[:800]}")
            # Try alternative: token-ID based matching (more robust)
            log.info("Falling back to token-ID based response template matching")

        try:
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_token_ids,
                tokenizer=tokenizer,
            )
            log.info("Using DataCollatorForCompletionOnlyLM (assistant-turn-only loss)")
        except TypeError:
            # Some TRL versions use processing_class instead of tokenizer
            try:
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_token_ids,
                    processing_class=tokenizer,
                )
                log.info("Using DataCollatorForCompletionOnlyLM (processing_class kwarg)")
            except Exception as e:
                log.warning(f"Could not build DataCollatorForCompletionOnlyLM: {e}")
                data_collator = None
    elif USE_PACKING:
        log.info("Packing is ON → skipping DataCollatorForCompletionOnlyLM")

    # ── Build training arguments ──────────────────────────────────────────────
    common_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,

        # Qwen 2.5 is bf16-native; use bf16 if GPU supports it, else fp16
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    if USE_SFT_CONFIG:
        sft_extras = dict(
            dataset_text_field="text",
            packing=USE_PACKING,
        )

        args = None
        for seq_key in ("max_length", "max_seq_length"):
            try:
                args = SFTConfig(**common_kwargs, **sft_extras, **{seq_key: MAX_SEQ_LEN})
                log.info(f"SFTConfig accepted '{seq_key}={MAX_SEQ_LEN}'")
                break
            except TypeError as te:
                log.info(f"SFTConfig rejected '{seq_key}': {te} — trying next ...")
                continue

        if args is None:
            log.warning("SFTConfig rejected all seq-length keys — using default.")
            args = SFTConfig(**common_kwargs, **sft_extras)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            args=args,
        )
    else:
        args = TrainingArguments(**common_kwargs)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            data_collator=data_collator,
            args=args,
            packing=USE_PACKING,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting SFT training (Qwen 2.5 3B) ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    log.info(f"SFT model saved -> {final_path}")

    saved_files = os.listdir(final_path)
    log.info(f"Files in {final_path}: {saved_files}")
    if not saved_files:
        log.error("WARNING: final directory is empty — save may have failed!")
    else:
        log.info(f"Stage 2 complete. {len(saved_files)} files saved.")

    log.info("Run GRPO training next.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        traceback.print_exc()
        raise 
