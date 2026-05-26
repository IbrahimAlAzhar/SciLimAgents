"""
DPO Training
============
Loads the SFT-trained LoRA model as the policy and uses it as the reference
model. Trains on (prompt, chosen, rejected) pairs from output/dpo_pair/.

Fix: merge+unload SFT LoRA into base weights first, then add a fresh
     DPO LoRA on top — TRL does not accept PeftModel + peft_config together.
"""

import os
import json
import glob
import logging
import torch

from config import get_config

logger = logging.getLogger(__name__)

DPO_PAIR_DIR  = "other_experiments/dpo_novagents/output/dpo_pair"
SFT_MODEL_DIR = "other_experiments/dpo_novagents/output/sft_model"
DPO_MODEL_DIR = "other_experiments/dpo_novagents/output/dpo_model"

# Intermediate directory: base model + SFT weights merged (full model, not adapter)
MERGED_SFT_DIR = "other_experiments/dpo_novagents/output/sft_model_merged"

def load_pairs():
    files = sorted(glob.glob(os.path.join(DPO_PAIR_DIR, "paper_*_pair.json")))
    logger.info(f"Found {len(files)} DPO pairs in {DPO_PAIR_DIR}")

    records = []
    for fp in files:
        with open(fp) as f:
            p = json.load(f)
        if p.get("prompt") and p.get("chosen") and p.get("rejected"):
            records.append({
                "prompt":   p["prompt"],
                "chosen":   p["chosen"],
                "rejected": p["rejected"],
            })
    logger.info(f"Loaded {len(records)} valid pairs")
    return records

def merge_sft_adapter(config):
    """
    Merge the SFT LoRA adapter into the base model weights and save to disk.
    This produces a full HuggingFace model (with config.json, model_type, etc.)
    that TRL's DPOTrainer can accept alongside a new peft_config.

    Only runs if MERGED_SFT_DIR does not already exist.
    """
    if os.path.exists(os.path.join(MERGED_SFT_DIR, "config.json")):
        logger.info(f"Merged SFT model already exists at {MERGED_SFT_DIR}, skipping merge.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.makedirs(MERGED_SFT_DIR, exist_ok=True)
    logger.info("Merging SFT LoRA adapter into base model weights...")

    base = AutoModelForCausalLM.from_pretrained(
        config.weak_model,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.hf_cache,
        trust_remote_code=True,
    )

    sft = PeftModel.from_pretrained(base, SFT_MODEL_DIR)

    logger.info("Running merge_and_unload()...")
    merged = sft.merge_and_unload()

    logger.info(f"Saving merged model to {MERGED_SFT_DIR} ...")
    merged.save_pretrained(MERGED_SFT_DIR)

    # Save tokenizer alongside so the directory is self-contained
    tokenizer = AutoTokenizer.from_pretrained(
        config.weak_model, cache_dir=config.hf_cache, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_SFT_DIR)

    logger.info("Merge complete.")

    # Free memory before DPO training loads the model again
    del merged, sft, base
    torch.cuda.empty_cache()

def main():
    config = get_config()
    os.makedirs(DPO_MODEL_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.logs_dir, "dpo_training.log")),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 60)
    logger.info("DPO TRAINING")
    logger.info(f"  Base model:         {config.weak_model}")
    logger.info(f"  SFT adapter:        {SFT_MODEL_DIR}")
    logger.info(f"  Merged SFT model:   {MERGED_SFT_DIR}")
    logger.info(f"  DPO output:         {DPO_MODEL_DIR}")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    from datasets import Dataset

    # Step 1: merge SFT LoRA into base weights
    merge_sft_adapter(config)

    # Step 2: load data
    records = load_pairs()
    if not records:
        logger.error("No DPO pairs found.")
        return

    dataset = Dataset.from_list(records).shuffle(seed=42)
    split = (dataset.train_test_split(test_size=0.1, seed=42)
             if len(dataset) >= 10
             else {"train": dataset, "test": dataset})
    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Step 3: load tokenizer from merged model dir (has full config)
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_SFT_DIR, cache_dir=config.hf_cache, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 4: load merged model as the policy base
    logger.info("Loading merged SFT model as DPO policy base...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_SFT_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.hf_cache,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Step 5: fresh LoRA config for DPO (no pre-existing adapter conflict)
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Step 6: DPO training args
    dpo_args = DPOConfig(
        output_dir=os.path.join(config.checkpoints_dir, "dpo"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        beta=config.dpo_beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    # ref_model=None: TRL uses the merged model with adapter disabled as reference.
    # This is the correct pattern when policy is a plain model + peft_config.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    trainer.save_model(DPO_MODEL_DIR)
    tokenizer.save_pretrained(DPO_MODEL_DIR)
    logger.info(f"Saved DPO model to {DPO_MODEL_DIR}")

    metrics_path = os.path.join(config.eval_dir, "dpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info("DPO TRAINING COMPLETE")

if __name__ == "__main__":
    main()