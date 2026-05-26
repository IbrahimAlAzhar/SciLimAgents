"""
SFT Training
============
Trains Llama-3-8B-Instruct on the BEST rollouts (one per paper) from the
sft_data/ folder. Each best rollout becomes a (prompt, response) pair where:
  - prompt   = the claim_extraction prompt (the input to the agent)
  - response = the concatenated raw_output of all 4 steps

Uses LoRA for memory efficiency. Saves to output/sft_model.
"""

import os
import json
import glob
import logging
import torch

from config import get_config

logger = logging.getLogger(__name__)

SFT_DATA_DIR = "other_experiments/dpo_novagents/output/sft_data"
SFT_MODEL_DIR = "other_experiments/dpo_novagents_llama_mistral/llama/sft_model"

def build_prompt(traj):
    return traj.get("steps", {}).get("claim_extraction", {}).get("prompt", "")

def build_response(traj):
    parts = []
    for s in ["claim_extraction", "novelty_technical",
              "experimental_scope", "limitation_synthesis"]:
        out = traj.get("steps", {}).get(s, {}).get("raw_output", "")
        if out:
            parts.append(f"=== {s} ===\n{out}")
    return "\n\n".join(parts)

def load_sft_records():
    files = sorted(glob.glob(os.path.join(SFT_DATA_DIR, "paper_*_best.json")))
    logger.info(f"Found {len(files)} best rollouts in {SFT_DATA_DIR}")

    records = []
    for fp in files:
        with open(fp) as f:
            traj = json.load(f)
        prompt = build_prompt(traj)
        response = build_response(traj)
        if prompt and response:
            records.append({"prompt": prompt, "response": response,
                            "paper_id": traj.get("paper_id", "")})
    logger.info(f"Built {len(records)} SFT records")
    return records

def format_chat(tokenizer, prompt, response):
    """Format as model chat template; return full text + prompt-only length."""
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    full = tokenizer.apply_chat_template(msgs, tokenize=False)

    msgs_p = [{"role": "user", "content": prompt}]
    prompt_only = tokenizer.apply_chat_template(
        msgs_p, tokenize=False, add_generation_prompt=True)
    return full, prompt_only

def main():
    config = get_config()
    os.makedirs(SFT_MODEL_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.logs_dir, "sft_training.log")),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 60)
    logger.info("SFT TRAINING")
    logger.info(f"  Base model: {config.weak_model}")
    logger.info(f"  Output:     {SFT_MODEL_DIR}")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    records = load_sft_records()
    if not records:
        logger.error("No SFT data found. Run scoring pipeline first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        config.weak_model, cache_dir=config.hf_cache, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = []
    for r in records:
        full, _ = format_chat(tokenizer, r["prompt"], r["response"])
        formatted.append({"text": full})

    dataset = Dataset.from_list(formatted).shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42) \
        if len(dataset) >= 10 else {"train": dataset, "test": dataset}
    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    model = AutoModelForCausalLM.from_pretrained(
        config.weak_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.hf_cache,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=os.path.join(config.checkpoints_dir, "sft"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=2e-5,
        num_train_epochs=config.num_epochs,
        max_length=config.max_length,
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
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    trainer.save_model(SFT_MODEL_DIR)
    tokenizer.save_pretrained(SFT_MODEL_DIR)
    logger.info(f"Saved SFT model to {SFT_MODEL_DIR}")

    metrics_path = os.path.join(config.eval_dir, "sft_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info("SFT TRAINING COMPLETE")

if __name__ == "__main__":
    main()
