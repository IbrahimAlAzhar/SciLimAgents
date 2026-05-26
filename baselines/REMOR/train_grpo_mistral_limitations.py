# wrote in chatgpt in paper 
import os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import nltk

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from nltk.translate.meteor_score import meteor_score

# -----------------------
# NLTK setup
# -----------------------
def ensure_nltk():
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

# -----------------------
# Seeds
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------
# Split & Helpers
# -----------------------
def fixed_index_split_df(df: pd.DataFrame):
    n = len(df)
    train_idx = list(range(0, min(101, n))) + list(range(201, min(301, n)))
    test_idx = list(range(101, min(201, n)))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def _safe_str(x) -> str:
    return str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else ""

def strip_think(text: str) -> str:
    parts = _safe_str(text).split("</think>")
    return parts[1].strip() if len(parts) > 1 else parts[0].strip()

# -----------------------
# Prompt template
# -----------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Write the LIMITATIONS section for the paper content. "
    "Use bullet points. Do not invent results."
)

USER_TEMPLATE = "Write the LIMITATIONS section for the following paper.\n\nPaper:\n{paper_text}"

def build_prompt(tokenizer, paper_text, system_prompt, max_prompt_length):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(paper_text=_safe_str(paper_text))},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_length:
        prompt_str = tokenizer.decode(ids[:max_prompt_length], skip_special_tokens=False)
    return prompt_str

# -----------------------
# Rewards
# -----------------------
def reward_meteor(completions, **kwargs):
    refs = kwargs["full_text"]
    return [float(meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(strip_think(c)))) for c, r in zip(completions, refs)]

def _lcs_length(x, y):
    n, m = len(x), len(y)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j-1] + 1 if x[i-1] == y[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return prev[m]

def reward_rouge_l(completions, **kwargs):
    results = []
    for c, r in zip(completions, kwargs["full_text"]):
        cand, ref = nltk.word_tokenize(strip_think(c)), nltk.word_tokenize(_safe_str(r))
        lcs = _lcs_length(cand, ref)
        if lcs == 0: results.append(0.0); continue
        p, r = lcs/len(cand), lcs/len(ref)
        results.append(float(2*p*r/(p+r)) if (p+r) > 0 else 0.0)
    return results

# -----------------------
# Main
# -----------------------
def main():
    ensure_nltk()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    args = parser.parse_args()
    
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model with 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        # IMPORTANT: device_map must be handled carefully in DDP
        device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
    )

    # 3. Prepare for PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 4. Dataset Prep
    df = pd.read_csv(args.input_csv)
    train_df, _ = fixed_index_split_df(df)
    prompts = [build_prompt(tokenizer, t, DEFAULT_SYSTEM_PROMPT, args.max_prompt_length) for t in train_df["input_text_cleaned"]]
    train_dataset = Dataset.from_pandas(pd.DataFrame({"prompt": prompts, "full_text": train_df["ground_truth_lim_peer"]}))

    # 5. GRPO Config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        remove_unused_columns=False
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_meteor, reward_rouge_l],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()