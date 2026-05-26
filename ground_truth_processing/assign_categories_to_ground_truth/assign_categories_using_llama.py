#!/usr/bin/env python3
"""
Assign handcrafted limitation categories to each GT entry using Llama 3 8B.

This script mirrors the GPT-based categorization pipeline but routes prompts
through a locally hosted Meta-Llama-3-8B-Instruct model (4-bit quantized).
It reads the balanced GT dataframe, categorizes every limitation list, and
stores the augmented data back into the same CSV under a new column.
"""

import ast
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model configuration (mirrors limitation_generation_3_agents.py)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "llama3_8b"
MAX_CONTEXT_TOKENS = 8000  # leave headroom for generation
MAX_NEW_TOKENS = 512

# --- MODIFIED: Set to None to process ALL rows ---
ROW_LIMIT = None 

# File paths
CSV_PATH = "df.csv"
OUTPUT_CSV = "df1.csv"
OUTPUT_COLUMN = "final_gt_author_peer_review_categorized_llama"

CATEGORIES = [
    "Novelty and Significance",
    "Experimental Validation and Rigor",
    "Comparisons and Baselines",
    "Methodological and Theoretical Soundness",
    "Generalization and Robustness",
    "Clarity, Presentation, and Definitions",
    "Computational Efficiency and Resources",
    "Evaluation Metrics and Analysis",
    "Ablation and Component Analysis",
    "Bias, Fairness, and Ethics",
    "Real-World Relevance",
    "Reproducibility and Open Science",
    "Interpretability",
    "Data Integrity",
]

def safe_literal_eval(val):
    """Safely parse list-like strings into Python objects."""
    try:
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, list):
            return val
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

def build_prompt(limitations: List[Dict[str, Any]]) -> str:
    """Build prompt instructing Llama to tag each limitation with a category."""
    categories_text = "\n".join(f"- {cat}" for cat in CATEGORIES)
    limitations_text = json.dumps(limitations, ensure_ascii=False, indent=2)
    return f"""
You are an expert reviewer assigning limitation categories to research paper critiques.

Use ONLY the following categories (pick the single best match for each limitation):
{categories_text}

For each limitation entry below, choose the category that best fits the limitation content.
Rules:
- Keep the output order identical to the input list.
- Do not invent new categories.
- If unsure, choose the closest category.

Input limitations:
{limitations_text}

Return ONLY a JSON array where each item has:
- "title": original title
- "limitation": original limitation text
- "category": one of the allowed categories
"""

# --- Llama model helpers ----------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

print("Loading Llama 3 8B model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def truncate_prompt_for_model(prompt: str, max_length: int = MAX_CONTEXT_TOKENS) -> str:
    """Truncate prompt tokens to fit the model context window."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return prompt
    print(f"⚠️ Prompt tokens ({len(token_ids)}) exceed limit ({max_length}). Truncating...")
    truncated_ids = token_ids[:max_length]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

def llama_generate(prompt: str) -> str:
    """Generate text using Llama 3 8B with sampling."""
    truncated_prompt = truncate_prompt_for_model(prompt, MAX_CONTEXT_TOKENS)
    inputs = tokenizer(
        truncated_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
        return_attention_mask=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# --- Response parsing -------------------------------------------------------

def parse_json_response(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract JSON array from model response."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    try:
        json_start = cleaned.find("[")
        json_end = cleaned.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned[json_start:json_end]
            return json.loads(json_str)
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"⚠️ Failed to parse JSON: {exc}")
        return None

def assign_categories_to_row(limitations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign categories to a single row's limitations using Llama."""
    if not limitations:
        return limitations

    prompt = build_prompt(limitations)
    response = llama_generate(prompt)
    parsed = parse_json_response(response)

    if not parsed or len(parsed) != len(limitations):
        print("⚠️ Model response invalid or mismatched length. Falling back to default category.")
        categorized = [dict(item) for item in limitations]
        for item in categorized:
            item["category"] = "Methodological and Theoretical Soundness"
        return categorized

    updated = []
    for original, enriched in zip(limitations, parsed):
        category = enriched.get("category", "").strip()
        if category not in CATEGORIES:
            category = "Methodological and Theoretical Soundness"
        updated_item = dict(original)
        updated_item["category"] = category
        updated.append(updated_item)
    return updated

def main():
    print(f"Loading dataframe from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataframe loaded. Shape: {df.shape}")

    df["final_gt_author_peer_review"] = df["final_gt_author_peer_review"].apply(safe_literal_eval)

    total_rows = len(df)
    categorized_col = [None] * total_rows

    # Initialize with originals to preserve structure when rows are empty
    for idx in range(total_rows):
        categorized_col[idx] = df.at[idx, "final_gt_author_peer_review"]

    target_indices = df.index
    if ROW_LIMIT is not None:
        target_indices = target_indices[:ROW_LIMIT]

    total_targets = len(target_indices)
    start_time = time.time()
    for counter, idx in enumerate(target_indices, start=1):
        limitations = df.at[idx, "final_gt_author_peer_review"]
        print(f"\n=== Processing row {counter}/{total_targets} (df idx {idx}) ===")
        print(f"Number of limitations: {len(limitations)}")

        categorized = assign_categories_to_row(limitations)
        categorized_col[idx] = categorized

        if counter % 10 == 0:
            df[OUTPUT_COLUMN] = categorized_col
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n💾 Saved progress after {counter} rows to: {OUTPUT_CSV}")

    df[OUTPUT_COLUMN] = categorized_col
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - start_time
    print(f"\n✅ Final save: Categorized dataframe saved to: {OUTPUT_CSV}")
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()