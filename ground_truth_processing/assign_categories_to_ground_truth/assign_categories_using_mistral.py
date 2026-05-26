#!/usr/bin/env python3
"""
Assign handcrafted limitation categories to each GT entry using the local Mistral model.

For every row in the dataframe:
- Parse `final_gt_author_peer_review` (list of dicts with "title" and "limitation").
- Ask Mistral to select the most appropriate category for each limitation from the
  predefined category list.
- Write the updated list (with an added "category" key) into a new dataframe column.
- Save the augmented dataframe for downstream processing.
"""

import ast
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"
MAX_CONTEXT_TOKENS = 32000

CSV_PATH = "df.csv"
OUTPUT_CSV = "df1.csv"

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

def load_model():
    """Load tokenizer and model once."""
    print("Loading Mistral model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

def truncate_prompt(tokenizer, prompt: str, max_length: int = MAX_CONTEXT_TOKENS) -> str:
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(f"⚠️ Prompt tokens ({tokens.shape[1]}) exceed limit ({max_length}). Truncating...")
        truncated_tokens = tokens[:, :max_length]
        return tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt

def mistral_generate(tokenizer, model, prompt: str, max_new_tokens: int = 1024) -> str:
    truncated_prompt = truncate_prompt(tokenizer, prompt, MAX_CONTEXT_TOKENS)
    messages = [{"role": "user", "content": truncated_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()

def safe_literal_eval(val):
    try:
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, list):
            return val
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

def build_prompt(limitations: List[Dict[str, Any]]) -> str:
    categories_text = "\n".join(f"- {cat}" for cat in CATEGORIES)
    limitations_text = json.dumps(limitations, ensure_ascii=False, indent=2)
    prompt = f"""
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
    return prompt

def parse_json_response(raw_text: str) -> Optional[List[Dict[str, Any]]]:
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

def assign_categories_to_row(limitations: List[Dict[str, Any]], tokenizer, model) -> List[Dict[str, Any]]:
    if not limitations:
        return limitations

    prompt = build_prompt(limitations)
    response = mistral_generate(tokenizer, model, prompt, max_new_tokens=1024)
    parsed = parse_json_response(response)

    if not parsed or len(parsed) != len(limitations):
        print("⚠️ Model response invalid or mismatched length. Falling back to default category 'Methodological and Theoretical Soundness'.")
        for item in limitations:
            item["category"] = "Methodological and Theoretical Soundness"
        return limitations

    # Merge categories back into original structures by position.
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
    tokenizer, model = load_model()

    print(f"Loading dataframe from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataframe loaded. Shape: {df.shape}")

    df["final_gt_author_peer_review"] = df["final_gt_author_peer_review"].apply(safe_literal_eval)

    total_rows = len(df)
    target_indices = df.index  # Process all rows
    categorized_col = [None] * total_rows

    start_time = time.time()
    for counter, idx in enumerate(target_indices, start=1):
        limitations = df.at[idx, "final_gt_author_peer_review"] 
        if idx % 50 == 0:
            print(f"Processing row {idx+1}/{total_rows} ...")
        categorized = assign_categories_to_row(limitations, tokenizer, model)
        categorized_col[idx] = categorized

        # Print sample output for the first processed row (testing view).
        if counter == 1:
            print("\n=== Sample categorized row ===")
            print(json.dumps(categorized, ensure_ascii=False, indent=2))

    # Fill any unprocessed rows with original limitations to avoid None entries.
    for idx in range(total_rows):
        if categorized_col[idx] is None:
            categorized_col[idx] = df.at[idx, "final_gt_author_peer_review"]

    df["final_gt_author_peer_review_categorized"] = categorized_col
   
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved categorized dataframe to: {OUTPUT_CSV}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

