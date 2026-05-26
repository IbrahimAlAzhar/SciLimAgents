#!/usr/bin/env python3
"""
Assign handcrafted limitation categories to each GT entry using GPT-4o-mini.

For every row in the dataframe:
- Parse `final_gt_author_peer_review` (list of dicts with "title" and "limitation").
- Ask GPT to select the most appropriate category for each limitation from the
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
import tiktoken
from openai import OpenAI

# OpenAI client setup
os.environ['OPENAI_API_KEY'] = ''
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 120000

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

def truncate_text_to_token_limit(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Truncate text to stay within token limit."""
    try:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    tokens = encoding.encode(text)
    token_count = len(tokens)

    if token_count <= max_tokens:
        return text

    print(f"⚠️ Prompt tokens ({token_count}) exceed limit ({max_tokens}). Truncating...")
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text

def gpt_generate(prompt: str) -> str:
    """Generate text using GPT-4o-mini with streaming."""
    truncated_prompt = truncate_text_to_token_limit(prompt, MAX_CONTEXT_TOKENS)
    
    summary_text = ""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": truncated_prompt}],
        stream=True,
        temperature=0
    )
    for chunk in stream:
        summary_text += chunk.choices[0].delta.content or ""
    return summary_text.strip()

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
    """Build prompt for GPT to assign categories."""
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
    """Assign categories to a single row's limitations using GPT."""
    if not limitations:
        return limitations

    prompt = build_prompt(limitations)
    response = gpt_generate(prompt)
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
    print(f"Loading dataframe from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataframe loaded. Shape: {df.shape}")

    # Parse the final_gt_author_peer_review column using ast
    df["final_gt_author_peer_review"] = df["final_gt_author_peer_review"].apply(safe_literal_eval)

    total_rows = len(df)
    # Process all rows
    target_indices = df.index
    categorized_col = [None] * total_rows

    # Initialize the new column with original data
    for idx in range(total_rows):
        categorized_col[idx] = df.at[idx, "final_gt_author_peer_review"]

    start_time = time.time()
    for counter, idx in enumerate(target_indices, start=1):
        limitations = df.at[idx, "final_gt_author_peer_review"]
        print(f"\n=== Processing row {counter}/{total_rows} ===")
        print(f"Number of limitations: {len(limitations)}")
        
        # Get GPT response and print it
        if limitations:
            prompt = build_prompt(limitations)
            response = gpt_generate(prompt)
            print(f"\n=== GPT Response for row {counter} ===")
            print(response)
            
            # Parse and assign categories
            parsed = parse_json_response(response)
            if not parsed or len(parsed) != len(limitations):
                print(f"⚠️ Row {counter}: Model response invalid or mismatched length. Falling back to default category.")
                categorized = [dict(item) for item in limitations]  # Deep copy
                for item in categorized:
                    item["category"] = "Methodological and Theoretical Soundness"
            else:
                updated = []
                for original, enriched in zip(limitations, parsed):
                    category = enriched.get("category", "").strip()
                    if category not in CATEGORIES:
                        category = "Methodological and Theoretical Soundness"
                    updated_item = dict(original)
                    updated_item["category"] = category
                    updated.append(updated_item)
                categorized = updated
        else:
            categorized = limitations
        
        categorized_col[idx] = categorized
        
        if counter % 10 == 0:
            df["final_gt_author_peer_review_categorized_gpt"] = categorized_col
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n💾 Saved progress after {counter} rows to: {OUTPUT_CSV}")

    # Final save with all rows
    df["final_gt_author_peer_review_categorized_gpt"] = categorized_col
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Final save: Categorized dataframe saved to: {OUTPUT_CSV}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

