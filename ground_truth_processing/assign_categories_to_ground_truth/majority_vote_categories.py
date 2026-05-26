# this code is for when gpt, mistral, and llama disagrees, take the combine output from gpt and mistral 
#!/usr/bin/env python3
"""
Compute a majority-vote category for each limitation across Mistral, GPT, and Llama outputs.

The script reads the existing GT CSV, parses the three categorized columns, and
adds a new column where the category for each limitation is chosen by majority
vote across the three model outputs (ties are broken by the earliest occurrence
order). If all three disagree, it combines the categories from GPT and Mistral.
The updated dataframe is written back to disk.
"""

import ast
import os
from collections import Counter
from typing import List, Optional

import pandas as pd

CSV_PATH = "df.csv"
OUTPUT_PATH = CSV_PATH  # update the same CSV with the new column

BASE_COLUMN = "final_gt_author_peer_review"
MODEL_COLUMNS = [
    "final_gt_author_peer_review_cat_mistral",      # Index 0
    "final_gt_author_peer_review_categorized_gpt",    # Index 1
    "final_gt_author_peer_review_categorized_llama",  # Index 2
]
OUTPUT_COLUMN = "final_gt_author_peer_review_categorized_majority"
HUMAN_FLAG = "human verification needed"

def safe_literal_eval(val):
    """Safely parse list-like strings into Python objects."""
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

def pick_majority(categories: List[str]) -> Optional[str]:
    """
    Return the majority category; break ties by earliest appearance.
    If all three model categories disagree, combine GPT and Mistral categories.
    """
    if not categories:
        return None

    # Explicitly handle the case where all three model votes differ (Total Disagreement)
    # MODEL_COLUMNS order is [Mistral, GPT, Llama] -> indices [0, 1, 2]
    if len(categories) == 3 and len(set(categories)) == 3:
        # User requested: GPT, Mistral (concatenated with comma)
        gpt_cat = categories[1]
        mistral_cat = categories[0]
        return f"{gpt_cat}, {mistral_cat}"

    first_seen = {}
    for idx, cat in enumerate(categories):
        first_seen.setdefault(cat, idx)

    counts = Counter(categories)
    winner, _ = min(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return winner

def merge_row(base_limitations: List[dict], categorized_lists: List[List[dict]]) -> List[dict]:
    """
    Merge categories for a single row by majority vote.

    Majority is computed across the same index in each categorized list; the
    base limitations list determines how many items are produced.
    """
    merged: List[dict] = []
    target_len = len(base_limitations)

    for idx in range(target_len):
        candidates: List[str] = []
        for cat_list in categorized_lists:
            if idx < len(cat_list):
                category = cat_list[idx].get("category")
                if category:
                    candidates.append(category)

        winner = pick_majority(candidates)
        entry = dict(base_limitations[idx]) if idx < len(base_limitations) else {}
        if winner is not None:
            entry["category"] = winner
        merged.append(entry)

    return merged

def main():
    print(f"Loading dataframe from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataframe with shape {df.shape}")

    # Parse columns into lists
    df[BASE_COLUMN] = df[BASE_COLUMN].apply(safe_literal_eval)
    for col in MODEL_COLUMNS:
        df[col] = df[col].apply(safe_literal_eval)

    print("Computing majority-vote categories...")
    total_rows = len(df)
    majority_col: List[Optional[List[dict]]] = [None] * total_rows
    mismatched_lengths = 0

    for counter, idx in enumerate(df.index, start=1):
        row = df.loc[idx]
        base_list = row[BASE_COLUMN]
        categorized = [row[col] for col in MODEL_COLUMNS]

        lengths = [len(base_list)] + [len(lst) for lst in categorized if isinstance(lst, list)]
        if len(set(lengths)) > 1:
            mismatched_lengths += 1

        merged = merge_row(base_list, categorized)
        majority_col[idx] = merged

        # Optional: Print progress periodically to reduce clutter
        if counter % 100 == 0:
            print(f"Processed row {counter}/{len(df)}")

    # Fill untouched rows with their base limitations (no category changes)
    for idx in df.index:
        if majority_col[idx] is None:
            majority_col[idx] = df.at[idx, BASE_COLUMN]

    df[OUTPUT_COLUMN] = majority_col
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataframe with majority vote to: {OUTPUT_PATH}")
    if mismatched_lengths:
        print(f"Note: {mismatched_lengths} rows had differing list lengths across inputs.")

if __name__ == "__main__":
    main()

