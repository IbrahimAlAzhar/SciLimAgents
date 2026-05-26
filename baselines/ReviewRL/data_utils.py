# =============================================================================
# data_utils.py
# -----------------------------------------------------------------------------
# CSV input/output helpers + incremental checkpointing.
#
# The user wants:
#   * to read N rows from a CSV (start, end indices)
#   * three columns: paper text, in-text citations, retrieved citations
#   * to save the resulting dataframe to a NEW CSV every K rows
#     (so that even if the cluster job dies mid-way nothing is lost)
#
# These helpers keep main.py readable.
# =============================================================================

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

def load_input_csv(
    input_csv: str,
    text_column: str,
    cited_in_text_column: Optional[str],
    cited_in_ret_column: Optional[str],
    start: int = 0,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the user's inference CSV and slice to [start:end].

    Required column: `text_column` (the paper body, e.g. `input_text_cleaned`).
    Optional columns: `cited_in_text_column`, `cited_in_ret_column`
                     (created with empty strings if missing, so the rest
                      of the pipeline never crashes on a KeyError).
    """
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise KeyError(
            f"text_column='{text_column}' not in CSV columns: {list(df.columns)[:20]}"
        )

    # Make sure the citation columns exist even if the user passed --no-citations.
    for col in (cited_in_text_column, cited_in_ret_column):
        if col and col not in df.columns:
            df[col] = ""

    if end is None:
        end = len(df)
    end = min(end, len(df))
    df = df.iloc[start:end].reset_index(drop=True).copy()
    return df

def save_checkpoint(
    df: pd.DataFrame,
    output_dir: str,
    output_name: str,
    overwrite: bool = True,
) -> str:
    """
    Save the dataframe to <output_dir>/<output_name>.csv and return the path.
    Creates the directory if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    if not overwrite and os.path.exists(out_path):
        # Avoid clobbering — append a numeric suffix.
        base, ext = os.path.splitext(out_path)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        out_path = f"{base}_{i}{ext}"
    df.to_csv(out_path, index=False)
    return out_path

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure every "agent / final" output column exists in `df` so we can
    safely write into it during streaming inference. We add:

        * reviewrl_queries          - the 3 generated queries (Section 3.2 step 1)
        * reviewrl_context          - the assembled retrieval context (Section 3.2 step 2)
        * reviewrl_raw_response     - full model output (with <think> block)
        * reviewrl_thinking         - extracted <think>...</think> trace
        * reviewrl_limitations      - parsed "## Limitations" section ONLY
        * reviewrl_format_reward    - format_reward dict serialized to JSON
        * reviewrl_total_reward     - scalar total reward
    """
    new_cols = [
        "reviewrl_queries",
        "reviewrl_context",
        "reviewrl_raw_response",
        "reviewrl_thinking",
        "reviewrl_limitations",
        "reviewrl_format_reward",
        "reviewrl_total_reward",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = ""
    return df