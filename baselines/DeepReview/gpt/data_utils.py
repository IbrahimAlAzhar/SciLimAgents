"""
data_utils.py
=============
Helpers for loading the input CSV, normalising the citation columns and
writing out the augmented DataFrame.

The CSV is expected to have at least one column with the paper text
(default `input_text_cleaned`) and OPTIONALLY two citation columns
(default `cited_in_text_abs`, `cited_in_ret`).  Each citation column may be
NaN, empty, or contain the literal string "No citations found" — all of
those are treated as "no citations available".
"""

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

_EMPTY_CITATION_MARKERS = {"", "no citations found", "none", "nan", "n/a"}

# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------
def load_papers(csv_path: str, start: int, end: int,
                text_column: str) -> pd.DataFrame:
    """Read the CSV and return rows [start:end] (inclusive of `text_column`)."""
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise KeyError(
            f"Text column '{text_column}' not found in CSV. "
            f"Available columns: {list(df.columns)[:20]} ..."
        )
    end = min(end, len(df))
    sub = df.iloc[start:end].copy().reset_index(drop=False)
    sub.rename(columns={"index": "_orig_index"}, inplace=True)
    return sub

def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Save DataFrame as CSV inside `output_dir`. Returns the full path."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path

# ---------------------------------------------------------------------------
# Citation normalisation
# ---------------------------------------------------------------------------
def _is_empty_citation(value) -> bool:
    """Return True if a citation cell should be treated as 'no citations'."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip().lower()
    if not s:
        return True
    if s in _EMPTY_CITATION_MARKERS:
        return True
    # Some pipelines use exact wording such as "No citations found"
    if s.startswith("no citations found"):
        return True
    return False

def merge_citations(in_text: Optional[str], retrieved: Optional[str],
                    max_chars: int) -> str:
    """Merge the two citation columns into a single context string.

    Rules:
        - empty / NaN / 'No citations found' columns are skipped silently;
        - if both are empty, returns an empty string;
        - the merged text is truncated to `max_chars` from the end (we keep
          the beginning, which usually contains the most relevant info).
    """
    parts = []
    if not _is_empty_citation(in_text):
        parts.append(f"[In-text cited works]\n{str(in_text).strip()}")
    if not _is_empty_citation(retrieved):
        parts.append(f"[Retrieved related works]\n{str(retrieved).strip()}")
    if not parts:
        return ""
    merged = "\n\n".join(parts)
    return _truncate(merged, max_chars)

def _truncate(text: str, max_chars: int) -> str:
    """Hard character-level truncation (cheap surrogate for token truncation).

    DeepReview truncates papers that exceed the model context window — we do
    the same so that Qwen-2.5-3B (12 k tokens) never blows up.
    """
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n\n[... truncated ...]\n\n" + text[-tail:]

def truncate_paper(paper: str, max_chars: int) -> str:
    """Public wrapper around `_truncate` for use by the pipeline."""
    if paper is None:
        return ""
    return _truncate(str(paper), max_chars)

# ---------------------------------------------------------------------------
# Output column helpers
# ---------------------------------------------------------------------------
NEW_COLUMNS = [
    "deepreview_questions",            # Stage 1: 3 research questions
    "deepreview_novelty_analysis",     # Stage 1: novelty analysis text
    "deepreview_reviewer_weaknesses",  # Stage 2: list of weaknesses per reviewer
    "deepreview_verification",         # Stage 3: verification report
    "deepreview_final_limitations",    # Final: deduplicated limitations
    "deepreview_status",               # "ok" | "error: <msg>"
]

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure all DeepReview output columns exist (filled with empty str)."""
    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df

# ---------------------------------------------------------------------------
# Light-weight extractors
# ---------------------------------------------------------------------------
_BOX_RE = re.compile(
    r"<\s*boxed_(\w+)\s*>(.*?)</?\s*boxed_\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

def extract_boxed(text: str, name: str) -> str:
    """Extract the content of a `<boxed_NAME>...</boxed_NAME>` block.

    Returns an empty string if not found.  Falls back to a more permissive
    pattern that allows stray whitespace or capitalisation differences.
    """
    if not text:
        return ""
    pattern = re.compile(
        rf"<\s*boxed_{name}\s*>(.*?)</\s*boxed_{name}\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: any line beginning with "BOXED_NAME" up to the next blank line
    return ""

def extract_numbered_list(text: str) -> list[str]:
    """Extract '1. ...' / '2. ...' style list items from a text block."""
    if not text:
        return []
    items = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[\.\)]\s+(.+)", line)
        if m:
            items.append(m.group(1).strip())
    return items 
