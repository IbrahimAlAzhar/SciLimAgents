"""
Merge limitation reports from 'df_lim' and 'df_nov' by matching 'id'.
Filters out "SKIPPED_SHORT_TEXT" and merges the reports using GPT-4o-mini.
Report A (df_lim) is strictly preserved as the anchor.
"""

import os
import time
import pandas as pd
import numpy as np
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

# ── Configuration ────────────────────────────────────────────────────
DF_LIM_PATH = "llm_agents/mistral_new/limagents/output/df_mistral_7b_v3_7_agents_novelty_output.csv"
DF_NOV_PATH = "llm_agents/mistral_new/novagents/df_mistral_7b_novagents_output.csv"

# Output Directory and Files
OUTPUT_DIR = "llm_agents/mistral_new/limagents+novagents"
CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, "checkpoint_merged_not_more_than_14.csv")
FINAL_CSV = os.path.join(OUTPUT_DIR, "df_lim_nov_merged_final_upd_not_more_than_14.csv")
CHECKPOINT_EVERY = 10

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds

# ── OpenAI client ────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Prompts ──────────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = """\
You are an expert scientific peer-review analyst. You receive two limitation \
reports written about the SAME research paper:
- Report A comes from the limitation agents (primary source).
- Report B comes from the novelty agents (secondary source).

Your job:
1. Produce a CONSOLIDATED list of between 10 and 14 limitations total \
2. Report A is the anchor. Use its limitations as the backbone of the final \
list, preserving their wording as much as possible (light edits for clarity \
are allowed).
3. Merge similar limitations:
   a. WITHIN Report A: if two or more Report A limitations cover the same \
underlying issue (e.g., several points about evaluation, several about \
dataset, several about scalability), merge them into a single richer \
limitation that retains all unique details.
   b. ACROSS reports: for each limitation in Report B, if it is similar, \
related, or overlapping with a limitation already in the consolidated list \
— even if worded differently — MERGE its unique details into that \
limitation, enriching it without losing information.
   c. If a Report B limitation has NO counterpart in the consolidated list, \
KEEP it as a separate, standalone limitation point (lightly edited for \
clarity).
4. Preservation rule: do not drop any substantive content from either report \
and do not invent new limitations. Every unique issue raised in A or B must \
appear somewhere in the final list, either as its own point or merged into a \
related point.
5. If after merging you have more than 14 points, continue grouping the most \
closely related ones until you are within the 10-14 range. If you have fewer \
than 10, split any overly broad merged point back into its distinct \
sub-issues until you reach at least 10.
6. Format each limitation as:
   **Short Descriptive Title**: 1-3 sentence explanation.
7. Output ONLY the list of limitations. No preamble, no commentary, no \
conclusion, no category headings, no numbering.
"""

def get_merge_user_prompt(lim_text: str, nov_text: str) -> str:
    return f"""\
### Report A — Limitation Agents (primary, must be kept):
{lim_text}

### Report B — Novelty Agents (merge into A when similar, otherwise keep separate):
{nov_text}

Produce the consolidated limitation report following the instructions provided.
"""

# ── Helper functions ─────────────────────────────────────────────────
def is_valid(text) -> bool:
    """Return True if text is a non-empty, non-NaN string."""
    if text is None:
        return False
    if isinstance(text, float) and np.isnan(text):
        return False
    if not isinstance(text, str):
        return False
    return len(text.strip()) > 0

def call_openai_with_retry(system_prompt: str, user_prompt: str) -> str:
    """Call GPT-4o-mini with exponential-backoff retry on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"  ⚠ Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). "
                  f"Retrying in {wait}s …")
            time.sleep(wait)

        except Exception as e:
            print(f"  ✗ Non-retryable error: {e}")
            return f"[ERROR] {e}"

    return "[ERROR] Max retries exceeded"

# ── Main pipeline ────────────────────────────────────────────────────
def main():
    print("Loading CSVs …")
    df_lim = pd.read_csv(DF_LIM_PATH)
    df_nov = pd.read_csv(DF_NOV_PATH)

    # Ensure IDs exist
    if 'id' not in df_lim.columns or 'id' not in df_nov.columns:
        raise ValueError("Both dataframes must contain an 'id' column.")

    print(f"Initial lengths - df_lim: {len(df_lim)}, df_nov: {len(df_nov)}")

    # 1. Identify IDs with "SKIPPED_SHORT_TEXT"
    lim_skipped_mask = df_lim['final_merged_limitations'].astype(str).str.contains("SKIPPED_SHORT_TEXT", na=False)
    nov_skipped_mask = df_nov['novelty_report'].astype(str).str.contains("SKIPPED_SHORT_TEXT", na=False)

    lim_skipped_ids = set(df_lim[lim_skipped_mask]['id'])
    nov_skipped_ids = set(df_nov[nov_skipped_mask]['id'])
    all_skipped_ids = lim_skipped_ids.union(nov_skipped_ids)

    # 2. Find common IDs that should NOT be skipped
    common_ids = set(df_lim['id']).intersection(set(df_nov['id']))
    valid_ids = common_ids - all_skipped_ids
    print(f"Found {len(valid_ids)} valid matching IDs to process (Skipped {len(all_skipped_ids)} IDs).")

    # 3. Load from checkpoint if available, otherwise create fresh copy
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_CSV):
        print("Resuming from checkpoint …")
        df_lim_copy = pd.read_csv(CHECKPOINT_CSV)
    else:
        df_lim_copy = df_lim.copy()

    if "novelty_report" not in df_lim_copy.columns:
        df_lim_copy["novelty_report"] = pd.Series(dtype='object')
    if "merged_report" not in df_lim_copy.columns:
        df_lim_copy["merged_report"] = pd.Series(dtype='object')

    # Convert df_nov into a dictionary for O(1) lookup by ID
    nov_dict = df_nov.set_index('id')['novelty_report'].to_dict()

    skipped = 0
    processed = 0

    for idx, row in df_lim_copy.iterrows():
        row_id = row['id']

        # Skip logic
        if row_id not in valid_ids:
            skipped += 1
            continue

        lim_master_text = row['final_merged_limitations']
        nov_text = nov_dict.get(row_id)

        if not is_valid(lim_master_text) or not is_valid(nov_text):
            skipped += 1
            continue

        # Skip if already processed in a previous run
        if is_valid(df_lim_copy.at[idx, "merged_report"]):
            skipped += 1
            continue

        print(f"[{idx + 1}/{len(df_lim_copy)}] Processing ID {row_id} …")

        # ── Call: Merge the limitations from df_lim and df_nov ──
        merge_user_msg = get_merge_user_prompt(lim_master_text, nov_text)
        merged_result = call_openai_with_retry(MERGE_SYSTEM_PROMPT, merge_user_msg)

        # Save results into df_lim_copy
        df_lim_copy.at[idx, "merged_report"] = merged_result
        df_lim_copy.at[idx, "novelty_report"] = nov_text  # Saving the nov column

        processed += 1

        # Checkpoint save
        if processed % CHECKPOINT_EVERY == 0:
            df_lim_copy.to_csv(CHECKPOINT_CSV, index=False)
            print(f"  -> Checkpoint saved.")

    # Final save
    df_lim_copy.to_csv(FINAL_CSV, index=False)

    print(f"\nDone. {processed} rows processed, {skipped} skipped, {len(df_lim_copy)} total rows in file.")
    print(f"Saved Final Output → {FINAL_CSV}")

if __name__ == "__main__":
    main()