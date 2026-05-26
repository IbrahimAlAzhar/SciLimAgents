import os
import re
import ast
import pandas as pd
from tqdm import tqdm

# =========================
# 1. Paths and basic setup
# =========================
INPUT_CSV = "df.csv"
OUTPUT_PATH = "df1.csv"
df = pd.read_csv(INPUT_CSV)

# =========================
# 2. Parsing helper functions
# =========================

def parse_gt_limitations(text_str):
    """
    Parse 'final_lim_gt_author_peer_cat_maj_hum_cleaned' which is a stringified list of dicts.
    Extract 'limitation' field and attach gt_id.
    Returns list of dicts: [{'gt_id': 0, 'gt_limitation': '...'}, ...]
    """
    try:
        data = ast.literal_eval(text_str)
        texts = [item.get('limitation') for item in data if isinstance(item, dict)]
        return [
            {'gt_id': i, 'gt_limitation': t}
            for i, t in enumerate(texts)
            if t is not None and str(t).strip()
        ]
    except (ValueError, SyntaxError, TypeError):
        return []

def parse_llama_master_dpo(text_str):
    """
    Parse 'llama_master_dpo' text into a list of numbered limitation blocks.
    Splits on '\n1', '\n2', '\n3', ... (with optional '.' and whitespace),
    and drops the header before the first number.

    Example input fragment:
        "Routing Problems\\n\\nIntroduction: ...\\n\\n1. ...\\n\\n2. ...\\n\\n3. ..."

    Returns list of dicts: [{'llm_id': 0, 'llm_limitation': '...'}, ...]
    """
    if not isinstance(text_str, str):
        return []
    
    # Split on newline + digits (+ optional dot + optional spaces)
    pattern = r'\n\d+\.?\s*'
    parts = re.split(pattern, text_str)

    # parts[0] = header ("Routing Problems...\n\nIntroduction: ...")
    # parts[1:], parts[2], ... = numbered limitations
    if len(parts) > 1:
        limitations = [p.strip() for p in parts[1:] if p.strip()]
        return [
            {'llm_id': j, 'llm_limitation': txt}
            for j, txt in enumerate(limitations)
        ]
    else:
        return []

# =========================
# 3a. Local heuristic evaluator (no external APIs)
# =========================
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def _normalize_words(s: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(str(s)) if len(w) >= 3}

def _yes_no_overlap(gt_text: str, llm_text: str, threshold: float = 0.2) -> str:
    """
    Lightweight, API-free proxy for "does one contain the other's topic".
    Returns "Yes" if Jaccard overlap over content words exceeds threshold.
    """
    a = _normalize_words(gt_text)
    b = _normalize_words(llm_text)
    if not a or not b:
        return "No"
    j = len(a & b) / len(a | b)
    return "Yes" if j >= threshold else "No"

# =========================
# 4. Apply parsing to dataframe
# =========================

df['gt_limitations_list'] = df['final_lim_gt_author_peer_cat_maj_hum_cleaned'].apply(parse_gt_limitations)
df['llama_master_dpo_list'] = df['llama_master_dpo'].apply(parse_llama_master_dpo)

# =========================
# 4. Create all GT × LLM pairs with IDs
# =========================

def build_pairs(row):
    pairs = []
    gt_list = row['gt_limitations_list']
    llm_list = row['llama_master_dpo_list']

    for gt in gt_list:
        for llm in llm_list:
            pairs.append({
                'gt_id': gt['gt_id'],
                'gt_limitation': gt['gt_limitation'],
                'llm_id': llm['llm_id'],
                'llm_limitation': llm['llm_limitation'],
            })
    return pairs

df['paired_limitations'] = df.apply(build_pairs, axis=1)

# (Optional sanity check for first row)
first_idx = df.index[0]
print("Sanity check on row 0:")
print(f"  GT count:  {len(df.loc[first_idx, 'gt_limitations_list'])}")
print(f"  LLM count: {len(df.loc[first_idx, 'llama_master_dpo_list'])}")
print(f"  Pair count: {len(df.loc[first_idx, 'paired_limitations'])}")

# =========================
# 5. LLM evaluation per pair
# =========================

def evaluate_pairs_with_llm(pairs_list):
    """
    Takes a list of pair dicts:
      {'gt_id', 'gt_limitation', 'llm_id', 'llm_limitation'}
    Evaluates each pair locally (no GPT/Gemini/OpenAI calls) and returns a list like:
      [
        ['Pair 1: Yes', 'gt_id:0', 'gt_limitation: ...', 'llm_id:0', 'llm_limitation: ...'],
        ...
      ]
    """
    results = []
    
    if not isinstance(pairs_list, list):
        return []

    for i, pair in enumerate(pairs_list):
        gt_text = pair['gt_limitation']
        llm_text = pair['llm_limitation']
        answer = _yes_no_overlap(gt_text, llm_text)

        result_entry = [
            f"Pair {i+1}: {answer}",
            f"gt_id:{pair['gt_id']}",
            f"gt_limitation:{gt_text}",
            f"llm_id:{pair['llm_id']}",
            f"llm_limitation:{llm_text}",
        ]
        results.append(result_entry)
        
    return results

# =========================
# 6. Apply LLM evaluation row by row
# =========================

df['llm_evaluation_results'] = None

print(f"Starting local evaluation on {len(df)} rows...")

for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing Rows")):
    print("i is",i)
    pairs = row['paired_limitations']
    row_results = evaluate_pairs_with_llm(pairs)
    df.at[index, 'llm_evaluation_results'] = row_results

    if (i + 1) % 10 == 0:
        df.to_csv(OUTPUT_PATH, index=False)

# Final save
df.to_csv(OUTPUT_PATH, index=False)
print("✅ Processing Complete. Final results saved.")
