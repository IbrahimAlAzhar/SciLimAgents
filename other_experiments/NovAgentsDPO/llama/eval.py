import pandas as pd 
df = pd.read_csv("other_experiments/dpo_novagents/llama/df_final_extracted_limitations.csv") 

import re
import pandas as pd 
import re 
import ast
import os
import time
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError

def parse_merged_limitations(text_str):
    if not isinstance(text_str, str) or not text_str:
        return []

    limitations = []
    current_category = "General"
    current_subcategory = ""
    lim_id = 0

    lines = text_str.splitlines() # This naturally handles the \n splits
    for line in lines:
        # Strip whitespace and non-breaking spaces
        line = line.strip(" \t\n\r\xa0")
        if not line:
            continue

        # 1. Main Category (e.g., "### Explicit Limitations")
        if line.startswith('###'):
            cleaned_header = line.lstrip('#').strip()
            if "Final List" not in cleaned_header:
                current_category = cleaned_header
            current_subcategory = "" # Reset sub-category on a new main category
            continue

        # 2. Sub-category (e.g., "1. **Scope:**")
        # Matches a number, a dot, optional space, and text wrapped in **
        subcat_match = re.match(r'^\d+\.\s*\*\*(.*?)\*\*:?$', line)
        if subcat_match:
            # Extract "Scope" and strip any trailing colons
            current_subcategory = subcat_match.group(1).strip().strip(':')
            continue

        # 3. NEW: Numbered Limitation Bullet Points (e.g., "1. The method is incremental...")
        # Matches a number, a dot, space, and then captures the rest of the text
        numbered_match = re.match(r'^\d+\.\s+(.*)$', line)
        if numbered_match:
            body = numbered_match.group(1).strip()
            
            # Combine Sub-category if it exists
            if current_subcategory:
                full_limitation = f"{current_subcategory} - {body} (- **{current_category}**)"
            else:
                full_limitation = f"{body} (- **{current_category}**)"
                
            limitations.append({
                'llm_id': lim_id,
                'llm_limitation': full_limitation
            })
            lim_id += 1
            continue

        # 4. Standard Bullet Points (e.g., "- **Temporal Coverage:** The study...")
        if line.startswith('-') or line.startswith('*'):
            body = line[1:].strip()
            
            # Check if the bullet starts with a bolded title
            if body.startswith('**') and '**' in body[2:]:
                parts = body.split('**')
                
                # parts[0] is empty, parts[1] is the title, parts[2:] is the description
                if len(parts) >= 3:
                    title = parts[1].strip().rstrip(':') # Removes trailing colon
                    desc = '**'.join(parts[2:]).lstrip(': ').strip()
                    
                    # Combine Sub-category and Title (e.g., "Scope - Temporal Coverage")
                    if current_subcategory:
                        full_title = f"{current_subcategory} - {title}"
                    else:
                        full_title = title
                        
                    full_limitation = f"{full_title}: {desc} (- **{current_category}**)"
                    
                    limitations.append({
                        'llm_id': lim_id,
                        'llm_limitation': full_limitation
                    })
                    lim_id += 1
                    continue
            
            # Fallback for standard bullets without ** formatting
            elif ':' in body:
                parts = body.split(':', 1)
                title = parts[0].strip()
                desc = parts[1].strip()
                full_limitation = f"{title}: {desc} (- **{current_category}**)"
            else:
                full_limitation = f"{body} (- **{current_category}**)"
                
            limitations.append({
                'llm_id': lim_id,
                'llm_limitation': full_limitation
            })
            lim_id += 1

    return limitations 

df['mistral_limitations_list'] = df['formatted_limitations'].apply(parse_merged_limitations) 

import re
import pandas as pd

def parse_gt_limitations(text_str):
    """
    Parses newline-separated text into a list of dicts.
    """
    if not isinstance(text_str, str) or not text_str.strip():
        return []
    
    results = []
    # Split the string by newlines
    lines = text_str.strip().split('\n')
    
    for i, line in enumerate(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            # Since the GT doesn't have categories in your sample, we just use the text.
            results.append({
                'gt_id': i, 
                'gt_limitation': cleaned_line
            })
            
    return results

# def parse_merged_limitations(text_str):
#     """
#     Parses text formatted as '**[Category]**: Limitation description'
#     """
#     if not isinstance(text_str, str) or not text_str.strip():
#         return []

#     limitations = []
#     lim_id = 0

#     # Regex to match "**[Category]**: Text..."
#     pattern = re.compile(r'^\*\*\[(.*?)\]\*\*:\s*(.*)')

#     # Split the text into lines
#     lines = text_str.strip().split('\n')

#     for ln in lines:
#         ln = ln.strip()
#         if not ln:
#             continue

#         match = pattern.match(ln)
#         if match:
#             category = match.group(1).strip()
#             description = match.group(2).strip()
            
#             # Format: "Limitation text (- **Category**)"
#             full_limitation = f"{description} (- **{category}**)"

#             limitations.append({
#                 "llm_id": lim_id,
#                 "llm_limitation": full_limitation
#             })
#             lim_id += 1

#     return limitations

# Apply functions to the dataframe
df['gt_limitations_list'] = df['ground_truth_lim_peer'].apply(parse_gt_limitations)

# =========================
# 4. Create all GT × LLM pairs with IDs
# =========================

def build_pairs(row):
    pairs = []
    gt_list = row['gt_limitations_list']
    llm_list = row['mistral_limitations_list']

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

# =========================
# 1. Paths and basic setup
# =========================
import re 

import ast
import os
import re
import ast
import pandas as pd
from tqdm import tqdm
from openai import OpenAI  
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-mini" 

# Make sure your key is set in the environment securely.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =========================
# 6. LLM evaluation per pair
# =========================

def call_openai_with_retry(prompt):
    delay = 1.0  # Start with a 1 second delay
    max_retries = 5

    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=False
            )
            return response

        except RateLimitError:
            # Add random "jitter" so concurrent scripts don't retry at the exact same time
            sleep_time = delay + random.uniform(0, 1)
            print(f"Rate limited. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

            delay *= 2  # Double the wait time for the next failure

    raise Exception("Failed after max retries due to rate limits.")

def evaluate_pairs_with_llm(pairs_list):
    """
    Takes a list of pair dicts:
      {'gt_id', 'gt_limitation', 'llm_id', 'llm_limitation'}
    Queries the LLM for each and returns a list like:
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

        description1 = f"ground truth limitations: {gt_text}"
        description2 = f"llm generated limitations: {llm_text}"
        
        prompt = (
            "Check whether 'list2' contains a topic or limitation from 'list1' "
            "or 'list1' contains a topic or limitation from 'list2'.\n\n"
            "Your answer should be \"Yes\" or \"No\".\n"
            f"List 1: {description1}\n"
            f"List 2: {description2}\n"
        )

        try:
            response = call_openai_with_retry(prompt)
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error: {str(e)}"

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
# 7. Apply LLM evaluation row by row
# =========================

df['llm_evaluation_results'] = None

print(f"Starting API Evaluation on {len(df)} rows...")

for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing Rows")):
    pairs = row['paired_limitations']
    row_results = evaluate_pairs_with_llm(pairs)
    df.at[index, 'llm_evaluation_results'] = row_results
    print("i is",i) 
    if (i + 1) % 10 == 0:
        df.to_csv("other_experiments/dpo_novagents/llama/df_eval_final_extracted_limitations.csv", index=False)

# # Final save
df.to_csv("other_experiments/dpo_novagents/llama/df_eval_final_extracted_limitations.csv", index=False)
# print("✅ Processing Complete. Final results saved.")

import pandas as pd
import ast

# # ==========================================
# # 1. Configuration
# # ==========================================
# io_csv = "SFT_from_zs_output/mistral/output/df_gpt_eval.csv"
col_eval = "llm_evaluation_results"

# print(f"Loading CSV: {io_csv} ...")
# df = pd.read_csv(io_csv)

# ==========================================
# 2. Convert 'llm_evaluation_results' from str to list using ast
# ==========================================
def parse_eval_list(val):
    """
    Convert a string representation of a Python list into a real list
    using ast.literal_eval. If already a list, return as-is.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        return ast.literal_eval(val)
    except (SyntaxError, ValueError, TypeError):
        return []

print(f"Parsing column '{col_eval}' with ast.literal_eval ...")
df[col_eval] = df[col_eval].apply(parse_eval_list)

# Quick sanity check on one row
print("\nExample parsed row 0 (first 2 items):")
print(df.loc[df.index[0], col_eval][:2])

# ==========================================
# 3. Compute recall, precision, F1 per row
# ==========================================
def compute_pair_metrics(row):
    """
    From llm_evaluation_results (list-of-lists), compute:
      - n_unique_gt
      - n_unique_llm
      - recall: (# gt_id with at least one Yes) / (total unique gt_id)
      - precision: (# llm_id with at least one Yes) / (total unique llm_id)
      - f1: harmonic mean of precision and recall
    Each item in llm_evaluation_results is:
      ['Pair 1: Yes/No', 'gt_id:0', 'gt_limitation:...', 'llm_id:0', 'llm_limitation:...']
    """
    items = row[col_eval]
    if not isinstance(items, list) or len(items) == 0:
        return pd.Series({
            "n_unique_gt": 0,
            "n_unique_llm": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0
        })
    
    all_gt_ids = set()
    all_llm_ids = set()
    yes_gt_ids = set()
    yes_llm_ids = set()
    
    for item in items:
        # Expect list like ['Pair 1: No', 'gt_id:0', 'gt_limitation:...', 'llm_id:0', 'llm_limitation:...']
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        
        # 1) Answer (Yes/No) from first element
        answer_str = str(item[0])
        is_yes = "YES" in answer_str.upper()  # robust yes-check
        
        # 2) Extract gt_id and llm_id from strings
        gid = None
        lid = None
        
        for elem in item:
            if isinstance(elem, str):
                if elem.startswith("gt_id"):
                    try:
                        gid = int(elem.split(":", 1)[1])
                    except Exception:
                        pass
                elif elem.startswith("llm_id"):
                    try:
                        lid = int(elem.split(":", 1)[1])
                    except Exception:
                        pass
        
        if gid is None or lid is None:
            continue
        
        all_gt_ids.add(gid)
        all_llm_ids.add(lid)
        
        if is_yes:
            yes_gt_ids.add(gid)
            yes_llm_ids.add(lid)
    
    n_unique_gt = len(all_gt_ids)
    n_unique_llm = len(all_llm_ids)
    
    recall = (len(yes_gt_ids) / n_unique_gt) if n_unique_gt > 0 else 0.0
    precision = (len(yes_llm_ids) / n_unique_llm) if n_unique_llm > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return pd.Series({
        "n_unique_gt": n_unique_gt,
        "n_unique_llm": n_unique_llm,
        "recall": recall,
        "precision": precision,
        "f1": f1
    })

print("\nComputing per-row precision, recall, and F1 ...")
metrics_df = df.apply(compute_pair_metrics, axis=1)

# Attach metrics to main df
df["n_unique_gt"] = metrics_df["n_unique_gt"]
df["n_unique_llm"] = metrics_df["n_unique_llm"]
df["recall"] = metrics_df["recall"]
df["precision"] = metrics_df["precision"]
df["f1"] = metrics_df["f1"]

# Save updated CSV
# df.to_csv(io_csv, index=False)
# print(f"\n✅ Metrics added and saved to: {io_csv}")

# Small preview
print(df[["n_unique_gt", "n_unique_llm", "recall", "precision", "f1"]].head())

# ==========================================
# 4. Print average precision, recall, and F1
# ==========================================

# If you want to include all rows (even those with 0/0 → 0 scores):
avg_precision = df["precision"].mean()
avg_recall = df["recall"].mean()
avg_f1 = df["f1"].mean()

print("\n=== Global Averages (including all rows) ===")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall:    {avg_recall:.4f}")
print(f"Average F1:        {avg_f1:.4f}")

# (Optional) If you want to ignore rows where there were no pairs (n_unique_gt == 0 or n_unique_llm == 0):
valid_mask = (df["n_unique_gt"] > 0) & (df["n_unique_llm"] > 0)
if valid_mask.any():
    avg_precision_valid = df.loc[valid_mask, "precision"].mean()
    avg_recall_valid = df.loc[valid_mask, "recall"].mean()
    avg_f1_valid = df.loc[valid_mask, "f1"].mean()

    print("\n=== Global Averages (only rows with at least one GT and one LLM) ===")
    print(f"Average Precision (valid): {avg_precision_valid:.4f}")
    print(f"Average Recall (valid):    {avg_recall_valid:.4f}")
    print(f"Average F1 (valid):        {avg_f1_valid:.4f}")
else:
    print("\n(No valid rows with both GT and LLM limitations found.)")

import pandas as pd
import ast
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ==========================================
# 1. Load CSV and column name
# ==========================================
# io_csv = "SFT_from_zs_output/mistral/output/df_gpt_eval.csv"
col_eval = "llm_evaluation_results"

# print(f"Loading CSV: {io_csv} ...")
# df = pd.read_csv(io_csv)

# ==========================================
# 2. Parse llm_evaluation_results from str -> list via ast
# ==========================================
def parse_eval_list(val):
    """
    Convert a string representation of a Python list into a real list
    using ast.literal_eval. If already a list, return as-is.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        return ast.literal_eval(val)
    except (SyntaxError, ValueError, TypeError):
        return []

print(f"Parsing '{col_eval}' with ast.literal_eval ...")
df[col_eval] = df[col_eval].apply(parse_eval_list)

# ==========================================
# 3. Prepare similarity helpers
# ==========================================

# Rouge-L scorer (create once)
rougeL_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def cosine_sim(gt_text, llm_text):
    """Cosine similarity using TF-IDF vectors."""
    vect = TfidfVectorizer().fit([gt_text, llm_text])
    tfidf = vect.transform([gt_text, llm_text])
    cos = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
    return float(cos)

def jaccard_sim(gt_text, llm_text):
    """Jaccard similarity over lowercased whitespace-token sets."""
    tokens1 = set(gt_text.lower().split())
    tokens2 = set(llm_text.lower().split())
    union = tokens1 | tokens2
    if not union:
        return 0.0
    inter = tokens1 & tokens2
    return float(len(inter) / len(union))

def rougeL_f1(gt_text, llm_text):
    """ROUGE-L F1 between reference (gt) and candidate (llm)."""
    scores = rougeL_scorer.score(gt_text, llm_text)
    return float(scores['rougeL'].fmeasure)

def bertscore_f1(gt_text, llm_text):
    """
    BERTScore F1 between reference (gt) and candidate (llm).
    We use llm_text as candidate and gt_text as reference.
    """
    P, R, F = bert_score([llm_text], [gt_text], lang='en', verbose=False)
    return float(F[0])

# ==========================================
# 4. Compute per-row averages over YES pairs
# ==========================================
def compute_similarity_metrics(row):
    """
    For this row's llm_evaluation_results:
      - Take only pairs where answer is 'Yes'
      - Extract gt_limitation and llm_limitation texts
      - Compute cosine, jaccard, rougeL, bertscore per pair
      - Return row-wise averages
    """
    items = row[col_eval]
    if not isinstance(items, list) or len(items) == 0:
        return pd.Series({
            "avg_cosine_sim": 0.0,
            "avg_jaccard_sim": 0.0,
            "avg_rougeL": 0.0,
            "avg_bertscore": 0.0,
            "n_yes_pairs": 0
        })
    
    gt_texts = []
    llm_texts = []

    for item in items:
        # item example:
        # ['Pair 1: Yes', 'gt_id:0', 'gt_limitation:TEXT...', 'llm_id:0', 'llm_limitation:TEXT...']
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        
        # Check if this pair is marked as Yes
        answer_str = str(item[0])
        is_yes = "YES" in answer_str.upper()
        if not is_yes:
            continue
        
        gt_text = None
        llm_text = None
        
        for elem in item:
            if isinstance(elem, str):
                if elem.startswith("gt_limitation:"):
                    gt_text = elem.split("gt_limitation:", 1)[1].strip()
                elif elem.startswith("llm_limitation:"):
                    llm_text = elem.split("llm_limitation:", 1)[1].strip()
        
        if gt_text and llm_text:
            gt_texts.append(gt_text)
            llm_texts.append(llm_text)
    
    n_yes = len(gt_texts)
    if n_yes == 0:
        return pd.Series({
            "avg_cosine_sim": 0.0,
            "avg_jaccard_sim": 0.0,
            "avg_rougeL": 0.0,
            "avg_bertscore": 0.0,
            "n_yes_pairs": 0
        })
    
    cos_vals = []
    jac_vals = []
    rougel_vals = []
    bert_vals = []
    
    for gt_text, llm_text in zip(gt_texts, llm_texts):
        try:
            cos_vals.append(cosine_sim(gt_text, llm_text))
        except Exception:
            cos_vals.append(0.0)
        try:
            jac_vals.append(jaccard_sim(gt_text, llm_text))
        except Exception:
            jac_vals.append(0.0)
        try:
            rougel_vals.append(rougeL_f1(gt_text, llm_text))
        except Exception:
            rougel_vals.append(0.0)
        try:
            bert_vals.append(bertscore_f1(gt_text, llm_text))
        except Exception:
            bert_vals.append(0.0)
    
    return pd.Series({
        "avg_cosine_sim": float(np.mean(cos_vals)) if cos_vals else 0.0,
        "avg_jaccard_sim": float(np.mean(jac_vals)) if jac_vals else 0.0,
        "avg_rougeL": float(np.mean(rougel_vals)) if rougel_vals else 0.0,
        "avg_bertscore": float(np.mean(bert_vals)) if bert_vals else 0.0,
        "n_yes_pairs": n_yes
    })

print("\nComputing similarity metrics (cosine, jaccard, ROUGE-L, BERTScore) for YES pairs...")
sim_metrics = df.apply(compute_similarity_metrics, axis=1)

df["avg_cosine_sim"] = sim_metrics["avg_cosine_sim"]
df["avg_jaccard_sim"] = sim_metrics["avg_jaccard_sim"]
df["avg_rougeL"] = sim_metrics["avg_rougeL"]
df["avg_bertscore"] = sim_metrics["avg_bertscore"]
df["n_yes_pairs"] = sim_metrics["n_yes_pairs"]

print("avg_cosine_sim",df["avg_cosine_sim"].mean())
print("avg_jaccard_sim",df["avg_jaccard_sim"].mean())
print("avg_rougeL",df["avg_rougeL"].mean())
print("avg_bertscore",df["avg_bertscore"].mean())
print("n_yes_pairs",df["n_yes_pairs"].mean())

# ==========================================
# 5. Save and quick preview
# ==========================================
df.to_csv("other_experiments/dpo_novagents/llama/df_eval_final_extracted_limitations.csv", index=False)

# print(f"\n✅ Similarity metrics added and saved back to: {io_csv}")

print("\nPreview of new columns:")
print(df[["n_yes_pairs", "avg_cosine_sim", "avg_jaccard_sim", "avg_rougeL", "avg_bertscore"]].head())
