import pandas as pd
import json
import ast
import numpy as np
import os

# 1. Configuration
io_csv = "df_mistral_evaluations_gemini_final.csv"
output_json_dir = "output"
output_json_path = os.path.join(output_json_dir, "gemini_final_result.json")

col_limitation = "gemini_limitation_eval"

print(f"Loading CSV: {io_csv} ...")
df = pd.read_csv(io_csv)

# 3. Parsing Function
def safe_parse(val):
    try:
        if pd.isna(val) or val == "" or str(val).startswith("Error"):
            return []
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

print("Parsing input column...")
df[col_limitation] = df[col_limitation].apply(safe_parse)

# ==========================================
# 4. METRIC CALCULATION LOGIC (Per Row)
# ==========================================
def calculate_metrics_to_dict(row):
    items = row[col_limitation]
    
    # Formatting Score Extraction
    formatting_score = 0.0
    data_items = items
    
    if items and isinstance(items[0], dict) and 'format_score' in items[0]:
        formatting_score = float(items[0]['format_score'])
        data_items = items[1:] 
    
    # --- Initialize Sets/Maps ---
    all_gt_ids = set()
    all_llm_ids = set()
    
    # Numerators (Unweighted)
    sim_gt_hits = set()
    sim_llm_hits = set()
    title_gt_hits = set()
    title_llm_hits = set()
    cat_gt_hits = set()
    cat_llm_hits = set()
    
    # Numerators (Weighted)
    sim_gt_max_scores = {}  
    sim_llm_max_scores = {} 
    
    matched_bert_scores = []
    
    # --- Loop Comparisons ---
    for item in data_items:
        gid = item.get('gt_id')
        lid = item.get('llm_id')
        
        if gid is None or lid is None:
            continue
            
        all_gt_ids.add(gid)
        all_llm_ids.add(lid)
        
        # 1. Similarity
        val_sim = str(item.get('is_similar', '')).strip().upper()
        sim_score = float(item.get('similarity_score', 0.0))
        
        if 'YES' in val_sim:
            sim_gt_hits.add(gid)
            sim_llm_hits.add(lid)
            sim_gt_max_scores[gid] = max(sim_gt_max_scores.get(gid, 0.0), sim_score)
            sim_llm_max_scores[lid] = max(sim_llm_max_scores.get(lid, 0.0), sim_score)
            
            b_score = item.get('bertscore')
            if b_score is not None and b_score != -1.0:
                matched_bert_scores.append(b_score)
        
        # 2. Title
        val_title = str(item.get('is_llm_in_title', '')).strip().upper()
        if 'YES' in val_title:
            title_gt_hits.add(gid)
            title_llm_hits.add(lid)
            
        # 3. Category
        val_cat = str(item.get('is_llm_in_category', '')).strip().upper()
        if 'YES' in val_cat:
            cat_gt_hits.add(gid)
            cat_llm_hits.add(lid)

    # --- Calculations ---
    n_unique_gt = len(all_gt_ids)
    n_unique_llm = len(all_llm_ids)
    
    def safe_div(num, den):
        return num / den if den > 0 else 0.0

    metrics_dict = {
        "formatting_score": formatting_score,
        
        "is_similar_precision": safe_div(len(sim_llm_hits), n_unique_llm),
        "is_similar_recall": safe_div(len(sim_gt_hits), n_unique_gt),
        
        "weighted_is_sim_precision": safe_div(sum(sim_llm_max_scores.values()), n_unique_llm),
        "weighted_is_sim_recall": safe_div(sum(sim_gt_max_scores.values()), n_unique_gt),
        
        "is_llm_in_title_precision": safe_div(len(title_llm_hits), n_unique_llm),
        "is_llm_in_title_recall": safe_div(len(title_gt_hits), n_unique_gt),
        
        "is_llm_in_category_precision": safe_div(len(cat_llm_hits), n_unique_llm),
        "is_llm_in_category_recall": safe_div(len(cat_gt_hits), n_unique_gt),
        
        "matched_avg_bscore": np.mean(matched_bert_scores) if matched_bert_scores else 0.0,
        
        # Keep counts just in case, but they won't be averaged in final JSON
        "n_unique_gt": n_unique_gt,
        "n_unique_llm": n_unique_llm
    }
    
    return [metrics_dict]

print("Calculating row metrics...")
df['result_gemini'] = df.apply(calculate_metrics_to_dict, axis=1)

# Save result (overwriting input file)
df.to_csv(io_csv, index=False)
print(f"✅ Row processing complete. Saved to: {io_csv}")

# ==========================================
# 5. GLOBAL AGGREGATION (Average All Rows)
# ==========================================
print("\nCalculating Global Averages...")

# Initialize accumulators
keys_to_average = [
    "formatting_score",
    "is_similar_precision", "is_similar_recall",
    "weighted_is_sim_precision", "weighted_is_sim_recall",
    "is_llm_in_title_precision", "is_llm_in_title_recall",
    "is_llm_in_category_precision", "is_llm_in_category_recall",
    "matched_avg_bscore"
]

total_sums = {key: 0.0 for key in keys_to_average}
valid_rows = 0

for _, row in df.iterrows():
    try:
        # Extract the dictionary from the list wrapper
        res_list = row['result_gemini']
        if isinstance(res_list, list) and len(res_list) > 0:
            res_dict = res_list[0]
            
            for key in keys_to_average:
                val = res_dict.get(key, 0.0)
                # Handle NaNs or Nones safely
                if pd.isna(val): val = 0.0
                total_sums[key] += float(val)
            
            valid_rows += 1
    except Exception as e:
        continue

# Calculate Averages
final_json_data = {}
if valid_rows > 0:
    for key, total_val in total_sums.items():
        final_json_data[key] = round(total_val / valid_rows, 4)
    final_json_data["total_rows_processed"] = valid_rows
else:
    final_json_data["error"] = "No valid rows found to average."

# Save to JSON
print(f"Saving final JSON to: {output_json_path}")
with open(output_json_path, 'w') as f:
    json.dump(final_json_data, f, indent=4)

print("\n--- FINAL RESULTS ---")
print(json.dumps(final_json_data, indent=4))