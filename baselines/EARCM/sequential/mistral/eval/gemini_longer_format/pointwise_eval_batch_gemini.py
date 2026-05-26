import pandas as pd
import ast
import re
import os
import time
import json
import torch
from tqdm import tqdm
from google import genai
from google.genai import types
from bert_score import score as bert_score_func 

# ==========================================
# 1. Configuration & Client Setup
# ==========================================

os.environ["GEMINI_API_KEY"] = "" 
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

print("Initializing Gemini Client...")
client = genai.Client(api_key=api_key)

# MODEL_ID = "gemini-2.0-flash" 
# This matches the stats you quoted: 1,000 RPD / 15 RPM
MODEL_ID = "gemini-2.5-flash-lite" 

# ==========================================
# 2. Parsing Helpers
# ==========================================

def parse_mistral_list(text):
    if pd.isna(text): return []
    start_match = re.search(r'(?:^|\n)\s*1[\.\)]', text)
    if not start_match: return []
    trimmed = "\n" + text[start_match.start():].strip()
    split_items = re.split(r'\n\s*\d+[\.\)]\s*', trimmed)
    return [i.strip() for i in split_items if len(i.strip()) > 5]

def parse_gt_list(text):
    try: 
        return ast.literal_eval(text) if pd.notna(text) else []
    except: 
        return []

def prepare_limitation_batch(row):
    """
    Creates a list of dictionaries for batch processing.
    Groups by GT_ID (Ground Truth) first.
    Includes 'gt_title' for the new check.
    """
    mistral_text = row.get('mistral_master_0.4', '')
    gt_text = row.get('final_lim_gt_author_peer_cat_maj_hum_cleaned', '')
    
    mistral_list = parse_mistral_list(mistral_text)
    gt_list = parse_gt_list(gt_text)
    
    lim_batch = []
    pair_counter = 0
    
    if mistral_list and gt_list:
        for gt_idx, gt_item in enumerate(gt_list):
            for m_idx, m_text in enumerate(mistral_list):
                
                lim_batch.append({
                    "pair_id": pair_counter,
                    "gt_id": gt_idx,
                    "llm_id": m_idx,
                    "gt_limitation": gt_item.get('limitation', ''),
                    "gt_title": gt_item.get('title', ''), # Added for the new check
                    "llm_limitation": m_text,
                    "category": gt_item.get('category', '')
                })
                pair_counter += 1
                
    return lim_batch

# ==========================================
# 3. Helpers: Gemini & BERTScore
# ==========================================

def run_gemini_json_batch(prompt: str, retries=3) -> list:
    """Standard JSON request for batches."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0 
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg:
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time)
            else:
                return "Error_Inference"
    return "Error_RateLimitExceeded"

def run_gemini_formatting_check(full_text: str) -> dict:
    """
    Evaluates the FULL text formatting once per row.
    Returns: {'format_score': 45}
    """
    if not full_text or pd.isna(full_text):
        return {"format_score": 0}

    prompt = f"""You are an expert formatting judge. Evaluate the following text **only** on formatting quality (structure, bullet points, clarity, consistency, visual hierarchy, grammar/punctuation that affect readability). Ignore factual accuracy, creativity, or tone.

    TEXT TO EVALUATE:
    \"\"\"{full_text[:10000]}\"\"\"

    CRITERIA (rate each 0–10, where 0 = unusable, 10 = perfect):
    1. **Bullet-Point Usage** – Are bullets used where lists exist? Consistent style?
    2. **Clarity & Readability** – Short sentences, active voice, no jargon walls, logical flow.
    3. **Structural Hierarchy** – Proper use of headings, bold/italic for emphasis.
    4. **Consistency** – Same bullet marker, punctuation, capitalization, spacing.
    5. **Overall Visual Appeal** – Balanced whitespace, no wall-of-text.

    Measure final score out of 50.

    Output strictly a JSON object: {{"format_score": <int>}}
    """
    
    try:
        result = run_gemini_json_batch(prompt)
        if isinstance(result, list):
            return result[0]
        return result
    except:
        return {"format_score": -1}

def calculate_bert_scores_for_batch(merged_data):
    if not merged_data or isinstance(merged_data, str):
        return merged_data

    cands = [item['llm_limitation'] for item in merged_data]
    refs = [item['gt_limitation'] for item in merged_data]
    
    try:
        P, R, F1 = bert_score_func(cands, refs, lang="en", verbose=False)
        for i, item in enumerate(merged_data):
            item['bertscore'] = float(F1[i])
    except Exception as e:
        for item in merged_data:
            item['bertscore'] = -1.0
            
    return merged_data

# ==========================================
# 4. Prompt Builders
# ==========================================

def get_limitation_prompt(batch_data):
    # Minimized data sent to LLM (Now includes gt_title)
    minimized_data = [{
        "pair_id": item['pair_id'],
        "llm_limitation": item['llm_limitation'],
        "gt_limitation": item['gt_limitation'],
        "gt_title": item['gt_title'], # NEW field
        "category": item['category']
    } for item in batch_data]
    
    data_str = json.dumps(minimized_data, indent=2)
    
    return f"""You are an expert scientific evaluator. 
    I will provide a JSON list of pairs containing a "Generated Limitation" (LLM) and a "Ground Truth Limitation" (GT), along with the GT Title.

    Input Data:
    {data_str}

    Task:
    For EACH pair in the list, determine:
    1. is_similar: Are they semantically similar? (YES/NO)
    2. similarity_score: A float between 0.0 and 1.0.
    3. better_one: Which is written better? (A, B, or Equal).
    4. better_margin: (High, Medium, Low, None).
    5. is_llm_in_category: Does the Generated Limitation (LLM) fit the provided category context? (YES/NO).
    6. is_llm_in_title: Does the Generated Limitation (LLM) support or align with the Ground Truth Title? (YES/NO).

    Output strictly a JSON list of objects matching this schema:
    [
        {{
            "pair_id": 0,
            "is_similar": "YES",
            "similarity_score": 0.9,
            "better_one": "B",
            "better_margin": "Low",
            "is_llm_in_category": "YES",
            "is_llm_in_title": "YES"
        }},
        ...
    ]
    """

# ==========================================
# 5. Result Merging Helper
# ==========================================

def merge_results(original_batch, gemini_results, keys_to_keep):
    if isinstance(gemini_results, str):
        return gemini_results 
        
    merged_list = []
    gemini_lookup = {item.get('pair_id'): item for item in gemini_results}
    
    for input_item in original_batch:
        pid = input_item['pair_id']
        eval_data = gemini_lookup.get(pid, {})
        
        merged_item = {k: input_item[k] for k in keys_to_keep if k in input_item}
        # Add the evaluation metrics
        for k, v in eval_data.items():
            if k != 'pair_id':
                merged_item[k] = v
        merged_list.append(merged_item)
        
    return merged_list

# ==========================================
# 6. Main Execution Loop
# ==========================================

INPUT_CSV = "zs_mistral_master_final.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = "df_mistral_evaluations_gemini_final_rows.csv"
FULL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)

df['gemini_limitation_eval'] = None
# gemini_title_eval column removed

print("Starting Batched Gemini Evaluation...")

for idx, row in tqdm(df.iterrows(), total=len(df)): 
    
    lim_batch_data = prepare_limitation_batch(row)
    
    if not lim_batch_data:
        continue

    # --- 1. Global Formatting Check (Once per row) ---
    try:
        mistral_full_text = row.get('mistral_master_0.4', '')
        formatting_result = run_gemini_formatting_check(mistral_full_text)
    except:
        formatting_result = {"format_score": 0}

    # --- 2. Limitations Batch ---
    try:
        # A. Get LLM Evaluation
        prompt_lim = get_limitation_prompt(lim_batch_data)
        raw_result_lim = run_gemini_json_batch(prompt_lim)
        
        # B. Merge Original Text + LLM Eval
        # Added gt_title to keys_to_keep so it shows in final CSV
        merged_lim_data = merge_results(
            lim_batch_data, 
            raw_result_lim, 
            keys_to_keep=['gt_id', 'llm_id', 'gt_limitation', 'gt_title', 'llm_limitation']
        )
        
        # C. Calculate BERTScore
        final_lim_data = calculate_bert_scores_for_batch(merged_lim_data)
        
        # D. Prepend Formatting Score
        if isinstance(final_lim_data, list):
            final_lim_data.insert(0, formatting_result)
            
        df.at[idx, 'gemini_limitation_eval'] = final_lim_data
        
    except Exception as e:
        print(f"Error Row {idx} (Lim): {e}")
        df.at[idx, 'gemini_limitation_eval'] = "Error_Inference"
        
    if idx % 10 == 0:
        df.to_csv(FULL_PATH, index=False)

df.to_csv(FULL_PATH, index=False)
print(f"✅ Processing Complete. Results saved to: {FULL_PATH}")