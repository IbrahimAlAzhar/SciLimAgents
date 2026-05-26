import json
import pandas as pd
import ast
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "models/mistral_7b_v3_instruct"
input_path = "df.csv"
output_path = "df1.csv"

# --- Model Loading ---
print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=cache_dir, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Instruction Prompt ---
instruction_batch = """
You are a research assistant comparing pairs of academic limitations.
For each pair provided, decide if they represent the SAME underlying issue ("similar") or DIFFERENT issues ("not similar").

Output Format:
You MUST respond with ONLY a valid JSON list of objects. No introductory text.
Format: [{"Pair": 1, "decision": "similar", "score": 0.9}, {"Pair": 2, "decision": "not similar", "score": 0.1}, ...]
"""

def mistral_generate(prompt):
    """Generates a response from the Mistral model."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=2048, # Increased to handle 20 pairs safely
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def process_pairs_chunk(pairs_chunk, start_idx):
    """Processes a chunk of 20 pairs and maps original text back to JSON."""
    formatted_pairs = ""
    for i, (gpt_text, gemini_text) in enumerate(pairs_chunk):
        formatted_pairs += f"Pair {start_idx + i + 1}:\nGPT: {gpt_text}\nGemini: {gemini_text}\n\n"
    
    prompt = f"{instruction_batch}\n\nPairs to evaluate:\n{formatted_pairs}"
    response = mistral_generate(prompt)
    
    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        parsed_results = json.loads(json_match.group(0)) if json_match else json.loads(response)
        
        final_results = []
        for i, res in enumerate(parsed_results):
            if i < len(pairs_chunk):
                # Manually injecting original texts from the Python object
                res['gt_gpt'] = pairs_chunk[i][0]
                res['gt_gemini'] = pairs_chunk[i][1]
                final_results.append(res)
        return final_results
    except Exception as e:
        print(f"⚠️ Error processing chunk: {e}")
        return [{"Pair": start_idx + i + 1, "decision": "Error", "score": 0, "gt_gpt": p[0], "gt_gemini": p[1]} for i, p in enumerate(pairs_chunk)]

# --- Main Execution ---
print("Reading input dataframe...")
df = pd.read_csv(input_path) 

df['limitation_pairs_auth_peer_gt_gpt_gemini'] = df['limitation_pairs_auth_peer_gt_gpt_gemini'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

all_decisions = []
total_rows = len(df)

print(f"Starting processing for {total_rows} rows...")

for idx, row in df.iterrows():
    pairs = row['limitation_pairs_auth_peer_gt_gpt_gemini']
    row_results = []
    
    # CHUNK SIZE SET TO 20
    chunk_size = 20 
    
    if isinstance(pairs, list) and len(pairs) > 0:
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i : i + chunk_size]
            # Slicing handles the leftovers (last chunk < 20) automatically
            results = process_pairs_chunk(chunk, i)
            row_results.extend(results)
    
    all_decisions.append(row_results)
    print(f"Progress: Row {idx + 1}/{total_rows}")

    # --- Checkpoint: Save every 10 iterations ---
    if (idx + 1) % 10 == 0:
        df_checkpoint = df.iloc[:idx+1].copy()
        df_checkpoint['llm_decisions'] = all_decisions
        df_checkpoint.to_csv(output_path, index=False)
        print(f"💾 [Row {idx + 1}] Checkpoint saved to: {output_path}")

# --- Final Save ---
df['llm_decisions'] = all_decisions
df.to_csv(output_path, index=False)
print(f"\n✅ All done! Results saved to: {output_path}")