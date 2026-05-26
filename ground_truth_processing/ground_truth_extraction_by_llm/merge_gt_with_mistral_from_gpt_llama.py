import json
import time
import pandas as pd
import ast
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "models/mistral_7b_v3_instruct"

print("Loading Mistral model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    cache_dir=cache_dir,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def truncate_prompt_for_model(prompt: str, max_length: int = 32000) -> str:
    """Truncate prompt to fit within model's context window, leaving room for generation"""
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    
    if tokens.shape[1] > max_length:
        print(f"⚠️ Prompt token count = {tokens.shape[1]} exceeds limit ({max_length}). Truncating...")
        truncated_tokens = tokens[:, :max_length]
        return tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    
    return prompt

def mistral_generate(prompt, max_new_tokens=512):
    """Generate text using Mistral model"""
    truncated_prompt = truncate_prompt_for_model(prompt, max_length=32000)
    
    messages = [
        {"role": "user", "content": truncated_prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

instruction_single = """
You are comparing two limitations.

- Decide whether they describe essentially the SAME underlying limitation or DIFFERENT ones.
- Consider them the SAME if they describe the same problem even with different wording.
- Consider them DIFFERENT if they focus on clearly different issues.

You must also output a relevance score between 0 and 1 that reflects how semantically similar the two limitations are:
- 1.0 = almost identical meaning
- 0.0 = completely unrelated
- If you judge them as "Same", the score should typically be above 0.5.

Respond ONLY with a single JSON object of the form:

{
  "decision": "Same",
  "score": 0.87
}

Use EXACTLY the strings "Same" or "Different" as values for "decision".
"""

def parse_limitations_column(x):
    """Parse limitations column that may contain JSON strings, markdown code blocks, or Python lists"""
    if pd.isna(x) or not str(x).strip():
        return []
    
    try:
        s = str(x).strip()
        
        # Remove markdown code blocks if present
        if s.startswith('```'):
            # Find the closing ```
            end_idx = s.rfind('```')
            if end_idx > 0:
                s = s[3:end_idx].strip()  # Remove opening ```
                # Remove language identifier if present (e.g., "json")
                if s.startswith('json'):
                    s = s[4:].strip()
                elif s.startswith('python'):
                    s = s[6:].strip()
        
        # Try json.loads first (handles JSON strings)
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            # Fall back to ast.literal_eval (handles Python literals)
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                # If both fail, try to extract JSON from the string
                # Look for JSON array pattern
                json_match = re.search(r'\[.*\]', s, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except:
                        pass
                print(f"⚠️ Warning: Could not parse limitations column. Returning empty list.")
                print(f"   First 200 chars: {s[:200]}")
                return []
    except Exception as e:
        print(f"⚠️ Error parsing limitations: {e}")
        return []

# Read the CSV file
csv_path = "df.csv"

df = pd.read_csv(csv_path) 
print("dataframe head:", df.head())

# Convert columns to lists using robust parsing function
print("Converting columns to lists...")
df['LLM_merged_author_peer_limitations_llama'] = df['LLM_merged_author_peer_limitations_llama'].apply(parse_limitations_column)
df['LLM_merged_author_peer_limitations_gpt'] = df['LLM_merged_author_peer_limitations_gpt'].apply(parse_limitations_column)

# Create pairs from the limitations
print("Creating pairs from limitations...")
def create_pairs(row):
    """Create all pairs between GPT and LLaMA limitations"""
    gpt_lims = row['LLM_merged_author_peer_limitations_gpt']
    llama_lims = row['LLM_merged_author_peer_limitations_llama']
    
    if not isinstance(gpt_lims, list):
        gpt_lims = []
    if not isinstance(llama_lims, list):
        llama_lims = []
    
    pairs = []
    pair_id = 0
    
    for gpt_idx, gpt_lim in enumerate(gpt_lims):
        for llama_idx, llama_lim in enumerate(llama_lims):
            # Extract limitation text and title (handle both dict and string formats)
            if isinstance(gpt_lim, dict):
                gpt_text = gpt_lim.get('limitation', gpt_lim.get('description', str(gpt_lim)))
                gpt_title = gpt_lim.get('title', '')
            else:
                gpt_text = str(gpt_lim)
                gpt_title = ''
            
            if isinstance(llama_lim, dict):
                llama_text = llama_lim.get('limitation', llama_lim.get('description', str(llama_lim)))
                llama_title = llama_lim.get('title', '')
            else:
                llama_text = str(llama_lim)
                llama_title = ''
            
            pairs.append({
                "pair_id": f"gpt_{gpt_idx}_llama_{llama_idx}",
                "gpt_limitation": gpt_text,
                "llama_limitation": llama_text,
                "gpt_title": gpt_title,
                "llama_title": llama_title
            })
            pair_id += 1
    
    return pairs

df['pair_content_all'] = df.apply(create_pairs, axis=1)

# New column to store per-pair decisions for each row
df['pair_decision_all'] = None

total_rows = len(df)
print(f"→ Evaluating pairs for {total_rows} rows")

for row_idx in range(total_rows):
    pairs = df.at[row_idx, "pair_content_all"]
    
    if not isinstance(pairs, list) or len(pairs) == 0:
        df.at[row_idx, "pair_decision_all"] = []
        continue
    
    print(f"\nRow {row_idx + 1}/{total_rows}: found {len(pairs)} pairs")
    
    row_results = []
    
    for pair in pairs:
        pair_id = pair.get("pair_id", "")
        
        g_lim = pair.get("gpt_limitation") or ""
        l_lim = pair.get("llama_limitation") or ""
        
        # If both empty, skip this pair
        if not str(g_lim).strip() and not str(l_lim).strip():
            row_results.append({
                "pair_id": pair_id,
                "decision": "Unknown",
                "score": None
            })
            continue
        
        # Build prompt for this pair
        full_prompt = (
            instruction_single
            + "\n\nGPT limitation:\n"
            + str(g_lim)
            + "\n\nLLaMA limitation:\n"
            + str(l_lim)
        )
        
        try:
            resp_text = mistral_generate(full_prompt, max_new_tokens=512)
            text = resp_text.strip() 
            print("text: ", text)
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response (might have extra text)
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(text)
                
                decision = parsed.get("decision", "Unknown")
                score = parsed.get("score", None)
            except Exception as e:
                print(f"  ⚠️ Could not parse JSON for pair {pair_id}: {e}")
                print(f"  Response was: {text[:200]}")
                decision = "Unknown"
                score = None
            
            row_results.append({
                "pair_id": pair_id,
                "decision": decision,
                "score": score
            })
            
        except Exception as e:
            print(f"  ✗ Error for row {row_idx}, pair {pair_id}: {e}")
            row_results.append({
                "pair_id": pair_id,
                "decision": "ERROR",
                "score": None,
                "error_message": str(e),
            })
    
    # Store list of decisions for this row
    df.at[row_idx, "pair_decision_all"] = row_results 
    print("row_results: ", row_results)
    
    if (row_idx + 1) % 10 == 0:
        output_file = "df1.csv"
        df.to_csv(output_file, index=False)
        print(f"  ✅ Checkpoint saved at row {row_idx + 1}")

# Save final results
output_file = "df1.csv"
df.to_csv(output_file, index=False)
print(f"\n→ Done. Results saved to: {output_file}")
print("→ You now have df['pair_content_all'] and df['pair_decision_all'] aligned by 'pair_id'.")

