import pandas as pd
import torch
import os
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

# --- Configuration ---
TEST_INPUT_FILE = "data/final_balanced_kde/df_balanced_kde_final.csv"

OUTPUT_DIR = 'SFT_from_zs_output/mistral/output_using_ground_truth'
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "mistral_sft_test_results.csv")

# 🔹 Choose one specific checkpoint here
CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints",
    "checkpoint-75"      # <-- change to actual checkpoint name (e.g. "checkpoint-25")
)

# --- 1. Load and Prepare Data ---
print(f"Loading Test Data from {TEST_INPUT_FILE}...")
try:
    df_full = pd.read_csv(TEST_INPUT_FILE)
    
    df_slice = df_full.copy()
    print(f"Sliced rows 100-199. Shape: {df_slice.shape}")
    
    df_test = df_slice.sample(n=50, random_state=42).copy()
    print(f"Selected 50 random samples. Shape: {df_test.shape}")
    
    if 'input_text' not in df_test.columns:
        print("Error: Column 'input_text' not found in dataframe.")
        print("Available columns:", df_test.columns.tolist())
        sys.exit(1)
        
except FileNotFoundError:
    print("Error: Input file not found.")
    sys.exit(1)

# --- 2. Load Model & Adapters ---
print("Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

print(f"Loading Fine-Tuned Adapters from {CHECKPOINT_DIR} ...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()

# 🔹 For checkpoints, just load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# generated output includes both input prompt and new tokens
# --- 3. Inference Loop ---
# def generate_sft_response(input_text):
#     system_prompt = "You are helpful assistant, generate limitations from these texts"
    
#     truncated_input = str(input_text)[:25000]
#     prompt = f"<s>[INST] {system_prompt}\n\nText:\n{truncated_input}\n\nLimitations: [/INST]"
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=1024,
#             do_sample=True,
#             temperature=0.4,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     if "[/INST]" in generated_text:
#         return generated_text.split("[/INST]")[-1].strip()
#     return generated_text.strip()

# generated output includes only new tokens
def generate_sft_response(input_text):
    system_prompt = "You are helpful assistant, generate limitations from these texts"
    
    truncated_input = str(input_text)[:25000]
    prompt = f"<s>[INST] {system_prompt}\n\nText:\n{truncated_input}\n\nLimitations: [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # --- THE FIX ---
    # 1. Calculate the length of the input prompt tokens
    input_length = inputs["input_ids"].shape[1]
    
    # 2. Slice the output tensor to keep ONLY the new tokens (everything after input_length)
    generated_tokens = outputs[0][input_length:]
    
    # 3. Decode only the new tokens
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

print("Running Inference...")
df_test["mistral_sft_generated"] = ""
df_test = df_test.reset_index(drop=True)

for i, row in df_test.iterrows():
    input_text = row['input_text']
    
    if pd.notna(input_text) and str(input_text).strip():
        try:
            response = generate_sft_response(input_text)
            df_test.at[i, "mistral_sft_generated"] = response
            print(f"  Processed sample {i+1}/50")
        except Exception as e:
            print(f"  Error at sample {i+1}: {e}")
            df_test.at[i, "mistral_sft_generated"] = "ERROR"
    else:
        print(f"  Skipping sample {i+1} (empty 'input_text')")

    if (i + 1) % 10 == 0:
        df_test.to_csv(RESULTS_FILE, index=False)

df_test.to_csv(RESULTS_FILE, index=False)
print(f"Done! Results saved to: {RESULTS_FILE}")
