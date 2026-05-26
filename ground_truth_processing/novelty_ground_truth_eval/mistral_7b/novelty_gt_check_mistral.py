import pandas as pd
import os
import torch
import signal
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. Configuration & Model Loading
# ==========================================

# --- PATHS (Updated for Mistral) ---
INPUT_CSV = "data/final_not_balanced_data/df_not_balanced_preprocessed_gt_final.csv"
OUTPUT_DIR = "data/final_not_balanced_data/novelty_gt"

# import pandas as pd 
# df = pd.read_csv("data/final_not_balanced_data/df_not_balanced_preprocessed_gt_final.csv") 
# df['limitations_autho_peer_gt']

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MODEL CONFIG
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"

MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 32000 # Mistral v0.3 supports larger context (32k)

print(f"Loading Mistral model from {MODEL_ID}...")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    cache_dir=CACHE_DIR, 
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model (Using bfloat16 as per your example)
# Note: If you run out of GPU memory, you can re-enable 4-bit quantization 
# by importing BitsAndBytesConfig and passing quantization_config=bnb_config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ==========================================
# 2. Signal Handling (Graceful Exit)
# ==========================================
global_df = None
global_current_row = 0

def signal_handler(signum, frame):
    print(f"\n⚠️  Received signal {signum}. Saving progress...")
    if global_df is not None:
        save_path = os.path.join(OUTPUT_DIR, f"emergency_novelty_save_mistral_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# 3. Inference Helper
# ==========================================

def truncate_prompt(text_input: str, max_length: int = 30000) -> str:
    """Truncates text safely to fit within Mistral's context."""
    tokens = tokenizer.encode(text_input, add_special_tokens=False)
    if len(tokens) > max_length:
        return tokenizer.decode(tokens[:max_length], skip_special_tokens=True)
    return text_input

def run_mistral(prompt_text: str, system_instruction: str = "") -> str:
    """
    Runs inference using Mistral's chat template.
    We combine system instruction into the user message for stricter adherence.
    """
    
    # Combine system instruction + user prompt for best 7B behavior
    full_content = f"{system_instruction}\n\n{prompt_text}"
    
    messages = [
        {"role": "user", "content": full_content}
    ]

    # Apply Mistral's Chat Template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Truncate if necessary (rarely needed for 32k window, but good safety)
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
         # Basic truncation strategy if template output is too huge
         inputs["input_ids"] = inputs["input_ids"][:, -MAX_INPUT_TOKENS:]
         inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_INPUT_TOKENS:]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,      
            temperature=0.1,  # Low temp for deterministic YES/NO
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# def get_novelty_check_prompt(text_to_analyze: str) -> str:
#     return f"""Analyze the text below, which contains peer review limitations or critiques of a paper.

# **Your Task:**
# Determine if the text mentions **ANY** limitation related to **Novelty** (e.g., lack of originality, incremental work, limited contribution, similar to existing methods, marginal improvements, or lack of surprise).

# **Input Text:**
# "{text_to_analyze}"

# **Output Format:**
# Start with **"YES"** or **"NO"**. Then provide a single sentence explaining why based on the text."""

# def get_novelty_check_prompt(text_to_analyze: str) -> str:
#     return f"""You are analyzing peer-review text that contains limitations/critiques of a paper.

# Your job is to judge how strongly the text raises NOVELTY/ORIGINALITY concerns.

# Novelty concerns include (but are not limited to): "incremental", "minor extension", "similar to prior work", "repackaging", "no new ideas", "not original", "limited contribution", "marginal improvement", "lack of surprise", "already known", "derivative", "unclear contribution vs baseline/prior art".

# INPUT TEXT:
# \"\"\"{text_to_analyze}\"\"\"

# SCORING (0 to 5): Output an integer from 0–5 indicating the DEGREE of novelty concern.

# Rubric:
# - 0 = No novelty concern mentioned at all (only talks about other issues like experiments, clarity, writing, etc.).
# - 1 = Very weak/implicit novelty concern: vague hints like “unclear contribution” or “limited novelty” without comparisons or strong wording.
# - 2 = Mild novelty concern: explicitly says novelty is limited/incremental, but without strong dismissal (may still suggest it’s somewhat useful).
# - 3 = Moderate novelty concern: clear claim of being derivative or close to prior work, OR says contribution is small and not well differentiated; may cite prior work/classes of methods.
# - 4 = Strong novelty concern: asserts it is not novel / mostly known / repackaging; strong language like “little to no novelty” with comparisons to existing methods.
# - 5 = Extreme novelty concern: explicitly says “not novel”, “no originality”, “already done”, “essentially identical to X”, or “no meaningful contribution”; treats novelty as a deal-breaker / primary rejection reason.

# OUTPUT FORMAT (STRICT):
# Return exactly two lines:
# Line 1: SCORE: <0-5>
# Line 2: EVIDENCE: <one sentence quoting or paraphrasing the key phrase(s) from the text that justify the score; if score=0, say 'No novelty-related critique is mentioned.'>

# """

def get_novelty_check_prompt(text_to_analyze: str) -> str:
    return f"""Analyze the text below, which contains peer review limitations or critiques of a paper.

**Your Task:**
1) Determine if the text mentions **ANY** limitation related to **Novelty** (originality) — including direct novelty critiques or clear synonyms/phrases such as: not novel, lack of originality, incremental, minor extension, derivative, similar to prior work, no new ideas, limited contribution, marginal improvement, already known, repackaging, little innovation, insufficient differentiation from prior work.
2) Produce two outputs:
   - **Binary Novelty Flag (0/1):**  
     - **0** = the text contains *any* novelty-related limitation/critique  
     - **1** = the text contains *no* novelty-related limitation/critique
   - **Degree of Novelty Score (0/1/2):** a coarse severity score based on how strongly novelty is criticized:
     - **0** = Direct/explicit novelty criticism (e.g., “not novel”, “lacks originality”, “incremental”, “similar to X”, “derivative”, “no new contribution”).
     - **1** = Strongly novelty-related but phrased indirectly (highly synonymous): clear implication of weak novelty (e.g., “limited contribution”, “unclear what is new”, “minimal innovation”, “insufficient differentiation from prior work”) without outright saying “not novel”.
     - **2** = Mild/moderate novelty-related hints (moderately synonymous): only soft or ambiguous signals that *could* relate to novelty (e.g., “incremental improvements are unclear”, “novelty could be better articulated”, “contribution seems small”) and novelty is not a central critique.

**Important Priority Rule:**
- If **any** novelty-related limitation is mentioned (direct or synonymous), set **Binary Novelty Flag = 0**.
- If **no** novelty-related limitation is mentioned, set **Binary Novelty Flag = 1** and **Degree of Novelty Score = 2** (since there is no novelty critique in the text).

**Input Text:**
"{text_to_analyze}"

**Output Format (STRICT):**
Line 1: YES or NO (YES = novelty limitation is mentioned; NO = not mentioned)
Line 2: BinaryNoveltyFlag: <0 or 1>
Line 3: DegreeNoveltyScore: <0 or 1 or 2>
Line 4: Explanation: <one single sentence explaining your decision based on the text>
"""

# ==========================================
# 4. Main Processing
# ==========================================

print("Loading CSV file...")
try:
    df = pd.read_csv(INPUT_CSV) 
    print(f"Loaded {len(df)} rows from: {INPUT_CSV}")
except FileNotFoundError:
    print(f"CSV not found at: {INPUT_CSV}")
    sys.exit(1)

# Ensure the column exists
if "novelty_limitation_check_mistral" not in df.columns:
    df["novelty_limitation_check_mistral"] = ""

global_df = df 

print(f"Starting Novelty Detection using {MODEL_ID}...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    global_current_row = i
    
    # 1. Retrieve the ground truth text
    target_text = str(row.get('limitations_autho_peer_gt', ''))

    # Skip if empty or too short
    if len(target_text) < 5 or target_text.lower() == 'nan':
        df.at[i, "novelty_limitation_check_mistral"] = "NO (Empty Input)"
        continue
    
    # Check if already processed (resume capability)
    if str(row.get('novelty_limitation_check_mistral', '')).strip() != "" and "emergency" not in OUTPUT_FILE:
         continue

    try:
        # 2. Run Novelty Check
        prompt_content = get_novelty_check_prompt(target_text)
        
        # We pass the persona/strictness as the 'system_instruction'
        response = run_mistral(
            prompt_text=prompt_content, 
            system_instruction="You are a strict classifier for scientific novelty issues."
        )
        
        # 3. Store result
        df.at[i, "novelty_limitation_check_mistral"] = response

    except Exception as e:
        print(f"Error at row {i}: {e}")
        df.at[i, "novelty_limitation_check_mistral"] = f"ERROR: {e}"
        continue

    if i % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False)

# Final Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Final file saved to: {OUTPUT_FILE}")