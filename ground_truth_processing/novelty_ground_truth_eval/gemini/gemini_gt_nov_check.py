import os
import sys
import signal
import time
import pandas as pd
from tqdm import tqdm
import autogen

# ==========================================
# 1. Configuration
# ==========================================

# ✅ GEMINI KEY (use env var on cluster if possible)
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please export it.")

# ✅ Gemini model
MODEL_ID = "gemini-3-flash-preview"

# --- PATHS (same as your Mistral script) ---
INPUT_CSV = "data/final_not_balanced_data/df_not_balanced_preprocessed_gt_final.csv"
OUTPUT_DIR = "data/final_not_balanced_data/novelty_gt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Gemini / AutoGen Setup
# ==========================================

llm_config = {
    "config_list": [
        {
            "model": MODEL_ID,
            "api_key": api_key,
            "api_type": "google",
        }
    ],
    "temperature": 0.1,
    "max_tokens": 256,
    "cache_seed": None,
}

novelty_agent = autogen.AssistantAgent(
    name="Novelty_Classifier",
    llm_config=llm_config,
    system_message=(
        "You are a scientific reviewer. Follow the required output format exactly. "
        "Do not add extra lines."
    ),
)

# ==========================================
# 3. Signal Handling (Graceful Exit)
# ==========================================

global_df = None
global_current_row = 0

def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Saving progress...")
    if global_df is not None:
        save_path = os.path.join(OUTPUT_DIR, f"emergency_novelty_save_gemini_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# 4. Prompt (YOUR UPDATED PROMPT WITH TWO OUTPUTS)
# ==========================================

def get_novelty_check_prompt(text_to_analyze: str) -> str:
    return f"""Analyze the text below, which contains peer review limitations or critiques of a paper.

**Your Task:**
1) Determine if the text mentions **ANY** limitation related to **Novelty** (originality), including synonyms such as:
not novel, lack of originality, incremental, minor extension, derivative, similar to prior work, no new ideas,
limited contribution, marginal improvement, already known, repackaging, little innovation, insufficient differentiation.

2) Produce two outputs:
- **Binary Novelty Flag (0/1):**
  - **0** = novelty limitation IS mentioned anywhere in the text
  - **1** = NO novelty limitation is mentioned

- **Degree of Novelty Score (0/1/2):**
  - **0** = Direct/explicit novelty criticism (e.g., “not novel”, “lacks originality”, “incremental”, “similar to X”).
  - **1** = Indirect but strongly novelty-related (e.g., “limited contribution”, “unclear what is new”, “minimal innovation”).
  - **2** = Mild/ambiguous novelty hint OR no novelty issue at all.

**Priority Rule:**
- If any novelty-related limitation is mentioned (direct OR indirect), set **BinaryNoveltyFlag = 0**.
- If no novelty limitation is mentioned, set **BinaryNoveltyFlag = 1** and **DegreeNoveltyScore = 2**.

**Input Text:**
"{text_to_analyze}"

**Output Format (STRICT):**
Line 1: YES or NO
Line 2: BinaryNoveltyFlag: <0 or 1>
Line 3: DegreeNoveltyScore: <0 or 1 or 2>
Line 4: Explanation: <one single sentence>
"""

# ==========================================
# 5. Gemini Inference Helper
# ==========================================

def run_gemini(prompt: str) -> str:
    reply = novelty_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
    if isinstance(reply, dict):
        return (reply.get("content") or "").strip()
    return str(reply).strip()

# ==========================================
# 6. Main Processing
# ==========================================

print("Loading CSV file...")
try:
    df = pd.read_csv(INPUT_CSV) 
    print(f"Loaded {len(df)} rows from: {INPUT_CSV}")
except FileNotFoundError:
    print(f"CSV not found at: {INPUT_CSV}")
    sys.exit(1)

# Ensure output column exists (same name as your Mistral script, but Gemini)
OUT_COL = "novelty_limitation_check_gemini"
if OUT_COL not in df.columns:
    df[OUT_COL] = ""

global_df = df

print(f"Starting Novelty Detection using Gemini ({MODEL_ID})...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    global_current_row = i

    target_text = str(row.get("limitations_autho_peer_gt", ""))

    # Skip empty / NaN
    if len(target_text) < 5 or target_text.lower() == "nan":
        df.at[i, OUT_COL] = "NO\nBinaryNoveltyFlag: 1\nDegreeNoveltyScore: 2\nExplanation: Empty or missing input text."
        continue

    # Resume: skip already processed
    if str(row.get(OUT_COL, "")).strip() != "":
        continue

    try:
        prompt = get_novelty_check_prompt(target_text)
        response = run_gemini(prompt)
        df.at[i, OUT_COL] = response

    except Exception as e:
        print(f"Error at row {i}: {e}")
        df.at[i, OUT_COL] = f"ERROR: {e}"
        continue

    if i % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False)

    # Small pause to reduce rate-limit risk
    time.sleep(0.1)

# Final save
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Final file saved to: {OUTPUT_FILE}")
