import ast
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Run the Extractor agent with Mistral on pdf_text content with multiple temperatures.
Reads the dataset, RENAMES the old 0.4 column, and generates new outputs for 0.6, 0.8, 1.
"""

start_time = time.time()

# --- Configuration ---
FILE_PATH = "llm_agents/output/zs_mistral_ext_analy_rev_cit.csv"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"
TEMPERATURES = [0.6, 0.8, 1]

# Global variables for graceful shutdown
global_df: Optional[pd.DataFrame] = None
global_current_row = 0

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\n⚠️  Received signal {signum}. Saving progress before termination...")
    if global_df is not None:
        base_path = FILE_PATH.replace(".csv", "")
        emergency_file = f"{base_path}_emergency_row_{global_current_row}.csv"
        
        global_df.to_csv(emergency_file, index=False)
        print(f"  🚨 Emergency save completed: {emergency_file}")

        global_df.to_csv(FILE_PATH, index=False)
        print(f"  🚨 Final output updated: {FILE_PATH}")

    print("  📊 Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load Mistral model and tokenizer
print("Loading Mistral model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def truncate_prompt_for_model(prompt: str, max_length: int = 32000) -> str:
    """Truncate prompt to fit within the model context window."""
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(
            f"⚠️  Prompt token count = {tokens.shape[1]} exceeds limit "
            f"({max_length}). Truncating..."
        )
        truncated_tokens = tokens[:, :max_length]
        return tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt

def mistral_generate(prompt: str, temperature: float, max_new_tokens: int = 1024) -> str:
    """Generate text using the Mistral model with specific temperature."""
    truncated_prompt = truncate_prompt_for_model(prompt, max_length=32000)
    messages = [{"role": "user", "content": truncated_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response

def get_extractor_prompt(paper_content: str) -> str:
    return f"""You are an expert in scientific literature analysis. Your task is to carefully read the provided scientific article 
and extract all explicitly stated limitations as mentioned by the authors. Focus on sections such as Discussion, Conclusion, or 
Limitations. List each limitation verbatim, including direct quotes where possible, and provide a brief context (e.g., what aspect of 
the study the limitation pertains to). Ensure accuracy and avoid inferring or adding limitations not explicitly stated. If no limitations 
are mentioned, state this clearly.

Workflow:
Plan: Outline which sections (e.g., Discussion, Conclusion, Limitations) to analyze and identify tools (e.g., text extraction) to 
access the article content. Justify the selection of sections based on their likelihood of containing limitation statements.

Reasoning: Let's think step by step to ensure thorough and accurate extraction of limitations:
Step 1: Identify all sections in the article that may contain limitations. For example, the Discussion often includes limitations as 
authors reflect on their findings, while a dedicated Limitations section is explicit.
Step 2: Use text extraction tools to retrieve content from these sections. Verify that the content is complete and accurate.
Step 3: Scan for explicit limitation statements, such as phrases like "a limitation of this study" or "we acknowledge that." 
Document why each statement qualifies as a limitation.
Step 4: For each identified limitation, extract the verbatim quote (if available) and note the context (e.g., related to sample size, 
methodology).
Step 5: Check for completeness by reviewing other potential sections (e.g., Conclusion) to ensure no limitations are missed.

Analyze: Use tools to extract and verify the article's content, focusing on explicit limitation statements. Cross-reference extracted 
quotes with the original text to ensure accuracy.

Reflect: Verify that all relevant sections were checked and no limitations were missed. Consider whether any section might have been 
overlooked and re-evaluate if necessary.

Output Format:
Bullet points listing each limitation.
For each: Verbatim quote (if available), context (e.g., aspect of the study), and section reference.
If none: "No limitations explicitly stated in the article."

Paper Content:
{paper_content}

Please extract and list the key limitations found in this paper. Be specific and provide clear reasoning for each limitation identified."""

def parse_sections(sections: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                heading = section.get("heading")
                text = section.get("text", "")
                if heading:
                    texts.append(f"{heading}: {text}")
                else:
                    texts.append(text)
            else:
                texts.append(str(section))
    return [t for t in texts if t]

def pdf_text_to_article(pdf_entry: Any) -> str:
    if pd.isna(pdf_entry):
        return ""

    parsed: Any = pdf_entry
    if isinstance(pdf_entry, str):
        try:
            parsed = ast.literal_eval(pdf_entry)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not parse pdf_text at row {global_current_row}: {exc}")
            return pdf_entry

    if isinstance(parsed, list):
        combined_parts: List[str] = []
        for item in parsed:
            combined_parts.append(pdf_text_to_article(item))
        return "\n\n".join([part for part in combined_parts if part])

    if isinstance(parsed, dict):
        parts: List[str] = []
        abstract = parsed.get("abstractText") or parsed.get("abstract") or ""
        if abstract:
            parts.append(f"Abstract: {abstract}")
        sections = parsed.get("sections", [])
        section_texts = parse_sections(sections)
        if section_texts:
            parts.append("\n\n".join(section_texts))
        return "\n\n".join([part for part in parts if part])

    return str(parsed)

def run_extractor_agent(paper_content: str, temperature: float) -> str:
    """Generate extractor output for a single article at a specific temperature."""
    print(f"    -> Running Extractor agent (Temp: {temperature})...")
    try:
        prompt = get_extractor_prompt(paper_content)
        response = mistral_generate(prompt, temperature=temperature, max_new_tokens=1024)
        return response.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"    Error in Extractor agent: {exc}")
        return f"ERROR in Extractor agent: {exc}"

print(f"Loading CSV file from: {FILE_PATH}")
try:
    df = pd.read_csv(FILE_PATH) 
    print(f"Successfully loaded CSV file with shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"Error loading CSV file: {exc}")
    sys.exit(1)

# --- RENAMING LOGIC ---
if "mistral_extractor" in df.columns:
    print("🔄 Renaming 'mistral_extractor' column to 'mistral_extractor_0.4'...")
    df.rename(columns={"mistral_extractor": "mistral_extractor_0.4"}, inplace=True)
else:
    print("ℹ️ Column 'mistral_extractor' not found. Skipping rename.")

global_df = df

# Initialize new columns for temperatures 0.6, 0.8, 1
for temp in TEMPERATURES:
    col_name = f"mistral_extractor_{temp}"
    if col_name not in df.columns:
        df[col_name] = ""

print("Processing samples with Extractor agent using Mistral...")
for i, row in df.iterrows():
    global_current_row = i + 1

    print(f"\n=== Processing row {i+1}/{len(df)} ===")
    
    # Use 'pdf_text_without_gt' if available, otherwise fallback
    content_col = "pdf_text_without_gt"
    if content_col not in df.columns and "pdf_text" in df.columns:
        content_col = "pdf_text"
        
    paper_content = pdf_text_to_article(row.get(content_col, ""))

    # Loop through the requested temperatures
    for temp in TEMPERATURES:
        col_name = f"mistral_extractor_{temp}"
        
        # Check if already processed to save time (optional)
        if pd.notna(df.at[i, col_name]) and str(df.at[i, col_name]).strip() != "":
             print(f"    Skipping Temp {temp} (already exists)")
             continue

        extractor_output = run_extractor_agent(paper_content, temperature=temp)
        df.at[i, col_name] = extractor_output
    
    print(f"  Row {i+1} completed all temperatures")

    # Checkpoint logic
    if i % 5 == 0:
        df.to_csv(FILE_PATH, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1} to {FILE_PATH}")

# Final Save
df.to_csv(FILE_PATH, index=False)
print(f"\nResults saved to: {FILE_PATH}")

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== Extractor Agent run complete ===")
print(f"Total samples processed: {len(df)}")
print(f"Temperatures used: {TEMPERATURES}")
print("Output columns: mistral_extractor_0.6, mistral_extractor_0.8, mistral_extractor_1")
print(f"Script started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")