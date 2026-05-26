import signal
import sys
import time
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Merge limitations from Extractor, Analyzer, Reviewer, and Citation agents using a merger prompt.
Generates 4 variations using different temperatures for GRPO/DPO training.
Reads:
  - llm_agents/output/zs_mistral_extractor.csv
  - llm_agents/output/zs_mistral_analyzer.csv
  - llm_agents/output/zs_mistral_reviewer.csv
  - llm_agents/output/zs_mistral_citation_context.csv
Writes merged results to llm_agents/output/zs_mistral_master.csv with multiple columns.
"""

start_time = time.time()

# --- CONFIGURATION ---
# We will generate 4 responses per paper using these 4 distinct temperatures
TEMPERATURES = [0.4, 0.6, 0.8, 1.0]

# Global references for signal handling
global_df: Optional[pd.DataFrame] = None
global_current_row = 0

OUTPUT_FILE = (
    "llm_agents/output/"
    "zs_mistral_master.csv"
)
EXTRACTOR_FILE = (
    "llm_agents/output/"
    "zs_mistral_extractor.csv"
)
ANALYZER_FILE = (
    "llm_agents/output/"
    "zs_mistral_analyzer.csv"
)
REVIEWER_FILE = (
    "llm_agents/output/"
    "zs_mistral_reviewer.csv"
)
CITATION_FILE = (
    "llm_agents/output/"
    "zs_mistral_citation_context.csv"
)

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\n⚠️  Received signal {signum}. Saving progress before termination...")
    if global_df is not None:
        emergency_file = (
            "llm_agents/output/"
            f"zs_mistral_master_emergency_{global_current_row}.csv"
        )
        global_df.to_csv(emergency_file, index=False)
        print(f"  🚨 Emergency save completed: {emergency_file}")

        global_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  🚨 Final output updated: {OUTPUT_FILE}")

    print("  📊 Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load merger model/tokenizer (Mistral)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "models/mistral_7b_v3_instruct"

print("Loading Mistral model and tokenizer for merger...")
merger_tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, trust_remote_code=True
)
merger_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
if merger_tokenizer.pad_token is None:
    merger_tokenizer.pad_token = merger_tokenizer.eos_token

def truncate_prompt_for_model(prompt: str, max_length: int = 32000) -> str:
    """Truncate prompt to fit within model's context window."""
    tokens = merger_tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(
            f"Prompt token count = {tokens.shape[1]} exceeds limit ({max_length}). Truncating..."
        )
        truncated_tokens = tokens[:, :max_length]
        return merger_tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt

def merger_generate(prompt: str, temperature: float, max_new_tokens: int = 1024) -> str:
    """Generate merged limitations using the merger LLM with a specific temperature."""
    truncated_prompt = truncate_prompt_for_model(prompt, max_length=32000)
    messages = [{"role": "user", "content": truncated_prompt}]

    inputs = merger_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(merger_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = merger_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,  # Dynamic temperature
            pad_token_id=merger_tokenizer.eos_token_id,
            eos_token_id=merger_tokenizer.eos_token_id,
        )

    response = merger_tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response

def get_merger_prompt(
    extractor_output: str,
    analyzer_output: str,
    reviewer_output: str,
    citation_output: str,
) -> str:
    return f"""You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by four specialized agents: 

**Agents:**
1. **Extractor** (explicit limitations from the article)
2. **Analyzer** (inferred limitations from critical analysis)
3. **Reviewer** (limitations from an open review perspective)
4. **Citation** (limitations inferred from the cited literature)

**Goals**:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content and cited works.
3. Prioritize critical limitations that affect the paper's validity and reproducibility.
4. Format the final list in a clear, concise, and professional manner, suitable for a scientific review or report.

**Workflow**:
1. **Plan**: Outline how to synthesize limitations, identify potential redundancies, and resolve discrepancies.
2. **Analyze**: Combine limitations, prioritizing critical ones, and verify alignment with the article and cited works.
3. **Reflect**: Check for completeness, scientific rigor, and clarity.
4. **Continue**: Iterate until the list is comprehensive, non-redundant, and professionally formatted.

**Output Format**:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived], [Citation-based]).

Extractor Agent Analysis:
{extractor_output}

Analyzer Agent Analysis:
{analyzer_output}

Reviewer Agent Analysis:
{reviewer_output}

Citation Agent Analysis:
{citation_output}

Please merge these four perspectives on the paper's limitations into a comprehensive, well-organized analysis. Synthesize the insights, resolve any contradictions, and provide a unified view of the paper's limitations. Don't miss any unique limitations."""

def load_agent_outputs() -> pd.DataFrame:
    """Load agent CSVs and align on submission."""
    base_df = pd.read_csv(EXTRACTOR_FILE)

    analyzer_df = pd.read_csv(ANALYZER_FILE, usecols=["submission", "mistral_analyzer"])
    reviewer_df = pd.read_csv(REVIEWER_FILE, usecols=["submission", "mistral_reviewer"])
    citation_df = pd.read_csv(
        CITATION_FILE, usecols=["submission", "citation_limitations"]
    ).rename(columns={"citation_limitations": "mistral_citation"})

    merged = base_df.merge(analyzer_df, on="submission", how="left", suffixes=("", "_an"))
    merged = merged.merge(reviewer_df, on="submission", how="left", suffixes=("", "_rev"))
    merged = merged.merge(citation_df, on="submission", how="left", suffixes=("", "_cit"))
    return merged

print("Loading agent outputs...")
try:
    df = load_agent_outputs()
    print(f"Loaded merged dataframe with shape: {df.shape}")
except FileNotFoundError as exc:
    print(f"Error: {exc}")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"Error loading agent outputs: {exc}")
    sys.exit(1)

global_df = df

# Initialize columns for each temperature
for temp in TEMPERATURES:
    col_name = f"mistral_master_{temp}"
    df[col_name] = ""

print(f"Merging limitations using the Master model with temperatures: {TEMPERATURES}...")

    global_current_row = i + 1

    extractor_output = row.get("mistral_extractor", "") or "" 
    print("extractor_output:", extractor_output)
    analyzer_output = row.get("mistral_analyzer", "") or ""
    print("analyzer_output:", analyzer_output)
    reviewer_output = row.get("mistral_reviewer", "") or "" 
    print("reviewer_output:", reviewer_output)
    citation_output = row.get("mistral_citation", "") or "" 
    print("citation_output:", citation_output)

    prompt = get_merger_prompt(
        str(extractor_output), str(analyzer_output), str(reviewer_output), str(citation_output)
    )
    
    # Generate response for each temperature
    for temp in TEMPERATURES:
        try:
            merged_output = merger_generate(prompt, temperature=temp, max_new_tokens=1024)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Merger generation error at row {i+1} (Temp {temp}): {exc}")
            merged_output = f"ERROR generating merged limitations: {exc}"
        
        # Save to the specific temperature column
        df.at[i, f"mistral_master_{temp}"] = merged_output

    if i % 5 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1}")

df.to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
elapsed = end_time - start_time
print("\n=== Master agent merge complete ===")
print(f"Total samples processed: {len(df)}")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Script started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")