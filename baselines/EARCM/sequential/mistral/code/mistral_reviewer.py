
#reviewer 
import ast
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Run the Reviewer agent with Mistral on pdf_text content.
Reads the balanced dataset, identifies review-style limitations, and saves results to
zs_mistral_reviewer.csv.
"""

start_time = time.time()

# Global variables for graceful shutdown
global_df: Optional[pd.DataFrame] = None
global_current_row = 0

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\n⚠️  Received signal {signum}. Saving progress before termination...")
    if global_df is not None:
        emergency_file = (
            ""
            f"llm_agents/output/zs_mistral_reviewer_emergency_{global_current_row}.csv"
        )
        global_df.to_csv(emergency_file, index=False)
        print(f"  🚨 Emergency save completed: {emergency_file}")

        final_file = (
            ""
            "llm_agents/output/zs_mistral_reviewer.csv"
        )
        global_df.to_csv(final_file, index=False)
        print(f"  🚨 Final output updated: {final_file}")

    print("  📊 Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "models/mistral_7b_v3_instruct"

print("Loading Mistral model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
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

def mistral_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate text using the Mistral model."""
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
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response

def get_reviewer_prompt(paper_content: str) -> str:
    return f"""You are an expert in open peer review with a focus on transparent and critical evaluation of scientific research. 
Your task is to review the provided scientific article from the perspective of an external peer reviewer. Identify potential limitations 
that might be raised in an open review process, considering common critiques such as reproducibility, transparency, generalizability, 
or ethical considerations. Leverage insights from similar studies or common methodological issues in the field.

Workflow:
Plan: Identify areas for review (e.g., reproducibility, transparency, ethics) and plan searches for external context
(e.g., similar studies, methodological critiques). Justify the selection based on peer review standards.

Reasoning: Let's think step by step to identify peer-review limitations:
Step 1: Select key areas for review (e.g., reproducibility, ethics) based on common peer review critiques.
Step 2: Use text analysis tools to extract relevant article details (e.g., methods, data reporting).
Step 3: Identify potential limitations, such as lack of transparency in data or ethical concerns, and justify using article content.
Step 4: Search web/X for external context (e.g., similar studies) to support limitations, rating source relevance 
(high, medium, low, none).
Step 5: Synthesize findings, ensuring limitations align with peer review standards and are supported by the article or external context.

Analyze: Critically review the article, integrating external context to identify limitations. Use tools to verify content and sources.
Reflect: Verify that limitations align with peer review standards and are supported by the article or external context. 
Re-evaluate overlooked areas if necessary.

Output Format:
Bullet points listing each limitation.
For each: Description, why it's a concern, and alignment with peer review standards.

Paper Content:
{paper_content}

Please provide a critical review identifying the limitations and areas of concern in this research. Consider what a peer reviewer would highlight as weaknesses or areas needing improvement."""

def parse_sections(sections: Any) -> List[str]:
    """Extract text from section dictionaries, preserving headings when available."""
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
    """
    Convert a pdf_text entry (often stored as a string) into a single text blob.
    Uses abstractText plus concatenated section text.
    """
    if pd.isna(pdf_entry):
        return ""

    parsed: Any = pdf_entry
    if isinstance(pdf_entry, str):
        try:
            parsed = ast.literal_eval(pdf_entry)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not parse pdf_text at row {global_current_row}: {exc}")
            return pdf_entry

    # Sometimes the parsed object may be a list of dicts; fold them together
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

def run_reviewer_agent(paper_content: str) -> str:
    """Generate reviewer output for a single article."""
    print("  Running Reviewer agent...")
    try:
        prompt = get_reviewer_prompt(paper_content)
        response = mistral_generate(prompt, max_new_tokens=1024)
        print("  Reviewer agent completed")
        return response.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  Error in Reviewer agent: {exc}")
        return f"ERROR in Reviewer agent: {exc}"

print("Loading CSV file...")
try:
    df = pd.read_csv(
        "data/final_gt/"
        "df_balanced_kde_with_pdf_gpt_llama_compared.csv"
    )
    print(f"Successfully loaded CSV file with shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"Error loading CSV file: {exc}")
    sys.exit(1)

global_df = df
df["mistral_reviewer"] = ""

print("Processing samples with Reviewer agent using Mistral...")
for i, row in df.iterrows():
    global_current_row = i + 1

    print(f"\n=== Processing row {i+1}/{len(df)} ===")
    paper_content = pdf_text_to_article(row.get("pdf_text_without_gt", ""))

    reviewer_output = run_reviewer_agent(paper_content)
    df.at[i, "mistral_reviewer"] = reviewer_output
    print(f"  Row {i+1} completed")

    if i % 5 == 0:
        checkpoint_file = (
            ""
            "llm_agents/output/zs_mistral_reviewer.csv"
        )
        df.to_csv(checkpoint_file, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1}")

final_output = (
    ""
    "llm_agents/output/zs_mistral_reviewer.csv"
)
df.to_csv(final_output, index=False)
print(f"\nResults saved to: {final_output}")

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== Reviewer Agent run complete ===")
print(f"Total samples processed: {len(df)}")
print("Agents used: Reviewer (Mistral-7B-Instruct-v0.3)")
print("Output column: mistral_reviewer")
print(f"Script started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")