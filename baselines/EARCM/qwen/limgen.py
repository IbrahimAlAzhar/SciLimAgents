import ast
import os
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_ID = "qwen2_5_3b_instruct"
CACHE_DIR = MODEL_ID  # Local path, no separate cache needed

INPUT_CSV = "df_updated_with_retrieval.csv"
OUTPUT_DIR = "Qwen/EARCLM"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df_qwen_agents_merged_200.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.time()

# Global variables for graceful shutdown
global_df: Optional[pd.DataFrame] = None
global_current_row = 0

# ==========================================
# SIGNAL HANDLING
# ==========================================

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\n⚠️  Received signal {signum}. Saving progress before termination...")
    if global_df is not None:
        emergency_file = os.path.join(
            OUTPUT_DIR, f"emergency_save_qwen_{global_current_row}.csv"
        )
        global_df.to_csv(emergency_file, index=False)
        print(f"  🚨 Emergency save completed: {emergency_file}")

        # Also try to save to main output
        global_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  Final output updated: {OUTPUT_FILE}")

    print("  📊 Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# MODEL LOADING
# ==========================================

print("Loading Qwen 2.5 3B Instruct model and tokenizer...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

# Set pad token for Qwen if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully.")

# ==========================================
# GENERATION HELPERS
# ==========================================

def truncate_prompt_for_model(prompt: str, max_length: int = 8000) -> str:
    """Truncate prompt to fit within context window (conservative 8k limit for input)."""
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(f"  ⚠️ Prompt token count = {tokens.shape[1]} exceeds limit ({max_length}). Truncating...")
        truncated_tokens = tokens[:, :max_length]
        return tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt

def qwen_generate(prompt: str, max_new_tokens: int = 1024, system_prompt: str = None) -> str:
    """Generate text using Qwen 2.5 3B Instruct."""
    truncated_prompt = truncate_prompt_for_model(prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": truncated_prompt})

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
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return response

# ==========================================
# PROMPTS
# ==========================================

def get_extractor_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are an expert in scientific literature analysis. Your task is to carefully read the provided scientific article 
and extract all explicitly stated limitations as mentioned by the authors. Focus on sections such as Discussion, Conclusion, or 
Limitations. List each limitation verbatim, including direct quotes where possible.

Output Format:
Bullet points listing each limitation.
For each: Verbatim quote (if available), context (e.g., aspect of the study), and section reference.
If none: "No limitations explicitly stated in the article."

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please extract and list the key limitations found in this paper."""

def get_analyzer_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are a critical scientific reviewer with expertise in research methodology and analysis. Your task is to analyze 
the provided scientific article and identify potential limitations not explicitly stated by the authors. Focus on aspects such as study 
design, sample size, data collection methods, statistical analysis, scope of findings, and underlying assumptions.

Output Format:
Bullet points listing each inferred limitation.
For each: Description, explanation, and impact on the study.

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please provide a detailed analysis of the limitations in this research."""

def get_reviewer_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are an expert in open peer review. Your task is to review the provided scientific article from the perspective 
of an external peer reviewer. Identify potential limitations that might be raised in an open review process, considering common 
critiques such as reproducibility, transparency, generalizability, or ethical considerations.

Output Format:
Bullet points listing each limitation.
For each: Description, why it's a concern, and alignment with peer review standards.

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please provide a critical review identifying the limitations and areas of concern in this research."""

def get_citation_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are an expert scientific research assistant tasked with inferring potential limitations for a current scientific article 
using both the article's own content and its cited papers. Your goal is to analyze the article together with these cited works 
and identify possible limitations that the current paper may have, by comparing its scope, methods, or results against the cited literature.

Objective:
Generate a list of scientifically grounded limitations that the current article might have, assuming it builds upon or is informed by the provided cited papers.

Workflow:
1. Review each cited paper to extract its methodology, findings, and scope.
2. Ask: If a paper cited this work but did not adopt or address its insights, what limitation might arise?
3. Formulate each limitation as a plausible shortcoming of the current article.

Output Format:
Bullet points listing each limitation.
For each: Description, explanation, and reference to the cited paper(s) in the format Paper Title.

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please identify limitations based on the citation context."""

def get_merger_prompt(extractor_out: str, analyzer_out: str, reviewer_out: str, citation_out: str) -> str:
    return f"""You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by four specialized agents: 

**Agents:**
1. **Extractor** (explicit limitations from the article).
2. **Analyzer** (inferred limitations from critical analysis).
3. **Reviewer** (limitations from an open review perspective).
4. **Citation** (limitations inferred from cited papers context).

**Goals**:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Format the final list in a clear, concise, and professional manner.

**Output Format**:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived], [Citation-context]).

Extractor Agent Analysis:
{extractor_out}

Analyzer Agent Analysis:
{analyzer_out}

Reviewer Agent Analysis:
{reviewer_out}

Citation Agent Analysis:
{citation_out}

Please merge these four different perspectives into a comprehensive, well-organized analysis."""

# ==========================================
# DATA PROCESSING HELPERS
# ==========================================

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
    """Convert a pdf_text entry into a single text blob."""
    if pd.isna(pdf_entry):
        return ""

    parsed: Any = pdf_entry
    if isinstance(pdf_entry, str):
        try:
            parsed = ast.literal_eval(pdf_entry)
        except Exception as exc:
            print(f"  ⚠️ Could not parse pdf_text: {exc}")
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

def cited_in_to_text_list(cited_entry: Any) -> List[str]:
    """Convert the cited_in column into a list of text blobs.

    For each cited_in_paper_N key, extracts:
      - abstractText
      - The Introduction section (heading containing 'Introduction')
    """
    if pd.isna(cited_entry):
        return []

    parsed: Any = cited_entry
    if isinstance(cited_entry, str):
        try:
            parsed = ast.literal_eval(cited_entry)
        except Exception as exc:
            print(f"  ⚠️ Could not parse cited_in: {exc}")
            return [str(cited_entry)]

    if not isinstance(parsed, dict):
        return [str(parsed)]

    papers: List[str] = []
    for key, val in parsed.items():
        # Only process keys like cited_in_paper_1, cited_in_paper_2, ...
        if not key.startswith("cited_in_paper_"):
            continue
        if not isinstance(val, dict):
            continue

        parts: List[str] = []

        # 1. Extract abstract
        abstract = val.get("abstractText") or val.get("abstract") or ""
        if abstract:
            parts.append(f"Abstract: {abstract}")

        # 2. Extract Introduction section (heading must contain 'Introduction')
        sections = val.get("sections", [])
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    heading = section.get("heading", "") or ""
                    if "introduction" in heading.lower():
                        intro_text = section.get("text", "").strip()
                        if intro_text:
                            parts.append(f"Introduction: {intro_text}")
                        break  # Only take the first matching Introduction section

        if parts:
            title = val.get("title", key)  # Use paper title if available, else key
            papers.append(f"[{title}]\n\n" + "\n\n".join(parts))

    return papers

# ==========================================
# MAIN EXECUTION
# ==========================================

print("Loading CSV file...")
try:
    df = pd.read_csv(INPUT_CSV) 
    print(f"Successfully loaded CSV file with shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    sys.exit(1)

# Initialize global reference for signal handler
global_df = df

# Initialize columns
df["qwen_extractor"] = ""
df["qwen_analyzer"] = ""
df["qwen_reviewer"] = ""
df["qwen_citation"] = ""
df["merged_limitations"] = ""
df["paper_content_processed"] = ""
df["cited_papers_context"] = ""

SAMPLES_TO_RUN = 200
print(f"Processing first {SAMPLES_TO_RUN} samples with 4 Agents + Merger using Qwen 2.5 3B...")

for i in range(min(SAMPLES_TO_RUN, len(df))):
    global_current_row = i + 1
    print(f"\n=== Processing row {i+1}/{SAMPLES_TO_RUN} ===")

    row = df.iloc[i]

    # 1. Prepare Content — read from 'input_text_cleaned' column
    paper_content = str(row.get("input_text_cleaned", "")) if not pd.isna(row.get("input_text_cleaned", "")) else ""
    # print('paper content',paper_content)
    cited_papers_list = cited_in_to_text_list(row.get("cited_in", ""))
    cited_papers_context = "\n\n---\n\n".join(cited_papers_list) 
    # print('cited paper context', cited_papers_context)

    # Store processed context for reference
    df.at[i, "paper_content_processed"] = paper_content
    df.at[i, "cited_papers_context"] = cited_papers_context

    # 2. Run Individual Agents
    print("  Running Extractor...")
    try:
        ext_prompt = get_extractor_prompt(paper_content, cited_papers_context)
        df.at[i, "qwen_extractor"] = qwen_generate(ext_prompt)
    except Exception as e:
        print(f"    Error: {e}")
        df.at[i, "qwen_extractor"] = f"Error: {e}"

    print("  Running Analyzer...")
    try:
        ana_prompt = get_analyzer_prompt(paper_content, cited_papers_context)
        df.at[i, "qwen_analyzer"] = qwen_generate(ana_prompt)
    except Exception as e:
        print(f"    Error: {e}")
        df.at[i, "qwen_analyzer"] = f"Error: {e}"

    print("  Running Reviewer...")
    try:
        rev_prompt = get_reviewer_prompt(paper_content, cited_papers_context)
        df.at[i, "qwen_reviewer"] = qwen_generate(rev_prompt)
    except Exception as e:
        print(f"    Error: {e}")
        df.at[i, "qwen_reviewer"] = f"Error: {e}"

    print("  Running Citation Agent...")
    try:
        cit_prompt = get_citation_prompt(paper_content, cited_papers_context)
        df.at[i, "qwen_citation"] = qwen_generate(cit_prompt)
    except Exception as e:
        print(f"    Error: {e}")
        df.at[i, "qwen_citation"] = f"Error: {e}"

    # 3. Run Merger
    print("  Running Master Coordinator (Merger)...")
    try:
        merger_prompt = get_merger_prompt(
            df.at[i, "qwen_extractor"],
            df.at[i, "qwen_analyzer"],
            df.at[i, "qwen_reviewer"],
            df.at[i, "qwen_citation"]
        )
        df.at[i, "merged_limitations"] = qwen_generate(merger_prompt)
        print("  ✅ Row completed.")
    except Exception as e:
        print(f"    Error in Merger: {e}")
        df.at[i, "merged_limitations"] = f"Error: {e}"

    if (i + 1) % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  💾 Checkpoint saved at row {i+1}")

# Final Save
df.to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== Qwen 2.5 3B Multi-Agent Analysis Complete ===")
print(f"Total samples processed: {min(SAMPLES_TO_RUN, len(df))}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")