

citation 
import ast
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Prepare citation-only context from df['pdf_text_without_gt'] and df['cited_in'].
For each row, build:
  - paper_content: normalized text from pdf_text_without_gt (abstract + sections)
  - cited_papers_list: list of strings, one per cited paper (abstract + sections)
  - cited_papers_context: cited_papers_list joined with separators for easy prompting
  - citation_limitations: model-generated limitations using the provided prompt

Writes the result to llm_agents/output/zs_mistral_citation_context.csv.
"""

start_time = time.time()

# Global references for signal handling
global_df: Optional[pd.DataFrame] = None
global_current_row = 0
OUTPUT_FILE = (
    "llm_agents/output/"
    "zs_mistral_citation_context.csv"
)

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\n⚠️  Received signal {signum}. Saving progress before termination...")
    if global_df is not None:
        emergency_file = (
            "llm_agents/output/"
            f"zs_mistral_citation_context_emergency_{global_current_row}.csv"
        )
        global_df.to_csv(emergency_file, index=False)
        print(f"  🚨 Emergency save completed: {emergency_file}")

    print("  📊 Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load Mistral model/tokenizer
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

def get_citation_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert scientific research assistant tasked with inferring potential limitations for an unspecified 
current scientific article using both the article's own content and its cited papers.
You are given the article content plus information from multiple cited papers, which are assumed to be referenced by the current article. 
Your goal is to analyze the article together with these cited works and identify possible limitations that the current paper may have, by 
comparing its scope, methods, or results against the cited literature.
If article content is sparse, rely more heavily on cited papers to surface plausible limitations.

Objective:

Generate a list of scientifically grounded limitations that the current article might have, assuming it builds upon or is informed by the provided cited papers.

Each limitation should:

Be concise

Reference the relevant cited paper(s) by title

Clearly explain how the cited paper exposes a potential limitation

Be plausible and insightful based on common scientific reasoning

Workflow:
Plan:
Identify key insights, strengths, and scopes of the cited papers that could set a high bar or reveal blind spots 
in a hypothetical citing article.

Reasoning: Let's think step by step to infer limitations:
Review each cited paper to extract its methodology, findings, and scope.
Ask: If a paper cited this work but did not adopt or address its insights, what limitation might arise?
Identify where the cited paper offers better methodology, broader scope, or contradicting findings.
Formulate each limitation as a plausible shortcoming of a hypothetical article that builds on—but possibly 
underutilizes—these cited works.

Justify each limitation based on specific attributes of the cited paper (e.g., "more comprehensive dataset", 
"stronger evaluation metric", etc.)

Analyze:
Develop a set of inferred limitations, each tied to specific cited paper(s) and grounded in logical comparison.

Reflect:
Ensure coverage of all relevant cited papers and validate that each limitation is scientifically plausible in 
context.

Output Format:
Bullet points listing each limitation.
For each: Description, explanation, and reference to the cited paper(s) in the format Paper Title.

Tool Use (if applicable):

Use citation lookup tools or document content to extract accurate summaries.
Do not assume details about the input paper—focus only on drawing limitations based on differences, omissions, 
or underuse of the cited works.

Chain of Thoughts:
During the Reasoning step, document the thought process explicitly. For example:
"I selected [Paper X] because it uses a more robust method than the current article."
"The current article's simpler method may limit accuracy compared to [Paper X]."
"I reviewed all cited papers to ensure no relevant gaps were missed."
This narrative ensures transparency and justifies each identified limitation.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please identify limitations that would be relevant for researchers who might cite this paper in future work. 
Consider what limitations future authors might mention when discussing this paper's contribution to the field, 
based on the cited papers context.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

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

def cited_in_to_text_list(cited_entry: Any) -> List[str]:
    """
    Convert the cited_in column into a list of text blobs (abstract + sections for each cited paper).
    """
    if pd.isna(cited_entry):
        return []

    parsed: Any = cited_entry
    if isinstance(cited_entry, str):
        try:
            parsed = ast.literal_eval(cited_entry)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not parse cited_in at row {global_current_row}: {exc}")
            return [str(cited_entry)]

    if not isinstance(parsed, dict):
        return [str(parsed)]

    papers: List[str] = []
    for _, val in parsed.items():
        if not isinstance(val, dict):
            continue
        parts: List[str] = []
        abstract = val.get("abstractText") or val.get("abstract") or ""
        if abstract:
            parts.append(f"Abstract: {abstract}")
        sections_text = parse_sections(val.get("sections", []))
        if sections_text:
            parts.append("\n\n".join(sections_text))
        if parts:
            papers.append("\n\n".join(parts))
    return papers

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

# Initialize global reference for signal handler
global_df = df

# Prepare output columns
df["paper_content"] = ""
df["cited_papers_list"] = [[] for _ in range(len(df))]
df["cited_papers_context"] = ""
df["citation_limitations"] = ""

print("Extracting citation contexts...")
for i, row in df.iterrows():
    global_current_row = i + 1

    paper_content = pdf_text_to_article(row.get("pdf_text_without_gt", ""))
    print("paper_content:", paper_content)
    cited_papers_list = cited_in_to_text_list(row.get("cited_in", ""))
    cited_papers_context = "\n\n---\n\n".join(cited_papers_list) 
    print("cited_papers_context:", cited_papers_context)

    df.at[i, "paper_content"] = paper_content
    df.at[i, "cited_papers_list"] = cited_papers_list
    df.at[i, "cited_papers_context"] = cited_papers_context
    try:
        prompt = get_citation_prompt(paper_content, cited_papers_context)
        citation_output = mistral_generate(prompt, max_new_tokens=1024)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Citation generation error at row {i+1}: {exc}")
        citation_output = f"ERROR generating citation-based limitations: {exc}"
    df.at[i, "citation_limitations"] = citation_output

    if i % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1}")

df.to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== Citation context prep complete ===")
print(f"Total samples processed: {len(df)}")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Script started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
