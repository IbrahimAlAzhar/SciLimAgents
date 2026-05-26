import ast
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Run Extractor, Analyzer, Reviewer, and Master Coordinator agents.
Reads df_balanced_kde_with_pdf_gpt_llama_compared.csv and writes outputs to
llm_agents/output/df_final_gt_mistral_agents_zs.csv.
"""

start_time = time.time()

# Global variables for graceful shutdown
global_df: Optional[pd.DataFrame] = None
global_current_row = 0

def signal_handler(signum, frame):
    """Handle termination signals to save progress before exiting."""
    print(f"\nReceived signal {signum}. Saving progress before termination...")
    if global_df is not None:
        emergency_file = (
            "data/final_gt/"
            f"emergency_save_mistral_{global_current_row}.csv"
        )
        global_df.to_csv(emergency_file, index=False)
        print(f"  Emergency save completed: {emergency_file}")

        output_file = (
            "llm_agents/output"
            "df_final_gt_mistral_agents_zs.csv"
        )
        global_df.to_csv(output_file, index=False)
        print(f"  Final output updated: {output_file}")

    print("  Progress saved. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load primary Mistral model/tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "models/mistral_7b_v3_instruct"
# Configure a separate model for merging (set to a different model_id/cache_dir to switch)
merger_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
merger_cache_dir = "models/mistral_7b_v3_instruct"

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

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Load merger model/tokenizer (reuse primary if identical to avoid double load)
if merger_model_id == model_id and merger_cache_dir == cache_dir:
    print("Merger model reuses the primary Mistral weights.")
    merger_tokenizer = tokenizer
    merger_model = model
else:
    print("Loading merger model and tokenizer...")
    merger_tokenizer = AutoTokenizer.from_pretrained(
        merger_model_id, cache_dir=merger_cache_dir, trust_remote_code=True
    )
    merger_model = AutoModelForCausalLM.from_pretrained(
        merger_model_id,
        cache_dir=merger_cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if merger_tokenizer.pad_token is None:
        merger_tokenizer.pad_token = merger_tokenizer.eos_token

def truncate_prompt_for_model(prompt: str, tokenizer_obj: AutoTokenizer, max_length: int = 32000) -> str:
    """Truncate prompt to fit within model's context window, leaving room for generation."""
    tokens = tokenizer_obj.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(f"Prompt token count = {tokens.shape[1]} exceeds limit ({max_length}). Truncating...")
        truncated_tokens = tokens[:, :max_length]
        return tokenizer_obj.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt

def mistral_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate text using Mistral model."""
    truncated_prompt = truncate_prompt_for_model(prompt, tokenizer, max_length=32000)
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

def merger_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate merged limitations using the (potentially different) merger LLM."""
    truncated_prompt = truncate_prompt_for_model(prompt, merger_tokenizer, max_length=32000)
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
            temperature=0.4,
            pad_token_id=merger_tokenizer.eos_token_id,
            eos_token_id=merger_tokenizer.eos_token_id,
        )

    response = merger_tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response

# Agent-specific prompts - each receives both paper_content and cited_papers
def get_extractor_prompt(paper_content: str, cited_papers: str) -> str:
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

Cited Papers Information:
{cited_papers}

Please extract and list the key limitations found in this paper. Be specific and provide clear reasoning for each limitation identified."""

def get_analyzer_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are a critical scientific reviewer with expertise in research methodology and analysis. Your task is to analyze 
the provided scientific article and identify potential limitations not explicitly stated by the authors. Focus on aspects such as study 
design, sample size, data collection methods, statistical analysis, scope of findings, and underlying assumptions. For each inferred 
limitation, provide a clear explanation of why it is a limitation and how it impacts the study's validity, reliability, 
or generalizability. Ensure inferences are grounded in the article's content and avoid speculative assumptions.

Workflow:
Plan: Identify key areas (e.g., methodology, sample size, statistical analysis) to analyze and select tools (e.g., text analysis) to 
verify article details. Justify the selection based on their potential to reveal limitations.

Reasoning: Let's think step by step to identify inferred limitations:
Step 1: Review the article's methodology to identify gaps (e.g., study design flaws, sampling issues).
Step 2: Use text analysis tools to extract relevant details (e.g., sample size, statistical methods).
Step 3: Evaluate each area for potential limitations, such as small sample size affecting generalizability or unaddressed assumptions.
Step 4: Document why each gap qualifies as a limitation and its impact on the study.
Step 5: Ensure all key areas are covered to avoid missing potential limitations.

Analyze: Critically evaluate the article, using tools to confirm content, and infer limitations based on methodological or analytical gaps.
Reflect: Assess whether inferred limitations are grounded in the article and relevant to its validity, reliability, or generalizability. 
Re-evaluate overlooked areas if necessary.

Output Format:
Bullet points listing each inferred limitation.
For each: Description, explanation, and impact on the study.

Paper Content:
{paper_content}

Cited Papers Information:
{cited_papers}

Please provide a detailed analysis of the limitations in this research. Consider both obvious and subtle limitations that could affect the validity and applicability of the findings."""

def get_reviewer_prompt(paper_content: str, cited_papers: str) -> str:
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

Cited Papers Information:
{cited_papers}

Please provide a critical review identifying the limitations and areas of concern in this research. Consider what a peer reviewer would highlight as weaknesses or areas needing improvement."""

def get_merger_prompt(
    extractor_output: str, analyzer_output: str, reviewer_output: str
) -> str:
    return f"""You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by three specialized agents: 

**Agents:**
1. **Extractor** (explicit limitations from the article), 
2. **Analyzer** (inferred limitations from critical analysis), 
3. **Reviewer** (limitations from an open review perspective).

**Goals**:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Prioritize critical limitations that affect the paper's validity and reproducibility.
4. Format the final list in a clear, concise, and professional manner, suitable for a scientific review or report.

**Workflow**:
1. **Plan**: Outline how to synthesize limitations, identify potential redundancies, and resolve discrepancies.
2. **Analyze**: Combine limitations, prioritizing critical ones, and verify alignment with the article.
3. **Reflect**: Check for completeness, scientific rigor, and clarity.
4. **Continue**: Iterate until the list is comprehensive, non-redundant, and professionally formatted.

**Output Format**:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived]).

Extractor Agent Analysis:
{extractor_output}

Analyzer Agent Analysis:
{analyzer_output}

Reviewer Agent Analysis:
{reviewer_output}

Please merge these three different perspectives on the paper's limitations into a comprehensive, well-organized analysis. Synthesize the insights, resolve any contradictions, and provide a unified view of the paper's limitations and Don't miss any unique limitations."""

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
            print(f"Could not parse pdf_text at row {global_current_row}: {exc}")
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

def extract_intro_from_sections(sections: Any) -> str:
    """Pull the Introduction text (best-effort) from a sections list/dict."""
    if not sections:
        return ""
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                heading = (section.get("heading") or "").lower()
                if "intro" in heading:
                    return section.get("text", "")
        # fallback to first text if no explicit intro
        for section in sections:
            if isinstance(section, dict):
                text = section.get("text")
                if text:
                    return text
    if isinstance(sections, dict):
        return extract_intro_from_sections(list(sections.values()))
    return ""

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
    for key, val in parsed.items():
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

def format_cited_in(cited_entry: Any) -> str:
    """
    Convert the cited_in column (stringified dict) into text with abstract/introduction snippets.
    """
    papers = cited_in_to_text_list(cited_entry)
    return "\n\n---\n\n".join(papers)

# Function to run a specific agent
def run_agent(agent_name: str, paper_content: str, cited_papers: str, agent_prompt_func) -> str:
    """Run a specific agent and return its output."""
    print(f"  Running {agent_name} agent...")
    try:
        prompt = agent_prompt_func(paper_content, cited_papers)
        response = mistral_generate(prompt, max_new_tokens=1024)
        print(f"  {agent_name} agent completed")
        return response.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  Error in {agent_name} agent: {exc}")
        return f"ERROR in {agent_name} agent: {exc}"

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

# Initialize new columns for each agent
df["extractor"] = ""
df["analyzer"] = ""
df["reviewer"] = ""
df["merged_limitations"] = ""
df["cited_papers_context"] = ""
df["cited_papers_list"] = [[] for _ in range(len(df))]

print("Processing all samples with three agents using Mistral...")
for i in range(1): # len(df)
    print("i is", i)
    global_current_row = i + 1  # Update global counter

    print(f"\n=== Processing row {i+1}/{len(df)} ===")
    row = df.iloc[i]
    paper_content = pdf_text_to_article(row.get("pdf_text_without_gt", ""))

    # Build cited_papers context from cited_in column (abstract + introduction snippets)
    cited_papers_list = cited_in_to_text_list(row.get("cited_in", ""))
    cited_papers = "\n\n---\n\n".join(cited_papers_list)
    df.at[i, "cited_papers_context"] = cited_papers
    df.at[i, "cited_papers_list"] = cited_papers_list
    print("cited_papers", cited_papers)
    # Run all three agents
    extractor_output = run_agent("Extractor", paper_content, cited_papers, get_extractor_prompt)
    analyzer_output = run_agent("Analyzer", paper_content, cited_papers, get_analyzer_prompt)
    reviewer_output = run_agent("Reviewer", paper_content, cited_papers, get_reviewer_prompt)

    # Store individual agent outputs
    df.at[i, "extractor"] = extractor_output
    df.at[i, "analyzer"] = analyzer_output
    df.at[i, "reviewer"] = reviewer_output

    # Merge agent outputs with Master Coordinator
    print("  Running Master Coordinator agent...")
    try:
        merger_prompt = get_merger_prompt(extractor_output, analyzer_output, reviewer_output)
        merged_output = merger_generate(merger_prompt, max_new_tokens=1024)
        df.at[i, "merged_limitations"] = merged_output.strip()
        print("  Master Coordinator agent completed")
    except Exception as exc:  # noqa: BLE001
        print(f"  Error in Master Coordinator agent: {exc}")
        df.at[i, "merged_limitations"] = f"ERROR in Master Coordinator agent: {exc}"

    print(f"  Row {i+1} completed")

    if i % 5 == 0:
        output_file = (
            "llm_agents/output/"
            "df_final_gt_mistral_agents_zs.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1}")

# Save results
output_file = (
    "llm_agents/output/"
    "df_final_gt_mistral_agents_zs.csv"
)
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Print summary
print(f"\n=== Three-Agent Limitation Analysis with Mistral Complete ===")
print(f"Total samples processed: {len(df)}")
print("Agents used: Extractor, Analyzer, Reviewer, Master Coordinator")
print(f"Model: {model_id}")
print(f"Merger model: {merger_model_id}")
print("Output columns: extractor, analyzer, reviewer, merged_limitations")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nScript started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
