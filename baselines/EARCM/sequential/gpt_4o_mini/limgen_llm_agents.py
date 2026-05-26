import ast
import os
import signal
import sys
import time
from typing import Any, List, Optional
import re 
import pandas as pd
from openai import OpenAI
import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = ''

# os.environ['OPENAI_API_KEY'] = ''
# 1. Setup OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-mini"

INPUT_CSV = "df_balanced_kde_final.csv"
OUTPUT_DIR = "gpt_4o_mini"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df_gpt4omini_ext_analy_rev_cit.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize OpenAI Client

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
            OUTPUT_DIR, f"emergency_save_gpt_{global_current_row}.csv"
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
# CLEANING HELPER
# ==========================================

def clean_text_detailed(text: Any) -> str:
    """
    Removes noise from text: newlines, 'et al' references, digits, and extra spaces.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # 1. Replace newlines with space
    text = text.replace('\n', ' ')
    
    # 2. Remove 'word et al' pattern (case insensitive)
    # \S+ matches the preceding word (e.g. "Smith" in "Smith et al")
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    
    # 3. Remove digits
    text = re.sub(r'\d+', '', text)
    
    # 4. Clean extra spaces
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# GENERATION HELPERS (GPT)
# ==========================================

def gpt_generate(prompt: str, system_prompt: str = "You are a helpful scientific research assistant.") -> str:
    """Generate text using GPT-4o-mini."""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temperature for factual consistency
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API Error: {e}")
        return f"Error: {e}"

# ==========================================
# PROMPTS
# ==========================================

# Agent-specific prompts - each receives both paper_content and cited_papers
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

Reflect: Verify that all relevant sections were checked and no limitations were missed. Consider whether any section might have been 
overlooked and re-evaluate if necessary.

Output Format:
Bullet points listing each limitation.
For each: Verbatim quote (if available), context (e.g., aspect of the study), and section reference.
If none: "No limitations explicitly stated in the article."

Paper Content:
{paper_content}

Please extract and list the key limitations found in this paper. Be specific and provide clear reasoning for each limitation identified."""

def get_analyzer_prompt(paper_content: str) -> str:
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

Please provide a detailed analysis of the limitations in this research. Consider both obvious and subtle limitations that could affect the validity and applicability of the findings."""
 
def get_reviewer_prompt(paper_content: str) -> str:
    return f"""You are a strict, constructive, and expert reviewer for the International Conference on Learning Representations (ICLR). Your task is to critically evaluate the provided research paper and strictly identify its **limitations, weaknesses, and flaws**. 

You must adhere to the official ICLR Review Guidelines, focusing on Technical Correctness, Clarity, Originality, and Significance.

**Input Paper Content:**
{paper_content}

---

### **Reviewer Objectives & Workflow**

**1. Analyze the Objective & Novelty:**
   - Does the paper actually solve the problem it claims to? 
   - Is the "novelty" just an incremental tweak (e.g., combining existing methods without new insight)?
   - *Limitation Trigger:* If the novelty is overstated or the contribution is marginal, this is a major limitation.

**2. Scrutinize Technical Correctness & Rigor (Crucial):**
   - Check the theoretical proofs (are assumptions too strong/unrealistic?).
   - Check the empirical results. Are the baselines weak? Are the improvements statistically significant?
   - *Limitation Trigger:* If the method only works under specific, unstated conditions, or if the baselines are outdated, flag this immediately.

**3. Evaluate Experimental Rigor & Reproducibility:**
   - Are the experiments sufficient to support the claims? (e.g., tested on enough seeds? multiple domains?)
   - Is the ablation study missing? Do we know *why* the method works?
   - *Limitation Trigger:* Lack of code, insufficient details to reproduce, or "cherry-picked" results.

**4. Assess Clarity & Placement in Literature:**
   - Does the paper ignore relevant prior work to make itself look better?
   - Is the writing confusing or hiding lack of substance?

---

### **Few-Shot Example (Tone and Depth):**
*Bad Limitation:* "The experiments are small."
*Good ICLR-Style Limitation:* "The experimental validation is limited to the Inverted Double Pendulum domain. To fully support the claim of 'superior performance,' the authors should demonstrate performance across a wider suite of standard RL benchmarks (e.g., MuJoCo) and compare against state-of-the-art baselines like SAC or TD3, not just PPO. Furthermore, the reliance on a specific regularization parameter without a sensitivity analysis raises concerns about the method's robustness."

---

### **Output Instructions**

Based on the analysis above, generate a **bulleted list of specific, scientifically grounded limitations**. 
Do not write a full review (no summary or acceptance decision needed)—**only list the Weak Points.**

**Format:**
* **[Category] Limitation Statement:**
    * *Explanation:* Why is this a problem? (e.g., undermines the central claim, reduces reproducibility).
    * *Evidence:* Reference specific sections, equations, or missing experiments from the text.

**Categories to consider:** [Theoretical Soundness], [Experimental Rigor], [Novelty/Significance], [Clarity/Reproducibility], [Related Work/Citation].

**Generate the limitations now:**""" 

# def get_reviewer_prompt(paper_content: str) -> str:
#     return f"""You are an expert in open peer review with a focus on transparent and critical evaluation of scientific research. 
# Your task is to review the provided scientific article from the perspective of an external peer reviewer. Identify potential limitations 
# that might be raised in an open review process, considering common critiques such as reproducibility, transparency, generalizability, 
# or ethical considerations. Leverage insights from similar studies or common methodological issues in the field.

# Workflow:
# Plan: Identify areas for review (e.g., reproducibility, transparency, ethics) and plan searches for external context
# (e.g., similar studies, methodological critiques). Justify the selection based on peer review standards.

# Reasoning: Let's think step by step to identify peer-review limitations:
# Step 1: Select key areas for review (e.g., reproducibility, ethics) based on common peer review critiques.
# Step 2: Use text analysis tools to extract relevant article details (e.g., methods, data reporting).
# Step 3: Identify potential limitations, such as lack of transparency in data or ethical concerns, and justify using article content.
# Step 4: Search web/X for external context (e.g., similar studies) to support limitations, rating source relevance 
# (high, medium, low, none).
# Step 5: Synthesize findings, ensuring limitations align with peer review standards and are supported by the article or external context.

# Analyze: Critically review the article, integrating external context to identify limitations. Use tools to verify content and sources.
# Reflect: Verify that limitations align with peer review standards and are supported by the article or external context. 
# Re-evaluate overlooked areas if necessary.

# Output Format:
# Bullet points listing each limitation.
# For each: Description, why it's a concern, and alignment with peer review standards.

# Paper Content:
# {paper_content}

# Please provide a critical review identifying the limitations and areas of concern in this research. Consider what a peer reviewer would highlight as weaknesses or areas needing improvement."""

def get_citation_prompt(paper_content: str, cited_papers: str) -> str:
    return f"""You are an expert scientific research assistant tasked with inferring potential limitations for a current scientific article 
using both the article's own content and its cited papers.

The cited papers are provided below in a structured format:
Paper1_Title, Paper1_Abstract, Paper1_Introduction
Paper2_Title, Paper2_Abstract, Paper2_Introduction
...and so on.

Objective:
Generate a list of scientifically grounded limitations that the current article might have, assuming it builds upon or is informed by these cited papers.

Workflow:
1. Review the Title, Abstract, and Introduction of each cited paper (Paper1, Paper2, etc.) to understand the foundation.
2. Ask: If a paper cited this work but did not adopt or address its insights, what limitation might arise?
3. Formulate each limitation as a plausible shortcoming of the current article.

Output Format:
Bullet points listing each limitation.
For each: Description, explanation, and reference to the cited paper (e.g., "Related to Paper1 [Title]").

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

def extract_intro_and_abstract(cited_entry: Any) -> str:
    """
    Parses the cited_in column. 
    Format: 'Paper1_Title:text, 'Paper1_Abstract': 'text', 'Paper1_Introduction: text'.
    Applies clean_text_detailed to the *content* strings to remove noise (digits, et al) 
    while preserving the Paper1/Paper2 structure.
    """
    if pd.isna(cited_entry):
        return ""

    parsed: Any = cited_entry
    if isinstance(cited_entry, str):
        try:
            parsed = ast.literal_eval(cited_entry)
        except Exception as exc:
            print(f"  ⚠️ Warning: Could not parse cited_in entry: {exc}")
            return ""

    if not isinstance(parsed, dict):
        return ""

    processed_papers: List[str] = []

    # Iterate over each cited paper in the dictionary
    # Use enumerate to create Paper1, Paper2, etc.
    for idx, (paper_id, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue
            
        # 1. Get raw content
        title_raw = data.get("title", "")
        abstract_raw = data.get("abstractText") or data.get("abstract") or ""
        
        # 2. Get Introduction from Sections
        intro_raw = ""
        sections = data.get("sections", [])
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    heading = str(section.get("heading", "")).lower()
                    if "introduction" in heading:
                        intro_raw = section.get("text", "")
                        break 
        
        # 3. Apply cleaning to the CONTENT only
        # We clean here so we don't accidentally remove the '1' in 'Paper1' later
        title_clean = clean_text_detailed(title_raw)
        abstract_clean = clean_text_detailed(abstract_raw)
        intro_clean = clean_text_detailed(intro_raw)

        # 4. Format string as requested
        # 'Paper1_Title:text, 'Paper1_Abstract': 'text', 'Paper1_Introduction: text'.
        
        # Only include if there is actually content to avoid empty tags
        if title_clean or abstract_clean or intro_clean:
            entry_str = (
                f"'Paper{idx}_Title: {title_clean}', "
                f"'Paper{idx}_Abstract': '{abstract_clean}', "
                f"'Paper{idx}_Introduction': '{intro_clean}'."
            )
            processed_papers.append(entry_str)

    # Join all papers
    return "\n".join(processed_papers)

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

# Initialize columns
df["extractor"] = ""
df["analyzer"] = ""
df["reviewer"] = ""
df["citation"] = ""
df["master"] = ""
df["cited_papers_context"] = ""

# SELECT ROW SLICE HERE
START_INDEX = 100
END_INDEX = 199 

# Slice the dataframe for processing
df_slice = df.iloc[START_INDEX:END_INDEX].copy()
global_df = df_slice 

total_samples = len(df_slice)
print(f"Processing {total_samples} samples (Rows {START_INDEX} to {END_INDEX}) with GPT-4o-mini...")

for idx, (i, row) in enumerate(df_slice.iterrows()):
    global_current_row = i
    print(f"\n=== Processing row {i} ({idx + 1}/{total_samples}) ===")
    
    # 1. Prepare Content
    paper_content = str(row.get("input_text_cleaned", ""))
    
    # Process citations (Extraction + Cleaning + Formatting)
    cited_papers_context = extract_intro_and_abstract(row.get("cited_in", ""))
    
    # Store processed context
    df_slice.at[i, "cited_papers_context"] = cited_papers_context

    # 2. Run Individual Agents
    print("  Running Extractor (Input: input_text_cleaned)...")
    ext_prompt = get_extractor_prompt(paper_content)
    df_slice.at[i, "extractor"] = gpt_generate(ext_prompt)

    print("  Running Analyzer (Input: input_text_cleaned)...")
    ana_prompt = get_analyzer_prompt(paper_content)
    df_slice.at[i, "analyzer"] = gpt_generate(ana_prompt)

    print("  Running Reviewer (Input: input_text_cleaned)...")
    rev_prompt = get_reviewer_prompt(paper_content)
    df_slice.at[i, "reviewer"] = gpt_generate(rev_prompt)
        
    print("  Running Citation Agent (Input: input_text_cleaned + Cited Abstracts/Intros)...")
    # Pass the specially formatted citation context
    cit_prompt = get_citation_prompt(paper_content, cited_papers_context)
    df_slice.at[i, "citation"] = gpt_generate(cit_prompt)

    # 3. Run Merger
    print("  Running Master Coordinator...")
    merger_prompt = get_merger_prompt(
        df_slice.at[i, "extractor"],
        df_slice.at[i, "analyzer"],
        df_slice.at[i, "reviewer"],
        df_slice.at[i, "citation"]
    )
    df_slice.at[i, "master"] = gpt_generate(merger_prompt)
    print("  ✅ Row completed.")

    if (idx + 1) % 5 == 0:
        df_slice.to_csv(OUTPUT_FILE, index=False)
        print(f"  💾 Checkpoint saved at row {i}") 

    time.sleep(1)  # Brief pause to avoid rate limits

# Final Save
df_slice.to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== GPT-4o-mini Multi-Agent Analysis Complete ===")
print(f"Total samples processed: {total_samples}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")