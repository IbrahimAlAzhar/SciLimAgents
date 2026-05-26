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

# 🔹 OpenAI client for Merger (GPT-4o mini)
from openai import OpenAI 
os.environ['OPENAI_API_KEY'] = ''
OPENAI_MODEL_ID = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ========= PROMPT FUNCTIONS (NEW) =========

def get_extractor_prompt(paper_content: str) -> str:
    return f"""You are an expert in scientific literature analysis. Your task is to carefully read the provided scientific article 
and extract all explicitly stated limitations as mentioned by the authors. Focus on sections such as Discussion, Conclusion, or 
Limitations. List each limitation verbatim, including direct quotes where possible, and provide a brief context (e.g., what aspect of 
the study the limitation pertains to). Ensure accuracy and avoid inferring or adding limitations not explicitly stated. If no limitations 
are mentioned, state this clearly.

Workflow:
Plan: Outline which sections (e.g., Discussion, Conclusion, Limitations) to analyze and identify tools to access the article content.
Reasoning: Let's think step by step to ensure thorough and accurate extraction of limitations:
Step 1: Identify all sections in the article that may contain limitations.
Step 2: Scan for explicit limitation statements, such as phrases like "a limitation of this study" or "we acknowledge that."
Step 3: Document why each statement qualifies as a limitation.
Step 4: For each identified limitation, extract the verbatim quote.
Step 5: Check for completeness by reviewing other potential sections.

Analyze: Use tools to extract and verify the article's content, focusing on explicit limitation statements.

Reflect: Verify that all relevant sections were checked and no limitations were missed.

Output Format:
- Bullet points listing each limitation.
- For each: Verbatim quote (if available), context (e.g., aspect of the study), and section reference.
- If none: "No limitations explicitly stated in the article."

Paper Content:
{paper_content}

Please extract and list the key limitations found in this paper. Be specific and provide clear reasoning for each limitation identified.
"""

def get_analyzer_prompt(paper_content: str) -> str:
    return f"""You are a critical scientific reviewer with expertise in research methodology and analysis. Your task is to analyze 
the provided scientific article and identify potential limitations not explicitly stated by the authors. Focus on aspects such as study 
design, sample size, data collection methods, statistical analysis, scope of findings, and underlying assumptions. For each inferred 
limitation, provide a clear explanation of why it is a limitation and how it impacts the study's validity, reliability, 
or generalizability. Ensure inferences are grounded in the article's content and avoid speculative assumptions.

Workflow:
Plan: Identify key areas (e.g., methodology, sample size, statistical analysis) to analyze.
Reasoning: Let's think step by step to identify inferred limitations:
Step 1: Review the article's methodology to identify gaps (e.g., study design flaws, sampling issues).
Step 2: Evaluate each area for potential limitations, such as small sample size affecting generalizability.
Step 3: Document why each gap qualifies as a limitation and its impact on the study.
Step 4: Ensure all key areas are covered.

Analyze: Critically evaluate the article and infer limitations based on methodological or analytical gaps.
Reflect: Assess whether inferred limitations are grounded in the article.

Output Format:
- Bullet points listing each inferred limitation.
- For each: Description, explanation, and impact on the study.

Paper Content:
{paper_content}

Please provide a detailed analysis of the limitations in this research. Consider both obvious and subtle limitations that could affect the validity and applicability of the findings.
"""

def get_reviewer_prompt(paper_content: str) -> str:
    return f"""You are an expert in open peer review with a focus on transparent and critical evaluation of scientific research. 
Your task is to review the provided scientific article from the perspective of an external peer reviewer. Identify potential limitations 
that might be raised in an open review process, considering common critiques such as reproducibility, transparency, generalizability, 
or ethical considerations. Leverage insights from similar studies or common methodological issues in the field.

Workflow:
Plan: Identify areas for review (e.g., reproducibility, transparency, ethics) and plan evaluations based on the article content.
Reasoning: Let's think step by step to identify peer-review limitations:
Step 1: Select key areas for review based on common peer review critiques.
Step 2: Identify potential limitations, such as lack of transparency in data or ethical concerns.
Step 3: Synthesize findings, ensuring limitations align with peer review standards.

Analyze: Critically review the article, integrating domain knowledge to identify limitations.
Reflect: Verify that limitations align with peer review standards.

Output Format:
- Bullet points listing each limitation.
- For each: Description, why it's a concern, and alignment with peer review standards.

Paper Content:
{paper_content}

Please provide a critical review identifying the limitations and areas of concern in this research. Consider what a peer reviewer would highlight as weaknesses or areas needing improvement.
"""

def get_merger_prompt(
    extractor_output: str, analyzer_output: str, reviewer_output: str
) -> str:
    return f"""You are a Master Coordinator, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by three specialized agents: 

Agents:
1. Extractor (explicit limitations from the article), 
2. Analyzer (inferred limitations from critical analysis), 
3. Reviewer (limitations from an open review perspective).

Goals:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Prioritize critical limitations that affect the paper's validity and reproducibility.
4. Format the final list in a clear, concise, and professional manner, suitable for a scientific review or report.

Workflow:
1. Plan: Outline how to synthesize limitations, identify potential redundancies, and resolve discrepancies.
2. Analyze: Combine limitations, prioritizing critical ones, and verify alignment with the article.
3. Reflect: Check for completeness, scientific rigor, and clarity.
4. Continue: Iterate until the list is comprehensive, non-redundant, and professionally formatted.

Output Format:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived]).

Extractor Agent Analysis:
{extractor_output}

Analyzer Agent Analysis:
{analyzer_output}

Reviewer Agent Analysis:
{reviewer_output}

Please merge these three different perspectives on the paper's limitations into a comprehensive, well-organized analysis. Synthesize the insights, resolve any contradictions, and provide a unified view of the paper's limitations and don't miss any unique limitations.
"""

# ========= ORIGINAL CONFIG =========

TEST_INPUT_FILE = "data/final_balanced_kde/df_balanced_kde_final.csv"

OUTPUT_DIR = 'SFT_from_zs_output/mistral/output_using_ground_truth'
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "mistral_sft_test_results_llm_agents.csv")

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints",
    "checkpoint-75"
)

# --- 1. Load and Prepare Data ---
print(f"Loading Test Data from {TEST_INPUT_FILE}...")
try:
    df_full = pd.read_csv(TEST_INPUT_FILE)
    print(f"Sliced rows 100-199. Shape: {df_slice.shape}")
    
    df_test = df_slice.sample(n=5, random_state=42).copy()
    print(f"Selected 50 random samples. Shape: {df_test.shape}")
    
    if 'input_text' not in df_test.columns:
        print("Error: Column 'input_text' not found in dataframe.")
        print("Available columns:", df_test.columns.tolist())
        sys.exit(1)
        
except FileNotFoundError:
    print("Error: Input file not found.")
    sys.exit(1)

# Initialize new columns for the four agents
df_test["extractor"] = ""
df_test["analyzer"] = ""
df_test["reviewer"] = ""
df_test["merger"] = ""

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

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========= GENERATION (KEEPING YOUR "ONLY NEW TOKENS" LOGIC) =========

def generate_sft_response(full_instruction_text: str) -> str:
    """
    full_instruction_text: the whole prompt body (Extractor/Analyzer/Reviewer/Merger),
    including paper content or agent outputs.
    """
    # Truncate to avoid overly long context
    truncated_prompt_body = str(full_instruction_text)[:25000]

    # Wrap in Mistral [INST] format
    prompt = f"<s>[INST] {truncated_prompt_body} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id
        )

    # 1. Length of the input tokens
    input_length = inputs["input_ids"].shape[1]
    # 2. Keep only newly generated tokens
    generated_tokens = outputs[0][input_length:]
    # 3. Decode only new tokens
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()

# ========= MERGER USING GPT-4o MINI =========

def generate_merger_with_gpt4o(
    extractor_output: str,
    analyzer_output: str,
    reviewer_output: str
) -> str:
    merger_prompt = get_merger_prompt(
        extractor_output=extractor_output,
        analyzer_output=analyzer_output,
        reviewer_output=reviewer_output
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_ID,
            input=merger_prompt,
            max_output_tokens=1024,
        )
        # responses API: first output, first content block, text field
        return resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"  GPT-4o mini merger error: {e}")
        return "ERROR_GPT4O"

# --- 3. Inference Loop ---

print("Running Inference...")
df_test = df_test.reset_index(drop=True)

for i, row in df_test.iterrows():
    input_text = row['input_text']
    
    if pd.notna(input_text) and str(input_text).strip():
        try:
            paper_content = str(input_text)

            # --- Extractor ---
            extractor_prompt = get_extractor_prompt(paper_content)
            extractor_out = generate_sft_response(extractor_prompt)
            df_test.at[i, "extractor"] = extractor_out

            # --- Analyzer ---
            analyzer_prompt = get_analyzer_prompt(paper_content)
            analyzer_out = generate_sft_response(analyzer_prompt)
            df_test.at[i, "analyzer"] = analyzer_out

            # --- Reviewer ---
            reviewer_prompt = get_reviewer_prompt(paper_content)
            reviewer_out = generate_sft_response(reviewer_prompt)
            df_test.at[i, "reviewer"] = reviewer_out

            # --- Merger using mistral ---
            # merger_prompt = get_merger_prompt(
            #     extractor_output=extractor_out,
            #     analyzer_output=analyzer_out,
            #     reviewer_output=reviewer_out
            # )
            # merger_out = generate_sft_response(merger_prompt)

            # --- Merger using GPT-4o mini --- 
            merger_out = generate_merger_with_gpt4o(
                extractor_output=extractor_out,
                analyzer_output=analyzer_out,
                reviewer_output=reviewer_out
            )
            df_test.at[i, "merger"] = merger_out

            print(f"  Processed sample {i+1}/50")

        except Exception as e:
            print(f"  Error at sample {i+1}: {e}")
            df_test.at[i, "extractor"] = "ERROR"
            df_test.at[i, "analyzer"] = "ERROR"
            df_test.at[i, "reviewer"] = "ERROR"
            df_test.at[i, "merger"] = "ERROR"
    else:
        print(f"  Skipping sample {i+1} (empty 'input_text')")

    # periodic save
    if (i + 1) % 10 == 0:
        df_test.to_csv(RESULTS_FILE, index=False)

df_test.to_csv(RESULTS_FILE, index=False)
print(f"Done! Results saved to: {RESULTS_FILE}")
