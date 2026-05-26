"""
Use Llama 3 8B to extract limitations/shortcomings from paper metadata.

Experiments (all using the same model):
  1) Extract from `Author_mention_limitation`
  2) Extract from `weaknesses`
  3) Extract from the merge of both fields

For each experiment, we ask the model to:
  - Extract explicit limitations/shortcomings only (no hallucinations)
  - Generate a short title for each limitation
  - Return JSON only (no extra text)

At first, we only run on 2 samples for a quick sanity check.
"""

import os
import time
import pandas as pd
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "llama3_8b"

INPUT_CSV = "df"
OUTPUT_CSV = "df1.csv" 

# If N_SAMPLES is None, process all rows in the input CSV.
N_SAMPLES = None

def load_model(model_id: str = MODEL_ID, cache_dir: str = CACHE_DIR):
    """Load Llama 3 8B model (4-bit quantized) and tokenizer."""
    print("Loading Llama 3 8B model and tokenizer...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Model loaded successfully.")
    return tokenizer, model

def build_prompt(text: str) -> str:
    """
    Build the extraction prompt.

    User requirements:
      - Prompt should be "extract limtiations or shortcomings"
      - Don't generate anything beyond what is in the text
      - Ask LLM to generate a title for each extracted limitation
    """
    return f"""
You are an expert scientific assistant.

Task: **extract limitations or shortcomings** from the given text. Work under these rules:
- Only extract limitations/shortcomings that are explicitly mentioned or clearly implied by the text.
- **Do not invent or hallucinate** any limitations that are not supported by the text.
- For each limitation, create:
  - a short, descriptive **title** (5–10 words)
  - a concise **description** (1–3 sentences) summarizing the limitation.

Strict output format:
- Respond **only** in valid JSON (no explanations, no markdown, no extra text).
- Use this exact JSON schema:
{{
  "limitations": [
    {{
      "title": "Short title of the limitation",
      "description": "limitation based only on the text."
    }}
    / ... more limitations ...
  ]
}}

If there are no clear limitations, return:
{{ "limitations": [] }}

Text:
\"\"\"{text}\"\"\"
""".strip()

def generate_limitations(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 512,
) -> str:
    """Run Llama 3 8B on the given text and return the raw JSON string."""
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()

def run_experiments(
    df: pd.DataFrame,
    tokenizer,
    model,
    n_samples: int | None = N_SAMPLES,
    col_author: str = "Author_mention_limitation",
    col_weakness: str = "weaknesses",
) -> pd.DataFrame:
    """Run the three extraction experiments and return a new DataFrame with results."""
    # If n_samples is None, process the full DataFrame
    if n_samples is None:
        df_sub = df.copy()
    else:
        df_sub = df.head(n_samples).copy()

    print(f"Running experiments on {len(df_sub)} samples...")

    # Prepare output columns
    col_exp1 = "Llama_Limitations_from_Author_mention_limitation"
    col_exp2 = "Llama_Limitations_from_weaknesses"
    col_exp3 = "Llama_Limitations_from_Author_and_weaknesses_merged"

    df_sub[col_exp1] = ""
    df_sub[col_exp2] = ""
    df_sub[col_exp3] = ""

    for idx, row in df_sub.iterrows():
        print("\n" + "=" * 80)
        print(f"Processing index {idx} ...")
        print("=" * 80)

        author_text = row.get(col_author, "")
        weakness_text = row.get(col_weakness, "")

        # Experiment 1: Author_mention_limitation only
        try:
            print("  [Exp1] Extracting from Author_mention_limitation ...")
            df_sub.at[idx, col_exp1] = generate_limitations(author_text, tokenizer, model)
            print("  [Exp1] Output:")
            print(df_sub.at[idx, col_exp1])
        except Exception as e:
            print(f"  [Exp1] Error: {e}")
            df_sub.at[idx, col_exp1] = "ERROR"

        # Experiment 2: weaknesses only
        try:
            print("  [Exp2] Extracting from weaknesses ...")
            df_sub.at[idx, col_exp2] = generate_limitations(weakness_text, tokenizer, model)
            print("  [Exp2] Output:")
            print(df_sub.at[idx, col_exp2])
        except Exception as e:
            print(f"  [Exp2] Error: {e}")
            df_sub.at[idx, col_exp2] = "ERROR"

        # Experiment 3: merged text
        try:
            print("  [Exp3] Extracting from merged Author_mention_limitation + weaknesses ...")
            merged_text = f"Author_mention_limitation:\n{author_text}\n\nweaknesses:\n{weakness_text}"
            df_sub.at[idx, col_exp3] = generate_limitations(merged_text, tokenizer, model)
            print("  [Exp3] Output:")
            print(df_sub.at[idx, col_exp3])
        except Exception as e:
            print(f"  [Exp3] Error: {e}")
            df_sub.at[idx, col_exp3] = "ERROR"

    return df_sub

def main():
    start_time = time.time()

    print(f"Reading input CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Basic sanity checks for required columns
    required_cols = ["Author_mention_limitation", "weaknesses"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV.")

    tokenizer, model = load_model()
    df_out = run_experiments(df, tokenizer, model, n_samples=N_SAMPLES)

    print(f"\nSaving results to: {OUTPUT_CSV}")
    df_out.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Limitation extraction completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    print("=" * 80)

if __name__ == "__main__":
    main()

