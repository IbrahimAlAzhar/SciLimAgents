"""
Inference
=========
Loads the DPO-trained model and runs limitation generation on the input dataset.
rows of df_updated_with_retrieval.csv. Reads the 'input_text_cleaned' column
(plain string).

Saves TWO columns:
  - generated_limitations_raw : full JSON output from the model (for debugging)
  - generated_limitations     : plain bullet list (compare to ground truth)

"""

import os
import re
import json
import logging
import torch
import pandas as pd

from config import get_config

logger = logging.getLogger(__name__)

INPUT_CSV     = "data/balanced_data/df_updated_with_retrieval.csv"
DPO_MODEL_DIR = "other_experiments/dpo_novagents/output/dpo_model"
INFERENCE_DIR = "other_experiments/dpo_novagents/output/inference"

NUM_ROWS    = 200
SAVE_EVERY  = 10
MAX_NEW_TOK = 1500
INPUT_COL   = "input_text_cleaned"
RAW_COL     = "generated_limitations_raw"
OUTPUT_COL  = "generated_limitations"

# ============================================================
# Prompt — aligned with the limitation_synthesis step in training
# ============================================================
PROMPT_TEMPLATE = """You are an extremely strict, harsh peer reviewer.

Read Paper A below and identify its key claims, then synthesize the most
important LIMITATIONS that a harsh reviewer would raise. Focus on:
- incremental contributions / weak novelty
- missing baselines or weak experimental validation
- overclaiming
- narrow scope or generalizability concerns
- methodological gaps

Paper A:
{paper_text}

Output a JSON array. For each limitation include:
- "limitation": a specific, concrete limitation statement (2-3 sentences)
- "severity": one of ["critical", "major", "minor"]
- "category": one of ["incremental_contribution", "missing_baseline",
  "overclaiming", "narrow_scope", "methodological_gap", "insufficient_differentiation"]

Generate between 6 and 8 limitations. Output ONLY a valid JSON array."""

def build_prompt(paper_text: str, max_chars: int = 8000) -> str:
    return PROMPT_TEMPLATE.format(paper_text=str(paper_text)[:max_chars])

# ============================================================
# Post-processing: JSON output -> plain "- bullet" list
# ============================================================
def to_plain_bullets(generated: str) -> str:
    """Extract the 'limitation' field from each JSON object and join as bullets."""
    if not isinstance(generated, str) or not generated.strip():
        return ""

    # Strip markdown code fences if any
    clean = re.sub(r'```(?:json)?', '', generated).strip()

    # Try to locate the JSON array if there's preamble/postamble
    start = clean.find('[')
    end   = clean.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = clean[start:end + 1]
    else:
        candidate = clean

    # Attempt 1: parse as a JSON array
    try:
        items = json.loads(candidate)
        if isinstance(items, list):
            bullets = [
                f"- {str(it['limitation']).strip()}"
                for it in items
                if isinstance(it, dict) and it.get("limitation")
            ]
            if bullets:
                return "\n".join(bullets)
    except Exception:
        pass

    # Attempt 2: regex out every "limitation": "..." entry (handles minor JSON breakage)
    matches = re.findall(
        r'"limitation"\s*:\s*"((?:[^"\\]|\\.)*)"',
        candidate,
    )
    if matches:
        return "\n".join(f"- {m.strip()}" for m in matches)

    # Fallback: keep raw output (better than nothing for debugging)
    return generated.strip()

# ============================================================
# Main
# ============================================================
def main():
    config = get_config()
    os.makedirs(INFERENCE_DIR, exist_ok=True)
    out_path = os.path.join(INFERENCE_DIR, "inference_results.csv")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.logs_dir, "inference.log")),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 60)
    logger.info("INFERENCE")
    logger.info(f"  Input CSV:  {INPUT_CSV}")
    logger.info(f"  DPO model:  {DPO_MODEL_DIR}")
    logger.info(f"  Output CSV: {out_path}")
    logger.info(f"  Rows:       first {NUM_ROWS}, save every {SAVE_EVERY}")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    df = pd.read_csv(INPUT_CSV).head(NUM_ROWS).copy().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")
    if INPUT_COL not in df.columns:
        raise ValueError(f"Column '{INPUT_COL}' not found. Have: {list(df.columns)}")

    for col in (RAW_COL, OUTPUT_COL):
        if col not in df.columns:
            df[col] = ""

    # Resume if results exist
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        if len(prev) == len(df):
            for col in (RAW_COL, OUTPUT_COL):
                if col in prev.columns:
                    df[col] = prev[col].fillna("").astype(str)
            done = (df[RAW_COL].str.len() > 0).sum()
            logger.info(f"Resuming: {done} rows already done")

    logger.info("Loading tokenizer + DPO model...")
    # tokenizer = AutoTokenizer.from_pretrained(DPO_MODEL_DIR, cache_dir=config.hf_cache, trust_remote_code=True) 
    
    tokenizer = AutoTokenizer.from_pretrained(config.weak_model, cache_dir=config.hf_cache, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.weak_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.hf_cache,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, DPO_MODEL_DIR)
    model.eval()

    for i, row in df.iterrows():
        # Skip if already processed
        if isinstance(df.at[i, RAW_COL], str) and len(df.at[i, RAW_COL]) > 0:
            continue

        paper_text = row.get(INPUT_COL, "")
        if not isinstance(paper_text, str) or len(paper_text.strip()) < 50:
            df.at[i, RAW_COL]    = ""
            df.at[i, OUTPUT_COL] = ""
            continue

        prompt = build_prompt(paper_text)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=config.max_prompt_length).to(model.device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOK,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"  Row {i} generation failed: {e}")
            generated = ""

        plain = to_plain_bullets(generated)

        df.at[i, RAW_COL]    = generated
        df.at[i, OUTPUT_COL] = plain

        n_bullets = plain.count("\n- ") + (1 if plain.startswith("- ") else 0)
        logger.info(f"  Row {i+1}/{len(df)} done "
                    f"(raw {len(generated)} chars, {n_bullets} bullets)")

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(out_path, index=False)
            logger.info(f"  -> Checkpointed at row {i+1} -> {out_path}")

    df.to_csv(out_path, index=False)
    logger.info(f"FINAL save: {out_path}")
    logger.info("INFERENCE COMPLETE")

if __name__ == "__main__":
    main()

# ============================================================
# Quick sanity check for to_plain_bullets() — run as:
#   python -c "from inference import to_plain_bullets; \
#              print(to_plain_bullets('[{\"limitation\":\"X\"},{\"limitation\":\"Y\"}]'))"
# ============================================================