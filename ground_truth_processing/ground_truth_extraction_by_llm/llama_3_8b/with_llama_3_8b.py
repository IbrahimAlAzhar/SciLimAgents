#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import re
import argparse
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------
# Defaults (your paths)
# -------------------------
INPUT_CSV = "df"
OUTPUT_CSV = "df1.csv" 

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "llama3_8b"

SRC_COL = "limitations_autho_peer_gt"
OUT_COL = "author_peer_llama_gt_llama_ext"

# Generation
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0
# TOP_P = 0.9

# Checkpointing / resume
SAVE_EVERY = 50  # checkpoint interval
# -------------------------
# Prompt (your exact prompt style)
# -------------------------
def build_prompt(text: str) -> str:
    return f"""
You are an expert scientific assistant.

Task: extract limitations or shortcomings from the given text. Work under these rules:
- Only extract limitations/shortcomings that are explicitly mentioned.
- Do not invent or hallucinate any limitations that are not supported by the text.

Strict output format:
- Respond only extracted limitations with newline.

Text:
\"\"\"{text}\"\"\"
""".strip() 

import re

# -------------------------
# Model loading
# -------------------------
def load_model(model_id: str, cache_dir: str):
    print(f"[INFO] Loading tokenizer/model: {model_id}")
    print(f"[INFO] cache_dir: {cache_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir) 
    # ✅ add this here
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()
    print("[INFO] Model loaded.")
    return tokenizer, model

# -------------------------
# Helpers: truncation + JSON cleanup
# -------------------------
def get_context_limit(model) -> int:
    # best-effort
    for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
        v = getattr(getattr(model, "config", object()), attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 8192  # safe default

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json_obj(s: str) -> str:
    """
    Best-effort: if the model outputs extra tokens, keep the first {...} block.
    If no block found, return raw string.
    """
    if not s:
        return s
    m = _JSON_OBJ_RE.search(s.strip())
    return m.group(0).strip() if m else s.strip()

def truncate_to_fit(prompt: str, tokenizer, model_ctx: int, max_new_tokens: int) -> str:
    """
    Ensure prompt tokens + max_new_tokens <= model_ctx (keep the END of the prompt if needed).
    """
    budget = model_ctx - max_new_tokens - 16
    if budget <= 0:
        return ""

    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() <= budget:
        return prompt

    # keep the end (often contains the actual text payload)
    ids = ids[-budget:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def generate_limitations(text: str, tokenizer, model, model_ctx: int) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    user_prompt = build_prompt(text)

    # Llama-3-Instruct works best with chat template
    messages = [
        {"role": "system", "content": "You are a careful scientific assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_str = truncate_to_fit(prompt_str, tokenizer, model_ctx, MAX_NEW_TOKENS)
    if not prompt_str:
        return ""

    inputs = tokenizer(prompt_str, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # ✅ More deterministic + less hallucination for extraction tasks
    with torch.no_grad():
        output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,            # smaller = less chance to spiral
        do_sample=False,               # extraction task -> deterministic
        temperature=0.0,
        repetition_penalty=1.15,       # key for “* * * *” loops
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # gen_text = clean_bullets(gen_text)

    return gen_text

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    ap.add_argument("--save_every", type=int, default=SAVE_EVERY)
    ap.add_argument("--row_start", type=int, default=-1)
    ap.add_argument("--row_end", type=int, default=-1)
    args = ap.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    save_every = args.save_every
    row_start = None if args.row_start < 0 else args.row_start
    row_end = None if args.row_end < 0 else args.row_end

    print(f"[INFO] Reading: {input_csv}")
    df = pd.read_csv(input_csv)

    if SRC_COL not in df.columns:
        raise ValueError(f"Missing required column: {SRC_COL}")

    # slice if requested
    if row_start is not None or row_end is not None:
        rs = 0 if row_start is None else row_start
        re_ = len(df) if row_end is None else row_end
        df = df.iloc[rs:re_].copy()
        print(f"[INFO] Sliced rows: {rs}..{re_-1} (n={len(df)})")
    else:
        df = df.copy()

    # prepare output column
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # Resume support: if output exists and same length, load existing column
    if os.path.exists(output_csv):
        try:
            old = pd.read_csv(output_csv)
            if OUT_COL in old.columns and len(old) == len(df):
                df[OUT_COL] = old[OUT_COL].fillna("")
                print(f"[RESUME] Loaded existing outputs from: {output_csv}")
        except Exception as e:
            print(f"[WARN] Could not resume from {output_csv}: {e}")

    tokenizer, model = load_model(MODEL_ID, CACHE_DIR)
    model_ctx = get_context_limit(model)
    print(f"[INFO] Detected model context limit: {model_ctx}")

    def is_done(x: str) -> bool:
        return isinstance(x, str) and len(x.strip()) > 0

    pending = [i for i in range(len(df)) if not is_done(df.at[df.index[i], OUT_COL])]
    print(f"[INFO] Total rows={len(df)} | Pending={len(pending)}")

    t0 = time.time()
    processed = 0

    for k, local_i in enumerate(tqdm(pending, desc="Extracting")):
        idx = df.index[local_i]
        text = df.at[idx, SRC_COL]

        try:
            out = generate_limitations(text, tokenizer, model, model_ctx)
        except Exception as e:
            print(f"[ERROR] row={local_i} idx={idx}: {e}")
            out = ""

        df.at[idx, OUT_COL] = out
        processed += 1

        if processed % save_every == 0:
            df.to_csv(output_csv, index=False)
            print(f"[SAVED] {output_csv} | processed={processed} | elapsed={(time.time()-t0)/60:.1f} min")

    df.to_csv(output_csv, index=False)
    print(f"[DONE] Saved: {output_csv}")
    print(f"[DONE] Total elapsed: {(time.time()-t0)/60:.2f} minutes")

if __name__ == "__main__":
    main()
