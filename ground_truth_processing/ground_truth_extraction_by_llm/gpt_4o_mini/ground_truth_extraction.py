#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import re
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# =========================
# Config
# =========================
INPUT_CSV = "df"
OUTPUT_CSV = "df1.csv"

SRC_COL = "limitations_autho_peer_gt"
OUT_COL = "author_peer_llama_gt_gpt4omini_ext"

# OpenAI model
MODEL_ID = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Generation
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_NEW_TOKENS = 1000

# Timeouts / retries
TIMEOUT_S = 600  # OpenAI Python SDK supports setting timeout. :contentReference[oaicite:6]{index=6}
SAVE_EVERY = 50
SLEEP_S = 1

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# =========================
# OpenAI client
# ========================= 
os.environ['OPENAI_API_KEY'] = ''
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(timeout=TIMEOUT_S, api_key=api_key)

# =========================
# Prompt builder
# =========================
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

# Optional: simple safety truncation by characters (since your inputs are short anyway)
# If you truly need token-accurate truncation, use tiktoken, but this is usually fine.
MAX_INPUT_CHARS = 300_000  # conservative; well below 128k tokens for typical text

def truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text[:max_chars] if len(text) > max_chars else text

# =========================
# OpenAI call (Responses API)
# =========================
def call_gpt(prompt_text: str) -> str:
    backoff = 2
    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=MODEL_ID,
                input=[{"role": "user", "content": prompt_text}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_NEW_TOKENS,
            )
            return (resp.output_text or "").strip()
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(30, backoff * 2)

    return ""

def is_done(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def main():
    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] INPUT_CSV={INPUT_CSV}")
    print(f"[INFO] OUTPUT_CSV={OUTPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if SRC_COL not in df.columns:
        raise ValueError(f"Missing required column: {SRC_COL}")

    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # Resume support: if output exists and has OUT_COL, load it
    if os.path.exists(OUTPUT_CSV):
        try:
            old = pd.read_csv(OUTPUT_CSV)
            if OUT_COL in old.columns and len(old) == len(df):
                df[OUT_COL] = old[OUT_COL].fillna("")
                print(f"[RESUME] loaded existing OUT_COL from {OUTPUT_CSV}")
        except Exception as e:
            print(f"[WARN] resume failed: {e}")

    pending = [i for i in range(len(df)) if not is_done(df.at[df.index[i], OUT_COL])]
    print(f"[INFO] total_rows={len(df)} pending={len(pending)}")

    processed = 0
    for local_i in tqdm(pending, desc="GPT-4o-mini extract"):
        idx = df.index[local_i]
        raw = df.at[idx, SRC_COL]

        if pd.isna(raw) or str(raw).strip() == "":
            df.at[idx, OUT_COL] = ""
            continue

        text = truncate_text(str(raw).strip(), MAX_INPUT_CHARS)
        prompt = build_prompt(text)

        try:
            out_text = call_gpt(prompt)
            df.at[idx, OUT_COL] = out_text
        except Exception as e:
            df.at[idx, OUT_COL] = f"ERROR: {e}"

        processed += 1
        if processed % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"[SAVED] {OUTPUT_CSV} (processed={processed})")

        time.sleep(SLEEP_S)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
