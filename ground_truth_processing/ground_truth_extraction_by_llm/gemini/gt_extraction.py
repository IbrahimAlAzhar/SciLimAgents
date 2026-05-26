#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
from tqdm import tqdm

import google.generativeai as genai

# Optional (for approximate truncation). If not installed, we fallback to char truncation.
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# =========================
# Config
# =========================
INPUT_CSV = "df.csv"
OUTPUT_CSV = "df1.csv"

SRC_COL = "limitations_autho_peer_gt"
OUT_COL = "author_peer_gemini_ext"

# Gemini model
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# MODEL_ID = "gemini-3-flash-preview" 
# gemini-2.5-flash

# Generation
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_NEW_TOKENS = 3500

# Timeouts / retries
TIMEOUT_S = 600
SAVE_EVERY = 50
SLEEP_S = 1

# Truncation budget (approx tokens; using cl100k_base if available)
INPUT_TOK_BUDGET = 50_000

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# =========================
# Gemini client
# ========================= 

os.environ["GEMINI_API_KEY"] = ""
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_output_tokens": MAX_NEW_TOKENS,
}

try:
    model = genai.GenerativeModel(model_name=MODEL_ID, generation_config=generation_config)
except Exception as e:
    raise RuntimeError(f"Error initializing Gemini model '{MODEL_ID}': {e}")

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

# =========================
# Truncation helpers
# =========================
def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Gemini has its own tokenizer; this is an approximation.
    If tiktoken is available, we use cl100k_base. Otherwise fallback to char truncation.
    """
    if not text:
        return ""

    text = str(text)

    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text)
            if len(ids) <= max_tokens:
                return text
            return enc.decode(ids[:max_tokens]) + "... [TRUNCATED]"
        except Exception:
            pass

    # Fallback: 1 token ~ 4 chars rough heuristic
    max_chars = max_tokens * 4
    return text[:max_chars] + ("... [TRUNCATED]" if len(text) > max_chars else "")

# =========================
# Gemini call
# =========================
def call_gemini(prompt_text: str) -> str:
    backoff = 2
    for attempt in range(6):
        try:
            # request_options timeout is supported in recent google.generativeai versions
            resp = model.generate_content(
                prompt_text,
                request_options={"timeout": TIMEOUT_S},
            )

            # Safely extract text
            out = getattr(resp, "text", None)
            if out:
                return out.strip()

            # Fallback: try candidates if .text is empty/unavailable
            try:
                cand = resp.candidates[0]
                parts = cand.content.parts
                if parts and hasattr(parts[0], "text"):
                    return (parts[0].text or "").strip()
            except Exception:
                pass

            return "NO_TEXT_GENERATED"

        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(30, backoff * 2)

    return "NO_TEXT_GENERATED"

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

    # Resume support
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
    for local_i in tqdm(pending, desc="Gemini extract"):
        idx = df.index[local_i]
        raw = df.at[idx, SRC_COL]

        if pd.isna(raw) or str(raw).strip() == "":
            df.at[idx, OUT_COL] = ""
            continue

        text = truncate_text_to_tokens(str(raw).strip(), INPUT_TOK_BUDGET)
        prompt = build_prompt(text)

        try:
            out_text = call_gemini(prompt)
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
