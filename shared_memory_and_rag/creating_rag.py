#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import json
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# MANUAL INPUT / OUTPUT
# ==========================================
KNOWLEDGE_CSV = "data/aspect_data_knowledge_source/df_iclr_2017_20_nips_output.csv"

# (Optional) keep pickle output if you still want it
RAG_PICKLE_OUT = "novelty_llm_agents_rag_based/rag_data_source.pkl"

# NEW: CSV output
RAG_CSV_OUT = "novelty_llm_agents_rag_based/rag_data_source.csv"

# ==========================================
# Helpers
# ==========================================
def coerce_text(x: Any) -> str:
    """Convert anything into a clean string (or '' if unusable)."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    if isinstance(x, (list, tuple, set)):
        parts = []
        for item in x:
            s = coerce_text(item)
            if s:
                parts.append(s)
        return " ".join(parts).strip()

    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x).strip()

    if isinstance(x, np.generic):
        try:
            x = x.item()
        except Exception:
            pass

    return str(x).strip()

def is_bad_text(s: Any) -> bool:
    if not isinstance(s, str):
        return True
    t = s.strip().lower()
    return t == "" or t in {"none", "nan", "<na>", "null"}

def parse_maybe_dict(x: Any) -> Any:
    """Parse string -> python object using ast.literal_eval, then json. Return x if already dict/list."""
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    try:
        return json.loads(s)
    except Exception:
        return None

# ==========================================
# Build RAG chunks as a list (and optional dict)
# ==========================================
def build_rag_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns: list of chunk dicts.
    Each chunk dict includes the unique chunk_key plus metadata fields and text.
    """
    chunks: List[Dict[str, Any]] = []
    seen_keys = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building RAG chunks"):
        try:
            meta_raw = row.get("metadata", None)
            meta = parse_maybe_dict(meta_raw)

            # metadata sometimes stored as [dict]
            if isinstance(meta, list) and meta:
                meta = meta[0]

            if not isinstance(meta, dict):
                continue

            paper_id = coerce_text(row.get("paper_id", ""))
            row_id = coerce_text(row.get("id", ""))
            conf = coerce_text(row.get("conference", ""))
            dec = coerce_text(row.get("decision", ""))
            title = coerce_text(meta.get("title", ""))

            def add_chunk(section_name: Any, content: Any):
                sect = coerce_text(section_name)
                txt = coerce_text(content)

                if is_bad_text(sect) or is_bad_text(txt):
                    return

                base_key = f"{paper_id}{row_id}{conf}{dec}{title}{sect}"
                chunk_key = base_key

                # avoid accidental overwrite if duplicates
                if chunk_key in seen_keys:
                    dup_i = 2
                    chunk_key = f"{base_key}__dup{dup_i}"
                    while chunk_key in seen_keys:
                        dup_i += 1
                        chunk_key = f"{base_key}__dup{dup_i}"

                seen_keys.add(chunk_key)

                chunks.append(
                    {
                        "chunk_key": chunk_key,
                        "paper_id": paper_id,
                        "id": row_id,
                        "conference": conf,
                        "decision": dec,
                        "title": title,
                        "section_name": sect,
                        "text": txt,  # ALWAYS STRING
                    }
                )

            # abstract
            add_chunk("abstract", meta.get("abstractText", ""))

            # sections
            sections = meta.get("sections", [])
            if isinstance(sections, list):
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    heading = sec.get("heading", "")
                    text = sec.get("text", "")
                    add_chunk(heading, text)

        except Exception:
            continue

    return chunks

def main():
    print("Loading knowledge CSV:")
    print(KNOWLEDGE_CSV)
    df = pd.read_csv(KNOWLEDGE_CSV)

    chunks = build_rag_chunks(df)
    print(f"Total chunks created: {len(chunks)}")

    # --------------------------
    # 1) Save as CSV with rag_db
    # --------------------------
    # rag_db column stores the full dict as a JSON string (safe for CSV)
    out_rows = []
    for ch in chunks:
        out_rows.append(
            {
                "chunk_key": ch["chunk_key"],
                "rag_db": json.dumps(ch, ensure_ascii=False),
            }
        )

    rag_df = pd.DataFrame(out_rows)

    os.makedirs(os.path.dirname(RAG_CSV_OUT), exist_ok=True)
    rag_df.to_csv(RAG_CSV_OUT, index=False)
    print("Saved CSV rag_db to:")
    print(RAG_CSV_OUT)

    # --------------------------
    # 2) (Optional) also save pickle dict for faster loading
    # --------------------------
    rag_dict = {c["chunk_key"]: c for c in chunks}

    os.makedirs(os.path.dirname(RAG_PICKLE_OUT), exist_ok=True)
    with open(RAG_PICKLE_OUT, "wb") as f:
        pickle.dump(rag_dict, f)
    print("Saved pickle rag_data_source to:")
    print(RAG_PICKLE_OUT)

    # sanity sample
    if chunks:
        print("\nSample chunk dict keys:")
        print(list(chunks[0].keys()))
        print("Sample chunk_key:")
        print(chunks[0]["chunk_key"])
        print("Sample rag_db JSON (first 300 chars):")
        print(json.dumps(chunks[0], ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()
