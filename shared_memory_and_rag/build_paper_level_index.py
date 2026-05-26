#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import ast
import re
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# PATHS
# =========================
DF_PATH = "data/nougat_output/nougat_all_papers_dataframe.csv"
OUT_DIR = "data_nougat_processing/paper_level_index"
os.makedirs(OUT_DIR, exist_ok=True)

META_PARQUET = os.path.join(OUT_DIR, "paper_meta.parquet")
FAISS_PATH   = os.path.join(OUT_DIR, "paper_faiss.index")

EMB_MODEL = "BAAI/bge-small-en-v1.5"

# =========================
# HELPERS
# =========================
def safe_ast(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def normalize_section_text(val):
    """paper_json sometimes stores list[str], sometimes str."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "\n".join(str(x) for x in val if str(x).strip())
    return str(val).strip()

def pick_conclusion_key(d):
    """Find a key that looks like conclusion."""
    if not isinstance(d, dict):
        return None
    for k in d.keys():
        kn = str(k).strip().lower()
        if "conclusion" in kn or "concluding" in kn:
            return k
    return None

def build_paper_text(paper_dict):
    """
    Extract only Abstract/Introduction/Conclusion from paper_json dict.
    paper_json example keys: "Abstract", "Introduction", "Conclusion" (sometimes)
    """
    if not isinstance(paper_dict, dict):
        return "", "", "", ""

    abs_txt = normalize_section_text(paper_dict.get("Abstract", "")) 
    # print('abstract text', abs_txt) 
    intro_txt = normalize_section_text(paper_dict.get("Introduction", ""))
    # print('intro text', intro_txt) 

    concl_key = "Conclusion" if "Conclusion" in paper_dict else pick_conclusion_key(paper_dict)
    concl_txt = normalize_section_text(paper_dict.get(concl_key, "")) if concl_key else "" 
    # print('conclusion text', concl_txt) 

    combined = "\n\n".join([
        f"Abstract:\n{abs_txt}" if abs_txt else "",
        f"Introduction:\n{intro_txt}" if intro_txt else "",
        f"Conclusion:\n{concl_txt}" if concl_txt else "",
    ]).strip() 
    # print('combined text', combined) 

    return abs_txt, intro_txt, concl_txt, combined

# =========================
# MAIN
# =========================
def main():
    # print("Loading df:", DF_PATH, flush=True)
    df = pd.read_csv(DF_PATH)

    # print("Parsing paper_json -> dict (ast)...", flush=True)
    paper_objs = df["paper_json"].apply(safe_ast)

    paper_ids = []
    abs_list = []
    intro_list = []
    concl_list = []
    combined_list = []

    # print("Extracting Abstract/Intro/Conclusion...", flush=True)
    for pid, obj in tqdm(zip(df.index.tolist(), paper_objs.tolist()), total=len(df)):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        abs_txt, intro_txt, concl_txt, combined = build_paper_text(obj)

        # Skip papers with no usable text in these sections
        if not combined:
            continue

        paper_ids.append(int(pid))
        abs_list.append(abs_txt)
        intro_list.append(intro_txt)
        concl_list.append(concl_txt)
        combined_list.append(combined)

    meta = pd.DataFrame({
        "paper_id": paper_ids,
        "abstract": abs_list,
        "introduction": intro_list,
        "conclusion": concl_list,
        "paper_text": combined_list
    })

    print("Saving meta parquet:", META_PARQUET, flush=True)
    meta.to_parquet(META_PARQUET, index=False)

    # Build FAISS (streaming add)
    print("Loading embedding model:", EMB_MODEL, flush=True)
    model = SentenceTransformer(EMB_MODEL)

    test = model.encode(["test"], convert_to_numpy=True).astype(np.float32)
    dim = int(test.shape[1])
    del test
    gc.collect()

    index = faiss.IndexFlatIP(dim)

    texts = meta["paper_text"].tolist()
    n = len(texts)
    print(f"Embedding and building FAISS: papers={n}, dim={dim}", flush=True)

    ENCODE_BATCH = 4096
    ST_BATCH = 256

    for start in tqdm(range(0, n, ENCODE_BATCH), desc="FAISS add"):
        batch_texts = texts[start:start+ENCODE_BATCH]
        emb = model.encode(batch_texts, batch_size=ST_BATCH, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype(np.float32, copy=False)
        faiss.normalize_L2(emb)
        index.add(emb)
        del emb
        gc.collect()

    faiss.write_index(index, FAISS_PATH)
    print("Saved FAISS index:", FAISS_PATH, flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()