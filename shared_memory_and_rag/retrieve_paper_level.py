#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import ast
import json
import re
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# PATHS
# =========================
DF1_PATH = "data/final_balanced_kde/df_updated_with_retrieval.csv"

INDEX_DIR = "data_nougat_processing/paper_level_index"
META_PARQUET = os.path.join(INDEX_DIR, "paper_meta.parquet")
FAISS_PATH   = os.path.join(INDEX_DIR, "paper_faiss.index")

OUT_CSV = "data_nougat_processing/df.csv"

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

def safe_to_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def normalize_text(s: str) -> str:
    s = str(s).replace("\\", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_query_sections(pdf_obj):
    """
    pdf_text format example has:
      - abstractText: str
      - sections: list of dicts with heading/text
    We will collect:
      - abstractText
      - any section whose heading contains 'introduction'
      - any section whose heading contains 'conclusion'
    """
    if not isinstance(pdf_obj, dict):
        return "", "", "", ""

    abs_txt = normalize_text(pdf_obj.get("abstractText", ""))

    intro_txt = ""
    concl_txt = ""

    secs = pdf_obj.get("sections", [])
    if isinstance(secs, list):
        for sec in secs:
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "")).lower()
            text = normalize_text(sec.get("text", ""))
            if not text:
                continue

            if "introduction" in heading and not intro_txt:
                intro_txt = text
            if "conclusion" in heading and not concl_txt:
                concl_txt = text

    combined = "\n\n".join([
        f"Abstract:\n{abs_txt}" if abs_txt else "",
        f"Introduction:\n{intro_txt}" if intro_txt else "",
        f"Conclusion:\n{concl_txt}" if concl_txt else "",
    ]).strip() 
    print('abstract text', abs_txt) 
    print('intro text', intro_txt) 
    print('conclusion text', concl_txt)
    print('combined text', combined)

    return abs_txt, intro_txt, concl_txt, combined

def cheap_overlap_score(q: str, doc: str) -> float:
    """
    Very fast lexical score for reranking within FAISS topK.
    This is NOT BM25; it's just a quick overlap proxy.
    """
    q_words = set(re.findall(r"[a-z0-9]+", q.lower()))
    if not q_words:
        return 0.0
    d_words = set(re.findall(r"[a-z0-9]+", doc.lower()))
    inter = len(q_words & d_words)
    return inter / (len(q_words) ** 0.5)

# =========================
# MAIN
# =========================
def main():
    print("Loading meta + FAISS index...", flush=True)
    meta = pd.read_parquet(META_PARQUET)
    index = faiss.read_index(FAISS_PATH)

    if index.ntotal != len(meta):
        raise RuntimeError("FAISS ntotal != meta rows. Index and parquet out of sync.")

    print("Loading embedding model:", EMB_MODEL, flush=True)
    model = SentenceTransformer(EMB_MODEL)

    print("Loading df1:", DF1_PATH, flush=True)
    df1 = pd.read_csv(DF1_PATH)

    if "pdf_text" not in df1.columns:
        raise ValueError("df1 missing column: pdf_text")

    # output columns
    df1["query_sections_used"] = None
    df1["top_paper_ids"] = None
    df1["top_paper_scores"] = None

    # retrieval settings
    TOPK_FAISS = 200   # fetch 200 candidates fast
    FINAL_TOPK = 20    # keep best 20 after rerank
    CHECKPOINT_EVERY = 25

    print("Running retrieval...", flush=True)
    try:
        for i in tqdm(range(len(df1)), desc="df1 retrieval"):
            pdf_obj = safe_ast(df1.at[df1.index[i], "pdf_text"])
            if isinstance(pdf_obj, list) and pdf_obj and isinstance(pdf_obj[0], dict):
                pdf_obj = pdf_obj[0]

            abs_txt, intro_txt, concl_txt, query_text = extract_query_sections(pdf_obj)

            df1.at[df1.index[i], "query_sections_used"] = compact_json({
                "abstract_len": len(abs_txt),
                "intro_len": len(intro_txt),
                "concl_len": len(concl_txt),
            })

            if not query_text:
                df1.at[df1.index[i], "top_paper_ids"] = compact_json([])
                df1.at[df1.index[i], "top_paper_scores"] = compact_json([])
                continue

            # FAISS search
            q_emb = model.encode(
                ["Represent this sentence for searching relevant passages: " + query_text],
                convert_to_numpy=True
            ).astype(np.float32, copy=False)
            faiss.normalize_L2(q_emb)

            scores, ids = index.search(q_emb, TOPK_FAISS)
            ids = ids[0].tolist()
            scores = scores[0].tolist()

            # Rerank cheaply within topK_FAISS
            candidates = []
            for cid, faiss_sc in zip(ids, scores):
                if cid == -1:
                    continue
                paper_id = int(meta.iloc[cid]["paper_id"])
                doc_text = meta.iloc[cid]["paper_text"]
                ov = cheap_overlap_score(query_text, doc_text)
                # fused score (tune weights if you want)
                fused = float(faiss_sc) + 0.05 * float(ov)
                candidates.append((paper_id, fused, float(faiss_sc), float(ov)))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:FINAL_TOPK]

            df1.at[df1.index[i], "top_paper_ids"] = compact_json([t[0] for t in top])
            df1.at[df1.index[i], "top_paper_scores"] = compact_json([
                {"paper_id": t[0], "fused": t[1], "faiss": t[2], "overlap": t[3]} for t in top
            ])

            if (i + 1) % CHECKPOINT_EVERY == 0:
                safe_to_csv(df1, OUT_CSV)
                print(f"\n--- checkpoint saved at row {i+1} ---", flush=True)

    finally:
        safe_to_csv(df1, OUT_CSV)

    print("DONE. Saved:", OUT_CSV, flush=True)

if __name__ == "__main__":
    main()