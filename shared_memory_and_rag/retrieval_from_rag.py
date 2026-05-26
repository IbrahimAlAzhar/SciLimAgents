#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ==========================================
# INPUT / OUTPUT
# ==========================================
RAG_CSV_IN = "novelty_llm_agents_rag_based/rag_data_source.csv"
RAG_DB_COL = "rag_db"

QUERY_CSV_IN = "data/final_balanced_kde/df_updated_with_retrieval.csv"
QUERY_COL = "pdf_text"  # stringified dict with abstractText/sections

OUT_CSV = "novelty_llm_agents_rag_based/df_with_retrieved_sections.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
TOP_K = 3
MAX_CHARS_PER_CHUNK = 20000

# Debug: set >0 to process only first N rows of query df
LIMIT_QUERY_ROWS = 0

# ==========================================
# Helpers
# ==========================================
def coerce_text(x: Any) -> str:
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

def parse_ast_dict(x: Any) -> Any:
    """Parse python-literal dict/list stored as string (your case)."""
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
        return None

def normalize_heading(h: str) -> str:
    """lower + collapse spaces; keep letters/numbers/spaces."""
    h = coerce_text(h).lower()
    h = re.sub(r"\s+", " ", h).strip()
    return h

# ==========================================
# Load RAG from CSV (rag_db str -> dict)
# ==========================================
def load_rag_from_csv(rag_csv: str, rag_db_col: str = "rag_db") -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(rag_csv)
    if rag_db_col not in df.columns:
        raise ValueError(f"Column '{rag_db_col}' not found in RAG CSV. Found: {list(df.columns)}")

    rag_dict: Dict[str, Dict[str, Any]] = {}
    bad = 0

    for i, s in enumerate(tqdm(df[rag_db_col].tolist(), desc="Parsing rag_db")):
        d = parse_ast_dict(s)
        if not isinstance(d, dict):
            bad += 1
            continue

        # must have chunk_key
        ck = coerce_text(d.get("chunk_key", ""))
        if not ck:
            bad += 1
            continue

        rag_dict[ck] = d

    print(f"Loaded RAG rows: {len(df)} | usable chunks: {len(rag_dict)} | bad rows skipped: {bad}")
    if not rag_dict:
        raise ValueError("No usable rag_db entries found after parsing. Check rag_db formatting.")
    return rag_dict

# ==========================================
# Hybrid Retriever (FAISS + BM25) over RAG dict
# ==========================================
class HybridRetriever:
    def __init__(self, data_source_dict: Dict[str, Dict[str, Any]], model_name: str = MODEL_NAME):
        print("Defensive cleaning of knowledge source...")

        raw_keys: List[str] = []
        raw_corpus: List[str] = []
        raw_data_source: Dict[str, Dict[str, Any]] = {}

        for k, v in tqdm(data_source_dict.items(), desc="Scrubbing Data"):
            if not isinstance(v, dict):
                continue

            txt = coerce_text(v.get("text", ""))
            if len(txt) > MAX_CHARS_PER_CHUNK:
                txt = txt[:MAX_CHARS_PER_CHUNK]

            if is_bad_text(txt):
                continue

            kk = str(k)
            raw_keys.append(kk)
            raw_corpus.append(txt)
            raw_data_source[kk] = v

        if not raw_corpus:
            raise ValueError("All data was scrubbed out! Your rag_data_source has no usable text.")

        self.encoder = SentenceTransformer(model_name)

        print(f"Encoding {len(raw_corpus)} chunks (safe mode)...")
        embeddings, good_keys, good_corpus = self._encode_filter_bad(raw_corpus, raw_keys, batch_size=BATCH_SIZE)

        if len(good_corpus) == 0:
            raise ValueError("All chunks failed encoding. Something is wrong with the text.")

        self.keys = good_keys
        self.corpus = good_corpus
        self.data_source = {k: raw_data_source[k] for k in self.keys}

        print(f"✅ Encoded successfully: {len(self.corpus)} chunks")
        dropped = len(raw_corpus) - len(self.corpus)
        if dropped:
            print(f"⚠️ Dropped {dropped} chunks that could not be encoded (printed during encoding).")

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Retriever ready.")

    def _encode_filter_bad(
        self, corpus: List[str], keys: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        all_embs = []
        good_keys = []
        good_corpus = []

        for start in tqdm(range(0, len(corpus), batch_size), desc="Encoding batches"):
            batch_texts = corpus[start:start + batch_size]
            batch_keys = keys[start:start + batch_size]

            batch_texts = [t if isinstance(t, str) else coerce_text(t) for t in batch_texts]

            try:
                embs = self.encoder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                all_embs.append(embs)
                good_keys.extend(batch_keys)
                good_corpus.extend(batch_texts)
            except Exception:
                print(f"\n❌ Batch failed [{start}:{start+len(batch_texts)}]. Falling back to item-by-item.")
                for t, k in zip(batch_texts, batch_keys):
                    try:
                        emb = self.encoder.encode([t], convert_to_numpy=True, show_progress_bar=False)
                        all_embs.append(emb)  # shape (1, dim)
                        good_keys.append(k)
                        good_corpus.append(t)
                    except Exception:
                        print("=== DROPPED BAD CHUNK ===")
                        print("key:", k)
                        print("type:", type(t))
                        print("repr(first 300 chars):", repr(t[:300] if isinstance(t, str) else str(t)[:300]))
                        print("=========================")
                        continue

        embeddings = np.vstack(all_embs)
        return embeddings, good_keys, good_corpus

    def search(self, query_text: Any, filter_keyword: Any = None, top_k: int = 3) -> List[Dict[str, Any]]:
        q = coerce_text(query_text)
        if is_bad_text(q):
            return []

        q_emb = self.encoder.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        d_scores, d_indices = self.index.search(q_emb, min(len(self.keys), 100))
        s_scores = self.bm25.get_scores(q.lower().split())
        s_indices = np.argsort(s_scores)[::-1][:100]

        combined: Dict[str, float] = {}
        max_s = float(max(s_scores)) if len(s_scores) and max(s_scores) > 0 else 1.0

        for pos, idx in enumerate(d_indices[0]):
            if idx == -1:
                continue
            key = self.keys[idx]
            combined[key] = combined.get(key, 0.0) + float(d_scores[0][pos]) * 0.5

        for idx in s_indices:
            key = self.keys[idx]
            combined[key] = combined.get(key, 0.0) + (float(s_scores[idx]) / max_s) * 0.5

        sorted_candidates = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        clean_filter = re.sub(r"[^a-zA-Z]", "", str(filter_keyword)).lower() if filter_keyword else ""

        final_list = []
        for key, score in sorted_candidates:
            meta = self.data_source[key]
            section = re.sub(r"[^a-zA-Z]", "", coerce_text(meta.get("section_name", ""))).lower()

            if (not clean_filter) or (clean_filter in section):
                entry = dict(meta)
                entry["hybrid_score"] = float(score)
                final_list.append(entry)

            if len(final_list) >= top_k:
                break

        return final_list

# ==========================================
# Extract target query sections from pdf_text
# ==========================================
TARGETS = [
    "introduction",
    "method",
    "methodology",
    "experiments",
    "experimental",
    "conclusion",
    "conclusions",
]

def extract_target_queries(pdf_text_val: Any) -> Dict[str, str]:
    """
    Returns dict: {target_name: query_text}
    - abstract uses abstractText
    - others from sections[].heading contains target substring
    """
    data = parse_ast_dict(pdf_text_val)
    if not isinstance(data, dict):
        return {}

    out: Dict[str, str] = {}

    # abstract
    abs_text = coerce_text(data.get("abstractText", ""))
    if not is_bad_text(abs_text):
        out["abstract"] = abs_text

    # sections
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        return out

    for sec in sections:
        if not isinstance(sec, dict):
            continue
        heading = normalize_heading(sec.get("heading", ""))
        text = coerce_text(sec.get("text", ""))

        if is_bad_text(heading) or is_bad_text(text):
            continue

        # match targets by substring in heading
        for t in TARGETS:
            if t in heading:
                # if multiple sections match same target, concatenate
                if t in out:
                    out[t] = (out[t] + "\n\n" + text).strip()
                else:
                    out[t] = text.strip()

    return out

def main():
    # --------------------------
    # 1) Load RAG from CSV
    # --------------------------
    print("Loading RAG CSV:")
    print(RAG_CSV_IN)
    rag_data_source = load_rag_from_csv(RAG_CSV_IN, rag_db_col=RAG_DB_COL)
    print(f"Loaded rag_data_source chunks: {len(rag_data_source)}")

    # --------------------------
    # 2) Build retriever
    # --------------------------
    retriever = HybridRetriever(rag_data_source, model_name=MODEL_NAME)

    # --------------------------
    # 3) Load query df
    # --------------------------
    print("\nLoading query CSV:")
    print(QUERY_CSV_IN)
    df_query = pd.read_csv(QUERY_CSV_IN) 

    if QUERY_COL not in df_query.columns:
        raise ValueError(f"Column '{QUERY_COL}' not found. Available columns: {list(df_query.columns)}")

    if LIMIT_QUERY_ROWS and LIMIT_QUERY_ROWS > 0:
        df_query = df_query.head(LIMIT_QUERY_ROWS).copy()
        print(f"Debug: processing only first {LIMIT_QUERY_ROWS} rows")

    df_query = df_query.reset_index(drop=True)

    # --------------------------
    # 4) Run retrieval for targets
    # --------------------------
    print("\nExtracting target sections and retrieving top-K...")
    tqdm.pandas()

    # create empty columns first
    out_cols = ["abstract"] + TARGETS
    for s in out_cols:
        df_query[f"retrieved_{s}_{TOP_K}"] = ""  # store JSON string

    def retrieve_one_row(pdf_text_val: Any) -> Dict[str, List[Dict[str, Any]]]:
        qmap = extract_target_queries(pdf_text_val)
        results: Dict[str, List[Dict[str, Any]]] = {}
        for sec_name, qtext in qmap.items():
            # filter_keyword is sec_name (or "abstract")
            results[sec_name] = retriever.search(qtext, filter_keyword=sec_name, top_k=TOP_K)
        return results

    all_results = df_query[QUERY_COL].progress_apply(retrieve_one_row)

    # write to columns
    for i, res in enumerate(all_results):
        if not isinstance(res, dict):
            continue
        for sec in out_cols:
            hits = res.get(sec, [])
            # store as JSON for CSV safety
            df_query.at[i, f"retrieved_{sec}_{TOP_K}"] = json.dumps(hits, ensure_ascii=False)

    # --------------------------
    # 5) Save
    # --------------------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_query.to_csv(OUT_CSV, index=False)
    print("\nSaved output CSV:")
    print(OUT_CSV)
    print("Done.")

if __name__ == "__main__":
    main()
