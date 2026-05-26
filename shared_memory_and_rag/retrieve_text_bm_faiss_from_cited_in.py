import os
import ast
import pandas as pd
from tqdm import tqdm

# ==========================================
# UPDATED IMPORTS (LangChain v0.2+ Compatible)
# ==========================================

# 1. Text Splitters (Moved to separate package)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Documents (Moved to core)
from langchain_core.documents import Document

# 3. Retrievers & Vector Stores
# Try modern imports first, keep fallback if you are using 'classic' shim
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ... rest of your code ...
# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these paths
INPUT_CSV = "data/final_balanced_kde/df_balanced_kde_final.csv"
OUTPUT_CSV = "data/final_balanced_kde/df_updated_with_retrieval.csv"

# Retrieval Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3 # Number of relevant chunks to retrieve per paper

# Initialize Embeddings (Global)
# We use a standard efficient model for scientific text
print("Loading Embeddings...")
hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def make_retriever_for_docs(docs, k=3):
    """
    Creates a hybrid FAISS + BM25 retriever for a specific set of documents.
    """
    if not docs:
        return None
        
    # FAISS (Dense)
    faiss_store = FAISS.from_documents(docs, hf_emb)
    faiss_r = faiss_store.as_retriever(search_kwargs={"k": k})

    # BM25 (Sparse)
    bm25_r = BM25Retriever.from_documents(docs)
    bm25_r.k = k

    # Ensemble (50/50 weight)
    return EnsembleRetriever(
        retrievers=[faiss_r, bm25_r],
        weights=[0.5, 0.5]
    )

def parse_citations_to_docs(cited_entry):
    """
    Converts the 'cited_in' column (string/dict) into a list of LangChain Documents.
    """
    if pd.isna(cited_entry):
        return []

    # 1. Parse string to dict if needed
    data = cited_entry
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except:
            return [Document(page_content=str(data))]

    # 2. Extract text from the dictionary structure
    # Structure assumed: {'PaperID': {'title': '...', 'abstract': '...', 'sections': [...]}}
    docs = []
    if isinstance(data, dict):
        for paper_id, details in data.items():
            if not isinstance(details, dict): 
                continue
                
            # Combine Title + Abstract + Intro/Sections
            text_parts = []
            title = details.get("title", "")
            if title: text_parts.append(f"Title: {title}")
            
            abstract = details.get("abstractText") or details.get("abstract") or ""
            if abstract: text_parts.append(f"Abstract: {abstract}")
            
            # Extract text from sections if available
            sections = details.get("sections", [])
            if isinstance(sections, list):
                for sec in sections:
                    if isinstance(sec, dict):
                        heading = str(sec.get("heading", ""))
                        content = str(sec.get("text", ""))
                        text_parts.append(f"{heading}: {content}")
            
            full_text = "\n\n".join(text_parts)
            if full_text.strip():
                docs.append(Document(page_content=full_text, metadata={"source": title}))
                
    elif isinstance(data, list):
        # Fallback if it's just a list of strings
        for item in data:
            docs.append(Document(page_content=str(item)))
            
    return docs

# ==========================================
# 3. MAIN LOOP
# ==========================================

def run_retrieval_update():
    # Load Data
    try:
        df = pd.read_csv(INPUT_CSV) 
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("Input CSV not found.")
        return

    # Initialize new column
    if "cited_in_ret" not in df.columns:
        df["cited_in_ret"] = ""

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("Starting Row-by-Row Retrieval Update...")
    
    # Iterate through DataFrame
    for i, row in tqdm(df.iterrows(), total=len(df)):
        
        # A. Prepare Query (Main Paper Text)
        # We use the first 2000 chars of the main paper as the "query" to find relevant citation info
        query_text = str(row.get("input_text_cleaned", ""))[:2000]
        
        # B. Prepare Citation Documents
        raw_citation_docs = parse_citations_to_docs(row.get("cited_in", ""))
        
        # C. Chunking
        if not raw_citation_docs:
            df.at[i, "cited_in_ret"] = "No citations found."
        else:
            chunked_docs = text_splitter.split_documents(raw_citation_docs)
            
            if not chunked_docs:
                 df.at[i, "cited_in_ret"] = "No content in citations."
            else:
                try:
                    # D. Build Retriever (Local Index for this row only)
                    ensemble_retriever = make_retriever_for_docs(chunked_docs, k=TOP_K)
                    
                    # E. Retrieve
                    relevant_docs = ensemble_retriever.invoke(query_text)
                    
                    # F. Format Result
                    # We combine the retrieved chunks into a single string
                    retrieved_text = "\n\n---\n\n".join(
                        [f"[From {d.metadata.get('source', 'Unknown')}]: {d.page_content}" for d in relevant_docs]
                    )
                    df.at[i, "cited_in_ret"] = retrieved_text
                    
                except Exception as e:
                    print(f"Error on row {i}: {e}")
                    df.at[i, "cited_in_ret"] = f"Error during retrieval: {e}"

        # G. Save Update (Overwrites file after EVERY row)
        # This ensures if the script crashes, you lose at most 1 row of progress
        df.to_csv(OUTPUT_CSV, index=False)

    print(f"Update complete. Final file saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_retrieval_update()