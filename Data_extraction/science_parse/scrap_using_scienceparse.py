import os
import tempfile
from pathlib import Path
import requests
from science_parse_api.api import parse_pdf
import time
import arxiv  # <--- NEW IMPORT

# --- Configuration ---
HOST = "http:/127.0.0.1"
PORT = "8080"
TIMEOUT_SECONDS = 120
SAVE_PATH = "df.csv"

import pandas as pd 
df = pd.read_csv("df.csv")

# Initialize Arxiv Client
client = arxiv.Client(
    page_size=1,
    delay_seconds=3,  # Respectful delay built-in
    num_retries=3
)

# ... [Keep your DataFrame setup code here] ...

print(f"Starting processing from index 504...\nSaving to: {SAVE_PATH}")

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    processed_count = 0

    # Iterate rows
    for i, pdf_dict in df["pdf_text"].items():
        
        if not isinstance(pdf_dict, dict): continue
        refs = pdf_dict.get("references")
        if not isinstance(refs, list) or not refs: continue

        row_cited_in_data = {}
        seen_ids = set()
        counter = 1

        for j, ref in enumerate(refs):
            if not isinstance(ref, dict): continue

            raw_arxiv_id = ref.get("arxiv_id")
            if not raw_arxiv_id: continue
            
            arxiv_id_str = str(raw_arxiv_id).strip()
            if not arxiv_id_str or arxiv_id_str in seen_ids: continue

            print(f"\n▶ Row {i}, Ref {j} (ID: {arxiv_id_str})")
            
            # --- NEW DOWNLOAD LOGIC USING LIBRARY ---
            pdf_path = tmpdir_path / f"row_{i}_ref_{j}.pdf"
            download_success = False
            
            try:
                # Search for the paper by ID
                search = arxiv.Search(id_list=[arxiv_id_str])
                paper = next(client.results(search))
                
                # Download to the specific temp path
                paper.download_pdf(dirpath=str(tmpdir_path), filename=f"row_{i}_ref_{j}.pdf")
                download_success = True
                print("  ✅ Downloaded via arxiv library")
                
            except Exception as e:
                print(f"  ❌ Arxiv Library Download failed: {e}")
                continue
            # ----------------------------------------

            if not download_success:
                continue

            # Parse
            output_dict = None
            try:
                output_dict = parse_pdf(HOST, pdf_path, port=PORT)
            except Exception as e:
                print(f"  ❌ Parse error: {e}")
            finally:
                pdf_path.unlink(missing_ok=True)

            # Store result
            if output_dict:
                # Check for empty parse
                if output_dict.get('id') == 'empty' and not output_dict.get('sections'):
                     print(f"  ⚠ Parse returned 'empty'")
                else:
                    key_name = f"cited_in_paper_{counter}"
                    output_dict['original_arxiv_id'] = arxiv_id_str 
                    row_cited_in_data[key_name] = output_dict
                    seen_ids.add(arxiv_id_str)
                    print(f"  ✅ Parsed & Saved as key: '{key_name}'")
                    counter += 1

        # Assign and Save
        if row_cited_in_data:
            df.at[i, "cited_in"] = row_cited_in_data
            print(f"💾 Row {i}: Stored {len(row_cited_in_data)} papers.")

        processed_count += 1
        if processed_count % 10 == 0:
            df.to_csv(SAVE_PATH, index=False)
            print(f"⚡ Checkpoint saved.")

# Final Save
df.to_csv(SAVE_PATH, index=False)
print(f"\n🎉 Completed.") 
