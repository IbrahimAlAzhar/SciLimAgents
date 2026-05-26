import pandas as pd
import ast
import os 
import pandas as pd  
QUERY_CSV_IN = "novelty_llm_agents_rag_based/df_with_retrieved_sections.csv"
df = pd.read_csv(QUERY_CSV_IN)  

df_large_source = pd.read_csv("data/aspect_data_knowledge_source/df_iclr_2017_20_nips_output.csv") 
# ============================================================
# 1) CONFIGURATION & PATHS
# ============================================================
# Update these paths to match your current session
# INPUT_PATH = "novelty_llm_agents_rag_based/df_with_retrieved_sections.csv"
# SOURCE_PATH = "data/iclr_icml_neurips_data/large_source.csv"
# OUTPUT_PATH = "novelty_llm_agents_rag_based/df_final_with_metadata.csv"

RETRIEVED_COLS = [
    'retrieved_abstract_3', 'retrieved_introduction_3', 'retrieved_method_3', 
    'retrieved_methodology_3', 'retrieved_experiments_3', 'retrieved_experimental_3', 
    'retrieved_conclusion_3', 'retrieved_conclusions_3'
]

# ============================================================
# 2) LOAD DATA
# ============================================================
print("Loading dataframes...")
# df = pd.read_csv(INPUT_PATH)
# df_large_source = pd.read_csv(SOURCE_PATH)

# Ensure ID types match (force to string to prevent mapping misses)
df_large_source['paper_id'] = df_large_source['paper_id'].astype(str)

# Create a high-performance metadata lookup map
metadata_lookup = df_large_source.set_index('paper_id')['metadata'].to_dict()

# ============================================================
# 3) PROCESSING FUNCTIONS
# ============================================================
def process_retrieval_row(row):
    """
    Parses strings into lists, extracts unique paper_ids, 
    and fetches corresponding metadata.
    """
    all_found_ids = []
    
    for col in RETRIEVED_COLS:
        raw_val = row.get(col, "")
        
        # Skip empty/NaN values
        if pd.isna(raw_val) or raw_val == "" or raw_val == "[]":
            continue
            
        try:
            # Convert string representation of list to Python list
            data_list = ast.literal_eval(str(raw_val))
            
            if isinstance(data_list, list):
                for item in data_list:
                    # Get the ID and ensure it's a string for the lookup
                    pid = str(item.get('paper_id', ""))
                    if pid:
                        all_found_ids.append(pid)
        except (ValueError, SyntaxError):
            continue
    
    # Deduplicate IDs while maintaining order
    unique_ids = list(dict.fromkeys(all_found_ids))
    
    # Map IDs to Metadata using our lookup dictionary
    relevant_meta = [metadata_lookup.get(pid) for pid in unique_ids if pid in metadata_lookup]
    
    return unique_ids, relevant_meta

# ============================================================
# 4) EXECUTION
# ============================================================
print(f"Processing {len(df)} rows across {len(RETRIEVED_COLS)} columns...")

# Apply the function (returns a Series of tuples)
combined_results = df.apply(process_retrieval_row, axis=1)

# Split the tuples into two distinct columns
df['similar_paper'] = combined_results.apply(lambda x: x[0])
df['relevant_papers'] = combined_results.apply(lambda x: x[1])

# ============================================================
# 5) VALIDATION & SAVE
# ============================================================
# Count how many rows successfully found metadata
success_count = df['relevant_papers'].apply(len).gt(0).sum()
print(f"Metadata mapping complete. Rows with successful matches: {success_count} / {len(df)}")

# df.to_csv(OUTPUT_PATH, index=False)
# print(f"File saved successfully to: {OUTPUT_PATH}")