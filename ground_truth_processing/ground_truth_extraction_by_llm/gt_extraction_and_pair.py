import pandas as pd
import ast

df = pd.read_csv("df.csv")

def get_best_matching_pairs(pairs_list):
    """
    Takes a list of comparison dictionaries.
    Groups by 'gt_gpt', finds the highest score, and keeps the first tie.
    """
    # 1. Safeguard for empty or malformed data
    if not isinstance(pairs_list, list) or len(pairs_list) == 0:
        return []
        
    best_matches = {}
    
    # 2. Loop through the list to find the highest scores
    for current_pair in pairs_list:
        gt_key = current_pair.get('gt_gpt')
        
        # If we haven't seen this gt_gpt before, save it as the current best
        if gt_key not in best_matches:
            best_matches[gt_key] = current_pair
        else:
            # If we HAVE seen it, check if the new score is strictly greater.
            # Using ">" instead of ">=" guarantees that if there is a tie (e.g., both are 1.0),
            # it ignores the new one and keeps the FIRST one it found.
            current_best_score = best_matches[gt_key].get('score', 0)
            new_score = current_pair.get('score', 0)
            
            if new_score > current_best_score:
                best_matches[gt_key] = current_pair
                
    # 3. Return the filtered list of the best pairs
    return list(best_matches.values())

# ==========================================
# How to apply it to your DataFrame
# ==========================================

# Assuming you already converted the string column to a list using the safe_list_convert 
# function we discussed earlier:
df['best_llm_decisions'] = df['llm_decisions'].apply(get_best_matching_pairs)

# If you specifically ONLY want to extract the 'gt_gpt' string from that winning pair 
# (and discard the gemini text and the score), you can add this quick extraction step:
df['winning_gt_gpt_only'] = df['best_llm_decisions'].apply(
    lambda x: [pair['gt_gpt'] for pair in x] if isinstance(x, list) else []
) 

import pandas as pd
import ast

def get_best_matching_pairs(pairs_list):
    """
    Takes a list of comparison dictionaries.
    Groups by 'gt_gpt', finds the highest score, and keeps the first tie.
    Safely handles None or missing scores.
    """
    if not isinstance(pairs_list, list) or len(pairs_list) == 0:
        return []
        
    best_matches = {}
    
    for current_pair in pairs_list:
        # Safeguard: ensure the item is actually a dictionary
        if not isinstance(current_pair, dict):
            continue
            
        gt_key = current_pair.get('gt_gpt')
        
        # Safely extract the new score. If it's None or missing, default to -1.0
        raw_new_score = current_pair.get('score')
        new_score = float(raw_new_score) if raw_new_score is not None else -1.0
        
        if gt_key not in best_matches:
            best_matches[gt_key] = current_pair
        else:
            # Safely extract the current best score for comparison
            raw_current_best = best_matches[gt_key].get('score')
            current_best_score = float(raw_current_best) if raw_current_best is not None else -1.0
            
            # Strict greater-than ensures we keep the first occurrence of a tie
            if new_score > current_best_score:
                best_matches[gt_key] = current_pair
                
    return list(best_matches.values())

# ==========================================
# Apply it to your DataFrame
# ==========================================
df['best_llm_decisions'] = df['llm_decisions'].apply(get_best_matching_pairs) 

import pandas as pd

# ==========================================
# Step 1: Filter out scores < 0.5
# ==========================================
def filter_low_scores(decisions_list):
    """Keeps only the dictionaries where the score is 0.5 or higher."""
    if not isinstance(decisions_list, list):
        return []
    
    valid_dicts = []
    for d in decisions_list:
        if not isinstance(d, dict):
            continue
        
        # Safely extract the score
        raw_score = d.get('score')
        score = float(raw_score) if raw_score is not None else -1.0
        
        # Keep the dictionary if the score is >= 0.5
        if score >= 0.5:
            valid_dicts.append(d)
            
    return valid_dicts

# Create the intermediate filtered column
df['filtered_llm_decisions'] = df['best_llm_decisions'].apply(filter_low_scores)

# ==========================================
# Step 2: Extract and join 'gt_gpt' strings
# ==========================================
def extract_and_join_gt(filtered_list):
    """Extracts 'gt_gpt' from the list of dicts and joins them with a newline."""
    if not isinstance(filtered_list, list) or len(filtered_list) == 0:
        return ""
    
    # Extract the string if the key exists
    extracted_texts = [str(d.get('gt_gpt')) for d in filtered_list if 'gt_gpt' in d]
    
    # Join the extracted texts with a newline character
    return "\n".join(extracted_texts)

# Create your final target column
df['ground_truth_lim_peer'] = df['filtered_llm_decisions'].apply(extract_and_join_gt) 

import pandas as pd

# ==========================================
# Step 1: Count the sentences safely
# ==========================================
def count_sentences(text):
    """Splits by newline and counts valid sentences, ignoring empty blanks."""
    # If the cell is completely empty or NaN, the count is 0
    if pd.isna(text) or str(text).strip() == "":
        return 0
    
    # Split by '\n' and only count the segments that actually contain text
    sentences = [s for s in str(text).split('\n') if s.strip() != ""]
    return len(sentences)

# Make a new column to store the exact count
df['limitation_count'] = df['ground_truth_lim_peer'].apply(count_sentences)

# ==========================================
# Step 2: Remove rows with 2 or fewer sentences
# ==========================================
# Create a new, clean dataframe that ONLY keeps rows with 3 or more sentences
df_filtered = df[df['limitation_count'] > 2].copy()

# (Optional Alternative) 
# If you wanted to keep ALL rows in your original dataframe, but just make a 
# new column that blanks out the text for the small ones, use this instead:
# df['ground_truth_final'] = df.apply(
#     lambda row: row['ground_truth_lim_peer'] if row['limitation_count'] > 2 else None, 
#     axis=1
# ) 

# Update df_filtered by only keeping rows where the count is NOT exactly 3
df_filtered = df_filtered[df_filtered['limitation_count'] != 3].copy()

# Alternatively, since we already filtered out 2 or less, 
# you can strictly ask for greater than 3:
# df_filtered = df_filtered[df_filtered['limitation_count'] > 3].copy() 

# The list of columns you want to remove
columns_to_drop = [
    "final_merged_limitations", 
    "full_chat_history", 
    "mistral_limitations_list"
]

# Safely drop the columns from both dataframes
df1 = df1.drop(columns=columns_to_drop, errors='ignore')
df2 = df2.drop(columns=columns_to_drop, errors='ignore')

# Alternatively, you can drop them in-place without reassigning the variable:
# df1.drop(columns=columns_to_drop, inplace=True, errors='ignore')
# df2.drop(columns=columns_to_drop, inplace=True, errors='ignore')