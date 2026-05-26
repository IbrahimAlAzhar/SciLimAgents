import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm

df = pd.read_csv("other_experiments/dpo_novagents_llama_mistral/mistral/inference/inference_results.csv") 
# 1. Set up the OpenAI Client
client = OpenAI() # Assumes OPENAI_API_KEY is in your environment

def extract_limitations_with_gpt(text: str) -> str:
    """Calls GPT-4o-mini to extract and format limitations."""
    if pd.isna(text) or not str(text).strip():
        return ""
    
    system_prompt = (
        "You are a helpful assistant that extracts and consolidates limitations from text. "
        "Identify all limitations from the input. You MUST group similar, related, or duplicate "
        "limitations together into a single, cohesive point. "
        "Output the final, categorized limitations ONLY as a numbered list (1., 2., 3.) "
        "separated by newlines. Do not include any introductory text, conversational filler, "
        "or markdown formatting like bolding."
    )
    user_prompt = f"Extract, group, and list the limitations from the following text:\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error on text generation: {e}")
        return "Error extracting limitations."

# 2. Define output paths
output_dir = "other_experiments/dpo_novagents_llama_mistral/mistral"
os.makedirs(output_dir, exist_ok=True) # Creates the folder if it doesn't exist
output_file = os.path.join(output_dir, "df_final_extracted_limitations.csv")

# Create the target column if it doesn't exist yet
if 'formatted_limitations' not in df.columns:
    df['formatted_limitations'] = None

# 3. Process and save iteratively
save_interval = 10

# Using df.index ensures this works even if your dataframe index isn't a perfect 0,1,2 sequence
for count, i in enumerate(tqdm(df.index, desc="Processing rows")):
    
    # Optional: Skip rows that already have a result (useful if your job gets interrupted and you restart)
    if pd.notna(df.at[i, 'formatted_limitations']):
        continue
        
    text = df.at[i, 'generated_limitations']
    df.at[i, 'formatted_limitations'] = extract_limitations_with_gpt(text)
    
    if (count + 1) % save_interval == 0:
        df.to_csv(output_file, index=False)
        # Uncomment the line below if you want to see exactly when it saves in the console
        # tqdm.write(f"Checkpoint saved at row {count + 1}")

df.to_csv(output_file, index=False)
print(f"\nProcessing complete! Final dataframe saved to: {output_file}")