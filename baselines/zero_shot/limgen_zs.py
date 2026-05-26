
import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
df = pd.read_csv("data/final_balanced_kde/df_balanced_kde_final.csv") 

# 1. Setup OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-mini"

# 2. Define the generation function
def generate_limitations_llm(text):
    """
    Sends text to the LLM with the prompt 'generate limitations'.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": f"You are a helpful assistant. Generate limitations based on the following text:\n\n{text}"}
            ],
            temperature=0.2, # Lower temperature for more deterministic/factual output
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# 3. Apply the function to the dataframe
# Using tqdm to show progress (install via 'pip install tqdm' if needed)
tqdm.pandas(desc="Generating Limitations")

# Assuming 'df' is your dataframe
# If you only want to run this on the 150 samples you created earlier, replace 'df' with 'df_sample'
df['zs_gpt_lim'] = df['input_text_cleaned'].progress_apply(generate_limitations_llm)

# 4. Save the result
output_dir = 'zero_shot/gpt_40_mini'
output_file = 'df_zs_gpt_output.csv'
full_path = os.path.join(output_dir, output_file)

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
df.to_csv(full_path, index=False)

print(f"Processing complete. Saved to: {full_path}") 

