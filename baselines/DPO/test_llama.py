import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

ADAPTER_DIR = "DPO_training_mist_gen_text/output"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading fine-tuned LLaMA with adapter in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = True  # OK for inference
model.eval()

import pandas as pd

TEST_CSV = "llm_agents/output/zs_mistral_master_final.csv" 

df_test_full = pd.read_csv(TEST_CSV)

# Keep only rows 100 to 200 (inclusive) based on 0-based index

def get_merger_prompt(extractor_out, analyzer_out, reviewer_out, citation_out):
    return f"""You are a **Master Coordinator**, an expert in scientific communication and synthesis. Your task is to integrate limitations provided by four specialized agents: 

**Agents:**
1. **Extractor** (explicit limitations from the article).
2. **Analyzer** (inferred limitations from critical analysis).
3. **Reviewer** (limitations from an open review perspective).
4. **Citation** (limitations inferred from cited papers context).

**Goals**:
1. Combine all limitations into a cohesive, non-redundant list.
2. Ensure each limitation is clearly stated, scientifically valid, and aligned with the article's content.
3. Format the final list in a clear, concise, and professional manner.

**Output Format**:
- Numbered list of final limitations.
- For each: Clear statement, brief justification, and source in brackets (e.g., [Author-stated], [Inferred], [Peer-review-derived], [Citation-context]).

Extractor Agent Analysis:
{extractor_out}

Analyzer Agent Analysis:
{analyzer_out}

Reviewer Agent Analysis:
{reviewer_out}

Citation Agent Analysis:
{citation_out}

Please merge these four different perspectives into a comprehensive, well-organized analysis."""
    

df_test["master_prompt"] = df_test.apply(
    lambda row: get_merger_prompt(
        row["mistral_extractor"],
        row["mistral_analyzer"],
        row["mistral_reviewer"],
        row["mistral_citation"],
    ),
    axis=1,
)

from tqdm import tqdm

def generate_master_output(prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # slice **only** the newly generated tokens
    generated_ids = outputs[0, input_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion.strip()

# Apply to the test set
master_outputs = []
for prompt in tqdm(df_test["master_prompt"], desc="Generating master outputs"):
    master_outputs.append(generate_master_output(prompt))

df_test["llama_master_dpo"] = master_outputs

# Save results
OUT_CSV = "DPO_training_mist_gen_text/output_csv/df_test_with_llama_master_dpo.csv"
df_test.to_csv(OUT_CSV, index=False)
print("Saved test predictions to:", OUT_CSV)
