# take ground truth with input text for fine tuning 
import pandas as pd
import torch
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
# UPDATED: Input file path to the balanced dataset
INPUT_FILE = 'df_balanced_kde_final.csv'

# Output directories
OUTPUT_DIR = 'SFT_from_zs_output/mistral/output_using_ground_truth'
MODEL_SAVE_DIR = 'SFT_from_zs_output/mistral/output_using_ground_truth/model'
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "models/mistral_7b_v3_instruct"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. Data Preparation ---
print("Loading Training Data...")
df = pd.read_csv(INPUT_FILE)

train_df = df.copy()
print(f"Training set size: {len(train_df)}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)

# --- 2. Formatting Function ---
def format_instruction(sample):
    """Formats the input into the instruction format Mistral expects."""
    system_prompt = "You are a helpful assistant, generate limitations from these texts"
    
    # Using 'input_text' as input
    user_text = f"Text:\n{sample['input_text']}\n\nLimitations:"
    
    # Using 'ground_truth_lim_peer' as target
    ground_truth = sample['ground_truth_lim_peer']
    
    # Construct the full prompt: System + User Instruction + Ground Truth Response
    full_text = f"<s>[INST] {system_prompt}\n\n{user_text} [/INST] {ground_truth}</s>"
    return {"text": full_text}

train_dataset = train_dataset.map(format_instruction)

# --- 3. Model Setup (QLoRA) ---
print("Loading Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# --- 4. Training ---

sft_config = SFTConfig(
    output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),

    # 🔹 These belong in SFTConfig (not SFTTrainer)
    dataset_text_field="text",   # name of column after your .map(format_instruction)
    max_length=4096,             # ✅ correct arg name (NOT max_seq_length)
    packing=False,

    # training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

print("Starting Training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=sft_config,     # pass the SFTConfig here
    # ❌ do NOT pass tokenizer, dataset_text_field, max_seq_length, packing here
)

trainer.train()

# --- 5. Saving ---
print(f"Saving Fine-Tuned Model to {MODEL_SAVE_DIR}...")
trainer.model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print("Training Complete.")
