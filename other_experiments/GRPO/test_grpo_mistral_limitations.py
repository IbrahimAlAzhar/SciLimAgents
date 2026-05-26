# wrote in chatgpt in paper 
import argparse
import numpy as np
import pandas as pd
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.meteor_score import meteor_score

# NLTK Setup
def ensure_nltk():
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception: pass
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

def _safe_str(x) -> str:
    return str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else ""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Write the LIMITATIONS section for the given paper content. "
    "Be specific and do not invent results."
)

USER_TEMPLATE = """Write the LIMITATIONS section for the following paper.

Paper:
{paper_text}

Requirements:
- Use bullet points.
- Each bullet should be specific and actionable.
- Do NOT include anything other than the limitations.
"""

def build_prompt(tokenizer, paper_text, system_prompt, max_prompt_length):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(paper_text=_safe_str(paper_text))},
    ]
    # Use the same chat template as training
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_length:
        prompt_str = tokenizer.decode(ids[:max_prompt_length], skip_special_tokens=False)
    return prompt_str

@torch.no_grad()
def generate_one(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, # Greedy for reproducibility
        pad_token_id=tokenizer.eos_token_id,
    )
    # Slice off the prompt tokens
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# Metrics
def calculate_meteor(candidate, reference):
    cand = nltk.word_tokenize(_safe_str(candidate))
    ref = [nltk.word_tokenize(_safe_str(reference))]
    return float(meteor_score(ref, cand)) if reference and candidate else 0.0

def rouge_l_f1(candidate, reference):
    from nltk.util import ngrams
    cand, ref = nltk.word_tokenize(_safe_str(candidate)), nltk.word_tokenize(_safe_str(reference))
    if not cand or not ref: return 0.0
    # Basic LCS based F1
    def lcs(x, y):
        n, m = len(x), len(y)
        table = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if x[i-1] == y[j-1]: table[i][j] = table[i-1][j-1] + 1
                else: table[i][j] = max(table[i-1][j], table[i][j-1])
        return table[n][m]
    match = lcs(cand, ref)
    p, r = match/len(cand), match/len(ref)
    return (2*p*r)/(p+r) if (p+r) > 0 else 0.0

class BERTScorerWrapper:
    def __init__(self, model_type, device, batch_size):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(model_type=model_type, lang="en", device=device)
        self.batch_size = batch_size
    def f1(self, cands, refs):
        _, _, F = self.scorer.score(cands, refs, batch_size=self.batch_size)
        return [float(x) for x in F.tolist()]

def main():
    ensure_nltk()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Load Tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Base Model in 4-bit (to save memory during test)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load the GRPO-trained Adapter
    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    bertscorer = BERTScorerWrapper("distilbert-base-uncased", "cuda", 16)

    preds, mets, rouges, berts = [], [], [], []
    for t, gt in tqdm(zip(df["input_text_cleaned"], df["ground_truth_lim_peer"]), total=len(df)):
        prompt = build_prompt(tokenizer, t, DEFAULT_SYSTEM_PROMPT, 1024)
        pred = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        preds.append(pred)
        mets.append(calculate_meteor(pred, gt))
        rouges.append(rouge_l_f1(pred, gt))

    # Batch BERTScore for speed
    berts = bertscorer.f1(preds, df["ground_truth_lim_peer"].tolist())

    df["prediction_limitations"] = preds
    df["meteor_vs_gt"] = mets
    df["rougeL_f1_vs_gt"] = rouges
    df["bertscore_f1_vs_gt"] = berts

    df.to_csv(args.out_csv, index=False)
    print(f"Mean METEOR: {np.mean(mets):.4f} | ROUGE: {np.mean(rouges):.4f} | BERT: {np.mean(berts):.4f}")

if __name__ == "__main__":
    main()