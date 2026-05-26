"""
Convert dpo_training_pairs.jsonl from [SYSTEM]/[USER] format
to Qwen2.5 chat template format (<|im_start|>/<|im_end|>).

Input:  dpo_training_pairs.jsonl  (prompt uses [SYSTEM]...[USER]...)
Output: dpo_training_pairs_chat.jsonl (prompt uses Qwen chat template)
"""

import json
import re

INPUT_JSONL = "other_experiments/DPO_perturb_based/dpo_train/final_dpo_pairs/dpo_training_pairs.jsonl"
OUTPUT_JSONL = "other_experiments/DPO_perturb_based/dpo_train/final_dpo_pairs/dpo_training_pairs_chat.jsonl"

def convert_prompt(old_prompt: str) -> str:
    """
    Convert:
        [SYSTEM] <system_text> [USER] <user_text>
    To:
        <|im_start|>system\n<system_text><|im_end|>\n<|im_start|>user\n<user_text><|im_end|>\n<|im_start|>assistant\n
    """
    # Extract system and user parts
    # Pattern: [SYSTEM] ... [USER] ...
    match = re.search(
        r"\[SYSTEM\]\s*(.*?)\s*\[USER\]\s*(.*)",
        old_prompt,
        re.DOTALL,
    )

    if match:
        system_text = match.group(1).strip()
        user_text = match.group(2).strip()
    else:
        # Fallback: no [SYSTEM] tag, treat entire prompt as user message
        system_text = (
            "You are an expert scientific paper reviewer. Your task is to identify ALL limitations "
            "of the given paper. Be thorough, specific, and evidence-grounded."
        )
        user_text = old_prompt.strip()

    new_prompt = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return new_prompt

# ---- Convert ----
converted = 0
with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)

        new_row = {
            "prompt": convert_prompt(row["prompt"]),
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
        converted += 1

print(f"Converted {converted} rows")
print(f"Saved to: {OUTPUT_JSONL}")

# ---- Verify first row ----
with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    first = json.loads(f.readline())

print("\n=== First row prompt (first 500 chars) ===")
print(first["prompt"][:500])
print("\n=== First row chosen (first 300 chars) ===")
print(first["chosen"][:300])