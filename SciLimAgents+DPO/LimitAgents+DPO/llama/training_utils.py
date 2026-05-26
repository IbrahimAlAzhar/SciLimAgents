import json
import os
import random
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainerCallback

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda name: int(name.rsplit("-", 1)[1]))
    return os.path.join(output_dir, ckpts[-1])

def load_sft_messages(path: str, min_assistant_chars: int = 1) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    processed: List[Dict] = []
    dropped = 0
    for row in records:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or not messages:
            dropped += 1
            continue
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            dropped += 1
            continue
        assistant_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "assistant"
        ).strip()
        if len(assistant_text) < min_assistant_chars:
            dropped += 1
            continue
        processed.append({"messages": messages})

    print(f"Loaded {len(records)} SFT records from {path}")
    print(f"After filtering: {len(processed)} valid samples, dropped {dropped}")
    if not processed:
        raise ValueError(f"No valid SFT samples found in {path}")
    return Dataset.from_list(processed)

def load_sft_prompt_completion(path: str, min_assistant_chars: int = 1) -> Dataset:
    """
    Convert chat messages into TRL prompt-completion conversational format.

    This avoids assistant_only_loss=True, which requires a tokenizer chat template
    with assistant masks. Instead, SFTTrainer can use completion_only_loss=True
    and train only on the final assistant answer.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    processed: List[Dict] = []
    dropped = 0
    for row in records:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or not messages:
            dropped += 1
            continue
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            dropped += 1
            continue

        assistant_indices = [
            idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"
        ]
        if not assistant_indices:
            dropped += 1
            continue

        last_assistant_idx = assistant_indices[-1]
        prompt = messages[:last_assistant_idx]
        completion_msg = messages[last_assistant_idx]
        completion_text = str(completion_msg.get("content", "")).strip()

        if not prompt or len(completion_text) < min_assistant_chars:
            dropped += 1
            continue

        processed.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": completion_text}],
            }
        )

    print(f"Loaded {len(records)} SFT records from {path}")
    print(f"After prompt-completion conversion: {len(processed)} valid samples, dropped {dropped}")
    if not processed:
        raise ValueError(f"No valid SFT prompt-completion samples found in {path}")
    return Dataset.from_list(processed)

def load_dpo_messages(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    processed: List[Dict] = []
    dropped = 0
    for row in records:
        prompt = row.get("input_messages", [])
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected:
            dropped += 1
            continue
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
            dropped += 1
            continue

        # TRL DPO prompt-completion conversational format:
        # prompt is the context, chosen/rejected are the assistant completions only.
        processed.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )

    print(f"Loaded {len(records)} DPO records from {path}")
    print(f"After filtering: {len(processed)} valid pairs, dropped {dropped}")
    if not processed:
        raise ValueError(f"No valid DPO pairs found in {path}")
    return Dataset.from_list(processed)

def split_dataset(dataset: Dataset, eval_split: float, seed: int):
    if eval_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
        return split["train"], split["test"]
    print(f"Train: {len(dataset)}, Eval: None")
    return dataset, None

class LossHistoryCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.rows: List[Dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                row[key] = value
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(
            os.path.join(self.output_dir, "train_history.csv"), index=False
        )

def plot_history(output_dir: str) -> None:
    history_path = os.path.join(output_dir, "train_history.csv")
    if not os.path.exists(history_path):
        print(f"No train_history.csv found at {history_path}")
        return

    df = pd.read_csv(history_path)
    if df.empty or "step" not in df.columns:
        print("History file is empty or missing the step column.")
        return

    metrics = [
        ("loss", "Training loss"),
        ("eval_loss", "Validation loss"),
        ("learning_rate", "Learning rate"),
        ("rewards/chosen", "DPO chosen reward"),
        ("rewards/rejected", "DPO rejected reward"),
        ("rewards/margins", "DPO reward margin"),
        ("logps/chosen", "DPO chosen logp"),
        ("logps/rejected", "DPO rejected logp"),
    ]
    available = [(col, title) for col, title in metrics if col in df.columns]
    if not available:
        print("No known plottable metrics found.")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(9, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        metric_df = df[["step", col]].dropna()
        ax.plot(metric_df["step"], metric_df[col], linewidth=1.8)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    out_png = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out_png, dpi=180)
    print(f"Saved training curves to {out_png}")
