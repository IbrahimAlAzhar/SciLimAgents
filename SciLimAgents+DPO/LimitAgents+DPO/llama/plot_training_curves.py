#!/usr/bin/env python
"""
Plot loss curves from a Hugging Face/TRL output directory.

It first uses train_history.csv from the callback in these scripts. If that file
does not exist, it falls back to trainer_state.json.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def load_history(output_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(output_dir, "train_history.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Could not find train_history.csv or trainer_state.json in {output_dir}"
        )
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return pd.DataFrame(state.get("log_history", []))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = load_history(args.output_dir)
    if "step" not in df.columns:
        raise ValueError("No step column found in the log history.")

    metrics = [
        "loss",
        "eval_loss",
        "learning_rate",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
    ]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not metrics:
        raise ValueError("No plottable metrics found.")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[["step", metric]].dropna()
        ax.plot(sub["step"], sub[metric], linewidth=1.8)
        ax.set_title(metric)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    out = args.out or os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(out, dpi=180)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

