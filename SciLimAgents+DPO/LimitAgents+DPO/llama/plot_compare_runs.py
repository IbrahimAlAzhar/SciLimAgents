#!/usr/bin/env python3
"""
Compare loss curves from multiple Hugging Face/TRL training output directories.

Works with either:
- train_history.csv, created by the improved trainers in this workspace
- trainer_state.json, created automatically by Hugging Face Trainer

Example:
  python3 plot_compare_runs.py \
    --run llama_sft_lr1e-5=/path/to/worker_sft_lr1e-5 \
    --run llama_sft_lr2e-5=/path/to/worker_sft_lr2e-5 \
    --metric eval_loss \
    --out /path/to/compare_eval_loss.png
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
            f"Missing train_history.csv and trainer_state.json in {output_dir}"
        )
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return pd.DataFrame(state.get("log_history", []))

def parse_run(value: str):
    if "=" not in value:
        name = os.path.basename(os.path.normpath(value))
        return name, value
    name, path = value.split("=", 1)
    return name.strip(), path.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in the form name=/path/to/output_dir. Can be repeated.",
    )
    p.add_argument("--metric", default="eval_loss")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    plt.figure(figsize=(9, 5.5))
    plotted = 0

    for run_arg in args.run:
        name, path = parse_run(run_arg)
        df = load_history(path)
        if "step" not in df.columns or args.metric not in df.columns:
            print(f"Skipping {name}: missing step or {args.metric}")
            continue
        sub = df[["step", args.metric]].dropna()
        if sub.empty:
            print(f"Skipping {name}: no values for {args.metric}")
            continue
        plt.plot(sub["step"], sub[args.metric], label=name, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No runs had metric {args.metric}")

    plt.title(f"{args.metric} comparison")
    plt.xlabel("Training step")
    plt.ylabel(args.metric)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
