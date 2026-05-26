"""
Evaluation Analysis
=====================
Paper-ready metrics and analysis utilities.
Compares SFT baseline vs GRPO-trained model performance.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

def compare_models(
    sft_results_path: str,
    grpo_results_path: str,
    output_dir: str,
) -> Dict:
    """
    Compare SFT baseline vs GRPO-trained model.
    Generates Table 1 for the paper.
    """
    sft_df = pd.read_csv(sft_results_path)
    grpo_df = pd.read_csv(grpo_results_path)

    metrics_cols = [
        "recall", "precision", "f1",
        "avg_cosine", "avg_jaccard", "avg_rougeL", "avg_bertscore",
        "reward_coverage", "reward_precision", "reward_groundedness",
        "reward_specificity", "reward_criticality", "reward_rule_based_total",
    ]

    results = {"Metric": [], "SFT Baseline": [], "GRPO Master": [], "Δ (Improvement)": []}

    for col in metrics_cols:
        if col in sft_df.columns and col in grpo_df.columns:
            sft_val = sft_df[col].mean()
            grpo_val = grpo_df[col].mean()
            delta = grpo_val - sft_val

            results["Metric"].append(col)
            results["SFT Baseline"].append(f"{sft_val:.4f}")
            results["GRPO Master"].append(f"{grpo_val:.4f}")
            results["Δ (Improvement)"].append(f"{delta:+.4f}")

    table_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("TABLE 1: SFT vs GRPO Comparison")
    print("=" * 80)
    print(table_df.to_string(index=False))

    table_path = os.path.join(output_dir, "comparison_table.csv")
    table_df.to_csv(table_path, index=False)

    return results

def analyze_grpo_convergence(metrics_path: str):
    """Analyze GRPO training convergence across iterations."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    print("\n" + "=" * 60)
    print("GRPO CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"{'Iteration':<12} {'Reward Mean':>14} {'Reward Std':>12} {'Avg Loss':>12} {'Avg KL':>10}")
    print("-" * 60)

    for m in metrics:
        it = m["iteration"]
        r_mean = m.get("reward_mean", 0)
        r_std = m.get("reward_std", 0)
        avg_loss = m["metrics"][-1]["avg_loss"] if m["metrics"] else 0
        avg_kl = m["metrics"][-1]["avg_kl"] if m["metrics"] else 0
        print(f"{it:<12} {r_mean:>14.4f} {r_std:>12.4f} {avg_loss:>12.4f} {avg_kl:>10.4f}")

def per_category_analysis(results_path: str):
    """Analyze performance across different limitation categories."""
    df = pd.read_csv(results_path)

    if "final_limitations" not in df.columns:
        log.warning("No final_limitations column found")
        return

    categories = {
        "Novelty/Significance": r"novel|significance|incremental|contribution",
        "Methodology": r"method|theoretical|proof|assumption|ablat",
        "Experimental": r"experiment|baseline|metric|comparison",
        "Generalization": r"general|robust|scalab|efficient",
        "Clarity/Reproducibility": r"clarity|reproduc|interpret",
        "Data/Ethics": r"data|bias|fairness|ethic",
    }

    import re

    print("\n" + "=" * 60)
    print("PER-CATEGORY COVERAGE ANALYSIS")
    print("=" * 60)

    for cat_name, pattern in categories.items():
        hits = df["final_limitations"].apply(
            lambda x: bool(re.search(pattern, str(x), re.IGNORECASE))
        )
        coverage = hits.mean() * 100
        print(f"  {cat_name:<30} {coverage:>6.1f}% of papers mention this category")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "other_experiments/grpo"

    # Convergence analysis
    metrics_path = os.path.join(output_dir, "grpo_master", "iteration_metrics.json")
    if os.path.exists(metrics_path):
        analyze_grpo_convergence(metrics_path)

    # Per-category analysis
    results_path = os.path.join(output_dir, "inference_results", "inference_results.csv")
    if os.path.exists(results_path):
        per_category_analysis(results_path) 
        