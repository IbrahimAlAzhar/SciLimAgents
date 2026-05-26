"""
End-to-End GRPO Pipeline (v2)
===============================
Training order: Worker → Master → Leader
Inference: uses GRPO-trained models for all three agent types

Training rationale:
  1. Worker FIRST: the unified worker model learns to produce better
     per-role limitation analysis. Master/leader benefit from better inputs.
  2. Master SECOND: learns to consolidate (now with better worker outputs).
  3. Leader LAST: learns which workers to call and how to give feedback.
     This needs good worker + master models to evaluate its decisions.

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage grpo_worker
  python run_pipeline.py --stage grpo_master
  python run_pipeline.py --stage grpo_leader
  python run_pipeline.py --stage inference
  python run_pipeline.py --stage evaluate
"""

import os
import gc
import json
import random
import logging
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

from config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

# ================================================================
# DATA LOADING
# ================================================================

def load_train_data(config: PipelineConfig) -> List[Dict]:
    """Load N random training samples."""
    log.info(f"Loading training data from {config.paths.train_csv}")
    df = pd.read_csv(config.paths.train_csv)
    df = df.dropna(subset=[config.paths.train_input_col, config.paths.train_gt_col])
    df = df[df[config.paths.train_input_col].str.len() > 100]
    log.info(f"Valid training rows: {len(df)}")

    if config.num_train_samples and len(df) > config.num_train_samples:
        df = df.sample(n=config.num_train_samples, random_state=config.seed)
    log.info(f"Using {len(df)} training samples")

    return [
        {"idx": i, "text": str(row[config.paths.train_input_col]),
         "ground_truth": str(row[config.paths.train_gt_col])}
        for i, (_, row) in enumerate(df.iterrows())
    ]

# ================================================================
# STAGE: REWARD MODEL
# ================================================================

def stage_train_reward_model(config: PipelineConfig, train_papers: List[Dict]):
    """Train reward model on preference pairs from initial rollouts."""
    from multi_agent_rollout import generate_all_rollouts, load_agent_model
    from reward_functions import (
        score_all_rollouts, create_preference_pairs, TrainedRewardModel,
    )

    log.info("=" * 60)
    log.info("STAGE: Training Reward Model")
    log.info("=" * 60)

    rm_papers = train_papers[:min(200, len(train_papers))]

    model, tokenizer = load_agent_model(config)
    rollout_path = os.path.join(config.paths.reward_model_dir, "rm_rollouts.json")
    all_trajs = generate_all_rollouts(model, tokenizer, rm_papers, config, save_path=rollout_path)
    all_scores = score_all_rollouts(all_trajs, config)
    pairs = create_preference_pairs(all_trajs, all_scores, config.reward)

    with open(os.path.join(config.paths.reward_model_dir, "preference_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=2, default=str)

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size if hasattr(model, "config") else 2048
    rm = TrainedRewardModel(model, tokenizer, device, hidden_size)
    rm.train_on_preferences(pairs, epochs=config.reward.reward_train_epochs)
    rm.save(os.path.join(config.paths.reward_model_dir, "reward_head.pt"))

    del model; gc.collect(); torch.cuda.empty_cache()
    return os.path.join(config.paths.reward_model_dir, "reward_head.pt")

# ================================================================
# STAGE: GRPO WORKER
# ================================================================

def stage_grpo_worker(config: PipelineConfig, train_papers: List[Dict]) -> str:
    """
    Train the unified worker agent via GRPO.
    One model handles all 7 roles; trained on per-role quality rewards.
    """
    from grpo_trainer import iterative_grpo

    log.info("=" * 60)
    log.info("STAGE: GRPO Worker Training (unified model, all roles)")
    log.info("=" * 60)

    final_dir, metrics = iterative_grpo(
        config, train_papers,
        agent_type="worker",
        num_iterations=config.grpo.worker_num_grpo_iterations,
    )
    log.info(f"Worker training complete: {final_dir}")
    return final_dir

# ================================================================
# STAGE: GRPO MASTER
# ================================================================

def stage_grpo_master(
    config: PipelineConfig,
    train_papers: List[Dict],
    worker_dir: Optional[str] = None,
) -> str:
    """
    Train the master agent via GRPO.
    Uses GRPO-trained worker model for generating better inputs.
    """
    from grpo_trainer import iterative_grpo

    log.info("=" * 60)
    log.info("STAGE: GRPO Master Training")
    log.info("=" * 60)

    final_dir, metrics = iterative_grpo(
        config, train_papers,
        agent_type="master",
        num_iterations=config.grpo.num_grpo_iterations,
        worker_model_dir=worker_dir,
    )
    log.info(f"Master training complete: {final_dir}")
    return final_dir

# ================================================================
# STAGE: GRPO LEADER
# ================================================================

def stage_grpo_leader(
    config: PipelineConfig,
    train_papers: List[Dict],
    worker_dir: Optional[str] = None,
    master_dir: Optional[str] = None,
) -> str:
    """
    Train the leader agent via GRPO.
    Uses GRPO-trained worker + master for rollout evaluation.
    The leader learns which workers to call and how to give feedback.
    """
    from grpo_trainer import iterative_grpo

    log.info("=" * 60)
    log.info("STAGE: GRPO Leader Training")
    log.info("=" * 60)

    final_dir, metrics = iterative_grpo(
        config, train_papers,
        agent_type="leader",
        num_iterations=config.grpo.leader_num_grpo_iterations,
        worker_model_dir=worker_dir,
        master_model_dir=master_dir,
    )
    log.info(f"Leader training complete: {final_dir}")
    return final_dir

# ================================================================
# STAGE: INFERENCE
# ================================================================

def stage_inference(
    config: PipelineConfig,
    worker_dir: Optional[str] = None,
    leader_dir: Optional[str] = None,
    master_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Run inference using all GRPO-trained models."""
    from inference import run_inference

    log.info("=" * 60)
    log.info("STAGE: Inference (all GRPO-trained agents)")
    log.info("=" * 60)

    results = run_inference(
        config,
        worker_dir=worker_dir,
        leader_dir=leader_dir,
        master_dir=master_dir,
        max_samples=max_samples,
    )
    log.info(f"Inference complete: {len(results)} samples")
    return results

# ================================================================
# STAGE: EVALUATE
# ================================================================

def stage_evaluate(
    config: PipelineConfig,
    worker_dir: Optional[str] = None,
    leader_dir: Optional[str] = None,
    master_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Full evaluation with GPT-4o-mini pointwise scoring."""
    from inference import full_evaluation

    log.info("=" * 60)
    log.info("STAGE: Evaluation")
    log.info("=" * 60)

    summary = full_evaluation(
        config,
        worker_dir=worker_dir,
        leader_dir=leader_dir,
        master_dir=master_dir,
        api_key=api_key,
        max_samples=max_samples,
    )
    return summary

# ================================================================
# FULL PIPELINE
# ================================================================

def _find_checkpoint(grpo_dir, n_iter):
    """Helper to find existing checkpoint."""
    for i in range(n_iter, 0, -1):
        p = os.path.join(grpo_dir, f"iteration_{i}", "final")
        if os.path.exists(p):
            return p
    return None

def run_full_pipeline(config: PipelineConfig, args):
    """
    Complete end-to-end pipeline:

    Training order: Worker → Master → Leader
    Then: Inference using all three GRPO-trained models

    Architecture:
    Paper → Leader decides workers → Selected Workers analyze
          → Leader reviews & gives feedback → Master consolidates
          → Final limitation set
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_papers = load_train_data(config)

    # Save config
    with open(os.path.join(config.paths.output_base, "pipeline_config.json"), "w") as f:
        json.dump({"paths": vars(config.paths), "grpo": vars(config.grpo)},
                  f, indent=2, default=str)

    worker_dir = None
    master_dir = None
    leader_dir = None

    # ── Stage 0: [Optional] Reward Model ──
    if args.stage in ("all", "reward_model"):
        if config.reward.train_reward_model:
            stage_train_reward_model(config, train_papers)

    # ── Stage 1: GRPO Worker (train FIRST) ──
    if args.stage in ("all", "grpo_worker"):
        worker_dir = stage_grpo_worker(config, train_papers)
    else:
        worker_dir = _find_checkpoint(
            config.paths.grpo_worker_dir, config.grpo.worker_num_grpo_iterations
        )

    # ── Stage 2: GRPO Master (uses trained worker) ──
    if args.stage in ("all", "grpo_master"):
        master_dir = stage_grpo_master(config, train_papers, worker_dir=worker_dir)
    else:
        master_dir = _find_checkpoint(
            config.paths.grpo_master_dir, config.grpo.num_grpo_iterations
        )

    # ── Stage 3: GRPO Leader (uses trained worker + master) ──
    if args.stage in ("all", "grpo_leader"):
        leader_dir = stage_grpo_leader(
            config, train_papers,
            worker_dir=worker_dir, master_dir=master_dir,
        )
    else:
        leader_dir = _find_checkpoint(
            config.paths.grpo_leader_dir, config.grpo.leader_num_grpo_iterations
        )

    # ── Stage 4: Inference with all GRPO models ──
    if args.stage in ("all", "inference"):
        stage_inference(
            config,
            worker_dir=worker_dir,
            leader_dir=leader_dir,
            master_dir=master_dir,
            max_samples=args.max_samples,
        )

    # ── Stage 5: Evaluation ──
    if args.stage in ("all", "evaluate"):
        stage_evaluate(
            config,
            worker_dir=worker_dir,
            leader_dir=leader_dir,
            master_dir=master_dir,
            api_key=args.api_key,
            max_samples=args.max_samples,
        )

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Worker model: {worker_dir or 'SFT'}")
    log.info(f"  Master model: {master_dir or 'SFT'}")
    log.info(f"  Leader model: {leader_dir or 'SFT'}")
    log.info("=" * 60)

# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End GRPO Pipeline v2")
    parser.add_argument(
        "--stage",
        choices=["all", "reward_model", "grpo_worker", "grpo_master",
                 "grpo_leader", "inference", "evaluate"],
        default="all",
    )
    parser.add_argument("--num_train_samples", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--num_grpo_iterations", type=int, default=3)
    parser.add_argument("--worker_iterations", type=int, default=2)
    parser.add_argument("--leader_iterations", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = PipelineConfig()
    config.num_train_samples = args.num_train_samples
    config.seed = args.seed
    config.agent.num_rollouts = args.num_rollouts
    config.grpo.num_grpo_iterations = args.num_grpo_iterations
    config.grpo.worker_num_grpo_iterations = args.worker_iterations
    config.grpo.leader_num_grpo_iterations = args.leader_iterations

    if args.output_dir:
        config.paths.output_base = args.output_dir
        config.paths.__post_init__()

    run_full_pipeline(config, args)

if __name__ == "__main__":
    main() 
    