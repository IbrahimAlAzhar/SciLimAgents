"""
GRPO Pipeline Configuration (v2)
==================================
Central config for multi-agent GRPO training on Qwen 2.5 3B.

v2 changes:
  - Worker GRPO: unified worker model with role parameter
  - Leader GRPO: enhanced decision-making (which workers, how many, feedback)
  - Training order: Worker → Master → Leader
  - Inference uses all three GRPO-trained models
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class PathConfig:
    """All file paths."""
    base_model_dir: str = "qwen2_5_3b_instruct"
    sft_model_dir: str = "other_experiments/sft/sft_qwen25_3b_model/final"

    train_csv: str = "data/not_balanced_data/df_filtered_not_bal_final.csv"
    inference_csv: str = "data/balanced_data/df_updated_with_retrieval.csv"

    train_input_col: str = "input_text_without_lim"
    train_gt_col: str = "ground_truth_lim_peer"
    inference_input_col: str = "input_text_cleaned"
    inference_gt_col: str = "ground_truth_lim_peer"

    output_base: str = "other_experiments/grpo"
    rollout_dir: str = ""
    grpo_worker_dir: str = ""
    grpo_master_dir: str = ""
    grpo_leader_dir: str = ""
    reward_model_dir: str = ""
    inference_output_dir: str = ""

    def __post_init__(self):
        self.rollout_dir = os.path.join(self.output_base, "rollouts")
        self.grpo_worker_dir = os.path.join(self.output_base, "grpo_worker")
        self.grpo_master_dir = os.path.join(self.output_base, "grpo_master")
        self.grpo_leader_dir = os.path.join(self.output_base, "grpo_leader")
        self.reward_model_dir = os.path.join(self.output_base, "reward_model")
        self.inference_output_dir = os.path.join(self.output_base, "inference_results")
        for d in [self.output_base, self.rollout_dir, self.grpo_worker_dir,
                  self.grpo_master_dir, self.grpo_leader_dir,
                  self.reward_model_dir, self.inference_output_dir]:
            os.makedirs(d, exist_ok=True)

# ── All available worker roles ──
WORKER_ROLES = [
    "novelty_significance",
    "theoretical_methodological",
    "experimental_evaluation",
    "generalization_robustness_efficiency",
    "clarity_interpretability_reproducibility",
    "data_ethics",
    "citation",
]

ROLE_DESCRIPTIONS = {
    "novelty_significance": "Novelty & significance of contributions",
    "theoretical_methodological": "Theoretical soundness & methodology",
    "experimental_evaluation": "Experimental rigor & baselines",
    "generalization_robustness_efficiency": "Generalization, robustness & efficiency",
    "clarity_interpretability_reproducibility": "Clarity, interpretability & reproducibility",
    "data_ethics": "Data integrity, bias & ethics",
    "citation": "Citation analysis & related work gaps",
}

# Category keywords for per-role reward scoring
ROLE_CATEGORY_KEYWORDS = {
    "novelty_significance": [
        "novel", "significance", "incremental", "contribution", "original",
        "prior work", "differentiation", "impact", "motivation",
    ],
    "theoretical_methodological": [
        "method", "theoretical", "proof", "assumption", "ablation",
        "algorithm", "formulation", "derivation", "approximation",
    ],
    "experimental_evaluation": [
        "experiment", "baseline", "metric", "comparison", "statistical",
        "evaluation", "benchmark", "dataset", "result", "performance",
    ],
    "generalization_robustness_efficiency": [
        "generalization", "robustness", "efficiency", "scalability",
        "computational", "resource", "deployment", "practical", "cost",
    ],
    "clarity_interpretability_reproducibility": [
        "clarity", "reproducibility", "interpretability", "documentation",
        "hyperparameter", "code", "notation", "explanation",
    ],
    "data_ethics": [
        "data", "bias", "fairness", "ethics", "privacy", "integrity",
        "annotation", "demographic", "consent", "transparency",
    ],
    "citation": [
        "citation", "related work", "prior", "reference", "comparison",
        "literature", "survey", "overlooked", "missing",
    ],
}

@dataclass
class AgentConfig:
    """Multi-agent rollout settings."""
    worker_roles: List[str] = field(default_factory=lambda: list(WORKER_ROLES))

    num_rollouts: int = 4
    max_new_tokens_worker: int = 512
    max_new_tokens_leader: int = 512
    max_new_tokens_master: int = 1024
    max_input_tokens: int = 3000

    temperatures: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0])
    top_p_values: List[float] = field(default_factory=lambda: [0.9, 0.92, 0.95, 0.98])

    leader_min_workers: int = 3
    leader_max_workers: int = 7

@dataclass
class RewardConfig:
    """Reward model settings."""
    coverage_weight: float = 2.0
    precision_weight: float = 1.5
    groundedness_weight: float = 1.0
    redundancy_penalty_weight: float = 0.5
    specificity_weight: float = 1.0
    criticality_weight: float = 1.0

    use_zero_shot_reward: bool = True
    zero_shot_weight: float = 1.0

    train_reward_model: bool = True
    reward_lora_r: int = 16
    reward_lora_alpha: int = 32
    reward_train_epochs: int = 2
    reward_train_lr: float = 2e-5
    reward_train_batch_size: int = 4

    chosen_threshold: float = 0.6
    rejected_threshold: float = 0.4

    # Worker-specific
    worker_category_coverage_weight: float = 2.0
    worker_specificity_weight: float = 1.5
    worker_evidence_weight: float = 1.0

    # Leader-specific
    leader_decision_quality_weight: float = 2.0
    leader_feedback_quality_weight: float = 1.5
    leader_coverage_weight: float = 1.0

@dataclass
class GRPOConfig:
    """GRPO training hyperparameters."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    num_grpo_iterations: int = 3
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 5e-6
    kl_coeff: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05

    num_generations: int = 4
    max_completion_len: int = 1024
    max_prompt_len: int = 3500
    temperature: float = 0.9

    advantage_norm: bool = True
    save_steps: int = 50
    log_steps: int = 5

    # ── Training flags ──
    train_worker: bool = True
    train_master: bool = True
    train_leader: bool = True

    # Worker GRPO specifics
    worker_num_grpo_iterations: int = 2
    worker_roles_per_batch: int = 3
    worker_learning_rate: float = 5e-6

    # Leader GRPO specifics
    leader_num_grpo_iterations: int = 2
    leader_learning_rate: float = 3e-6

@dataclass
class PipelineConfig:
    """Top-level config."""
    paths: PathConfig = field(default_factory=PathConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    num_train_samples: int = 0  # 0 = use all
    seed: int = 42
    use_bf16: bool = True
    use_4bit: bool = True