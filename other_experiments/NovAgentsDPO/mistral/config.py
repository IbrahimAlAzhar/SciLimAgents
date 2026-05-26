"""
Shared configuration for the SD-DPO pipeline.
All paths, model names, and hyperparameters live here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

# ============================================================
# PATHS
# ============================================================
OUTPUT_BASE_DIR = "other_experiments/dpo_novagents_llama_mistral/mistral"

@dataclass
class Config:
    # --- Data paths ---
    rollout_data_path: str = "data/not_balanced_data/df_not_bal_final_strat_samp.csv"
    paper_data_path: str = "data/balanced_data/df_updated_with_retrieval.csv"
    nougat_data_path: str = "data/nougat_data/nougat_all_papers_dataframe_excluding_bal_not_bal_data.csv"

    rollout_sample_size: int = 150
    rollout_random_seed: int = 42

    output_dir: str = OUTPUT_BASE_DIR

    # --- Sub-directories ---
    rollouts_dir: str = os.path.join(OUTPUT_BASE_DIR, "rollouts")
    scores_dir: str = os.path.join(OUTPUT_BASE_DIR, "scores")
    pairs_dir: str = os.path.join(OUTPUT_BASE_DIR, "pairs")
    checkpoints_dir: str = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
    eval_dir: str = os.path.join(OUTPUT_BASE_DIR, "eval")
    logs_dir: str = os.path.join(OUTPUT_BASE_DIR, "logs")

    # --- Models ---
    strong_model: str = "gpt-4o-mini"
    weak_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    reward_judge_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_cache: str = "models/mistral_7b_v3_instruct"

    # --- Step-level reward weights ---
    step_weights: Dict[str, float] = field(default_factory=lambda: {
        "claim_extraction": 0.15,
        "novelty_technical": 0.25,
        "experimental_scope": 0.25,
        "limitation_synthesis": 0.35,
    })

    # --- Rollout generation ---
    max_papers: int = 0  # 0 = no limit
    num_rollouts_per_model: int = 3
    temperature_strong: float = 0.7
    temperature_weak: float = 0.9
    max_gen_tokens: int = 2000

    # --- DPO training ---
    dpo_beta: float = 0.1
    learning_rate: float = 5e-7
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_prompt_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- Preference pair construction ---
    min_reward_gap: float = 0.05

    # --- OpenAI API ---
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_requests_per_minute: int = 30

    def __post_init__(self):
        """Create all output directories."""
        for d in [self.output_dir, self.rollouts_dir, self.scores_dir,
                  self.pairs_dir, self.checkpoints_dir, self.eval_dir,
                  self.logs_dir]:
            os.makedirs(d, exist_ok=True)

def get_config(**overrides) -> Config:
    """Get config with optional overrides."""
    cfg = Config()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
