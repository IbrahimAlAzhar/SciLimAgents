"""
GRPO Training Pipeline for Limitation Generation
==================================================
Zero-shot reward model + Qwen 2.5 3B Instruct
Optimized for 40GB GPU: LoRA + gradient checkpointing
Rollout diversity: seed variation + prompt variation
Iterative: rollout → score → advantage → policy update → repeat
"""

import os
import json
import math
import copy
import random
import logging
import gc
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from rouge_score import rouge_scorer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def log_gpu_memory(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"[GPU MEM {tag}] Allocated: {alloc:.2f} GB | Reserved: {reserved:.2f} GB")

# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class GRPOConfig:
    # Paths
    model_path: str = "qwen2_5_3b_instruct"
    train_csv: str = "data/balanced_data/df_updated_with_retrieval.csv"
    test_csv: str = "data/not_balanced_data/df_not_bal_strat_samp.csv"
    train_input_col: str = "input_text_cleaned"
    test_input_col: str = "input_text_without_lim"
    ground_truth_col: str = "ground_truth_lim_peer"
    output_dir: str = "GRPO/grpo_op_train_seed_prompt_k"

    # Rollout
    num_rollouts: int = 4  # k=4 generations per input
    rollout_temperature: float = 0.7  # single fixed temperature for all rollouts
    rollout_top_p: float = 0.95  # single fixed top_p for all rollouts
    max_new_tokens: int = 512  # reduced for memory

    # GRPO
    grpo_iterations: int = 3  # outer loops (2-4 recommended)
    grpo_epochs_per_iter: int = 2  # gradient steps per iteration
    batch_size: int = 1  # keep at 1 for memory safety
    grad_accum_steps: int = 8  # effective batch = 8
    lr: float = 5e-5  # slightly higher LR for LoRA
    kl_coeff: float = 0.05  # KL penalty weight
    max_grad_norm: float = 1.0

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Reward weights (zero-shot)
    w_coverage: float = 0.35
    w_precision: float = 0.25
    w_groundedness: float = 0.15
    w_redundancy: float = 0.10
    w_severity: float = 0.15

    # Data
    max_train_samples: int = -1  # -1 = all
    max_input_length: int = 1024  # reduced for memory
    max_response_length: int = 512  # truncate responses for training
    samples_per_iteration: int = 200  # subset per GRPO iteration
    max_eval_samples: int = 50
    seed: int = 42

# ============================================================
# 2. PROMPT VARIANTS (for diverse rollouts)
# ============================================================
#
# WHY seed + prompt variation instead of top-p / top-k variation?
#
# top-p variation problem:
#   Small top-p (0.7) → conservative, repeats same obvious limitations.
#   Large top-p (0.99) → noisy, hallucinated limitations with fake citations.
#   The "useful diversity" window is too narrow for structured tasks.
#   4 different top-p values often produce nearly identical outputs or garbage.
#
# top-k variation problem:
#   top-k is distribution-shape-agnostic — it cuts a fixed count of tokens
#   regardless of whether the probability mass is concentrated or spread.
#   top-k=20 → too restrictive (same output every time).
#   top-k=200 → noise from irrelevant tokens.
#   Known to behave poorly with LLMs because vocab distributions vary
#   wildly across positions.
#
# seed + prompt variation wins because:
#   - Seeds give STOCHASTIC diversity: same distribution, different samples.
#     The model already "knows" multiple valid limitations — different seeds
#     surface different ones naturally.
#   - Prompt variants give SEMANTIC diversity: different framings make the
#     model attend to different aspects (methodology vs generalizability vs
#     comprehensive). This creates meaningfully different outputs, not just
#     token-level noise.
#   - Together, GRPO sees responses that differ in WHAT they find (prompt)
#     and HOW they express it (seed), giving richer advantage signal.
#

PROMPT_VARIANTS = [
    {
        # Variant 0: Standard — balanced limitation identification
        "system": (
            "You are a scientific peer reviewer. Given a research paper's text, "
            "identify and list the **limitations** of the study. For each limitation:\n"
            "- State it clearly and concisely\n"
            "- Indicate severity (major or minor)\n"
            "- Reference the relevant section/evidence from the paper when possible\n\n"
            "Output a numbered list of limitations."
        ),
        "user_suffix": "List the limitations:",
    },
    {
        # Variant 1: Methodology-focused — targets internal validity
        "system": (
            "You are an expert methodologist reviewing a research paper. "
            "Focus on identifying limitations related to:\n"
            "- Study design and experimental setup\n"
            "- Sample size, selection bias, and representativeness\n"
            "- Statistical methods and their appropriateness\n"
            "- Control variables and confounding factors\n"
            "- Data collection procedures and measurement validity\n\n"
            "For each limitation, state it clearly, classify as major or minor, "
            "and cite the specific section or evidence from the paper."
        ),
        "user_suffix": "Identify the methodological limitations:",
    },
    {
        # Variant 2: Generalizability & impact — targets external validity
        "system": (
            "You are a critical reviewer assessing a research paper's broader impact. "
            "Focus on identifying limitations related to:\n"
            "- Generalizability to other populations, settings, or contexts\n"
            "- Ecological validity and real-world applicability\n"
            "- Reproducibility and replication concerns\n"
            "- Scope of conclusions relative to evidence presented\n"
            "- Missing comparisons with alternative approaches\n\n"
            "For each limitation, state it clearly, classify as major or minor, "
            "and reference specific parts of the paper."
        ),
        "user_suffix": "What are the limitations regarding generalizability and broader impact?",
    },
    {
        # Variant 3: Comprehensive critical review — structured deep analysis
        "system": (
            "You are a thorough peer reviewer conducting a detailed critical analysis. "
            "Systematically evaluate the paper for ALL types of limitations including:\n"
            "- Theoretical framework gaps\n"
            "- Methodological weaknesses (design, sampling, analysis)\n"
            "- Data quality and completeness issues\n"
            "- Threats to internal and external validity\n"
            "- Presentation clarity and missing details\n"
            "- Ethical considerations overlooked\n\n"
            "For each limitation found, provide: (1) clear statement, "
            "(2) severity (major/minor), (3) evidence from the paper, "
            "(4) potential impact on the study's conclusions."
        ),
        "user_suffix": "Provide a comprehensive list of all limitations:",
    },
]

# ============================================================
# 3. ZERO-SHOT REWARD MODEL (Rule-Based)
# ============================================================

class ZeroShotRewardModel:
    """
    Scores generated limitations against ground truth using:
    - Coverage:      fraction of GT limitations captured
    - Precision:     fraction of generated limitations that are valid
    - Groundedness:  whether evidence pointers / citations exist
    - Redundancy:    penalty for duplicate / near-duplicate limitations
    - Severity:      calibration of severity labels (major/minor)
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    @staticmethod
    def split_limitations(text: str) -> List[str]:
        """Split a limitation text into individual limitation items."""
        if not isinstance(text, str) or not text.strip():
            return []
        # Try numbered list first (1. ... 2. ... )
        items = re.split(r"\n\s*\d+[\.\)]\s*", text)
        items = [x.strip() for x in items if x.strip()]
        if len(items) <= 1:
            # Try bullet points
            items = re.split(r"\n\s*[-•\*]\s*", text)
            items = [x.strip() for x in items if x.strip()]
        if len(items) <= 1:
            # Try sentence split
            items = re.split(r"(?<=[.!?])\s+", text)
            items = [x.strip() for x in items if len(x.strip()) > 20]
        return items if items else [text.strip()]

    def _pairwise_rouge(self, a: str, b: str) -> float:
        scores = self.rouge.score(a, b)
        return scores["rougeL"].fmeasure

    def coverage_score(self, gen_items: List[str], gt_items: List[str]) -> float:
        if not gt_items:
            return 1.0
        matched = 0
        for gt in gt_items:
            best = max((self._pairwise_rouge(gt, g) for g in gen_items), default=0.0)
            if best > 0.3:
                matched += 1
        return matched / len(gt_items)

    def precision_score(self, gen_items: List[str], gt_items: List[str]) -> float:
        if not gen_items:
            return 0.0
        matched = 0
        for g in gen_items:
            best = max((self._pairwise_rouge(g, gt) for gt in gt_items), default=0.0)
            if best > 0.3:
                matched += 1
        return matched / len(gen_items)

    def groundedness_score(self, generated_text: str) -> float:
        evidence_patterns = [
            r"section\s+\d", r"table\s+\d", r"figure\s+\d", r"page\s+\d",
            r"\(.*?et al\..*?\)", r"\[[\d,\s]+\]",
            r"as (stated|mentioned|shown|discussed) in",
        ]
        hits = sum(1 for p in evidence_patterns if re.search(p, generated_text, re.I))
        return min(hits / 3.0, 1.0)

    def redundancy_penalty(self, gen_items: List[str]) -> float:
        if len(gen_items) <= 1:
            return 0.0
        dups = 0
        total_pairs = 0
        for i in range(len(gen_items)):
            for j in range(i + 1, len(gen_items)):
                total_pairs += 1
                if self._pairwise_rouge(gen_items[i], gen_items[j]) > 0.6:
                    dups += 1
        return dups / total_pairs if total_pairs else 0.0

    def severity_calibration(self, gen_items: List[str], gt_items: List[str]) -> float:
        sev_words_major = {"major", "significant", "critical", "severe", "fundamental", "serious"}
        sev_words_minor = {"minor", "small", "slight", "marginal", "modest", "limited"}

        def sev_label(text):
            low = text.lower()
            major_hits = sum(1 for w in sev_words_major if w in low)
            minor_hits = sum(1 for w in sev_words_minor if w in low)
            if major_hits > minor_hits:
                return "major"
            elif minor_hits > major_hits:
                return "minor"
            return "neutral"

        if not gen_items or not gt_items:
            return 0.5
        gen_major = [sev_label(g) for g in gen_items].count("major") / len(gen_items)
        gt_major = [sev_label(g) for g in gt_items].count("major") / len(gt_items)
        return 1.0 - abs(gen_major - gt_major)

    def score(self, generated_text: str, ground_truth_text: str) -> Dict[str, float]:
        gen_items = self.split_limitations(generated_text)
        gt_items = self.split_limitations(ground_truth_text)

        cov = self.coverage_score(gen_items, gt_items)
        prec = self.precision_score(gen_items, gt_items)
        grd = self.groundedness_score(generated_text)
        red = self.redundancy_penalty(gen_items)
        sev = self.severity_calibration(gen_items, gt_items)

        c = self.config
        composite = (
            c.w_coverage * cov
            + c.w_precision * prec
            + c.w_groundedness * grd
            - c.w_redundancy * red
            + c.w_severity * sev
        )
        composite = max(0.0, min(1.0, composite))

        return {
            "composite": composite,
            "coverage": cov,
            "precision": prec,
            "groundedness": grd,
            "redundancy": red,
            "severity_cal": sev,
        }

# ============================================================
# 4. DATASET
# ============================================================

class LimitationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_col: str, gt_col: str, max_samples: int = -1):
        self.data = []
        self.original_indices = []  # track which rows survived filtering
        for idx, row in df.iterrows():
            inp = str(row[input_col]) if pd.notna(row[input_col]) else ""
            gt = str(row[gt_col]) if pd.notna(row[gt_col]) else ""
            if inp.strip() and gt.strip():
                self.data.append({"input_text": inp, "ground_truth": gt})
                self.original_indices.append(idx)
        if max_samples > 0:
            self.data = self.data[:max_samples]
            self.original_indices = self.original_indices[:max_samples]
        logger.info(f"Dataset loaded: {len(self.data)} samples (from {len(df)} rows)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================
# 5. PROMPT BUILDING
# ============================================================

def build_prompt(input_text: str, tokenizer, prompt_variant_idx: int = 0) -> str:
    """
    Build prompt using the specified prompt variant.
    Different variants focus on different aspects of limitation analysis.
    """
    variant = PROMPT_VARIANTS[prompt_variant_idx % len(PROMPT_VARIANTS)]

    messages = [
        {"role": "system", "content": variant["system"]},
        {"role": "user", "content": f"Paper text:\n\n{input_text[:4000]}\n\n{variant['user_suffix']}"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = (
            f"<|im_start|>system\n{variant['system']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return prompt

# ============================================================
# 6. ROLLOUT: Generate k responses with seed + prompt variation
# ============================================================

@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    samples: List[Dict],
    config: GRPOConfig,
    device: torch.device,
) -> List[Dict]:
    """
    For each sample, generate k responses using:
      - Different random seeds (natural sampling diversity)
      - Different prompt variants (semantic diversity —
        methodology vs generalizability vs comprehensive)

    Same temperature and top_p across all k rollouts.
    Seed provides stochastic diversity; prompt provides semantic diversity.
    """
    model.eval()

    # Disable gradient checkpointing for generation (causes issues with use_cache)
    if hasattr(model, "disable_adapter_layers"):
        pass  # keep adapters
    try:
        model.base_model.model.gradient_checkpointing_disable()
    except AttributeError:
        try:
            model.gradient_checkpointing_disable()
        except AttributeError:
            pass

    results = []

    for sample in tqdm(samples, desc="Generating rollouts"):
        responses = []
        response_ids_list = []
        prompt_ids_list = []  # each rollout may have different prompt

        for k in range(config.num_rollouts):
            # --- Seed variation: different seed per rollout ---
            rollout_seed = config.seed + k * 1000 + hash(sample["input_text"][:50]) % 10000
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(rollout_seed)

            # --- Prompt variation: different framing per rollout ---
            prompt = build_prompt(sample["input_text"], tokenizer, prompt_variant_idx=k)
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=config.max_input_length,
            ).to(device)
            prompt_ids = enc["input_ids"][0]

            out = model.generate(
                **enc,
                max_new_tokens=config.max_new_tokens,
                temperature=config.rollout_temperature,
                top_p=config.rollout_top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_ids = out[0][len(prompt_ids):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            responses.append(text)
            response_ids_list.append(gen_ids.cpu())
            prompt_ids_list.append(prompt_ids.cpu())

        results.append({
            "input_text": sample["input_text"],
            "ground_truth": sample["ground_truth"],
            "prompt_ids_list": prompt_ids_list,  # one per rollout (different prompts)
            "responses": responses,
            "response_ids": response_ids_list,
        })

    # Restore global seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Re-enable gradient checkpointing for training
    try:
        model.base_model.model.gradient_checkpointing_enable()
    except AttributeError:
        try:
            model.gradient_checkpointing_enable()
        except AttributeError:
            pass

    return results

# ============================================================
# 7. SCORING + ADVANTAGE COMPUTATION
# ============================================================

def compute_advantages(
    rollout_data: List[Dict],
    reward_model: ZeroShotRewardModel,
) -> List[Dict]:
    """
    Score each rollout response, compute group-relative advantages.
    Advantage = (score - group_mean) / (group_std + eps)
    """
    scored = []
    for item in rollout_data:
        scores = []
        score_details = []
        for resp in item["responses"]:
            s = reward_model.score(resp, item["ground_truth"])
            scores.append(s["composite"])
            score_details.append(s)

        mean_s = np.mean(scores)
        std_s = np.std(scores) + 1e-8
        advantages = [(s - mean_s) / std_s for s in scores]

        scored.append({
            **item,
            "scores": scores,
            "score_details": score_details,
            "advantages": advantages,
            "mean_score": mean_s,
        })

    return scored

# ============================================================
# 8. GRPO POLICY UPDATE (LoRA-aware)
# ============================================================

def grpo_loss(
    policy_model,
    ref_model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    advantage: float,
    config: GRPOConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """
    GRPO loss for a single (prompt, response, advantage) triple.
    L = -advantage * log_pi(response|prompt) + kl_coeff * KL(pi || pi_ref)

    ref_model stays on GPU (frozen base model, no LoRA overhead).
    """
    # Truncate response to save memory
    if len(response_ids) > config.max_response_length:
        response_ids = response_ids[:config.max_response_length]

    full_ids = torch.cat([prompt_ids, response_ids]).to(device)
    prompt_len = len(prompt_ids)

    # --- Policy forward (with gradients, LoRA active) ---
    policy_model.train()
    outputs = policy_model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    logits = outputs.logits[0, :-1, :]
    targets = full_ids[1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # Only response tokens
    resp_lp = token_lp[prompt_len - 1:]

    # --- Reference forward (frozen, no grad) ---
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=full_ids.unsqueeze(0), use_cache=False)
        ref_logits = ref_outputs.logits[0, :-1, :]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        ref_resp_lp = ref_token_lp[prompt_len - 1:]

    # KL divergence: E_pi[log pi - log pi_ref]
    kl = (resp_lp - ref_resp_lp).mean()

    # Policy gradient with advantage weighting
    pg_loss = -advantage * resp_lp.mean()

    # Total
    loss = pg_loss + config.kl_coeff * kl

    return loss, {
        "pg_loss": pg_loss.item(),
        "kl": kl.item(),
        "resp_log_prob": resp_lp.mean().item(),
    }

def grpo_update_step(
    policy_model,
    ref_model,
    scored_data: List[Dict],
    optimizer,
    scheduler,
    config: GRPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """One epoch of GRPO updates over all scored rollouts."""
    policy_model.train()

    # Enable gradient checkpointing
    try:
        policy_model.base_model.model.gradient_checkpointing_enable()
    except AttributeError:
        try:
            policy_model.gradient_checkpointing_enable()
        except AttributeError:
            pass

    total_loss = 0.0
    total_kl = 0.0
    total_pg = 0.0
    n_updates = 0

    random.shuffle(scored_data)
    optimizer.zero_grad()

    for item in tqdm(scored_data, desc="GRPO update"):
        for k in range(len(item["responses"])):
            advantage = item["advantages"][k]
            # Each rollout has its own prompt_ids (different prompt variant)
            prompt_ids = item["prompt_ids_list"][k]
            response_ids = item["response_ids"][k]

            try:
                loss, info = grpo_loss(
                    policy_model, ref_model,
                    prompt_ids, response_ids,
                    advantage, config, device,
                )

                scaled_loss = loss / config.grad_accum_steps
                scaled_loss.backward()

                total_loss += loss.item()
                total_kl += info["kl"]
                total_pg += info["pg_loss"]
                n_updates += 1

                if n_updates % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Periodically clear cache
                if n_updates % 32 == 0:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM on sample, skipping (resp len={len(response_ids)})")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

    # Flush remaining gradients
    if n_updates % config.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {
        "avg_loss": total_loss / max(n_updates, 1),
        "avg_kl": total_kl / max(n_updates, 1),
        "avg_pg_loss": total_pg / max(n_updates, 1),
        "n_updates": n_updates,
    }

# ============================================================
# 9. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    eval_samples: List[Dict],
    reward_model: ZeroShotRewardModel,
    config: GRPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate policy on held-out data (greedy generation, standard prompt)."""
    model.eval()

    # Disable gradient checkpointing for generation
    try:
        model.base_model.model.gradient_checkpointing_disable()
    except AttributeError:
        try:
            model.gradient_checkpointing_disable()
        except AttributeError:
            pass

    all_scores = []
    sample_outputs = []

    for i, sample in enumerate(tqdm(eval_samples[:config.max_eval_samples], desc="Evaluating")):
        # Always use variant 0 (standard) for evaluation consistency
        prompt = build_prompt(sample["input_text"], tokenizer, prompt_variant_idx=0)
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.max_input_length,
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=config.max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        s = reward_model.score(text, sample["ground_truth"])
        all_scores.append(s)

        # Save a few examples
        if i < 3:
            sample_outputs.append({
                "generated": text[:500],
                "ground_truth": sample["ground_truth"][:500],
                "scores": s,
            })

    avg = {k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0]}

    # Log sample outputs
    for i, ex in enumerate(sample_outputs):
        logger.info(f"\n--- Eval Example {i+1} ---")
        logger.info(f"Generated: {ex['generated'][:300]}...")
        logger.info(f"Scores: {ex['scores']}")

    return avg

# ============================================================
# 10. FULL TEST INFERENCE + PERFORMANCE REPORT
# ============================================================

@torch.no_grad()
def test_inference(
    model,
    tokenizer,
    test_samples: List[Dict],
    reward_model: ZeroShotRewardModel,
    config: GRPOConfig,
    device: torch.device,
    save_path: str,
    label: str = "policy",
    original_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Run full inference on ALL test samples.
    Uses standard prompt (variant 0) for fair comparison.
    Saves TWO files:
      1. save_path                         → scores CSV for automated evaluation
      2. save_path.replace('.csv', '_full_for_human_review.csv')
         → original dataframe + generated_limitations column + all scores
         → ready for human review or further evaluation
    Returns aggregate scores.
    """
    model.eval()

    # Disable gradient checkpointing for generation
    try:
        model.base_model.model.gradient_checkpointing_disable()
    except AttributeError:
        try:
            model.gradient_checkpointing_disable()
        except AttributeError:
            pass

    generated_texts = []
    all_score_dicts = []

    # Check for partial progress to resume from
    checkpoint_path = save_path.replace(".csv", "_checkpoint.csv")
    start_idx = 0
    if os.path.exists(checkpoint_path):
        partial_df = pd.read_csv(checkpoint_path)
        start_idx = len(partial_df)
        generated_texts = partial_df["generated_limitations"].tolist()
        for _, row in partial_df.iterrows():
            all_score_dicts.append({
                "composite": row["composite"],
                "coverage": row["coverage"],
                "precision": row["precision"],
                "groundedness": row["groundedness"],
                "redundancy": row["redundancy"],
                "severity_cal": row["severity_cal"],
            })
        logger.info(f"[{label}] Resuming from checkpoint at row {start_idx}/{len(test_samples)}")

    save_every = 10  # save checkpoint every N rows

    for i, sample in enumerate(tqdm(
        test_samples[start_idx:],
        initial=start_idx,
        total=len(test_samples),
        desc=f"Test inference [{label}]",
    )):
        # Always use standard prompt (variant 0) for test — fair comparison
        prompt = build_prompt(sample["input_text"], tokenizer, prompt_variant_idx=0)
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.max_input_length,
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=config.max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        scores = reward_model.score(generated_text, sample["ground_truth"])

        generated_texts.append(generated_text)
        all_score_dicts.append(scores)

        current_idx = start_idx + i + 1

        # Incremental save every N rows
        if current_idx % save_every == 0 or current_idx == len(test_samples):
            checkpoint_rows = []
            for j in range(current_idx):
                checkpoint_rows.append({
                    "sample_idx": j,
                    "input_text": test_samples[j]["input_text"],
                    "ground_truth": test_samples[j]["ground_truth"],
                    "generated_limitations": generated_texts[j],
                    **all_score_dicts[j],
                })
            ckpt_df = pd.DataFrame(checkpoint_rows)
            ckpt_df.to_csv(checkpoint_path, index=False)
            logger.info(f"[{label}] Checkpoint saved: {current_idx}/{len(test_samples)} rows")

        if current_idx % 50 == 0:
            torch.cuda.empty_cache()

    # ---- File 1: Scores CSV (for automated evaluation / comparison) ----
    results = []
    for i, (sample, gen_text, scores) in enumerate(zip(test_samples, generated_texts, all_score_dicts)):
        results.append({
            "sample_idx": i,
            "input_text": sample["input_text"],
            "ground_truth": sample["ground_truth"],
            "generated_limitations": gen_text,
            "composite": scores["composite"],
            "coverage": scores["coverage"],
            "precision": scores["precision"],
            "groundedness": scores["groundedness"],
            "redundancy": scores["redundancy"],
            "severity_cal": scores["severity_cal"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    logger.info(f"[{label}] Scores CSV saved to {save_path} ({len(results_df)} rows)")

    # Remove checkpoint file since full save succeeded
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"[{label}] Checkpoint file removed (full save complete)")

    # ---- File 2: Full DataFrame for human review ----
    # Append generated limitations + scores to original test DataFrame
    human_review_path = save_path.replace(".csv", "_full_for_human_review.csv")

    if original_df is not None and len(original_df) == len(generated_texts):
        # Copy original df, add new columns
        review_df = original_df.copy()
        review_df[f"generated_limitations_{label}"] = generated_texts
        review_df[f"reward_composite_{label}"] = [s["composite"] for s in all_score_dicts]
        review_df[f"reward_coverage_{label}"] = [s["coverage"] for s in all_score_dicts]
        review_df[f"reward_precision_{label}"] = [s["precision"] for s in all_score_dicts]
        review_df[f"reward_groundedness_{label}"] = [s["groundedness"] for s in all_score_dicts]
        review_df[f"reward_redundancy_{label}"] = [s["redundancy"] for s in all_score_dicts]
        review_df[f"reward_severity_cal_{label}"] = [s["severity_cal"] for s in all_score_dicts]
        # Add a blank column for human annotation
        review_df[f"human_rating_{label}"] = ""
        review_df[f"human_notes_{label}"] = ""
    else:
        # Fallback: build from scratch with full text
        review_df = results_df.copy()
        review_df["human_rating"] = ""
        review_df["human_notes"] = ""

    review_df.to_csv(human_review_path, index=False)
    logger.info(f"[{label}] Human review CSV saved to {human_review_path} ({len(review_df)} rows)")

    # ---- Aggregate metrics ----
    score_cols = ["composite", "coverage", "precision", "groundedness", "redundancy", "severity_cal"]
    agg = {}
    for col in score_cols:
        vals = results_df[col].values
        agg[f"{col}_mean"] = float(np.mean(vals))
        agg[f"{col}_std"] = float(np.std(vals))
        agg[f"{col}_median"] = float(np.median(vals))
        agg[f"{col}_min"] = float(np.min(vals))
        agg[f"{col}_max"] = float(np.max(vals))

    return agg

def compare_base_vs_policy(base_results_path: str, policy_results_path: str, output_dir: str):
    """
    Compare base model vs GRPO-trained policy model on test set.
    Prints table + saves comparison CSV.
    """
    base_df = pd.read_csv(base_results_path)
    policy_df = pd.read_csv(policy_results_path)

    score_cols = ["composite", "coverage", "precision", "groundedness", "redundancy", "severity_cal"]

    logger.info("\n" + "=" * 80)
    logger.info("BASE vs GRPO POLICY — TEST SET COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<20} {'Base Mean':>12} {'Policy Mean':>12} {'Delta':>10} {'Δ%':>8}")
    logger.info("-" * 62)

    comparison_rows = []
    for col in score_cols:
        base_mean = base_df[col].mean()
        policy_mean = policy_df[col].mean()
        delta = policy_mean - base_mean
        delta_pct = (delta / (abs(base_mean) + 1e-8)) * 100
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
        # For redundancy, lower is better
        if col == "redundancy":
            direction = "↑" if delta < 0 else "↓" if delta > 0 else "="

        logger.info(f"{col:<20} {base_mean:>12.4f} {policy_mean:>12.4f} {delta:>+10.4f} {delta_pct:>+7.1f}% {direction}")

        comparison_rows.append({
            "metric": col,
            "base_mean": base_mean,
            "base_std": base_df[col].std(),
            "policy_mean": policy_mean,
            "policy_std": policy_df[col].std(),
            "delta": delta,
            "delta_pct": delta_pct,
        })

    # Per-sample win/loss/tie
    wins = (policy_df["composite"] > base_df["composite"]).sum()
    losses = (policy_df["composite"] < base_df["composite"]).sum()
    ties = (policy_df["composite"] == base_df["composite"]).sum()
    total = len(policy_df)
    logger.info(f"\nPer-sample: Policy wins {wins}/{total} ({100*wins/total:.1f}%), "
                f"losses {losses}/{total} ({100*losses/total:.1f}%), "
                f"ties {ties}/{total}")

    # Save comparison
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(output_dir, "base_vs_policy_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Comparison saved to {comp_path}")

# ============================================================
# 11. MAIN TRAINING LOOP
# ============================================================

def main():
    config = GRPOConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    log_gpu_memory("startup")

    # Log rollout strategy
    logger.info(f"Rollout strategy: seed + prompt variation (k={config.num_rollouts})")
    logger.info(f"  Fixed temperature: {config.rollout_temperature}, top_p: {config.rollout_top_p}")
    logger.info(f"  Prompt variants: {len(PROMPT_VARIANTS)}")
    for i, v in enumerate(PROMPT_VARIANTS):
        logger.info(f"    Variant {i}: ...{v['user_suffix']}")

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load base model (shared weights for ref) ----
    logger.info("Loading base model in bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # remove if not available
        attn_implementation="sdpa",
    )
    log_gpu_memory("base model loaded")

    # ---- Frozen reference model (base without LoRA) ----
    # Freeze base for reference
    ref_model = base_model
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    log_gpu_memory("ref model frozen")

    # ---- Create LoRA policy model (separate copy) ----
    logger.info("Loading policy model with LoRA...")
    policy_base = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # remove if not available
        attn_implementation="sdpa",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_base, lora_config)
    policy_model.print_trainable_parameters()

    # Enable gradient checkpointing
    policy_model.gradient_checkpointing_enable()
    # Required for gradient checkpointing with LoRA
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()

    log_gpu_memory("policy LoRA model loaded")

    # ---- Load data ----
    logger.info("Loading datasets...")
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    train_dataset = LimitationDataset(
        train_df, config.train_input_col, config.ground_truth_col, config.max_train_samples
    )
    test_dataset = LimitationDataset(
        test_df, config.test_input_col, config.ground_truth_col
    )

    reward_model = ZeroShotRewardModel(config)

    # ---- Pre-training evaluation ----
    logger.info("=" * 60)
    logger.info("PRE-GRPO EVALUATION")
    logger.info("=" * 60)
    pre_eval = evaluate(policy_model, tokenizer, test_dataset.data, reward_model, config, device)
    logger.info(f"Pre-GRPO scores: {json.dumps(pre_eval, indent=2)}")

    # Filter test_df to match test_dataset (same rows, same order)
    test_df_filtered = test_df.loc[test_dataset.original_indices].reset_index(drop=True)

    # ---- Full test inference on BASE model (before training) ----
    logger.info("=" * 60)
    logger.info("BASE MODEL — FULL TEST INFERENCE")
    logger.info("=" * 60)
    base_test_path = os.path.join(config.output_dir, "test_results_base.csv")
    base_test_agg = test_inference(
        policy_model, tokenizer, test_dataset.data,
        reward_model, config, device,
        save_path=base_test_path, label="base",
        original_df=test_df_filtered,
    )
    logger.info(f"Base test aggregate: {json.dumps(base_test_agg, indent=2)}")

    # ---- GRPO Iterative Loop ----
    all_metrics = [{"iteration": 0, "eval": pre_eval}]

    for iteration in range(1, config.grpo_iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"GRPO ITERATION {iteration}/{config.grpo_iterations}")
        logger.info(f"{'='*60}")
        log_gpu_memory(f"iter {iteration} start")

        # Step 1: Rollout
        iter_samples = random.sample(
            train_dataset.data,
            min(len(train_dataset.data), config.samples_per_iteration),
        )
        logger.info(f"Generating rollouts for {len(iter_samples)} samples, k={config.num_rollouts}...")
        rollout_data = generate_rollouts(policy_model, tokenizer, iter_samples, config, device)

        # Step 2: Score + advantages
        logger.info("Scoring rollouts and computing advantages...")
        scored_data = compute_advantages(rollout_data, reward_model)

        all_scores_flat = [s for item in scored_data for s in item["scores"]]
        above_avg_count = sum(1 for item in scored_data for a in item["advantages"] if a > 0)
        total_count = len(all_scores_flat)
        logger.info(
            f"Rollout scores — mean: {np.mean(all_scores_flat):.4f}, "
            f"std: {np.std(all_scores_flat):.4f}, "
            f"min: {np.min(all_scores_flat):.4f}, "
            f"max: {np.max(all_scores_flat):.4f}"
        )
        logger.info(f"Above-average responses: {above_avg_count}/{total_count}")

        # Step 3: GRPO update
        # Only train on LoRA params
        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        total_steps = (len(scored_data) * config.num_rollouts * config.grpo_epochs_per_iter) / config.grad_accum_steps
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(total_steps // 10, 1),
            num_training_steps=max(total_steps, 1),
        )

        for epoch in range(config.grpo_epochs_per_iter):
            logger.info(f"  Epoch {epoch+1}/{config.grpo_epochs_per_iter}")
            update_info = grpo_update_step(
                policy_model, ref_model, scored_data,
                optimizer, scheduler, config, device,
            )
            logger.info(
                f"  Loss: {update_info['avg_loss']:.4f}, "
                f"KL: {update_info['avg_kl']:.4f}, "
                f"PG: {update_info['avg_pg_loss']:.4f}, "
                f"Updates: {update_info['n_updates']}"
            )
            log_gpu_memory(f"iter {iteration} epoch {epoch+1}")

        # Cleanup between iterations
        del scored_data, rollout_data, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        # Step 4: Evaluate
        logger.info("Evaluating after iteration...")
        eval_scores = evaluate(policy_model, tokenizer, test_dataset.data, reward_model, config, device)
        logger.info(f"Post-iter-{iteration} scores: {json.dumps(eval_scores, indent=2)}")

        iter_metrics = {
            "iteration": iteration,
            "rollout_mean_score": float(np.mean(all_scores_flat)),
            "eval": eval_scores,
            "update": update_info,
        }
        all_metrics.append(iter_metrics)

        # Save LoRA checkpoint (small, fast)
        ckpt_dir = os.path.join(config.output_dir, f"lora_checkpoint_iter_{iteration}")
        policy_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"LoRA checkpoint saved to {ckpt_dir}")

        # Save metrics
        with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

    # ---- Final summary ----
    logger.info("\n" + "=" * 60)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info("=" * 60)
    for m in all_metrics:
        composite = m["eval"]["composite"] if "eval" in m else "N/A"
        rollout = m.get("rollout_mean_score", "N/A")
        logger.info(f"Iter {m['iteration']}: eval_composite={composite:.4f}" +
                     (f", rollout_mean={rollout:.4f}" if isinstance(rollout, float) else ""))

    # Save final merged model (optional — merges LoRA into base)
    logger.info("Merging LoRA weights into base model...")
    merged_dir = os.path.join(config.output_dir, "final_merged_model")
    merged_model = policy_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Final merged model saved to {merged_dir}")

    # ---- Full test inference on TRAINED policy model ----
    logger.info("=" * 60)
    logger.info("GRPO POLICY — FULL TEST INFERENCE")
    logger.info("=" * 60)
    # Reload merged model for clean inference
    trained_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    policy_test_path = os.path.join(config.output_dir, "test_results_policy.csv")
    policy_test_agg = test_inference(
        trained_model, tokenizer, test_dataset.data,
        reward_model, config, device,
        save_path=policy_test_path, label="policy",
        original_df=test_df_filtered,
    )
    logger.info(f"Policy test aggregate: {json.dumps(policy_test_agg, indent=2)}")

    # ---- Compare base vs policy ----
    compare_base_vs_policy(base_test_path, policy_test_path, config.output_dir)

    # Save final metrics
    all_metrics.append({
        "stage": "final_test",
        "base_agg": base_test_agg,
        "policy_agg": policy_test_agg,
    })
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Done!")

if __name__ == "__main__":
    main()