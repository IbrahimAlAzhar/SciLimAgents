"""
GRPO Trainer (v2)
==================
Trains all three agent types via GRPO:

Training order: Worker → Master → Leader

1. WORKER GRPO:
   - Unified model with role parameter
   - K rollouts per (paper, role) → score per-role quality
   - Advantage: group-relative within same (paper, role)
   - Reward: category coverage for assigned role + specificity + evidence

2. MASTER GRPO (same as v1):
   - Input: worker outputs + paper → Output: consolidated limitations
   - K rollouts per paper → score final quality
   - Advantage: group-relative across K

3. LEADER GRPO:
   - Input: paper → Output: planning decisions + feedback
   - Reward based on whether leader's decisions led to good final output
   - This is trained LAST because it needs good worker + master models

All use LoRA adapter toggle for KL penalty vs frozen SFT.
"""

import os
import gc
import json
import re
import logging
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from peft import PeftModel, LoraConfig, get_peft_model

from config import PipelineConfig, GRPOConfig, ROLE_CATEGORY_KEYWORDS, WORKER_ROLES
from agent_prompts import (
    get_unified_worker_prompt,
    get_leader_prompt,
    get_leader_feedback_prompt,
    get_master_prompt,
)
from multi_agent_rollout import truncate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ================================================================
# 1. MODEL LOADING FOR GRPO
# ================================================================

def load_grpo_model(
    config: PipelineConfig,
    model_dir: Optional[str] = None,
    learning_rate: Optional[float] = None,
) -> Tuple:
    """
    Load model for GRPO training:
    1. Load base + SFT/previous adapter → merge
    2. Attach NEW LoRA adapters for GRPO
    3. LoRA toggle: enable = policy, disable = frozen SFT ref
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = config.paths.base_model_dir
    sft_dir = model_dir or config.paths.sft_model_dir

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        sft_dir, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, sft_dir, local_files_only=True)
    model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=config.grpo.lora_r,
        lora_alpha=config.grpo.lora_alpha,
        lora_dropout=config.grpo.lora_dropout,
        target_modules=config.grpo.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    log.info(f"GRPO model ready on {device}")
    return model, tokenizer, device

# ================================================================
# 2. WORKER-SPECIFIC REWARD
# ================================================================

def compute_worker_role_reward(
    worker_output: str,
    role: str,
    ground_truth: str,
    paper_text: str,
    config: PipelineConfig,
) -> Dict[str, float]:
    """
    Compute reward for a single worker output for a specific role.

    Components:
    1. Category coverage: Does the output cover keywords for its assigned category?
    2. Specificity: Evidence-grounded, cites sections/tables/figures
    3. GT overlap: Does the output capture GT limitations relevant to this role?
    4. Quality: Non-generic, sufficient detail
    """
    scores = {}

    # 1. Category coverage (does output cover its assigned role?)
    keywords = ROLE_CATEGORY_KEYWORDS.get(role, [])
    if keywords:
        hits = sum(1 for kw in keywords if re.search(kw, worker_output, re.IGNORECASE))
        scores["category_coverage"] = min(1.0, hits / max(3, len(keywords) * 0.4))
    else:
        scores["category_coverage"] = 0.5

    # 2. Specificity
    specific_patterns = [
        r"\d+\.?\d*\s*%", r"table\s*\d+", r"figure\s*\d+",
        r"section\s*\d+", r"equation\s*\d+", r"\d+\.\d+",
    ]
    spec_hits = sum(1 for p in specific_patterns if re.search(p, worker_output, re.IGNORECASE))
    scores["specificity"] = min(1.0, spec_hits / 3)

    # 3. Evidence grounding
    evidence_patterns = [
        r"(section|table|figure|equation|experiment|appendix|algorithm)\s*\w*",
        r'"[^"]{5,}"',  # Quoted evidence
    ]
    ev_hits = sum(len(re.findall(p, worker_output, re.IGNORECASE)) for p in evidence_patterns)
    scores["evidence"] = min(1.0, ev_hits * 0.15)

    # 4. GT overlap (role-relevant GT items)
    gt_keywords = ROLE_CATEGORY_KEYWORDS.get(role, [])
    gt_lines = [l.strip() for l in ground_truth.split("\n") if l.strip()]
    role_relevant_gt = []
    for line in gt_lines:
        if any(re.search(kw, line, re.IGNORECASE) for kw in gt_keywords):
            role_relevant_gt.append(line)

    if role_relevant_gt:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        matched = 0
        for gt_item in role_relevant_gt:
            try:
                vect = TfidfVectorizer().fit([gt_item, worker_output])
                tfidf = vect.transform([gt_item, worker_output])
                sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
                if sim > 0.25:
                    matched += 1
            except Exception:
                pass
        scores["gt_overlap"] = matched / len(role_relevant_gt)
    else:
        scores["gt_overlap"] = 0.5  # Neutral if no role-relevant GT

    # 5. Quality: sufficient detail (not too short, not too generic)
    words = worker_output.split()
    if len(words) < 30:
        scores["quality"] = 0.2
    elif len(words) < 80:
        scores["quality"] = 0.5
    else:
        scores["quality"] = min(1.0, len(words) / 200)

    # Generic phrase penalty
    generic = ["the paper should", "future work could", "it would be better",
               "more experiments are needed", "the authors could"]
    gen_hits = sum(1 for g in generic if g.lower() in worker_output.lower())
    scores["quality"] -= gen_hits * 0.1
    scores["quality"] = max(0.0, scores["quality"])

    # Weighted combination
    w = config.reward
    total = (
        scores["category_coverage"] * w.worker_category_coverage_weight
        + scores["specificity"] * w.worker_specificity_weight
        + scores["evidence"] * w.worker_evidence_weight
        + scores["gt_overlap"] * 1.5
        + scores["quality"] * 1.0
    )
    denom = (w.worker_category_coverage_weight + w.worker_specificity_weight
             + w.worker_evidence_weight + 1.5 + 1.0)
    scores["worker_total"] = total / denom

    return scores

# ================================================================
# 3. LEADER-SPECIFIC REWARD
# ================================================================

def compute_leader_reward(
    leader_planning_text: str,
    leader_feedback_text: str,
    final_limitations: str,
    ground_truth: str,
    selected_workers: List[str],
    config: PipelineConfig,
) -> Dict[str, float]:
    """
    Reward for leader's decisions.

    Components:
    1. Decision quality: Did leader pick the RIGHT workers?
       (measured by final output quality)
    2. Feedback quality: Was feedback structured and actionable?
    3. Coverage: Did the final output cover all critical categories?
    4. Efficiency: Didn't over-select (parsimony bonus)
    """
    from reward_functions import compute_coverage, compute_precision

    scores = {}

    # 1. Decision quality (proxy: final output quality)
    coverage = compute_coverage(final_limitations, ground_truth)
    precision = compute_precision(final_limitations, ground_truth)
    scores["decision_quality"] = (coverage + precision) / 2

    # 2. Feedback quality: structured output with specific recommendations
    feedback_structured = 0
    for marker in ["COVERAGE_GAPS", "FEEDBACK_FOR", "ADDITIONAL_WORKERS",
                    "PRIORITY_GUIDANCE", "FINAL_ASSESSMENT"]:
        if marker in (leader_feedback_text or "").upper():
            feedback_structured += 1
    scores["feedback_structure"] = feedback_structured / 5

    # Planning structure
    planning_structured = 0
    for marker in ["SELECTED_WORKERS", "GUIDANCE_FOR", "NUM_ROUNDS", "CRITICAL_CATEGORIES"]:
        if marker in (leader_planning_text or "").upper():
            planning_structured += 1
    scores["planning_structure"] = planning_structured / 4

    # 3. Coverage across all categories
    all_category_keywords = []
    for kws in ROLE_CATEGORY_KEYWORDS.values():
        all_category_keywords.extend(kws)
    cat_hits = sum(
        1 for kw in set(all_category_keywords)
        if re.search(kw, final_limitations, re.IGNORECASE)
    )
    scores["category_breadth"] = min(1.0, cat_hits / 15)

    # 4. Efficiency bonus: not selecting all 7 workers unnecessarily
    n_selected = len(selected_workers)
    if 3 <= n_selected <= 5:
        scores["efficiency"] = 1.0
    elif n_selected <= 7:
        scores["efficiency"] = 0.7
    else:
        scores["efficiency"] = 0.5

    # Weighted combination
    w = config.reward
    total = (
        scores["decision_quality"] * w.leader_decision_quality_weight
        + scores["feedback_structure"] * w.leader_feedback_quality_weight
        + scores["planning_structure"] * w.leader_feedback_quality_weight
        + scores["category_breadth"] * w.leader_coverage_weight
        + scores["efficiency"] * 0.5
    )
    denom = (w.leader_decision_quality_weight + 2 * w.leader_feedback_quality_weight
             + w.leader_coverage_weight + 0.5)
    scores["leader_total"] = total / denom

    return scores

# ================================================================
# 4. GRPO DATA PREPARATION (per agent type)
# ================================================================

def prepare_worker_grpo_data(
    all_trajectories: List[List],
    tokenizer,
    config: PipelineConfig,
) -> List[Dict]:
    """
    Prepare GRPO data for WORKER training.

    For each paper, for each role, across K rollouts:
    - prompt = unified_worker_prompt(paper, role)
    - completion = worker output for that role
    - reward = per-role quality score
    - advantage = group-relative within same (paper, role)
    """
    # Group by (paper_idx, role)
    from collections import defaultdict
    role_groups = defaultdict(list)  # key: (paper_idx, role) → list of items

    for paper_trajs in all_trajectories:
        for traj in paper_trajs:
            paper_idx = traj.paper_idx if hasattr(traj, "paper_idx") else traj.get("paper_idx", 0)
            paper_text = traj.paper_text if hasattr(traj, "paper_text") else traj.get("paper_text", "")
            gt = traj.ground_truth if hasattr(traj, "ground_truth") else traj.get("ground_truth", "")
            workers = traj.worker_outputs if hasattr(traj, "worker_outputs") else traj.get("worker_outputs", [])

            for w in workers:
                if hasattr(w, "role"):
                    role = w.role
                    output = w.output_text
                    prompt = w.prompt_text
                else:
                    role = w.get("role", "unknown")
                    output = w.get("output", "")
                    prompt = w.get("prompt", "")

                # Compute per-role reward
                reward_scores = compute_worker_role_reward(
                    output, role, gt, paper_text, config,
                )

                role_groups[(paper_idx, role)].append({
                    "prompt": prompt,
                    "completion": output,
                    "reward": reward_scores["worker_total"],
                    "role": role,
                    "paper_idx": paper_idx,
                    "ground_truth": gt,
                })

    # Compute group-relative advantages
    grpo_data = []
    for (pidx, role), items in role_groups.items():
        if len(items) < 2:
            continue

        rewards = [it["reward"] for it in items]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8

        for item in items:
            adv = (item["reward"] - mean_r) / std_r
            item["advantage"] = adv
            grpo_data.append(item)

    log.info(f"Worker GRPO data: {len(grpo_data)} items from "
             f"{len(role_groups)} (paper, role) groups")
    return grpo_data

def prepare_master_grpo_data(
    all_trajectories: List[List],
    all_scores: List[List[Dict]],
    tokenizer,
    config: PipelineConfig,
) -> List[Dict]:
    """
    Prepare GRPO data for MASTER training.
    Same as v1: group-relative advantages across K rollouts per paper.
    """
    grpo_data = []

    for paper_idx, (trajs, scores) in enumerate(zip(all_trajectories, all_scores)):
        if len(trajs) < 2:
            continue

        rewards = [s["combined_reward"] for s in scores]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8

        for k, (traj, score) in enumerate(zip(trajs, scores)):
            adv = (rewards[k] - mean_r) / std_r

            if hasattr(traj, "master_output") and traj.master_output:
                prompt = traj.master_output.prompt_text
                completion = traj.master_output.output_text
            elif isinstance(traj, dict):
                prompt = traj.get("master_output", {}).get("prompt", "")
                completion = traj.get("master_output", {}).get("output", "")
            else:
                continue

            if not completion:
                continue

            grpo_data.append({
                "prompt": prompt,
                "completion": completion,
                "reward": rewards[k],
                "advantage": adv,
                "paper_idx": paper_idx,
                "ground_truth": traj.ground_truth if hasattr(traj, "ground_truth") else traj.get("ground_truth", ""),
            })

    log.info(f"Master GRPO data: {len(grpo_data)} items")
    return grpo_data

def prepare_leader_grpo_data(
    all_trajectories: List[List],
    all_scores: List[List[Dict]],
    tokenizer,
    config: PipelineConfig,
) -> List[Dict]:
    """
    Prepare GRPO data for LEADER training.

    The leader has TWO outputs per trajectory:
    1. Planning output (which workers, guidance)
    2. Feedback output (assessment, priority guidance)

    We train on BOTH, using the final output quality as reward.
    """
    grpo_data = []

    for paper_idx, (trajs, scores) in enumerate(zip(all_trajectories, all_scores)):
        if len(trajs) < 2:
            continue

        # Compute leader-specific rewards
        leader_rewards = []
        for traj in trajs:
            if hasattr(traj, "leader_planning"):
                planning_text = traj.leader_planning.output_text if traj.leader_planning else ""
                feedback_text = traj.leader_feedback.output_text if traj.leader_feedback else ""
                final = traj.final_limitations
                gt = traj.ground_truth
                selected = traj.leader_decisions.get("selected_workers", []) if traj.leader_decisions else []
            else:
                planning_text = traj.get("leader_planning", {}).get("output", "")
                feedback_text = traj.get("leader_feedback", {}).get("output", "")
                final = traj.get("final_limitations", "")
                gt = traj.get("ground_truth", "")
                selected = traj.get("leader_planning", {}).get("decisions", {}).get("selected_workers", [])

            lr = compute_leader_reward(
                planning_text, feedback_text, final, gt, selected, config,
            )
            leader_rewards.append(lr["leader_total"])

        mean_r = np.mean(leader_rewards)
        std_r = np.std(leader_rewards) + 1e-8

        for k, traj in enumerate(trajs):
            adv = (leader_rewards[k] - mean_r) / std_r

            # Planning output
            if hasattr(traj, "leader_planning") and traj.leader_planning:
                grpo_data.append({
                    "prompt": traj.leader_planning.prompt_text,
                    "completion": traj.leader_planning.output_text,
                    "reward": leader_rewards[k],
                    "advantage": adv,
                    "paper_idx": paper_idx,
                    "sub_type": "planning",
                })

            # Feedback output
            if hasattr(traj, "leader_feedback") and traj.leader_feedback:
                grpo_data.append({
                    "prompt": traj.leader_feedback.prompt_text,
                    "completion": traj.leader_feedback.output_text,
                    "reward": leader_rewards[k],
                    "advantage": adv,
                    "paper_idx": paper_idx,
                    "sub_type": "feedback",
                })

    log.info(f"Leader GRPO data: {len(grpo_data)} items")
    return grpo_data

# ================================================================
# 5. GRPO LOSS (shared across all agent types)
# ================================================================

def compute_grpo_loss(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    advantage: float,
    kl_coeff: float,
    device: torch.device,
    max_prompt_len: int = 3500,
    max_completion_len: int = 1024,
) -> Tuple[torch.Tensor, Dict]:
    """GRPO loss with LoRA adapter toggle for KL."""
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[:max_prompt_len]
    if len(response_ids) > max_completion_len:
        response_ids = response_ids[:max_completion_len]
    if len(response_ids) == 0:
        return torch.tensor(0.0, device=device), {"pg_loss": 0, "kl": 0}

    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
    prompt_len = len(prompt_ids)
    targets = full_ids[0, 1:]

    # Policy forward (LoRA ON)
    model.enable_adapter_layers()
    model.train()
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits[0, :-1, :]
    lp = F.log_softmax(logits, dim=-1)
    token_lp = lp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    resp_lp = token_lp[prompt_len - 1:]

    # Ref forward (LoRA OFF = frozen SFT)
    model.disable_adapter_layers()
    with torch.no_grad():
        ref_out = model(input_ids=full_ids, use_cache=False)
        ref_logits = ref_out.logits[0, :-1, :]
        ref_lp = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_lp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        ref_resp_lp = ref_token_lp[prompt_len - 1:]
    model.enable_adapter_layers()

    adv_t = torch.tensor(advantage, device=device, dtype=resp_lp.dtype)
    pg_loss = -(adv_t * resp_lp).mean()
    kl = (resp_lp - ref_resp_lp).mean()
    total = pg_loss + kl_coeff * kl

    return total, {"pg_loss": pg_loss.item(), "kl": kl.item()}

# ================================================================
# 6. GENERIC GRPO TRAINING LOOP
# ================================================================

def train_grpo_epoch(
    model, tokenizer,
    grpo_data: List[Dict],
    config: GRPOConfig,
    device: torch.device,
    optimizer, scheduler,
    agent_type: str = "master",
) -> Dict:
    """One epoch of GRPO training."""
    model.train()
    model.enable_adapter_layers()

    random.shuffle(grpo_data)

    total_loss, total_pg, total_kl, n_updates = 0, 0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(grpo_data, desc=f"GRPO [{agent_type}]")
    for item in pbar:
        prompt = item["prompt"]
        completion = item["completion"]
        advantage = item["advantage"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
        response_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")[0]

        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        try:
            loss, info = compute_grpo_loss(
                model, prompt_ids, response_ids,
                advantage, config.kl_coeff, device,
                max_prompt_len=config.max_prompt_len,
                max_completion_len=config.max_completion_len,
            )

            (loss / config.grad_accum_steps).backward()
            total_loss += loss.item()
            total_pg += info["pg_loss"]
            total_kl += info["kl"]
            n_updates += 1

            if n_updates % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if n_updates % 32 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({"loss": f"{total_loss/max(n_updates,1):.4f}"})

        except torch.cuda.OutOfMemoryError:
            log.warning("OOM, skipping")
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    # Final gradient step
    if n_updates % config.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {
        "avg_loss": total_loss / max(n_updates, 1),
        "avg_kl": total_kl / max(n_updates, 1),
        "n_updates": n_updates,
        "agent_type": agent_type,
    }

def run_grpo_training(
    model, tokenizer,
    grpo_data: List[Dict],
    config: PipelineConfig,
    device: torch.device,
    output_dir: str,
    agent_type: str = "master",
    learning_rate: Optional[float] = None,
) -> List[Dict]:
    """Full GRPO training run."""
    grpo_cfg = config.grpo
    lr = learning_rate or grpo_cfg.learning_rate

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_steps = len(grpo_data) * grpo_cfg.num_train_epochs / grpo_cfg.grad_accum_steps
    warmup_steps = int(total_steps * grpo_cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    all_metrics = []
    for epoch in range(grpo_cfg.num_train_epochs):
        log.info(f"GRPO [{agent_type}] Epoch {epoch+1}/{grpo_cfg.num_train_epochs}")
        metrics = train_grpo_epoch(
            model, tokenizer, grpo_data,
            grpo_cfg, device, optimizer, scheduler,
            agent_type=agent_type,
        )
        metrics["epoch"] = epoch + 1
        all_metrics.append(metrics)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log.info(f"GRPO [{agent_type}] saved to {final_path}")

    with open(os.path.join(output_dir, f"grpo_{agent_type}_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

# ================================================================
# 7. ITERATIVE GRPO (per agent type)
# ================================================================

def iterative_grpo(
    config: PipelineConfig,
    train_papers: List[Dict],
    agent_type: str = "master",
    num_iterations: Optional[int] = None,
    starting_model_dir: Optional[str] = None,
    # For leader/master training, optionally use trained worker model
    worker_model_dir: Optional[str] = None,
    master_model_dir: Optional[str] = None,
):
    """
    Iterative GRPO for a specific agent type.
    Rollout → Score → GRPO update → repeat.
    """
    from multi_agent_rollout import generate_all_rollouts, load_agent_model
    from reward_functions import score_all_rollouts

    if agent_type == "worker":
        output_dir = config.paths.grpo_worker_dir
        n_iter = num_iterations or config.grpo.worker_num_grpo_iterations
        lr = config.grpo.worker_learning_rate
    elif agent_type == "master":
        output_dir = config.paths.grpo_master_dir
        n_iter = num_iterations or config.grpo.num_grpo_iterations
        lr = config.grpo.learning_rate
    elif agent_type == "leader":
        output_dir = config.paths.grpo_leader_dir
        n_iter = num_iterations or config.grpo.leader_num_grpo_iterations
        lr = config.grpo.leader_learning_rate
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    current_model_dir = starting_model_dir or config.paths.sft_model_dir
    all_iteration_metrics = []

    for iteration in range(n_iter):
        log.info(f"\n{'='*60}")
        log.info(f"GRPO [{agent_type.upper()}] ITERATION {iteration+1}/{n_iter}")
        log.info(f"{'='*60}")

        iter_dir = os.path.join(output_dir, f"iteration_{iteration+1}")
        os.makedirs(iter_dir, exist_ok=True)

        # ── Rollout ──
        log.info("Generating rollouts...")
        rollout_model, rollout_tok = load_agent_model(config, current_model_dir)

        # Use trained worker/master models if available
        w_model, w_tok = None, None
        m_model, m_tok = None, None

        if worker_model_dir and agent_type != "worker":
            w_model, w_tok = load_agent_model(config, worker_model_dir)
        if master_model_dir and agent_type == "leader":
            m_model, m_tok = load_agent_model(config, master_model_dir)

        all_trajectories = generate_all_rollouts(
            rollout_model, rollout_tok,
            train_papers, config,
            save_path=os.path.join(iter_dir, "rollouts.json"),
            worker_model=w_model, worker_tokenizer=w_tok,
            master_model=m_model, master_tokenizer=m_tok,
        )

        # ── Score ──
        log.info("Scoring rollouts...")
        all_scores = score_all_rollouts(
            all_trajectories, config,
            model=rollout_model, tokenizer=rollout_tok,
        )

        # Free rollout models
        del rollout_model
        if w_model is not None:
            del w_model
        if m_model is not None:
            del m_model
        gc.collect()
        torch.cuda.empty_cache()

        # ── Prepare GRPO data ──
        log.info("Preparing GRPO data...")
        grpo_model, grpo_tok, device = load_grpo_model(config, current_model_dir)

        if agent_type == "worker":
            grpo_data = prepare_worker_grpo_data(all_trajectories, grpo_tok, config)
        elif agent_type == "master":
            grpo_data = prepare_master_grpo_data(all_trajectories, all_scores, grpo_tok, config)
        elif agent_type == "leader":
            grpo_data = prepare_leader_grpo_data(all_trajectories, all_scores, grpo_tok, config)

        # ── GRPO training ──
        log.info("GRPO training...")
        metrics = run_grpo_training(
            grpo_model, grpo_tok, grpo_data,
            config, device, iter_dir,
            agent_type=agent_type,
            learning_rate=lr,
        )

        all_rewards = [s["combined_reward"] for ps in all_scores for s in ps]
        all_iteration_metrics.append({
            "iteration": iteration + 1,
            "metrics": metrics,
            "reward_mean": float(np.mean(all_rewards)),
        })

        current_model_dir = os.path.join(iter_dir, "final")

        del grpo_model
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "iteration_metrics.json"), "w") as f:
        json.dump(all_iteration_metrics, f, indent=2)

    log.info(f"Iterative GRPO [{agent_type}] complete. Final: {current_model_dir}")
    return current_model_dir, all_iteration_metrics