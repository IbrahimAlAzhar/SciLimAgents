"""
Multi-Agent Rollout Pipeline (v2)
===================================
Key v2 changes:
  - Unified worker: one model, role injected via prompt
  - Enhanced leader: decides which workers, provides feedback
  - Supports loading separate GRPO-trained checkpoints per agent type
  - Full trajectory tracking for all agents (used by GRPO trainer)
"""

import os
import gc
import json
import logging
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import PipelineConfig, WORKER_ROLES
from agent_prompts import (
    get_unified_worker_prompt,
    get_leader_prompt,
    get_leader_feedback_prompt,
    get_master_prompt,
    parse_leader_planning,
    parse_leader_feedback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class AgentOutput:
    """Output from a single agent call."""
    agent_name: str
    agent_type: str          # "worker", "leader", "master"
    role: str                # For worker: the role name. For leader/master: "leader"/"master"
    prompt_text: str         # Full prompt text (for GRPO training)
    output_text: str
    num_tokens: int = 0

@dataclass
class Trajectory:
    """Full trajectory for one rollout of one paper."""
    paper_idx: int
    rollout_idx: int
    paper_text: str
    ground_truth: str

    # Agent outputs
    leader_planning: Optional[AgentOutput] = None
    leader_decisions: Optional[dict] = None    # Parsed leader decisions
    worker_outputs: List[AgentOutput] = field(default_factory=list)
    leader_feedback: Optional[AgentOutput] = None
    leader_feedback_parsed: Optional[dict] = None
    master_output: Optional[AgentOutput] = None

    # Final result
    final_limitations: str = ""
    temperature: float = 0.7
    seed: int = 42

    def to_dict(self) -> Dict:
        return {
            "paper_idx": self.paper_idx,
            "rollout_idx": self.rollout_idx,
            "ground_truth": self.ground_truth,
            "final_limitations": self.final_limitations,
            "temperature": self.temperature,
            "seed": self.seed,
            "leader_planning": {
                "output": self.leader_planning.output_text if self.leader_planning else "",
                "prompt": self.leader_planning.prompt_text[:500] if self.leader_planning else "",
                "decisions": self.leader_decisions or {},
            },
            "worker_outputs": [
                {
                    "name": w.agent_name,
                    "role": w.role,
                    "output": w.output_text,
                    "prompt": w.prompt_text[:500],
                }
                for w in self.worker_outputs
            ],
            "leader_feedback": {
                "output": self.leader_feedback.output_text if self.leader_feedback else "",
                "prompt": self.leader_feedback.prompt_text[:500] if self.leader_feedback else "",
                "parsed": self.leader_feedback_parsed or {},
            },
            "master_output": {
                "output": self.master_output.output_text if self.master_output else "",
                "prompt": self.master_output.prompt_text[:500] if self.master_output else "",
            },
        }

# ================================================================
# MODEL LOADING
# ================================================================

def load_agent_model(
    config: PipelineConfig,
    adapter_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model with an adapter for rollout generation.
    Used for worker, leader, or master — each can have its own checkpoint.
    """
    base_dir = config.paths.base_model_dir
    adapter = adapter_dir or config.paths.sft_model_dir

    log.info(f"Loading model: base={base_dir}, adapter={adapter}")

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter, local_files_only=True)
    model.eval()
    return model, tokenizer

def load_all_agent_models(
    config: PipelineConfig,
    worker_dir: Optional[str] = None,
    leader_dir: Optional[str] = None,
    master_dir: Optional[str] = None,
) -> Dict:
    """
    Load all three agent models.
    If a GRPO-trained checkpoint exists, use it; otherwise fall back to SFT.

    For memory efficiency on 2x40GB GPUs:
    - We load ONE model at a time during rollout (sequential generation)
    - Or share a single model if all use the same checkpoint
    """
    sft_dir = config.paths.sft_model_dir
    w_dir = worker_dir or sft_dir
    l_dir = leader_dir or sft_dir
    m_dir = master_dir or sft_dir

    # Check if all the same → load once
    all_same = (w_dir == l_dir == m_dir)

    if all_same:
        log.info("All agents use the same model checkpoint — loading once")
        model, tokenizer = load_agent_model(config, w_dir)
        return {
            "worker": (model, tokenizer),
            "leader": (model, tokenizer),
            "master": (model, tokenizer),
            "shared": True,
        }

    log.info("Loading separate models for worker, leader, master")
    # Load them one at a time; free previous before loading next during rollout
    return {
        "worker_dir": w_dir,
        "leader_dir": l_dir,
        "master_dir": m_dir,
        "shared": False,
    }

# ================================================================
# GENERATION
# ================================================================

def generate_response(
    model, tokenizer,
    user_prompt: str,
    system_msg: str = "You are a helpful research paper reviewer.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Tuple[str, str]:
    """
    Generate a response using the Qwen ChatML format.
    Returns: (response_text, full_prompt_text)
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=config_max_len(max_new_tokens),
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, prompt_text

def config_max_len(max_new_tokens: int) -> int:
    """Max input length = context budget minus generation budget."""
    return 3500

def truncate_text(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    if not text:
        return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

# ================================================================
# FULL MULTI-AGENT ROLLOUT
# ================================================================

def run_single_rollout(
    model, tokenizer,
    paper_text: str,
    paper_idx: int,
    rollout_idx: int,
    ground_truth: str,
    config: PipelineConfig,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    # Separate models (if available)
    worker_model=None, worker_tokenizer=None,
    leader_model=None, leader_tokenizer=None,
    master_model=None, master_tokenizer=None,
) -> Trajectory:
    """
    Run one complete multi-agent rollout:
      1. Leader plans (decides which workers, gives guidance)
      2. Selected workers analyze paper (unified model, different roles)
      3. Leader reviews & provides feedback
      4. Master consolidates with leader feedback
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Use separate models if provided, else fallback to shared model
    w_model = worker_model or model
    w_tok = worker_tokenizer or tokenizer
    l_model = leader_model or model
    l_tok = leader_tokenizer or tokenizer
    m_model = master_model or model
    m_tok = master_tokenizer or tokenizer

    traj = Trajectory(
        paper_idx=paper_idx,
        rollout_idx=rollout_idx,
        paper_text=paper_text,
        ground_truth=ground_truth,
        temperature=temperature,
        seed=seed,
    )

    paper_trunc = truncate_text(paper_text, tokenizer, config.agent.max_input_tokens)

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Leader Planning
    # ══════════════════════════════════════════════════════════════
    leader_prompt = get_leader_prompt(paper_trunc)
    leader_response, leader_full_prompt = generate_response(
        l_model, l_tok, leader_prompt,
        max_new_tokens=config.agent.max_new_tokens_leader,
        temperature=temperature, top_p=top_p,
    )
    traj.leader_planning = AgentOutput(
        agent_name="leader_planning",
        agent_type="leader",
        role="leader",
        prompt_text=leader_full_prompt,
        output_text=leader_response,
    )

    # Parse leader decisions
    decisions = parse_leader_planning(leader_response)
    traj.leader_decisions = decisions
    selected_workers = decisions["selected_workers"]

    # Enforce min/max worker constraints
    if len(selected_workers) < config.agent.leader_min_workers:
        # Add workers until minimum met
        for role in WORKER_ROLES:
            if role not in selected_workers:
                selected_workers.append(role)
            if len(selected_workers) >= config.agent.leader_min_workers:
                break
    selected_workers = selected_workers[:config.agent.leader_max_workers]

    log.info(f"  Leader selected {len(selected_workers)} workers: {selected_workers}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Worker Agents (unified model, different roles)
    # ══════════════════════════════════════════════════════════════
    all_worker_texts = []
    for role in selected_workers:
        # Get role-specific guidance from leader (if any)
        guidance = decisions.get("worker_guidance", {}).get(role, "")
        worker_prompt_base = get_unified_worker_prompt(paper_trunc, role)

        # Append leader guidance if available
        if guidance:
            worker_prompt_base += f"\n\nLeader's specific guidance for you: {guidance}"

        worker_response, worker_full_prompt = generate_response(
            w_model, w_tok, worker_prompt_base,
            max_new_tokens=config.agent.max_new_tokens_worker,
            temperature=temperature, top_p=top_p,
        )

        agent_out = AgentOutput(
            agent_name=f"worker_{role}",
            agent_type="worker",
            role=role,
            prompt_text=worker_full_prompt,
            output_text=worker_response,
        )
        traj.worker_outputs.append(agent_out)
        all_worker_texts.append(f"[{role.upper()}]:\n{worker_response}")

    combined_workers = "\n\n".join(all_worker_texts)
    combined_workers_trunc = truncate_text(combined_workers, tokenizer, 2000)

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Leader Feedback
    # ══════════════════════════════════════════════════════════════
    feedback_prompt = get_leader_feedback_prompt(
        paper_trunc, combined_workers_trunc, round_number=1,
    )
    feedback_response, feedback_full_prompt = generate_response(
        l_model, l_tok, feedback_prompt,
        max_new_tokens=config.agent.max_new_tokens_leader,
        temperature=max(0.5, temperature - 0.2),  # Slightly lower temp for feedback
        top_p=top_p,
    )
    traj.leader_feedback = AgentOutput(
        agent_name="leader_feedback",
        agent_type="leader",
        role="leader",
        prompt_text=feedback_full_prompt,
        output_text=feedback_response,
    )
    feedback_parsed = parse_leader_feedback(feedback_response)
    traj.leader_feedback_parsed = feedback_parsed

    # ── Round 2: Workers REGENERATE based on leader feedback ──
    # Workers that got "needs_improvement" regenerate with leader's feedback
    # Additional workers requested by leader are also activated
    worker_feedback_map = feedback_parsed.get("worker_feedback", {})
    regenerated_roles = []
    additional_roles = []

    # Identify workers that need to regenerate
    for role, fb_text in worker_feedback_map.items():
        if "needs_improvement" in fb_text.lower() or "improve" in fb_text.lower():
            # Match to valid role names
            for valid in WORKER_ROLES:
                if role.lower() in valid.lower() or valid.lower() in role.lower():
                    regenerated_roles.append((valid, fb_text))
                    break

    # Identify new workers requested by leader
    if feedback_parsed.get("additional_workers"):
        for r in feedback_parsed["additional_workers"]:
            r_clean = r.strip().lower().replace(" ", "_")
            for valid in WORKER_ROLES:
                if r_clean in valid or valid in r_clean:
                    existing_roles = [w.role for w in traj.worker_outputs]
                    if valid not in existing_roles:
                        additional_roles.append(valid)
                    break

    # Regenerate workers that received "needs_improvement" feedback
    if regenerated_roles:
        log.info(f"  Round 2: Regenerating {len(regenerated_roles)} workers with leader feedback")
        for role, fb_text in regenerated_roles:
            # Build prompt with original role + leader's specific feedback
            worker_prompt_base = get_unified_worker_prompt(paper_trunc, role)
            # Find original output for this role
            original_output = ""
            for w in traj.worker_outputs:
                if w.role == role:
                    original_output = w.output_text
                    break

            regeneration_prompt = (
                f"{worker_prompt_base}\n\n"
                f"YOUR PREVIOUS ANALYSIS:\n{original_output}\n\n"
                f"LEADER FEEDBACK ON YOUR ANALYSIS:\n{fb_text}\n\n"
                f"Please revise and improve your analysis based on the leader's feedback. "
                f"Address the gaps and weaknesses identified. Provide an updated bullet list."
            )

            worker_response, worker_full_prompt = generate_response(
                w_model, w_tok, regeneration_prompt,
                max_new_tokens=config.agent.max_new_tokens_worker,
                temperature=temperature, top_p=top_p,
            )

            traj.worker_outputs.append(AgentOutput(
                agent_name=f"worker_{role}_revised",
                agent_type="worker",
                role=role,
                prompt_text=worker_full_prompt,
                output_text=worker_response,
            ))
            # Replace original output in the text list with revised version
            for idx, txt in enumerate(all_worker_texts):
                if f"[{role.upper()}]:" in txt:
                    all_worker_texts[idx] = f"[{role.upper()} (Revised)]:\n{worker_response}"
                    break
            else:
                all_worker_texts.append(f"[{role.upper()} (Revised)]:\n{worker_response}")

    # Add new workers requested by leader
    if additional_roles:
        log.info(f"  Round 2: Adding {len(additional_roles)} new workers: {additional_roles}")
        for role in additional_roles[:3]:  # Cap at 3 additional
            worker_prompt = get_unified_worker_prompt(paper_trunc, role)
            worker_response, worker_full_prompt = generate_response(
                w_model, w_tok, worker_prompt,
                max_new_tokens=config.agent.max_new_tokens_worker,
                temperature=temperature, top_p=top_p,
            )
            traj.worker_outputs.append(AgentOutput(
                agent_name=f"worker_{role}_r2",
                agent_type="worker",
                role=role,
                prompt_text=worker_full_prompt,
                output_text=worker_response,
            ))
            all_worker_texts.append(f"[{role.upper()} (Round 2)]:\n{worker_response}")

    # Update combined workers text for master
    if regenerated_roles or additional_roles:
        combined_workers = "\n\n".join(all_worker_texts)
        combined_workers_trunc = truncate_text(combined_workers, tokenizer, 2000)

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Master Consolidation (with leader feedback)
    # ══════════════════════════════════════════════════════════════
    leader_feedback_for_master = feedback_parsed.get("priority_guidance", "")
    master_prompt = get_master_prompt(
        paper_trunc, combined_workers_trunc, leader_feedback_for_master,
    )
    master_response, master_full_prompt = generate_response(
        m_model, m_tok, master_prompt,
        max_new_tokens=config.agent.max_new_tokens_master,
        temperature=max(0.5, temperature - 0.2),  # Lower temp for consolidation
        top_p=top_p,
    )
    traj.master_output = AgentOutput(
        agent_name="master",
        agent_type="master",
        role="master",
        prompt_text=master_full_prompt,
        output_text=master_response,
    )
    traj.final_limitations = master_response

    return traj

# ================================================================
# K ROLLOUTS
# ================================================================

def run_k_rollouts(
    model, tokenizer,
    paper_text: str,
    paper_idx: int,
    ground_truth: str,
    config: PipelineConfig,
    worker_model=None, worker_tokenizer=None,
    leader_model=None, leader_tokenizer=None,
    master_model=None, master_tokenizer=None,
) -> List[Trajectory]:
    """Run K diverse rollouts for one paper."""
    trajectories = []
    K = config.agent.num_rollouts

    for k in range(K):
        temp = config.agent.temperatures[k % len(config.agent.temperatures)]
        top_p = config.agent.top_p_values[k % len(config.agent.top_p_values)]
        seed = config.seed + paper_idx * 100 + k

        log.info(f"  Rollout {k+1}/{K} | temp={temp}, top_p={top_p}")

        traj = run_single_rollout(
            model, tokenizer,
            paper_text, paper_idx, k,
            ground_truth, config,
            temperature=temp, top_p=top_p, seed=seed,
            worker_model=worker_model, worker_tokenizer=worker_tokenizer,
            leader_model=leader_model, leader_tokenizer=leader_tokenizer,
            master_model=master_model, master_tokenizer=master_tokenizer,
        )
        trajectories.append(traj)

    return trajectories

def generate_all_rollouts(
    model, tokenizer,
    papers: List[Dict],
    config: PipelineConfig,
    save_path: Optional[str] = None,
    worker_model=None, worker_tokenizer=None,
    leader_model=None, leader_tokenizer=None,
    master_model=None, master_tokenizer=None,
) -> List[List[Trajectory]]:
    """Generate K rollouts for all papers."""
    all_trajectories = []

    for i, paper in enumerate(papers):
        log.info(f"Paper {i+1}/{len(papers)} (idx={paper['idx']})")

        trajs = run_k_rollouts(
            model, tokenizer,
            paper["text"], paper["idx"], paper["ground_truth"],
            config,
            worker_model=worker_model, worker_tokenizer=worker_tokenizer,
            leader_model=leader_model, leader_tokenizer=leader_tokenizer,
            master_model=master_model, master_tokenizer=master_tokenizer,
        )
        all_trajectories.append(trajs)

        if save_path and (i + 1) % 10 == 0:
            _save_rollouts(all_trajectories, save_path)

    if save_path:
        _save_rollouts(all_trajectories, save_path)
        log.info(f"Rollouts saved to {save_path}")

    return all_trajectories

def _save_rollouts(all_trajectories, path):
    data = []
    for paper_trajs in all_trajectories:
        data.append([t.to_dict() for t in paper_trajs])
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)