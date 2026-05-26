"""
GRPO-Trained Multi-Agent Inference
=====================================
Loads three separate GRPO-trained models (worker, leader, master)
and runs the full agent pipeline for limitation generation.

Flow (mirrors AutoGen round-robin):
  1. Leader plans: selects workers, gives guidance
  2. Workers analyze: unified GRPO-trained model with different role prompts
  3. Leader reviews: provides feedback, identifies gaps, requests round 2
  4. Master consolidates: merges, deduplicates, produces final limitation list

Usage:
  python run_inference.py
  python run_inference.py --row_start 0 --row_end 100
  python run_inference.py --worker_dir /path/to/worker --master_dir /path/to/master --leader_dir /path/to/leader
"""

import os
import gc
import re
import sys
import json
import time
import logging
import argparse
import traceback
from typing import List, Dict, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ================================================================
# PATHS & CONFIG
# ================================================================

# ── Model checkpoints ──
BASE_MODEL_DIR = "qwen2_5_3b_instruct"
SFT_MODEL_DIR  = "other_experiments/sft/sft_qwen25_3b_model/final"

# GRPO-trained checkpoints (update these after training completes)

GRPO_BASE = "other_experiments/grpo"
GRPO_WORKER_DIR = os.path.join(GRPO_BASE, "grpo_worker/iteration_2/final")

GRPO_MASTER_DIR = os.path.join(GRPO_BASE, "grpo_master/iteration_3/final")

GRPO_LEADER_DIR = os.path.join(GRPO_BASE, "grpo_leader/iteration_2/final")

# ── Data ──

INPUT_CSV  = "data/balanced_data/df_updated_with_retrieval.csv"

OUTPUT_DIR = "other_experiments/grpo/output"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "inference_grpo_agents.csv")

TEXT_COL   = "input_text_cleaned"
ROW_START  = 0
ROW_END    = None  # None = all rows

# ── Generation settings ──
MAX_INPUT_TOKENS         = 3000
MAX_NEW_TOKENS_WORKER    = 512
MAX_NEW_TOKENS_LEADER    = 512
MAX_NEW_TOKENS_MASTER    = 1024
TEMPERATURE              = 0.7
TOP_P                    = 0.9
SAVE_EVERY               = 10

# ================================================================
# WORKER ROLE PROMPTS (unified model, different roles)
# ================================================================

WORKER_ROLES = {
    "Novelty_Significance": """You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
When finished, provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.""",

    "Theoretical_Methodological": """You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
When done, deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.""",

    "Experimental_Evaluation": """You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
When finished, provide a bullet list of experimental evaluation-related limitations with evidence.""",

    "Generalization_Robustness_Efficiency": """You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings, is practical in resources, and addresses deployment constraints.
When finished, provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations with evidence.""",

    "Clarity_Interpretability_Reproducibility": """You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations and missing details needed to reproduce results.
When finished, provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations with evidence.""",

    "Data_Ethics": """You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations.
When finished, provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations with evidence.""",

    "Citation": """You are the **Citation Agent**.
Task: Analyze the paper's use of citations and related work.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Are important related works missing?
Output: bullet list of citation-related limitations with evidence.""",
}

ALL_AGENT_NAMES = [f"{name}_Agent" for name in WORKER_ROLES.keys()]
AGENT_NAMES_STR = ", ".join(ALL_AGENT_NAMES)

# ================================================================
# LEADER PROMPTS
# ================================================================

def build_leader_planning_prompt(paper_content: str) -> str:
    """Leader decides which workers to activate and gives guidance."""
    roles_menu = "\n".join(f"  - {name}_Agent" for name in WORKER_ROLES.keys())
    return f"""You are the **Leader Agent**.
You are coordinating a group of specialist agents to produce a comprehensive limitation list.

Available specialist agents:
{roles_menu}

Your task:
1. Decide which specialist agents are most relevant for THIS paper (select 3-7).
   Output as: SELECTED_WORKERS: [Agent1, Agent2, ...]
2. For each selected agent, give 1-2 sentences of specific guidance about what to focus on.
   Output as: GUIDANCE_FOR_<AgentName>: <specific instructions>
3. How many review rounds? (1 for simple papers, 2 for complex)
   Output as: NUM_ROUNDS: <1 or 2>

PAPER CONTENT:
{paper_content}"""

def build_leader_feedback_prompt(paper_content: str, worker_outputs: str) -> str:
    """Leader reviews worker outputs and provides feedback."""
    return f"""You are the **Leader Agent**. You have received analyses from your specialist workers.

Review their outputs and provide:
1. COVERAGE_GAPS: Which limitation categories are missing or underrepresented?
2. QUALITY_FEEDBACK: For each worker, is the output good or needs improvement?
3. ADDITIONAL_WORKERS: Should any additional specialists be activated? (or "none")
4. PRIORITY_GUIDANCE: What should the Master Agent prioritize when consolidating?
5. FINAL_ASSESSMENT: Overall quality score (1-10) and brief summary.

WORKER OUTPUTS:
{worker_outputs}

PAPER (for reference):
{paper_content}"""

# ================================================================
# MASTER PROMPT
# ================================================================

def build_master_prompt(paper_content: str, worker_outputs: str, leader_feedback: str = "") -> str:
    """Master consolidates all worker outputs into final limitation list."""
    leader_section = ""
    if leader_feedback:
        leader_section = f"\n\nLEADER FEEDBACK & PRIORITY GUIDANCE:\n{leader_feedback}"

    return f"""You are the **Master Agent**.
You will receive limitation analyses from specialist agents and produce ONE consolidated list.
Rules:
- Integrate specialist outputs.
- Remove redundancy (merge similar limitations).
- Keep specificity and evidence.
- Do NOT invent new limitations beyond what specialists raised.
- Follow the Leader's priority guidance if provided.
{leader_section}

SPECIALIST ANALYSES:
{worker_outputs}

Output format:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then bullets, grouped by category.

PAPER CONTENT (for context):
{paper_content}"""

# ================================================================
# MODEL LOADING
# ================================================================

def load_model(base_model_dir: str, adapter_dir: str, device_map: str = "auto"):
    """Load base model + LoRA adapter."""
    log.info(f"Loading model: base={base_model_dir}, adapter={adapter_dir}")

    # Check if adapter exists, fall back to SFT if not
    if not os.path.exists(adapter_dir):
        log.warning(f"Adapter not found at {adapter_dir}, falling back to SFT: {SFT_MODEL_DIR}")
        adapter_dir = SFT_MODEL_DIR

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    model.eval()

    alloc_gb = torch.cuda.memory_allocated() / 1e9
    log.info(f"Model loaded. GPU memory: {alloc_gb:.2f} GB")

    return model, tokenizer

def load_all_models(worker_dir, leader_dir, master_dir):
    """
    Load all three agent models.
    If all point to the same checkpoint, load once to save memory.
    """
    all_same = (worker_dir == leader_dir == master_dir)

    if all_same:
        log.info("All agents use same checkpoint — loading once")
        model, tokenizer = load_model(BASE_MODEL_DIR, worker_dir)
        return {
            "worker":  {"model": model, "tokenizer": tokenizer},
            "leader":  {"model": model, "tokenizer": tokenizer},
            "master":  {"model": model, "tokenizer": tokenizer},
            "shared": True,
        }

    log.info("Loading SEPARATE models for worker, leader, master")

    # Strategy: load on different GPUs if available
    n_gpus = torch.cuda.device_count()
    log.info(f"Available GPUs: {n_gpus}")

    if n_gpus >= 2:
        # Spread across GPUs
        log.info("Distributing models across 2 GPUs")
        worker_model, worker_tok = load_model(BASE_MODEL_DIR, worker_dir, device_map="auto")
        leader_model, leader_tok = load_model(BASE_MODEL_DIR, leader_dir, device_map="auto")
        master_model, master_tok = load_model(BASE_MODEL_DIR, master_dir, device_map="auto")
    else:
        # Single GPU: load all with auto device map
        worker_model, worker_tok = load_model(BASE_MODEL_DIR, worker_dir)
        leader_model, leader_tok = load_model(BASE_MODEL_DIR, leader_dir)
        master_model, master_tok = load_model(BASE_MODEL_DIR, master_dir)

    return {
        "worker":  {"model": worker_model, "tokenizer": worker_tok},
        "leader":  {"model": leader_model, "tokenizer": leader_tok},
        "master":  {"model": master_model, "tokenizer": master_tok},
        "shared": False,
    }

# ================================================================
# GENERATION
# ================================================================

def generate(model, tokenizer, user_prompt: str,
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:
    """Generate response using Qwen ChatML format."""
    messages = [
        {"role": "system", "content": "You are a helpful and critical research paper reviewer."},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=MAX_INPUT_TOKENS + 500,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

def truncate_text(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to max_tokens."""
    if not text:
        return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

# ================================================================
# LEADER OUTPUT PARSING
# ================================================================

def parse_leader_planning(leader_output: str) -> Dict:
    """Parse leader's planning output to extract selected workers."""
    decisions = {
        "selected_workers": [],
        "worker_guidance": {},
        "num_rounds": 1,
    }

    # Parse SELECTED_WORKERS
    match = re.search(r"SELECTED_WORKERS:\s*\[([^\]]+)\]", leader_output, re.IGNORECASE)
    if match:
        raw_names = [n.strip().strip("'\"") for n in match.group(1).split(",")]
        # Fuzzy match to valid role names
        valid = []
        for raw in raw_names:
            raw_clean = raw.replace("_Agent", "").replace(" ", "_")
            for role_name in WORKER_ROLES:
                if raw_clean.lower() in role_name.lower() or role_name.lower() in raw_clean.lower():
                    if role_name not in valid:
                        valid.append(role_name)
                    break
        decisions["selected_workers"] = valid if valid else list(WORKER_ROLES.keys())[:5]
    else:
        # Default: all workers
        decisions["selected_workers"] = list(WORKER_ROLES.keys())

    # Parse guidance per worker
    for role in WORKER_ROLES:
        pattern = rf"GUIDANCE_FOR_{role}.*?:\s*(.+?)(?=\nGUIDANCE_FOR|NUM_ROUNDS|$)"
        match = re.search(pattern, leader_output, re.IGNORECASE | re.DOTALL)
        if match:
            decisions["worker_guidance"][role] = match.group(1).strip()

    # Parse NUM_ROUNDS
    match = re.search(r"NUM_ROUNDS:\s*(\d+)", leader_output, re.IGNORECASE)
    if match:
        decisions["num_rounds"] = min(int(match.group(1)), 2)

    return decisions

# ================================================================
# MAIN INFERENCE PIPELINE (per paper)
# ================================================================

def process_single_paper(
    paper_text: str,
    paper_idx: int,
    models: Dict,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> Dict:
    """
    Full multi-agent pipeline for one paper.
    Mirrors AutoGen round-robin: Leader → Workers → Leader feedback → Master

    Returns dict with all outputs and final limitations.
    """
    w_model = models["worker"]["model"]
    w_tok   = models["worker"]["tokenizer"]
    l_model = models["leader"]["model"]
    l_tok   = models["leader"]["tokenizer"]
    m_model = models["master"]["model"]
    m_tok   = models["master"]["tokenizer"]

    paper_trunc = truncate_text(paper_text, w_tok, MAX_INPUT_TOKENS)

    result = {
        "paper_idx": paper_idx,
        "leader_planning": "",
        "selected_workers": [],
        "worker_outputs": {},
        "leader_feedback": "",
        "master_output": "",
        "final_limitations": "",
        "num_workers_used": 0,
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Leader Planning (GRPO-trained leader model)
    # ══════════════════════════════════════════════════════════════
    log.info(f"  [Paper {paper_idx}] Step 1: Leader planning...")
    leader_plan_prompt = build_leader_planning_prompt(paper_trunc)
    leader_plan_output = generate(
        l_model, l_tok, leader_plan_prompt,
        max_new_tokens=MAX_NEW_TOKENS_LEADER,
        temperature=temperature, top_p=top_p,
    )
    result["leader_planning"] = leader_plan_output

    # Parse leader decisions
    decisions = parse_leader_planning(leader_plan_output)
    selected_workers = decisions["selected_workers"]
    num_rounds = decisions["num_rounds"]

    # Enforce minimum 3, maximum 7
    if len(selected_workers) < 3:
        for role in WORKER_ROLES:
            if role not in selected_workers:
                selected_workers.append(role)
            if len(selected_workers) >= 3:
                break
    selected_workers = selected_workers[:7]
    result["selected_workers"] = selected_workers
    result["num_workers_used"] = len(selected_workers)

    log.info(f"  [Paper {paper_idx}] Leader selected {len(selected_workers)} workers: {selected_workers}")
    log.info(f"  [Paper {paper_idx}] Leader requested {num_rounds} round(s)")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Worker Agents — Round 1 (GRPO-trained worker model)
    # ══════════════════════════════════════════════════════════════
    log.info(f"  [Paper {paper_idx}] Step 2: Running {len(selected_workers)} workers (round 1)...")
    all_worker_texts = []

    for role_name in selected_workers:
        role_prompt = WORKER_ROLES[role_name]
        guidance = decisions.get("worker_guidance", {}).get(role_name, "")

        full_prompt = f"{role_prompt}\n\nPAPER CONTENT:\n{paper_trunc}"
        if guidance:
            full_prompt += f"\n\nLeader's specific guidance for you: {guidance}"

        worker_output = generate(
            w_model, w_tok, full_prompt,
            max_new_tokens=MAX_NEW_TOKENS_WORKER,
            temperature=temperature, top_p=top_p,
        )

        result["worker_outputs"][role_name] = worker_output
        all_worker_texts.append(f"=== {role_name}_Agent ===\n{worker_output}")
        log.info(f"    {role_name}_Agent: {len(worker_output)} chars")

    combined_workers = "\n\n".join(all_worker_texts)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Leader Feedback (GRPO-trained leader model)
    # ══════════════════════════════════════════════════════════════
    log.info(f"  [Paper {paper_idx}] Step 3: Leader reviewing worker outputs...")
    workers_for_leader = truncate_text(combined_workers, l_tok, 2000)
    feedback_prompt = build_leader_feedback_prompt(paper_trunc, workers_for_leader)
    leader_feedback = generate(
        l_model, l_tok, feedback_prompt,
        max_new_tokens=MAX_NEW_TOKENS_LEADER,
        temperature=max(0.5, temperature - 0.2),
        top_p=top_p,
    )
    result["leader_feedback"] = leader_feedback

    # ── Round 2: Workers REGENERATE based on leader feedback ──
    # Parse leader feedback for per-worker quality assessment
    regenerated_roles = []
    additional_roles = []

    # Find workers that need to regenerate (leader said "needs_improvement")
    feedback_lines = leader_feedback.split("\n")
    for line in feedback_lines:
        if "FEEDBACK_FOR" in line.upper():
            for role_name in WORKER_ROLES:
                if role_name.upper() in line.upper() or role_name.replace("_", " ").upper() in line.upper():
                    if "needs_improvement" in line.lower() or "improve" in line.lower() or "weak" in line.lower():
                        # Extract the feedback text after the colon
                        fb_text = line.split(":", 1)[-1].strip() if ":" in line else line
                        regenerated_roles.append((role_name, fb_text))
                    break

    # Find additional workers requested by leader
    add_match = re.search(
        r"ADDITIONAL_WORKERS:\s*\[([^\]]+)\]", leader_feedback, re.IGNORECASE,
    )
    if add_match and "none" not in add_match.group(1).lower():
        additional_raw = [n.strip().strip("'\"") for n in add_match.group(1).split(",")]
        for raw in additional_raw:
            raw_clean = raw.replace("_Agent", "").replace(" ", "_")
            for role_name in WORKER_ROLES:
                if raw_clean.lower() in role_name.lower() or role_name.lower() in raw_clean.lower():
                    if role_name not in [r for r in result["worker_outputs"]]:
                        additional_roles.append(role_name)
                    break

    # Regenerate workers with leader feedback
    if regenerated_roles:
        log.info(f"  [Paper {paper_idx}] Round 2: Regenerating {len(regenerated_roles)} workers with leader feedback")
        for role_name, fb_text in regenerated_roles:
            role_prompt = WORKER_ROLES[role_name]
            original_output = result["worker_outputs"].get(role_name, "")

            regeneration_prompt = (
                f"{role_prompt}\n\n"
                f"PAPER CONTENT:\n{paper_trunc}\n\n"
                f"YOUR PREVIOUS ANALYSIS:\n{original_output}\n\n"
                f"LEADER FEEDBACK ON YOUR ANALYSIS:\n{fb_text}\n\n"
                f"Please revise and improve your analysis based on the leader's feedback. "
                f"Address the gaps and weaknesses identified. Provide an updated bullet list."
            )

            worker_output = generate(
                w_model, w_tok, regeneration_prompt,
                max_new_tokens=MAX_NEW_TOKENS_WORKER,
                temperature=temperature, top_p=top_p,
            )

            # Replace original with revised version
            result["worker_outputs"][role_name] = worker_output
            # Update in the text list
            for idx, txt in enumerate(all_worker_texts):
                if f"=== {role_name}_Agent ===" in txt:
                    all_worker_texts[idx] = f"=== {role_name}_Agent (Revised) ===\n{worker_output}"
                    break
            log.info(f"    {role_name}_Agent REVISED: {len(worker_output)} chars")

    # Add new workers requested by leader
    if additional_roles:
        log.info(f"  [Paper {paper_idx}] Round 2: Adding {len(additional_roles)} new workers: {additional_roles}")
        for role_name in additional_roles[:3]:
            role_prompt = WORKER_ROLES[role_name]
            full_prompt = f"{role_prompt}\n\nPAPER CONTENT:\n{paper_trunc}"

            worker_output = generate(
                w_model, w_tok, full_prompt,
                max_new_tokens=MAX_NEW_TOKENS_WORKER,
                temperature=temperature, top_p=top_p,
            )
            result["worker_outputs"][f"{role_name}_R2"] = worker_output
            all_worker_texts.append(f"=== {role_name}_Agent (Round 2) ===\n{worker_output}")
            result["num_workers_used"] += 1

    # Rebuild combined workers text after regeneration
    if regenerated_roles or additional_roles:
        combined_workers = "\n\n".join(all_worker_texts)

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Master Consolidation (GRPO-trained master model)
    # ══════════════════════════════════════════════════════════════
    log.info(f"  [Paper {paper_idx}] Step 4: Master consolidating...")
    workers_for_master = truncate_text(combined_workers, m_tok, 2500)

    # Extract leader's priority guidance for master
    priority_match = re.search(
        r"PRIORITY_GUIDANCE:\s*(.+?)(?=\n(?:FINAL_ASSESSMENT|$))",
        leader_feedback, re.IGNORECASE | re.DOTALL,
    )
    leader_priority = priority_match.group(1).strip() if priority_match else ""

    master_prompt = build_master_prompt(paper_trunc, workers_for_master, leader_priority)
    master_output = generate(
        m_model, m_tok, master_prompt,
        max_new_tokens=MAX_NEW_TOKENS_MASTER,
        temperature=max(0.5, temperature - 0.2),
        top_p=top_p,
    )

    result["master_output"] = master_output
    result["final_limitations"] = master_output

    log.info(f"  [Paper {paper_idx}] Done. Final output: {len(master_output)} chars, "
             f"{len(master_output.split(chr(10)))} lines")

    return result

# ================================================================
# MAIN
# ================================================================

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resolve model paths ──
    worker_dir = args.worker_dir
    leader_dir = args.leader_dir
    master_dir = args.master_dir

    # Auto-find latest GRPO checkpoints if not specified
    for agent_type, grpo_dir, default_dir in [
        ("worker", os.path.join(GRPO_BASE, "grpo_worker"), worker_dir),
        ("leader", os.path.join(GRPO_BASE, "grpo_leader"), leader_dir),
        ("master", os.path.join(GRPO_BASE, "grpo_master"), master_dir),
    ]:
        if default_dir and os.path.exists(default_dir):
            continue
        # Search for latest iteration
        for i in range(10, 0, -1):
            candidate = os.path.join(grpo_dir, f"iteration_{i}", "final")
            if os.path.exists(candidate):
                if agent_type == "worker": worker_dir = candidate
                elif agent_type == "leader": leader_dir = candidate
                elif agent_type == "master": master_dir = candidate
                log.info(f"Found {agent_type} checkpoint: {candidate}")
                break
        else:
            log.warning(f"No GRPO checkpoint for {agent_type}, using SFT: {SFT_MODEL_DIR}")
            if agent_type == "worker": worker_dir = SFT_MODEL_DIR
            elif agent_type == "leader": leader_dir = SFT_MODEL_DIR
            elif agent_type == "master": master_dir = SFT_MODEL_DIR

    log.info("=" * 70)
    log.info("GRPO Multi-Agent Inference")
    log.info("=" * 70)
    log.info(f"Worker model: {worker_dir}")
    log.info(f"Leader model: {leader_dir}")
    log.info(f"Master model: {master_dir}")
    log.info(f"Input CSV:    {INPUT_CSV}")
    log.info(f"Output CSV:   {args.output_csv}")
    log.info(f"Rows:         {args.row_start} to {args.row_end or 'end'}")

    # ── Load models ──
    log.info("\nLoading models...")
    models = load_all_models(worker_dir, leader_dir, master_dir)
    log.info("All models loaded.\n")

    # ── Load data ──
    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df)} rows from {INPUT_CSV}")

    row_end = args.row_end or len(df)
    df = df.iloc[args.row_start:row_end].copy()
    log.info(f"Processing rows {args.row_start} to {row_end} ({len(df)} rows)")

    # ── Initialize output columns ──
    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "num_workers_used" not in df.columns:
        df["num_workers_used"] = 0
    if "selected_workers" not in df.columns:
        df["selected_workers"] = ""
    if "leader_planning" not in df.columns:
        df["leader_planning"] = ""
    if "leader_feedback" not in df.columns:
        df["leader_feedback"] = ""
    if "full_worker_outputs" not in df.columns:
        df["full_worker_outputs"] = ""

    # ── Process each paper ──
    log.info("\n" + "=" * 70)
    log.info("Starting inference...")
    log.info("=" * 70)

    for i in tqdm(range(len(df)), desc="Processing papers"):
        row = df.iloc[i]
        idx = df.index[i]
        paper_text = str(row.get(TEXT_COL, "") or "")

        if len(paper_text) < 100:
            df.at[idx, "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            log.warning(f"  Row {args.row_start + i}: Skipped (text too short)")
            continue

        try:
            result = process_single_paper(
                paper_text=paper_text,
                paper_idx=args.row_start + i,
                models=models,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            # Save results
            df.at[idx, "final_merged_limitations"] = result["final_limitations"]
            df.at[idx, "num_workers_used"] = result["num_workers_used"]
            df.at[idx, "selected_workers"] = json.dumps(result["selected_workers"])
            df.at[idx, "leader_planning"] = result["leader_planning"]
            df.at[idx, "leader_feedback"] = result["leader_feedback"]
            df.at[idx, "full_worker_outputs"] = json.dumps(result["worker_outputs"], default=str)

        except Exception as e:
            df.at[idx, "final_merged_limitations"] = f"ERROR: {str(e)}"
            log.error(f"  Row {args.row_start + i}: Error — {e}")
            traceback.print_exc()

        # ── Save every N rows ──
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(args.output_csv, index=False)
            log.info(f"  Checkpoint saved ({i + 1}/{len(df)} rows) → {args.output_csv}")

        # Clear GPU cache periodically
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # ── Final save ──
    df.to_csv(args.output_csv, index=False)
    log.info(f"\n{'=' * 70}")
    log.info(f"INFERENCE COMPLETE")
    log.info(f"{'=' * 70}")
    log.info(f"Processed: {len(df)} papers")
    log.info(f"Saved to:  {args.output_csv}")

    # Stats
    completed = (df["final_merged_limitations"] != "PENDING") & \
                (df["final_merged_limitations"] != "SKIPPED_SHORT_TEXT") & \
                (~df["final_merged_limitations"].str.startswith("ERROR", na=False))
    log.info(f"Successful: {completed.sum()}/{len(df)}")
    log.info(f"Avg workers used: {df.loc[completed, 'num_workers_used'].mean():.1f}")

# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Multi-Agent Inference")
    parser.add_argument("--worker_dir", type=str, default=GRPO_WORKER_DIR)
    parser.add_argument("--leader_dir", type=str, default=GRPO_LEADER_DIR)
    parser.add_argument("--master_dir", type=str, default=GRPO_MASTER_DIR)
    parser.add_argument("--row_start", type=int, default=ROW_START)
    parser.add_argument("--row_end", type=int, default=ROW_END)
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    args = parser.parse_args()

    TEMPERATURE = args.temperature
    SAVE_EVERY = args.save_every

    print(f"\n[CLI args] {sys.argv}")
    print(f"[GPUs] {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    main(args)