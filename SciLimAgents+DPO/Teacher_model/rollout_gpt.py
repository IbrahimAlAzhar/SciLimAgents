"""
Rollout generation for multi-agent limitation extraction — SIMPLIFIED.

Per paper, per rollout:
  For each of 6 workers:
      1. Leader -> Worker  (initial prompt)
      2. Worker -> Leader  (first answer)          [training sample: worker]
      3. Leader -> Worker  (feedback)              [training sample: leader]
      4. Worker -> Leader  (revised answer)        [training sample: worker]
  Then:
      5. Leader -> Master  (bundle all worker outputs)   [training sample: leader]
      6. Master -> final consolidated limitations        [training sample: master]

4 rollouts per paper with different (temperature, top_p, seed) for diversity.
Output: JSON files ready for SFT and DPO training.
"""

import os
import json
import time
import ast
import re

import pandas as pd
import autogen
import tiktoken
from tqdm import tqdm

# ==========================================
# 1. CONFIG
# ==========================================

os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or api_key == "YOUR_KEY_HERE":
    raise ValueError("Valid OPENAI_API_KEY environment variable not found!")

MODEL_ID = "gpt-4o"

INPUT_CSV = "data/not_balanced_data/df_not_bal_final_strat_samp.csv"
OUTPUT_DIR = "other_experiments/dpo/output_gpt"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_per_paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

OUTPUT_FULL_JSON   = os.path.join(OUTPUT_DIR, "rollout_data_full.json")
OUTPUT_WORKER_JSON = os.path.join(OUTPUT_DIR, "rollout_data_worker.json")
OUTPUT_LEADER_JSON = os.path.join(OUTPUT_DIR, "rollout_data_leader.json")
OUTPUT_MASTER_JSON = os.path.join(OUTPUT_DIR, "rollout_data_master.json")

INPUT_COL  = "input_text_without_lim"
GT_COL     = "ground_truth_lim_peer"

# 4 rollouts with diverse sampling
ROLLOUT_CONFIGS = [
    {"rollout_id": 0, "temperature": 0.7, "top_p": 0.90, "seed": 42},
    {"rollout_id": 1, "temperature": 0.9, "top_p": 0.95, "seed": 123},
    {"rollout_id": 2, "temperature": 1.1, "top_p": 1.00, "seed": 456},
    {"rollout_id": 3, "temperature": 0.5, "top_p": 0.85, "seed": 789},
]

PAPER_TOKEN_LIMIT    = 40000

# Slice of the CSV to process — start with 150 as a pilot

# ==========================================
# 2. WORKER PROMPTS
# ==========================================

def get_novelty_significance_prompt(paper_content):
    return f"""You are a highly skeptical expert focused on limitations related to novelty and significance. Scrutinize whether contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for: rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters.
Provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""

def get_theoretical_methodological_prompt(paper_content):
    return f"""You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws: unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, missing ablations, or ablations that do not convincingly attribute performance gains.
Provide a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content):
    return f"""You specialize in experimental evaluation: validation, rigor, comparisons, baselines, and metrics. Find weaknesses: insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate or outdated baselines, incomplete comparisons, misleading metrics, missing error bars, or overemphasis on minor gains.
Provide a bullet list of experimental evaluation-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content):
    return f"""Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings, is practical in terms of resources, and addresses genuine deployment needs.
Point out: overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, excessive training/inference demands, reliance on synthetic data, ignoring cost/latency constraints, or lack of user studies.
Provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content):
    return f"""You focus on clarity, interpretability, and reproducibility. Scrutinize for: unclear explanations, lack of explainability or insights into decisions, and insufficient details for replication (code, data, hyperparameters, protocols, seeds).
Provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, with suggestions where relevant.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content):
    return f"""You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for: collection, labeling, cleaning, representativeness, or documentation issues; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""

# --- Worker registry ---
WORKER_CONFIGS = [
    ("Novelty_Significance_Agent",                    get_novelty_significance_prompt,            "novelty and significance"),
    ("Theoretical_Methodological_Agent",              get_theoretical_methodological_prompt,      "theoretical and methodological soundness (including ablations)"),
    ("Experimental_Evaluation_Agent",                 get_experimental_evaluation_prompt,         "experimental evaluation, baselines, and metrics"),
    ("Generalization_Robustness_Efficiency_Agent",    get_generalization_robustness_efficiency_prompt, "generalization, robustness, efficiency, and applicability"),
    ("Clarity_Interpretability_Reproducibility_Agent",get_clarity_interpretability_reproducibility_prompt, "clarity, interpretability, and reproducibility"),
    ("Data_Ethics_Agent",                             get_data_ethics_prompt,                     "data integrity, bias, fairness, and ethics"),
]

# ==========================================
# 3. LEADER / MASTER PROMPTS
# ==========================================

def get_leader_system_prompt():
    return """You are the Leader Agent coordinating a team of 6 specialist agents to identify limitations in a scientific paper.

Your job has TWO modes:

MODE A — Providing FEEDBACK to a worker:
When a worker has just submitted their initial bullet list of limitations, give targeted feedback in ONE message:
- Identify vague statements and demand specificity.
- Flag limitations that lack evidence from the paper.
- Point out missing angles in the worker's specialty area.
- If a limitation is generic (could apply to any paper), say so.
- Be strict but constructive. Keep feedback focused — 3 to 6 concrete points.
- End with: "Please revise and send your updated bullet list."

MODE B — Handing off to the Master Agent:
Once you have collected revised outputs from all 6 workers, summarize and forward them. Format:
"Master Agent, here are the limitation analyses from the team:
[Worker name]: [bullet list]
[Worker name]: [bullet list]
...
Please synthesize them into a single, consolidated, non-redundant, high-quality list of limitations, grouped by category."

Do NOT generate limitations yourself. Only orchestrate, critique, and forward."""

def get_master_system_prompt():
    return """You are the Master Agent. You receive limitation analyses from 6 specialist workers (via the Leader Agent) and produce ONE final consolidated list.

TASK:
- Integrate all specialist outputs.
- Remove redundancies (merge similar limitations).
- Prioritize the most severe and well-justified ones.
- Preserve specificity and evidence from the originals.
- Organize by category (Novelty & Significance, Theoretical, Experimental, Generalization, Clarity, Data & Ethics).
- Do NOT introduce new limitations not raised by specialists.
- Aim for 10-20 strong limitations.

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then a bulleted list:
- **Category:** Specific limitation (with brief explanation and evidence)."""

# ==========================================
# 4. TEXT HELPERS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def truncate_text_to_tokens(text, max_tokens):
    if not text:
        return ""
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + " ... [TRUNCATED]"

# ==========================================
# 5. LLM CALL HELPER
# ==========================================

def llm_call(system_prompt, user_message, rollout_config):
    """One direct call to the model — used for orchestrated turns."""
    llm_config = {
        "config_list": [{"model": MODEL_ID, "api_key": api_key}],
        "temperature": rollout_config["temperature"],
        "top_p":       rollout_config["top_p"],
        "seed":        rollout_config["seed"],
        "timeout":     120,
        "cache_seed":  None,
    }
    agent = autogen.AssistantAgent(
        name="temp_agent",
        system_message=system_prompt,
        llm_config=llm_config,
    )
    user = autogen.UserProxyAgent(
        name="temp_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    user.initiate_chat(agent, message=user_message, clear_history=True, silent=True)
    # The agent's reply is the last assistant message
    reply = agent.last_message(user)["content"]
    return reply.strip() if reply else ""

# ==========================================
# 6. ONE ROLLOUT (deterministic orchestrated loop)
# ==========================================

def execute_single_rollout(paper_idx, paper_text, ground_truth, rollout_config):
    """
    Deterministic flow:
      For each worker:
          W1 = worker(initial prompt)
          F  = leader_feedback(W1)
          W2 = worker(original + F)
      After all workers:
          M  = master(leader bundle of all W2)
    """
    samples = []
    all_worker_revised = {}   # worker_name -> revised output

    leader_sys = get_leader_system_prompt()
    master_sys = get_master_system_prompt()

    # ---------- WORKERS ----------
    for worker_name, prompt_fn, specialty in WORKER_CONFIGS:
        # System prompt for this worker
        worker_sys = prompt_fn(paper_text)

        # ---- Turn 1: Leader asks worker for initial list ----
        leader_initial_ask = (
            f"{worker_name}, please analyze the paper and produce a bullet list of "
            f"limitations focused on {specialty}. Be specific, evidence-based, and "
            "tied directly to paper content. Return only the bullet list."
        )

        worker_initial_input = [
            {"role": "system", "content": worker_sys},
            {"role": "user",   "content": f"[Leader_Agent]: {leader_initial_ask}"},
        ]
        worker_initial_output = llm_call(worker_sys, f"[Leader_Agent]: {leader_initial_ask}", rollout_config)

        # Training sample: Worker Turn 1
        samples.append({
            "paper_idx": paper_idx,
            "rollout_id": rollout_config["rollout_id"],
            "rollout_config": {k: rollout_config[k] for k in ("temperature", "top_p", "seed")},
            "agent_role": "worker",
            "agent_name": worker_name,
            "turn_type": "initial",
            "input_messages": worker_initial_input,
            "output": worker_initial_output,
            "ground_truth": ground_truth,
        })

        # ---- Turn 2: Leader gives feedback ----
        leader_feedback_user_msg = (
            f"Worker '{worker_name}' (specialty: {specialty}) submitted this initial "
            f"list of limitations:\n\n{worker_initial_output}\n\n"
            "Please provide targeted feedback (MODE A) to improve specificity, "
            "evidence grounding, and coverage. End with the required instruction."
        )
        leader_feedback_input = [
            {"role": "system", "content": leader_sys},
            {"role": "user",   "content": leader_feedback_user_msg},
        ]
        leader_feedback_output = llm_call(leader_sys, leader_feedback_user_msg, rollout_config)

        # Training sample: Leader feedback
        samples.append({
            "paper_idx": paper_idx,
            "rollout_id": rollout_config["rollout_id"],
            "rollout_config": {k: rollout_config[k] for k in ("temperature", "top_p", "seed")},
            "agent_role": "leader",
            "agent_name": "Leader_Agent",
            "turn_type": "feedback_to_worker",
            "target_worker": worker_name,
            "input_messages": leader_feedback_input,
            "output": leader_feedback_output,
            "ground_truth": ground_truth,
        })

        # ---- Turn 3: Worker revises based on feedback ----
        worker_revise_user_msg = (
            f"[Leader_Agent]: {leader_initial_ask}\n\n"
            f"[Your previous answer]:\n{worker_initial_output}\n\n"
            f"[Leader_Agent feedback]:\n{leader_feedback_output}\n\n"
            "Now return ONLY your revised bullet list."
        )
        worker_revise_input = [
            {"role": "system", "content": worker_sys},
            {"role": "user",   "content": worker_revise_user_msg},
        ]
        worker_revised_output = llm_call(worker_sys, worker_revise_user_msg, rollout_config)

        # Training sample: Worker Turn 2 (revised)
        samples.append({
            "paper_idx": paper_idx,
            "rollout_id": rollout_config["rollout_id"],
            "rollout_config": {k: rollout_config[k] for k in ("temperature", "top_p", "seed")},
            "agent_role": "worker",
            "agent_name": worker_name,
            "turn_type": "revised",
            "input_messages": worker_revise_input,
            "output": worker_revised_output,
            "ground_truth": ground_truth,
        })

        all_worker_revised[worker_name] = worker_revised_output

    # ---------- LEADER HANDOFF TO MASTER ----------
    bundle_lines = ["Master Agent, here are the limitation analyses from the team:"]
    for wname, out in all_worker_revised.items():
        bundle_lines.append(f"\n### {wname}:\n{out}")
    bundle_lines.append(
        "\nPlease synthesize them into a single, consolidated, non-redundant, "
        "high-quality list of limitations, grouped by category."
    )
    leader_bundle_user_msg = (
        "You have just collected revised outputs from all 6 specialist workers. "
        "Produce the MODE B handoff message to the Master Agent. "
        "Here are the 6 revised outputs:\n\n"
        + "\n\n".join([f"### {w}:\n{o}" for w, o in all_worker_revised.items()])
    )
    leader_handoff_input = [
        {"role": "system", "content": leader_sys},
        {"role": "user",   "content": leader_bundle_user_msg},
    ]
    leader_handoff_output = llm_call(leader_sys, leader_bundle_user_msg, rollout_config)

    samples.append({
        "paper_idx": paper_idx,
        "rollout_id": rollout_config["rollout_id"],
        "rollout_config": {k: rollout_config[k] for k in ("temperature", "top_p", "seed")},
        "agent_role": "leader",
        "agent_name": "Leader_Agent",
        "turn_type": "handoff_to_master",
        "input_messages": leader_handoff_input,
        "output": leader_handoff_output,
        "ground_truth": ground_truth,
    })

    # ---------- MASTER SYNTHESIS ----------
    master_input = [
        {"role": "system", "content": master_sys},
        {"role": "user",   "content": leader_handoff_output},
    ]
    master_output = llm_call(master_sys, leader_handoff_output, rollout_config)

    samples.append({
        "paper_idx": paper_idx,
        "rollout_id": rollout_config["rollout_id"],
        "rollout_config": {k: rollout_config[k] for k in ("temperature", "top_p", "seed")},
        "agent_role": "master",
        "agent_name": "Master_Agent",
        "turn_type": "synthesis",
        "input_messages": master_input,
        "output": master_output,
        "ground_truth": ground_truth,
    })

    return {
        "samples": samples,
        "final_master_output": master_output,
    }

# ==========================================
# 7. SAVE / CONSOLIDATE
# ==========================================

def consolidate_and_save(all_paper_records):
    with open(OUTPUT_FULL_JSON, "w") as f:
        json.dump(all_paper_records, f, indent=2)

    worker, leader, master = [], [], []
    for paper in all_paper_records:
        for rollout in paper.get("rollouts", []):
            for s in rollout.get("samples", []):
                if   s["agent_role"] == "worker": worker.append(s)
                elif s["agent_role"] == "leader": leader.append(s)
                elif s["agent_role"] == "master": master.append(s)

    with open(OUTPUT_WORKER_JSON, "w") as f: json.dump(worker, f, indent=2)
    with open(OUTPUT_LEADER_JSON, "w") as f: json.dump(leader, f, indent=2)
    with open(OUTPUT_MASTER_JSON, "w") as f: json.dump(master, f, indent=2)

    print(f"  💾 workers: {len(worker)} | leader: {len(leader)} | master: {len(master)}")

# ==========================================
# 8. MAIN
# ==========================================

def run_pipeline():
    print(f"Loading CSV: {INPUT_CSV}")
    df_full = pd.read_csv(INPUT_CSV)
    df = df_full.copy().reset_index(drop=True)
    print(f"Processing {len(df)} papers × {len(ROLLOUT_CONFIGS)} rollouts.")

    all_paper_records = []
    existing_checkpoints = set(os.listdir(CHECKPOINT_DIR))

    for local_i in tqdm(range(len(df)), desc="Papers"):
        global_i = local_i
        ckpt_name = f"paper_{global_i:05d}.json"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name) 

        if ckpt_name in existing_checkpoints:
            try:
                with open(ckpt_path, "r") as f:
                    cached = json.load(f)
                # Only reuse the checkpoint if at least one rollout has real samples
                has_real_data = any(
                    len(r.get("samples", [])) > 0 for r in cached.get("rollouts", [])
                )
                if has_real_data:
                    all_paper_records.append(cached)
                    continue
                else:
                    print(f"[RETRY] paper {global_i}: checkpoint has no samples, re-running")
            except Exception:
                pass

        row = df.iloc[local_i]
        paper_text    = str(row.get(INPUT_COL, ""))
        ground_truth  = str(row.get(GT_COL, ""))

        paper_text    = truncate_text_to_tokens(paper_text, PAPER_TOKEN_LIMIT)

        if len(paper_text) < 100:
            print(f"[SKIP] paper {global_i}: text too short")
            continue

        paper_record = {
            "paper_idx":     int(global_i),
            "paper_text":    paper_text,
            "ground_truth":  ground_truth,
            "rollouts":      []
        }

        for rc in ROLLOUT_CONFIGS:
            print(f"  → paper {global_i} | rollout {rc['rollout_id']} (T={rc['temperature']}, seed={rc['seed']})")
            try:
                result = execute_single_rollout(global_i, paper_text, ground_truth, rc)
                paper_record["rollouts"].append({
                    "rollout_id":          rc["rollout_id"],
                    "rollout_config":      rc,
                    "samples":             result["samples"],
                    "final_master_output": result["final_master_output"],
                })
                time.sleep(0.5)
            except Exception as e:
                print(f"  [ERROR] paper {global_i} rollout {rc['rollout_id']}: {e}")
                paper_record["rollouts"].append({
                    "rollout_id":     rc["rollout_id"],
                    "rollout_config": rc,
                    "error":          str(e),
                    "samples":        [],
                    "final_master_output": "",
                })

        with open(ckpt_path, "w") as f:
            json.dump(paper_record, f, indent=2)
        all_paper_records.append(paper_record)

        if (local_i + 1) % 5 == 0:
            consolidate_and_save(all_paper_records)

    consolidate_and_save(all_paper_records)
    print(f"\n✅ Done. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_pipeline()