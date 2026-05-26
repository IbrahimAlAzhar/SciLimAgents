"""
Inference & Evaluation Pipeline (v2)
=======================================
Key v2 changes:
  - Uses GRPO-trained models for ALL agent types (worker, leader, master)
  - Each agent type can have its own checkpoint
  - Supports fallback to SFT if a GRPO checkpoint doesn't exist
  - Full trajectory logging for qualitative analysis
"""

import os
import re
import gc
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
from config import PipelineConfig
from multi_agent_rollout import (
    load_agent_model, run_single_rollout, truncate_text, Trajectory,
)
from reward_functions import rule_based_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ================================================================
# 1. FIND BEST CHECKPOINTS
# ================================================================

def find_best_checkpoint(grpo_dir: str, num_iterations: int = 3) -> Optional[str]:
    """Find the latest GRPO iteration checkpoint."""
    for i in range(num_iterations, 0, -1):
        path = os.path.join(grpo_dir, f"iteration_{i}", "final")
        if os.path.exists(path):
            log.info(f"Found checkpoint: {path}")
            return path
    return None

def resolve_model_paths(config: PipelineConfig) -> Dict[str, str]:
    """
    Resolve the best available model checkpoint for each agent type.
    Priority: GRPO checkpoint > SFT checkpoint
    """
    sft = config.paths.sft_model_dir

    worker_dir = find_best_checkpoint(
        config.paths.grpo_worker_dir,
        config.grpo.worker_num_grpo_iterations,
    ) or sft

    master_dir = find_best_checkpoint(
        config.paths.grpo_master_dir,
        config.grpo.num_grpo_iterations,
    ) or sft

    leader_dir = find_best_checkpoint(
        config.paths.grpo_leader_dir,
        config.grpo.leader_num_grpo_iterations,
    ) or sft

    paths = {
        "worker": worker_dir,
        "master": master_dir,
        "leader": leader_dir,
    }

    log.info("Resolved model paths:")
    for agent, path in paths.items():
        is_grpo = "grpo" in path.lower()
        log.info(f"  {agent}: {path} ({'GRPO' if is_grpo else 'SFT'})")

    return paths

# ================================================================
# 2. MULTI-AGENT INFERENCE WITH SEPARATE MODELS
# ================================================================

def run_inference(
    config: PipelineConfig,
    worker_dir: Optional[str] = None,
    leader_dir: Optional[str] = None,
    master_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run inference using GRPO-trained models for each agent type.

    Memory strategy for 2x40GB GPUs:
    - If all models are the same checkpoint → load once (~10GB with QLoRA)
    - If different checkpoints → load sequentially, free between
      OR use device_map to spread across GPUs
    """
    # Resolve paths
    model_paths = resolve_model_paths(config)
    w_dir = worker_dir or model_paths["worker"]
    l_dir = leader_dir or model_paths["leader"]
    m_dir = master_dir or model_paths["master"]

    log.info(f"Inference with: worker={w_dir}, leader={l_dir}, master={m_dir}")

    # Load data
    df = pd.read_csv(config.paths.inference_csv)
    if max_samples:
        df = df.head(max_samples)
    log.info(f"Inference dataset: {len(df)} samples")

    # Check if all models are the same
    all_same = (w_dir == l_dir == m_dir)

    if all_same:
        log.info("All agents use same model — loading once")
        model, tokenizer = load_agent_model(config, w_dir)
        w_model, w_tok = model, tokenizer
        l_model, l_tok = model, tokenizer
        m_model, m_tok = model, tokenizer
    else:
        log.info("Loading separate models for each agent type")
        w_model, w_tok = load_agent_model(config, w_dir)
        l_model, l_tok = load_agent_model(config, l_dir)
        m_model, m_tok = load_agent_model(config, m_dir)

    # Use the first available as the "base" model for the rollout function
    base_model = w_model
    base_tok = w_tok

    # Run inference
    results = []
    trajectories_log = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        paper_text = str(row.get(config.paths.inference_input_col, "") or "")
        ground_truth = str(row.get(config.paths.inference_gt_col, "") or "")

        if len(paper_text) < 100:
            results.append({
                "idx": i,
                "final_limitations": "SKIPPED_SHORT_TEXT",
                "ground_truth": ground_truth,
            })
            continue

        traj = run_single_rollout(
            base_model, base_tok,
            paper_text, i, 0, ground_truth, config,
            temperature=0.7, top_p=0.9, seed=config.seed + i,
            worker_model=w_model, worker_tokenizer=w_tok,
            leader_model=l_model, leader_tokenizer=l_tok,
            master_model=m_model, master_tokenizer=m_tok,
        )

        # Compute scores
        scores = rule_based_reward(
            traj.final_limitations, ground_truth, paper_text, config.reward,
        )

        # Extract leader decisions for analysis
        leader_info = {}
        if traj.leader_decisions:
            leader_info = {
                "selected_workers": traj.leader_decisions.get("selected_workers", []),
                "num_rounds": traj.leader_decisions.get("num_rounds", 1),
            }
        if traj.leader_feedback_parsed:
            leader_info["final_score"] = traj.leader_feedback_parsed.get("final_score", 0)
            leader_info["coverage_gaps"] = traj.leader_feedback_parsed.get("coverage_gaps", [])

        result = {
            "idx": i,
            "final_limitations": traj.final_limitations,
            "ground_truth": ground_truth,
            "num_workers_used": len(traj.worker_outputs),
            "workers_selected": json.dumps(leader_info.get("selected_workers", [])),
            "leader_rounds": leader_info.get("num_rounds", 1),
            "leader_score": leader_info.get("final_score", 0),
            **{f"reward_{k}": v for k, v in scores.items()},
        }
        results.append(result)
        trajectories_log.append(traj.to_dict())

        if (i + 1) % 20 == 0:
            _save_partial(results, trajectories_log, config)

    # Final save
    results_df = pd.DataFrame(results)
    output_path = os.path.join(config.paths.inference_output_dir, "inference_results.csv")
    results_df.to_csv(output_path, index=False)

    traj_path = os.path.join(config.paths.inference_output_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(trajectories_log, f, indent=2, default=str)

    log.info(f"Inference complete: {output_path}")

    # Cleanup
    del w_model, l_model, m_model
    gc.collect()
    torch.cuda.empty_cache()

    return results_df

def _save_partial(results, trajectories, config):
    pd.DataFrame(results).to_csv(
        os.path.join(config.paths.inference_output_dir, "inference_partial.csv"),
        index=False,
    )

# ================================================================
# 3. POINTWISE EVALUATION (GPT-4o-mini)
# ================================================================

def parse_generated_limitations(text: str) -> List[Dict]:
    """Parse generated limitations into structured list."""
    if not isinstance(text, str) or not text.strip():
        return []
    limitations = []
    # Bold-key format
    matches = re.finditer(r'\*\*(.*?)\*\*:?\s*(.*?)(?=\*\*|$)', text, re.DOTALL)
    for m in matches:
        key = m.group(1).strip().rstrip(':')
        val = m.group(2).strip().replace('\n', ' ').strip()
        if val and len(val.split()) >= 5:
            limitations.append({"llm_id": len(limitations)+1, "llm_limitation": f"{key}: {val}"})
    # Fallback
    if not limitations:
        for line in text.strip().split("\n"):
            line = re.sub(r"^[\d]+[.)]\s*", "", line.strip())
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            if len(line.split()) >= 5:
                limitations.append({"llm_id": len(limitations)+1, "llm_limitation": line})
    return limitations

def parse_gt_limitations(text: str) -> List[Dict]:
    if not isinstance(text, str) or not text.strip():
        return []
    results = []
    for i, line in enumerate(text.strip().split("\n")):
        cleaned = re.sub(r"^-\s*", "", line.strip()).strip()
        if cleaned and len(cleaned.split()) >= 3:
            results.append({"gt_id": i, "gt_limitation": cleaned})
    return results

def evaluate_pairs_with_llm(pairs_list, api_key=None, model_id="gpt-4o-mini"):
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai not installed")
        return []

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    results = []

    for i, pair in enumerate(pairs_list or []):
        prompt = (
            "Check whether 'list2' contains a topic or limitation from 'list1' "
            "or 'list1' contains a topic or limitation from 'list2'.\n\n"
            "Your answer should be \"Yes\" or \"No\".\n"
            f"List 1: ground truth limitations: {pair['gt_limitation']}\n"
            f"List 2: llm generated limitations: {pair['llm_limitation']}\n"
        )
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error: {e}"
        results.append([
            f"Pair {i+1}: {answer}",
            f"gt_id:{pair['gt_id']}", f"gt_limitation:{pair['gt_limitation']}",
            f"llm_id:{pair['llm_id']}", f"llm_limitation:{pair['llm_limitation']}",
        ])
    return results

def run_pointwise_evaluation(df, api_key=None, save_path=None):
    df["llm_lim_list"] = df["final_limitations"].apply(parse_generated_limitations)
    df["gt_lim_list"] = df["ground_truth"].apply(parse_gt_limitations)

    def build_pairs(row):
        return [
            {"gt_id": g["gt_id"], "gt_limitation": g["gt_limitation"],
             "llm_id": l["llm_id"], "llm_limitation": l["llm_limitation"]}
            for g in row["gt_lim_list"] for l in row["llm_lim_list"]
        ]

    df["paired_limitations"] = df.apply(build_pairs, axis=1)
    df["llm_evaluation_results"] = None

    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating")):
        df.at[idx, "llm_evaluation_results"] = evaluate_pairs_with_llm(
            row["paired_limitations"], api_key=api_key,
        )
        if save_path and (i+1) % 10 == 0:
            df.to_csv(save_path, index=False)

    if save_path:
        df.to_csv(save_path, index=False)
    return df

# ================================================================
# 4. METRICS
# ================================================================

def compute_recall_precision_f1(df):
    def _compute(row):
        items = row.get("llm_evaluation_results", [])
        if not isinstance(items, list):
            return pd.Series({"recall": 0, "precision": 0, "f1": 0})
        all_gt, all_llm, yes_gt, yes_llm = set(), set(), set(), set()
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            is_yes = "YES" in str(item[0]).upper()
            gid, lid = None, None
            for elem in item:
                if isinstance(elem, str):
                    if elem.startswith("gt_id"):
                        try: gid = int(elem.split(":", 1)[1])
                        except: pass
                    elif elem.startswith("llm_id"):
                        try: lid = int(elem.split(":", 1)[1])
                        except: pass
            if gid is None or lid is None:
                continue
            all_gt.add(gid); all_llm.add(lid)
            if is_yes:
                yes_gt.add(gid); yes_llm.add(lid)
        n_gt, n_llm = len(all_gt), len(all_llm)
        r = len(yes_gt)/n_gt if n_gt else 0
        p = len(yes_llm)/n_llm if n_llm else 0
        f = 2*p*r/(p+r) if (p+r) else 0
        return pd.Series({"recall": r, "precision": p, "f1": f})

    m = df.apply(_compute, axis=1)
    df["recall"] = m["recall"]; df["precision"] = m["precision"]; df["f1"] = m["f1"]
    return df

def compute_similarity_metrics(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False
    try:
        from bert_score import score as bs_fn
        has_bert = True
    except ImportError:
        has_bert = False

    def _sim(row):
        items = row.get("llm_evaluation_results", [])
        if not isinstance(items, list):
            return pd.Series({"avg_cosine": 0, "avg_jaccard": 0, "avg_rougeL": 0, "avg_bertscore": 0})
        gts, llms = [], []
        for item in items:
            if not isinstance(item, (list, tuple)) or "YES" not in str(item[0]).upper():
                continue
            gt_t, llm_t = None, None
            for e in item:
                if isinstance(e, str):
                    if e.startswith("gt_limitation:"): gt_t = e.split("gt_limitation:", 1)[1].strip()
                    elif e.startswith("llm_limitation:"): llm_t = e.split("llm_limitation:", 1)[1].strip()
            if gt_t and llm_t:
                gts.append(gt_t); llms.append(llm_t)
        if not gts:
            return pd.Series({"avg_cosine": 0, "avg_jaccard": 0, "avg_rougeL": 0, "avg_bertscore": 0})
        cos, jac, rou, ber = [], [], [], []
        for g, l in zip(gts, llms):
            try:
                v = TfidfVectorizer().fit([g, l])
                cos.append(float(cosine_similarity(v.transform([g, l])[0], v.transform([g, l])[1])[0, 0]))
            except: cos.append(0)
            t1, t2 = set(g.lower().split()), set(l.lower().split())
            u = t1 | t2
            jac.append(len(t1 & t2)/len(u) if u else 0)
            if has_rouge:
                try: rou.append(float(rs.score(g, l)["rougeL"].fmeasure))
                except: rou.append(0)
        if has_bert:
            try:
                P, R, F = bs_fn(llms, gts, lang="en", verbose=False)
                ber = [float(f) for f in F]
            except: ber = [0]*len(gts)
        return pd.Series({
            "avg_cosine": np.mean(cos) if cos else 0,
            "avg_jaccard": np.mean(jac) if jac else 0,
            "avg_rougeL": np.mean(rou) if rou else 0,
            "avg_bertscore": np.mean(ber) if ber else 0,
        })

    s = df.apply(_sim, axis=1)
    for c in s.columns:
        df[c] = s[c]
    return df

# ================================================================
# 5. FULL EVALUATION
# ================================================================

def full_evaluation(config, worker_dir=None, leader_dir=None, master_dir=None,
                    api_key=None, max_samples=None):
    output_dir = config.paths.inference_output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_df = run_inference(
        config, worker_dir=worker_dir, leader_dir=leader_dir,
        master_dir=master_dir, max_samples=max_samples,
    )

    eval_path = os.path.join(output_dir, "eval_results.csv")
    if api_key:
        results_df = run_pointwise_evaluation(results_df, api_key=api_key, save_path=eval_path)
        results_df = compute_recall_precision_f1(results_df)
        results_df = compute_similarity_metrics(results_df)

    final_path = os.path.join(output_dir, "final_evaluation.csv")
    results_df.to_csv(final_path, index=False)

    summary = {}
    for col in ["recall", "precision", "f1", "avg_cosine", "avg_jaccard",
                "avg_rougeL", "avg_bertscore", "reward_rule_based_total",
                "num_workers_used", "leader_score"]:
        if col in results_df.columns:
            summary[col] = float(results_df[col].mean())

    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    for k, v in summary.items():
        log.info(f"  {k:<25}: {v:.4f}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary