"""
Single-file Rollout Pipeline for SD-DPO
========================================

Merges: config.py, data_loader.py, step_prompts.py, 01_generate_rollouts.py

Usage:
  python rollout_pipeline.py --model strong
  python rollout_pipeline.py --model weak
  python rollout_pipeline.py --model both
"""

import os
import ast
import json
import re
import time
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import pandas as pd
import tiktoken
import openai

logger = logging.getLogger(__name__)

# ============================================================
# CONFIG (from config.py)
# ============================================================

OUTPUT_BASE_DIR = "other_experiments/dpo_novagents/output"

@dataclass
class Config:
    # --- Data paths ---
    # ROLLOUT INPUT: Using the newly requested balanced CSV
    rollout_data_path: str = "data/balanced_data/df_with_retrieved_sections.csv"

    # MAIN INPUT: has 'pdf_text_without_gt' column (Paper A — the paper being reviewed)
    paper_data_path: str = "data/balanced_data/df_updated_with_retrieval.csv"

    # PAPER B POOL: nougat papers used for comparison / retrieval simulation
    nougat_data_path: str = "data/nougat_data/nougat_all_papers_dataframe_excluding_bal_not_bal_data.csv"

    # How many rows to take from the end of the rollout CSV
    rollout_sample_size: int = 150
    rollout_random_seed: int = 42

    # OUTPUT: all results saved here
    output_dir: str = OUTPUT_BASE_DIR

    # --- Sub-directories (created automatically) ---
    rollouts_dir: str = os.path.join(OUTPUT_BASE_DIR, "rollouts")
    scores_dir: str = os.path.join(OUTPUT_BASE_DIR, "scores")
    pairs_dir: str = os.path.join(OUTPUT_BASE_DIR, "pairs")
    checkpoints_dir: str = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
    eval_dir: str = os.path.join(OUTPUT_BASE_DIR, "eval")
    logs_dir: str = os.path.join(OUTPUT_BASE_DIR, "logs")

    # --- Models ---
    strong_model: str = "gpt-4o-mini"
    weak_model: str = "qwen2_5_3b_instruct"
    reward_judge_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_cache: str = "hf_cache"

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

    # --- Paper B Summarization Limits ---
    per_retr_paper_input_tok: int = 3000
    per_retr_paper_summary_tok: int = 700

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

# ============================================================
# UTILITY FUNCTIONS & LLM HELPERS (Updated with tiktoken)
# ============================================================

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def tok_len(text: str) -> int:
    return len(tokenizer.encode(text or ""))

def truncate_to_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    """Truncate text to a maximum number of tokens."""
    if not text: return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens: return text
    ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
    return tokenizer.decode(ids)

def openai_chat_completion(content: str, max_tokens: int, temperature: float = 0.2) -> str:
    """Direct OpenAI API call for summarization tasks."""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"OpenAI chat/completions failed: {e}")
        return ""

def build_b_summaries_from_list(lst: list, k: int = 3) -> str:
    """
    Takes the parsed relevant_papers_list, summarizes each paper up to k items, 
    and combines them into a single Paper B context string.
    """
    config = Config()
    summaries = []
    
    if not lst:
        return "(No related papers available for comparison)"

    for idx in range(k):
        if idx < len(lst):
            item_text = str(lst[idx]).strip()
            # Truncate raw retrieved paper before sending to summarizer
            item_text = truncate_to_tokens(item_text, config.per_retr_paper_input_tok)
            prompt = f"Summarize Paper B for limitations comparison:\n{item_text}"
            summary = openai_chat_completion(prompt, config.per_retr_paper_summary_tok).replace("{", "")
            summaries.append(summary)
            
    combined = "\n\n".join([f"--- Retrieved Paper #{i+1} ---\n{summaries[i]}" for i in range(len(summaries))]).strip()
    return combined

def safe_parse_json(text: str, default: Any = None) -> Any:
    """Robustly parse JSON from LLM output."""
    if default is None:
        default = []
    if not text or not isinstance(text, str):
        return default
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    return default

def truncate_for_prompt(text: str, max_chars: int = 4000) -> str:
    """Truncate text to fit in prompt (char based fallback)."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        truncated = truncated[:last_period + 1]
    return truncated

# ============================================================
# DATA LOADER
# ============================================================

def load_rollout_papers(config: Config) -> List[Dict]:
    """
    Load papers for rollout generation from the single CSV file.
    Extracts main text as a raw string and evaluates relevant_papers_list for summarization.
    """
    csv_path = config.rollout_data_path
    sample_size = config.rollout_sample_size

    logger.info(f"Loading ROLLOUT papers from: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} total rows initially")

    # Adhere to User Correction Ledger: skip row 100 to 200 explicitly
    df = df.drop(index=range(100, 201), errors='ignore')
    
    df = df.tail(sample_size)
    logger.info(f"  Selected last {len(df)} rows")

    papers = []
    skipped = 0
    parse_errors = 0

    for idx, row in df.iterrows():

        # ---- 1. Extract Paper A Text (Raw String, NO PARSING) ----
        if "input_text_cleaned" in row and pd.notna(row["input_text_cleaned"]):
            raw_text = str(row["input_text_cleaned"]).strip()
        else:
            skipped += 1
            continue

        if len(raw_text) < 100:
            skipped += 1
            continue

        # ---- 2. Extract and Parse relevant_papers_list ----
        rel_list = []
        if "relevant_papers_list" in row and pd.notna(row["relevant_papers_list"]):
            rel_raw = str(row["relevant_papers_list"]).strip()
            # Strip explicit outer single quotes if they exist in the CSV string
            if rel_raw.startswith("'") and rel_raw.endswith("'"):
                rel_raw = rel_raw[1:-1]
            try:
                # Apply literal eval to make dict/list from string
                rel_list = ast.literal_eval(rel_raw)
                if not isinstance(rel_list, list):
                    rel_list = [rel_list]
            except Exception as e:
                logger.warning(f"  Row {idx} relevant_papers_list parse failed: {e}")
                parse_errors += 1
                pass

        # ---- 3. Extract Title (Fallback if missing) ----
        title = f"Paper_{idx}"
        if "title" in row and pd.notna(row["title"]):
            title = str(row["title"]).strip()

        paper = {
            "id": str(idx),
            "title": title,
            "text": raw_text,
            "relevant_papers_list": rel_list,
        }

        papers.append(paper)

    logger.info(f"  Prepared {len(papers)} rollout papers "
                f"(skipped {skipped} empty, {parse_errors} parse errors)")

    if papers:
        s = papers[0]
        logger.info(f"  Sample rollout paper:")
        logger.info(f"    title      = '{s['title'][:80]}'")
        logger.info(f"    text len   = {len(s['text'])} chars")
        logger.info(f"    relevant   = {len(s['relevant_papers_list'])} items to summarize")
        
    return papers

# ============================================================
# STEP PROMPTS (from step_prompts.py)
# ============================================================

STEP_NAMES = [
    "claim_extraction",       
    "novelty_technical",      
    "experimental_scope",     
    "limitation_synthesis",   
]

HARSH_REVIEWER_POLICY = """
HARSH REVIEWER MODE (STRICT):
You are an extremely skeptical reviewer. Assume low novelty unless proven otherwise with direct evidence.

Scoring rule (score in [0,1]):
- Start from 0.0 by default.
- Increase score only if you can point to CLEAR, SPECIFIC differences between Paper A and Paper B.
- If there is ANY meaningful overlap (same task framing, same method family, same experiment setup), score MUST be <= 0.30.
- If Paper A's contribution is a known variant / incremental tweak / combination of known components already in Paper B, score MUST be <= 0.20.
- Score > 0.50 ONLY if Paper A introduces a new capability that Paper B does not cover AND you cite explicit evidence.
- If evidence is missing or ambiguous, score MUST be <= 0.15 and say "insufficient evidence".

Hard constraints:
- Do NOT reward novelty because Paper A mentions something Paper B does not; absence of mention is NOT novelty.
- Penalize vague claims ("novel", "first", "unique") unless backed by concrete differences.
- Treat re-framing/rewording as NOT novel.
- Prefer false negatives over false positives.
""".strip()

STEP_PROMPTS = {

    "claim_extraction": """You are a scientific paper analyst. Read Paper A below and extract its KEY CLAIMS 
about contributions, novelty, and significance.

Paper A:
{paper_a_text}

Extract claims as a JSON list. Each claim should have:
- "claim": the verbatim or closely paraphrased claim from the paper
- "section": which section it appears in (e.g., "abstract", "introduction", "method", "results")
- "type": one of ["novelty", "performance", "methodology", "scope", "theoretical"]
- "evidence_quote": a short quote from Paper A supporting this claim (max 30 words)

Return between 3 and 8 claims. Focus on novelty and performance claims.
Output ONLY valid JSON array. No preamble, no markdown fences.""",

    "novelty_technical": """Evaluate the novelty of Paper A compared to Paper B.

{harsh_reviewer}

Focus on TWO dimensions:

DIMENSION 1 — Technical Contributions & Novelty:
Assess whether Paper A introduces significant new concepts or insights, or if it primarily offers 
incremental improvements, weak technical contributions, or simplistic adaptations of existing methods.
Consider if the proposed methods closely resemble Paper B or rely heavily on established techniques.

DIMENSION 2 — Methodological Clarity & Rigor:
Assess the clarity and rigor of Paper A's methodology compared to Paper B, including whether Paper A 
articulates experimental setups in detail for reproducibility, or lacks sufficient detail.

Paper A Title: {title_a}
Paper A Claims:
{claims_json}

Paper A Key Sections:
{paper_a_text}

Paper B Summary:
{paper_b_text}

Output format (STRICT):
Technical Novelty Score: <float 0..1>
Methodological Rigor Score: <float 0..1>
Reasons:
- <bullet 1 with comparison against Paper B>
- <bullet 2>
- <bullet 3>
Evidence from Paper A:
- <pointer 1>
Evidence from Paper B:
- <pointer 1>
Identified Technical Limitations:
- <limitation 1 with grounding>
- <limitation 2 with grounding>""",

    "experimental_scope": """Evaluate Paper A compared to Paper B on THREE dimensions.

{harsh_reviewer}

DIMENSION 1 — Experimental Validation & Comparative Analysis:
Assess whether Paper A provides comprehensive experimental validation, adequate comparisons with 
benchmarks and state-of-the-art techniques, or lacks sufficient benchmarking and fails to 
demonstrate significant performance improvements.

DIMENSION 2 — Literature Review & Contextualization:
Assess the thoroughness of Paper A's literature review and how well it contextualizes its 
contributions within existing research. Does it overlook prior studies (like Paper B)?

DIMENSION 3 — Scope of Analysis & Generalizability:
Assess the breadth of Paper A's analysis — does it explore diverse datasets and broader implications, 
or is it limited to narrow tasks restricting generalizability?

Paper A Title: {title_a}
Paper A Claims:
{claims_json}

Technical Analysis from Previous Step:
{previous_step_output}

Paper A Key Sections:
{paper_a_text}

Paper B Summary:
{paper_b_text}

Output format (STRICT):
Experimental Validation Score: <float 0..1>
Literature Contextualization Score: <float 0..1>
Scope & Generalizability Score: <float 0..1>
Reasons:
- <bullet 1 comparing Paper A vs Paper B>
- <bullet 2>
- <bullet 3>
Evidence from Paper A:
- <pointer 1>
Evidence from Paper B:
- <pointer 1>
Identified Experimental/Scope Limitations:
- <limitation 1 with grounding>
- <limitation 2 with grounding>""",

    "limitation_synthesis": """You are an extremely strict, harsh peer reviewer synthesizing final 
novelty limitations for Paper A based on all previous analysis.

{harsh_reviewer}

FOCUS — Claim Accuracy & Overclaiming:
Assess whether Paper A's claims about contributions are substantiated by results, or if it 
exaggerates novelty, effectiveness, or impact without adequate evidence.

Paper A Title: {title_a}
Paper A Claims:
{claims_json}

Technical Analysis (Step 2):
{step2_output}

Experimental & Scope Analysis (Step 3):
{step3_output}

Paper B Summary:
{paper_b_text}

SYNTHESIZE all analysis into final limitations. For each limitation:
- "limitation": a specific, concrete limitation statement (2-3 sentences, NOT generic)
- "grounding": which evidence from Steps 2-3 supports this
- "severity": one of ["critical", "major", "minor"]
- "category": one of ["incremental_contribution", "missing_baseline", "overclaiming", 
  "narrow_scope", "methodological_gap", "insufficient_differentiation"]
- "novelty_score": float 0..1 for this specific aspect
- "suggestion": what the authors could do to address this

Generate between 3 and 6 limitations. Each MUST reference specific evidence.
Output ONLY valid JSON array. No preamble, no markdown fences.""",
}

# ============================================================
# MODEL WRAPPERS
# ============================================================

class StrongModelClient:
    """GPT-4o-mini via OpenAI API."""

    def __init__(self, config):
        self.client = openai.OpenAI()
        self.model = config.strong_model
        self.temperature = config.temperature_strong
        self.max_tokens = config.max_gen_tokens
        self.rpm_limit = config.openai_requests_per_minute
        self._last_call_time = 0

    def generate(self, prompt: str) -> str:
        elapsed = time.time() - self._last_call_time
        min_interval = 60.0 / self.rpm_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self._last_call_time = time.time()
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            time.sleep(5)
            return ""

class WeakModelClient:
    """Qwen-3B (or any local HuggingFace model)."""

    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading weak model: {config.weak_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.weak_model, cache_dir=config.hf_cache
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.weak_model,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=config.hf_cache,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature = config.temperature_weak
        self.max_tokens = config.max_gen_tokens
        logger.info(f"Weak model loaded on {self.model.device}")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=3072).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

# ============================================================
# TRAJECTORY GENERATION
# ============================================================

def generate_trajectory(paper: dict, client, rollout_idx: int, model_type: str) -> dict:
    """
    Generate a complete 4-step trajectory for one paper.
    """
    trajectory = {
        "paper_id": paper["id"],
        "paper_title": paper["title"],
        "model_type": model_type,
        "rollout_idx": rollout_idx,
        "timestamp": datetime.now().isoformat(),
        "steps": {},
    }

    # Fetch the pre-summarized Paper B text
    paper_b_text = paper.get("paper_b_text_summarized", "(No related papers available for comparison)")

    # Truncated Paper A text for prompts
    paper_a_text = truncate_to_tokens(paper["text"], max_tokens=2500)

    # ==============================================================
    # Step 1: Claim Extraction
    # ==============================================================
    logger.info(f"    Step 1: Claim Extraction")

    prompt_1 = STEP_PROMPTS["claim_extraction"].format(
        paper_a_text=paper_a_text
    )

    raw_1 = client.generate(prompt_1)
    claims = safe_parse_json(raw_1, default=[])

    trajectory["steps"]["claim_extraction"] = {
        "prompt": prompt_1,         
        "raw_output": raw_1,        
        "parsed": claims,           
        "num_items": len(claims),
    }
    logger.info(f"      Extracted {len(claims)} claims")

    # ==============================================================
    # Step 2: Novelty & Technical Analysis
    # ==============================================================
    logger.info(f"    Step 2: Novelty Technical")

    claims_json = json.dumps(claims[:6], indent=2, default=str)

    prompt_2 = STEP_PROMPTS["novelty_technical"].format(
        harsh_reviewer=HARSH_REVIEWER_POLICY,
        title_a=paper["title"],
        claims_json=truncate_for_prompt(claims_json, 1500),
        paper_a_text=paper_a_text,
        paper_b_text=paper_b_text,
    )

    raw_2 = client.generate(prompt_2)

    trajectory["steps"]["novelty_technical"] = {
        "prompt": prompt_2,         
        "raw_output": raw_2,
        "parsed": raw_2,            
        "num_items": 1,
    }
    logger.info(f"      Generated technical analysis ({len(raw_2)} chars)")

    # ==============================================================
    # Step 3: Experimental & Scope
    # ==============================================================
    logger.info(f"    Step 3: Experimental & Scope")

    prompt_3 = STEP_PROMPTS["experimental_scope"].format(
        harsh_reviewer=HARSH_REVIEWER_POLICY,
        title_a=paper["title"],
        claims_json=truncate_for_prompt(claims_json, 1500),
        previous_step_output=truncate_for_prompt(raw_2, 1500),  
        paper_a_text=paper_a_text,
        paper_b_text=paper_b_text,
    )

    raw_3 = client.generate(prompt_3)

    trajectory["steps"]["experimental_scope"] = {
        "prompt": prompt_3,         
        "raw_output": raw_3,
        "parsed": raw_3,
        "num_items": 1,
    }
    logger.info(f"      Generated experimental analysis ({len(raw_3)} chars)")

    # ==============================================================
    # Step 4: Limitation Synthesis
    # ==============================================================
    logger.info(f"    Step 4: Limitation Synthesis")

    prompt_4 = STEP_PROMPTS["limitation_synthesis"].format(
        harsh_reviewer=HARSH_REVIEWER_POLICY,
        title_a=paper["title"],
        claims_json=truncate_for_prompt(claims_json, 1500),
        step2_output=truncate_for_prompt(raw_2, 1500),   
        step3_output=truncate_for_prompt(raw_3, 1500),   
        paper_b_text=paper_b_text,
    )

    raw_4 = client.generate(prompt_4)
    limitations = safe_parse_json(raw_4, default=[])

    trajectory["steps"]["limitation_synthesis"] = {
        "prompt": prompt_4,         
        "raw_output": raw_4,
        "parsed": limitations,
        "num_items": len(limitations),
    }
    logger.info(f"      Generated {len(limitations)} limitations")

    return trajectory

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for SD-DPO")
    parser.add_argument("--model", choices=["strong", "weak", "both"], default="both",
                        help="Which model to generate rollouts from")
    parser.add_argument("--max_papers", type=int, default=None,
                        help="Override max papers to process")
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Override number of rollouts per paper")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this paper index (for resuming)")
    args = parser.parse_args()

    config = get_config()
    if args.max_papers:
        config.max_papers = args.max_papers
    if args.num_rollouts:
        config.num_rollouts_per_model = args.num_rollouts

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.logs_dir, "01_rollouts.log")),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 60)
    logger.info("STEP 1: GENERATE ROLLOUTS")
    logger.info(f"  Model(s): {args.model}")
    logger.info(f"  Max papers: {config.max_papers}")
    logger.info(f"  Rollouts per model: {config.num_rollouts_per_model}")
    logger.info(f"  Start index: {args.start_idx}")
    logger.info("=" * 60)

    papers = load_rollout_papers(config)
    papers = papers[args.start_idx:]
    logger.info(f"Processing {len(papers)} papers")

    # Initialize model clients
    models_to_run = []
    if args.model in ["strong", "both"]:
        logger.info("Initializing strong model (GPT-4o-mini)...")
        models_to_run.append(("strong", StrongModelClient(config)))
    if args.model in ["weak", "both"]:
        logger.info("Initializing weak model (Qwen)...")
        models_to_run.append(("weak", WeakModelClient(config)))

    # Generate rollouts
    total_trajectories = 0

    for paper_idx, paper in enumerate(papers):
        logger.info(f"\n--- Paper {paper_idx + 1}/{len(papers)}: "
                     f"{paper['title'][:60]}... ---")
        
        # -------------------------------------------------------------
        # LLM SUMMARIZATION STEP: Extract and summarize Paper B baseline
        # Done once per paper before looping through the rollout models
        # -------------------------------------------------------------
        logger.info(f"    Summarizing {len(paper.get('relevant_papers_list', []))} relevant papers to build Paper B baseline...")
        b_combined = build_b_summaries_from_list(paper.get("relevant_papers_list", []), k=3)
        paper["paper_b_text_summarized"] = b_combined

        for model_type, client in models_to_run:
            paper_rollouts = []

            for rollout_idx in range(config.num_rollouts_per_model):
                logger.info(f"  Rollout {rollout_idx + 1}/"
                             f"{config.num_rollouts_per_model} ({model_type})")

                try:
                    traj = generate_trajectory(
                        paper, client, rollout_idx, model_type
                    )
                    paper_rollouts.append(traj)
                    total_trajectories += 1
                except Exception as e:
                    logger.error(f"  ERROR in rollout: {e}", exc_info=True)
                    continue

            # Save rollouts for this paper + model
            if paper_rollouts:
                out_file = os.path.join(
                    config.rollouts_dir,
                    f"paper_{paper['id']}_{model_type}.json"
                )
                with open(out_file, "w") as f:
                    json.dump(paper_rollouts, f, indent=2, default=str)
                logger.info(f"  Saved {len(paper_rollouts)} rollouts → {out_file}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"COMPLETE: {total_trajectories} trajectories generated")
    logger.info(f"Output: {config.rollouts_dir}")
    logger.info(f"{'=' * 60}")

if __name__ == "__main__":
    main()