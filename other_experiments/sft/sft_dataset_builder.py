"""
SFT Dataset Builder for Paper Limitation Extraction (GRPO-Ready)
=================================================================
Pipeline:
  Stage 1 — Best-of-N Generation:  Generate N reasoning traces per paper
  Stage 2 — Multi-Signal Scoring:  Score each trace via semantic similarity,
             LLM-as-judge, F1 overlap with ground truth, and structural quality
  Stage 3 — Rejection Sampling:    Keep top-K per paper by composite reward
  Stage 4 — Diversity-Aware Filter: Cluster by weakness type and balance
  Stage 5 — Contrastive Pair Gen:  Pair best vs worst for optional DPO/preference data
  Stage 6 — Curriculum Ordering:   Sort by difficulty for staged training
  Stage 7 — Export:                 Train/val splits in JSONL

Requirements:
  pip install openai pandas scikit-learn numpy tqdm sentence-transformers
"""

import os, re, json, time, logging, argparse, random, hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------------------------------------------------------------------------
# Lazy imports (so the script can still be inspected without all deps)
# ---------------------------------------------------------------------------
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def _load_sklearn():
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    return cosine_similarity, KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — override via CLI flags
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CFG = dict(
    source_csv="data/not_balanced_data/df_not_bal_final_strat_samp.csv",
    output_dir="other_experiments/sft/output",
    text_col="input_text_without_lim",
    gt_col="ground_truth_lim_peer",
    best_of_n=3,                # rejection-sampling width
    keep_top_k=1,               # keep top-K per paper after scoring
    global_top_frac=0.70,       # then keep top 60% globally
    val_frac=0.15,
    seed=42,
    paper_max_chars=10_000,
    gpt_model="gpt-4o-mini", # gpt-4o-mini
    judge_model="gpt-4o-mini",  # can be a stronger model
    max_tokens=4096,
    temperature=0.7,            # higher than 0.4 → more diversity across N samples
    sleep_sec=0.3,
    enable_contrastive=True,    # generate DPO-style pairs
    enable_curriculum=True,     # sort by difficulty
    diversity_clusters=6,       # K for weakness-type clustering
    use_embeddings=True,        # semantic similarity to GT
)

# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

# --- Generation prompt (teacher model) ---
SYSTEM_PROMPT_GEN = """\
You are an expert scientific reviewer specialised in critically identifying
limitations of research papers.

You MUST reason through the paper step by step.
Enclose your ENTIRE reasoning inside <think>...</think> tags using these 8 steps:

  Step 1 - UNDERSTAND      : Summarise the paper's main claim and method in 2-3 sentences.
  Step 2 - NOVELTY CHECK   : Is the contribution genuinely novel or incremental?
                              Are significance claims over-stated?
  Step 3 - METHOD AUDIT    : Are theoretical assumptions reasonable?
                              Are there logical gaps, missing proofs, or weak ablations?
  Step 4 - EXPERIMENT AUDIT: Are baselines fair and up-to-date?
                              Are results statistically robust? Are metrics appropriate?
  Step 5 - GENERALISATION  : Does the method work outside the tested settings?
                              Are compute requirements practical?
  Step 6 - CLARITY & REPRO : Can the work be independently replicated?
                              Is the writing precise and complete?
  Step 7 - DATA & ETHICS   : Are datasets representative and well-documented?
                              Bias, fairness, privacy, or dual-use concerns?
  Step 8 - PRIORITISE      : Rank identified limitations from most to least critical.

After </think>, output a clean structured bullet list grouped by category:
  - **Novelty/Significance:** ...
  - **Methodology:** ...
  - **Experiments:** ...
  - **Generalisation/Robustness:** ...
  - **Clarity/Reproducibility:** ...
  - **Data/Ethics:** ...

Be specific, critical, and ground every point in evidence from the paper.
Do NOT hallucinate or invent facts not present in the text.
"""

USER_TMPL_GEN = """\
Identify all key limitations of the paper below using the 8-step reasoning
process described in your instructions.

### Paper Content
{paper}
"""

# --- LLM-as-Judge prompt ---
SYSTEM_PROMPT_JUDGE = """\
You are a meta-reviewer evaluating the quality of a paper limitation analysis.
You will be given:
  1. The paper text
  2. A candidate limitation analysis (with reasoning)
  3. A ground-truth reference list of known limitations

Score the candidate on these five dimensions (each 1-5):
  - COVERAGE:   How many ground-truth limitations are captured?
  - PRECISION:  Are the stated limitations actually valid (no hallucinations)?
  - SPECIFICITY: Are limitations grounded in concrete paper details, not generic?
  - REASONING:  Is the <think> chain-of-thought logical and step-by-step?
  - DEPTH:      Does it identify non-obvious or subtle weaknesses?

Return ONLY a JSON object (no markdown fences) with keys:
  coverage, precision, specificity, reasoning, depth, overall, explanation
where "overall" is the average and "explanation" is one sentence.
"""

USER_TMPL_JUDGE = """\
### Paper (truncated)
{paper}

### Candidate Analysis
{candidate}

### Ground-Truth Limitations
{ground_truth}
"""

# ═══════════════════════════════════════════════════════════════════════════
# PARSING & UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if pd.isna(text) or not text:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()

def parse_response(raw: str) -> Dict[str, str]:
    """Split a raw model response into thinking + limitations."""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        limitations = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        bullet_start = re.search(
            r"^[\-\*\u2022]|\*\*(Novelty|Method|Experiment|Clarity|Data|General)",
            raw, re.MULTILINE,
        )
        if bullet_start:
            thinking = raw[: bullet_start.start()].strip()
            limitations = raw[bullet_start.start() :].strip()
        else:
            cut = int(len(raw) * 0.6)
            thinking = raw[:cut].strip()
            limitations = raw[cut:].strip()
    return {"thinking": thinking, "limitations": limitations}

def extract_bullet_points(text: str) -> List[str]:
    """Extract individual bullet points from a limitation block."""
    lines = re.split(r"\n", text)
    bullets = []
    current = ""
    for line in lines:
        line = line.strip()
        if re.match(r"^[\-\*\u2022]", line):
            if current:
                bullets.append(current.strip())
            current = re.sub(r"^[\-\*\u2022]\s*", "", line)
        elif line:
            current += " " + line
    if current:
        bullets.append(current.strip())
    # Also handle \n-separated ground truths
    if not bullets:
        bullets = [b.strip() for b in text.split("\n") if b.strip()]
    return bullets

def compute_f1_overlap(pred_bullets: List[str], gt_bullets: List[str]) -> float:
    """
    Token-level F1 between predicted and ground-truth limitation sets.
    Each GT bullet is matched to its best-matching predicted bullet.
    """
    if not pred_bullets or not gt_bullets:
        return 0.0

    def _tok(s):
        return set(re.findall(r"\w+", s.lower()))

    precisions, recalls = [], []
    for gt in gt_bullets:
        gt_tokens = _tok(gt)
        if not gt_tokens:
            continue
        best_f1 = 0.0
        for pred in pred_bullets:
            pred_tokens = _tok(pred)
            if not pred_tokens:
                continue
            overlap = gt_tokens & pred_tokens
            p = len(overlap) / len(pred_tokens)
            r = len(overlap) / len(gt_tokens)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            best_f1 = max(best_f1, f1)
        recalls.append(best_f1)

    for pred in pred_bullets:
        pred_tokens = _tok(pred)
        if not pred_tokens:
            continue
        best_f1 = 0.0
        for gt in gt_bullets:
            gt_tokens = _tok(gt)
            if not gt_tokens:
                continue
            overlap = gt_tokens & pred_tokens
            p = len(overlap) / len(pred_tokens)
            r = len(overlap) / len(gt_tokens)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            best_f1 = max(best_f1, f1)
        precisions.append(best_f1)

    avg_p = np.mean(precisions) if precisions else 0.0
    avg_r = np.mean(recalls) if recalls else 0.0
    if avg_p + avg_r == 0:
        return 0.0
    return float(2 * avg_p * avg_r / (avg_p + avg_r))

def structural_quality_score(thinking: str, limitations: str) -> float:
    """
    Heuristic structural quality check (0-1).
    Checks: bullet count, category coverage, reasoning step coverage, length.
    """
    if not limitations:
        return 0.0
    bullets = len(re.findall(r"^[\-\*\u2022]", limitations, re.MULTILINE))
    categories = len(re.findall(
        r"novelty|method|experiment|generaliz|clarity|data|ethic|robustness|reproducib",
        limitations, re.IGNORECASE,
    ))
    steps_found = len(re.findall(
        r"step\s*[1-8]|understand|novelty.?check|method.?audit|experiment|"
        r"generalisa|clarity|data|prioriti",
        thinking, re.IGNORECASE,
    ))
    words = len(limitations.split())

    # Normalised sub-scores
    bullet_score = min(1.0, bullets / 8)
    cat_score = min(1.0, categories / 5)
    step_score = min(1.0, steps_found / 6)
    length_score = min(1.0, words / 300)

    return 0.25 * bullet_score + 0.25 * cat_score + 0.25 * step_score + 0.25 * length_score

# ═══════════════════════════════════════════════════════════════════════════
# GROUNDING SCORE  (inspired by RAGEN §3.3 + VAGEN §4.2)
# ═══════════════════════════════════════════════════════════════════════════
# RAGEN shows that <think> traces trained on sparse rewards hallucinate
# reasoning disconnected from the actual environment state ("thought-state
# mismatch").  VAGEN confirms that state estimation must reference concrete
# entities.  A reasoning trace that says "the authors lack baselines" without
# naming *which* baselines or *which* method is generic boilerplate —
# exactly the template collapse Paper 1 warns about.
#
# This score measures how many paper-specific entities (proper nouns, method
# names, dataset names, numbers, acronyms) from the input paper appear in
# the <think> block.  Higher = more grounded = less template risk.
# ─────────────────────────────────────────────────────────────────────────

def extract_paper_entities(paper: str) -> set:
    """
    Extract a set of 'grounding entities' from the paper text:
      - Capitalised multi-word names  (e.g. "Proximal Policy Optimization")
      - Acronyms / abbreviations      (e.g. "PPO", "GRPO", "KL")
      - Dataset / benchmark names     (e.g. "MATH500", "GSM8K", "ImageNet")
      - Specific numbers/percentages  (e.g. "59.8%", "4x4")
    """
    entities = set()

    # Acronyms (2-6 uppercase letters, optionally with digits)
    for m in re.finditer(r"\b[A-Z][A-Z0-9]{1,5}\b", paper):
        tok = m.group()
        # skip very common English words that happen to match
        if tok not in {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT",
                       "ARE", "NOT", "BUT", "HAS", "WAS", "HIS", "HER",
                       "OUR", "CAN", "ALL", "ITS"}:
            entities.add(tok.lower())

    # Capitalised phrases (2-4 words, likely method/model names)
    for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+|-)){1,3}[A-Z][a-z]+", paper):
        entities.add(m.group().lower())

    # Dataset/benchmark patterns (word + digits)
    for m in re.finditer(r"\b[A-Za-z]+[-_]?\d+[A-Za-z]*\b", paper):
        entities.add(m.group().lower())

    # Specific numeric results (percentages, dimensions)
    for m in re.finditer(r"\d+\.?\d*\s*%", paper):
        entities.add(m.group().strip())

    return entities

def compute_grounding_score(thinking: str, limitations: str, paper: str) -> float:
    """
    Fraction of paper-specific entities that appear in the reasoning trace.
    Returns 0-1; higher means the reasoning is more grounded in the paper.
    """
    entities = extract_paper_entities(paper)
    if not entities:
        return 0.5  # can't measure; neutral

    combined = (thinking + " " + limitations).lower()
    hits = sum(1 for e in entities if e in combined)

    # Normalise: even hitting 30% of entities is strong grounding
    raw = hits / len(entities)
    return min(1.0, raw / 0.30)  # saturates at 30% recall

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-SAMPLE TEMPLATE DETECTION  (inspired by Paper 1 §2 MI diagnostic)
# ═══════════════════════════════════════════════════════════════════════════
# Paper 1 decomposes reasoning variation into H(Z|X) and I(X;Z).  Template
# collapse = high H(Z|X) but low I(X;Z), meaning traces look diverse per
# input but are interchangeable *across* inputs.
#
# We approximate this: after collecting all selected reasoning traces, we
# embed them and compute pairwise cosine similarity.  If a trace is too
# similar to traces from *other* papers, it's likely templated and should
# be penalised.  This is a dataset-level post-hoc filter, not per-sample.
# ─────────────────────────────────────────────────────────────────────────

def detect_and_penalise_templates(
    samples: List[Dict],
    embed_model,
    similarity_threshold: float = 0.85,
    penalty_factor: float = 0.3,
) -> List[Dict]:
    """
    For each sample, compute average cosine similarity of its reasoning
    trace against traces from OTHER papers.  If avg similarity exceeds
    threshold, discount its composite_score by penalty_factor.

    Returns samples with updated scores and a 'template_risk' field.
    """
    if not embed_model or len(samples) < 5:
        for s in samples:
            s["template_risk"] = 0.0
            s["cross_paper_sim"] = 0.0
        return samples

    cosine_similarity, _ = _load_sklearn()

    # Embed all reasoning traces
    traces = [s.get("thinking", "") + " " + s.get("limitations", "") for s in samples]
    embeddings = embed_model.encode(traces, show_progress_bar=False)

    # Compute full pairwise similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # For each sample, compute mean similarity to samples from different papers
    row_indices = [s.get("row_idx", i) for i, s in enumerate(samples)]

    for i, s in enumerate(samples):
        cross_sims = []
        for j in range(len(samples)):
            if i != j and row_indices[i] != row_indices[j]:
                cross_sims.append(sim_matrix[i][j])

        if cross_sims:
            avg_cross_sim = float(np.mean(cross_sims))
            max_cross_sim = float(np.max(cross_sims))
        else:
            avg_cross_sim = 0.0
            max_cross_sim = 0.0

        s["cross_paper_sim"] = round(avg_cross_sim, 4)

        # Flag as template risk if highly similar to other papers' traces
        if avg_cross_sim > similarity_threshold:
            s["template_risk"] = round(avg_cross_sim, 4)
            original = s.get("composite_score", 0)
            s["composite_score"] = round(original * (1 - penalty_factor), 4)
            log.debug(f"  Template penalty: row {s.get('row_idx')} "
                      f"sim={avg_cross_sim:.3f} score {original:.3f}->{s['composite_score']:.3f}")
        else:
            s["template_risk"] = 0.0

    n_flagged = sum(1 for s in samples if s["template_risk"] > 0)
    log.info(f"Template detection: {n_flagged}/{len(samples)} samples flagged "
             f"(threshold={similarity_threshold:.2f}, penalty={penalty_factor:.0%})")
    return samples

# ═══════════════════════════════════════════════════════════════════════════
# WEAKNESS-TYPE TAGGING (for diversity-aware sampling)
# ═══════════════════════════════════════════════════════════════════════════

WEAKNESS_CATEGORIES = {
    "missing_baseline":    r"baseline|comparison|benchmark|state.of.the.art|sota",
    "limited_scope":       r"narrow|limited|specific|only\s+tested|single\s+(domain|dataset|task)",
    "scalability":         r"scal|large.scale|compute|cost|memory|overhead|efficient",
    "statistical":         r"statistic|significance|variance|confidence|error.bar|p.value|repeated",
    "novelty":             r"incremental|marginal|trivial|well.explored|prior.work|already",
    "reproducibility":     r"reproduc|replicate|code|implementation|detail|hyperparameter",
    "ablation":            r"ablation|component|contribution|isolat",
    "generalization":      r"generaliz|transfer|out.of.distribution|ood|domain.shift|robustness",
    "data_quality":        r"bias|imbalance|representative|annotation|noise|label",
    "clarity":             r"clarity|writing|unclear|ambiguous|notation|definition",
}

def tag_weakness_types(text: str) -> List[str]:
    """Return a list of weakness categories present in the text."""
    tags = []
    for cat, pattern in WEAKNESS_CATEGORIES.items():
        if re.search(pattern, text, re.IGNORECASE):
            tags.append(cat)
    return tags if tags else ["other"]

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — BEST-OF-N GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_n_responses(
    client: OpenAI, paper: str, n: int, cfg: dict
) -> List[Dict]:
    """Generate N candidate limitation analyses for one paper."""
    candidates = []
    for i in range(n):
        try:
            resp = client.chat.completions.create(
                model=cfg["gpt_model"],
                max_completion_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_GEN},
                    {"role": "user", "content": USER_TMPL_GEN.format(paper=paper)},
                ],
            )
            raw = resp.choices[0].message.content or ""
            parsed = parse_response(raw)
            candidates.append({
                "raw": raw,
                "thinking": parsed["thinking"],
                "limitations": parsed["limitations"],
                "sample_idx": i,
            })
        except Exception as e:
            log.warning(f"  Generation attempt {i} failed: {e}")
        time.sleep(cfg["sleep_sec"])
    return candidates

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — MULTI-SIGNAL SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_with_llm_judge(
    client: OpenAI, paper: str, candidate: str, ground_truth: str, cfg: dict
) -> Dict:
    """Use LLM-as-Judge to score a candidate analysis against ground truth."""
    try:
        resp = client.chat.completions.create(
            model=cfg["judge_model"],
            max_completion_tokens=512,
            temperature=0.1,  # low temp for consistent judging
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                {"role": "user", "content": USER_TMPL_JUDGE.format(
                    paper=paper[:5000],
                    candidate=candidate[:3000],
                    ground_truth=ground_truth[:2000],
                )},
            ],
        )
        raw = resp.choices[0].message.content or ""
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        scores = json.loads(raw.strip())
        return {
            "judge_coverage": float(scores.get("coverage", 3)),
            "judge_precision": float(scores.get("precision", 3)),
            "judge_specificity": float(scores.get("specificity", 3)),
            "judge_reasoning": float(scores.get("reasoning", 3)),
            "judge_depth": float(scores.get("depth", 3)),
            "judge_overall": float(scores.get("overall", 3)),
            "judge_explanation": scores.get("explanation", ""),
        }
    except Exception as e:
        log.warning(f"  Judge scoring failed: {e}")
        return {
            "judge_coverage": 3.0, "judge_precision": 3.0,
            "judge_specificity": 3.0, "judge_reasoning": 3.0,
            "judge_depth": 3.0, "judge_overall": 3.0,
            "judge_explanation": "judge_error",
        }

def compute_semantic_similarity(
    embed_model, candidate_text: str, gt_text: str
) -> float:
    """Cosine similarity between candidate and GT embeddings."""
    cosine_similarity, _ = _load_sklearn()
    emb = embed_model.encode([candidate_text, gt_text])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def compute_composite_score(
    candidate: Dict,
    ground_truth: str,
    client: OpenAI,
    cfg: dict,
    embed_model=None,
) -> Dict:
    """
    Compute a composite reward score from multiple signals:
      - F1 token overlap with GT                    (weight 0.20)
      - Semantic similarity to GT                   (weight 0.15)
      - LLM-as-Judge overall                        (weight 0.30)
      - Structural quality heuristic                (weight 0.15)
      - Grounding score (RAGEN/VAGEN-inspired)      (weight 0.20)

    The grounding score (Signal 5) directly addresses the "thought-state
    mismatch" problem from RAGEN: reasoning traces that don't reference
    concrete entities from the paper are hallucinated boilerplate.
    """
    limitations = candidate["limitations"]
    thinking = candidate["thinking"]
    paper = candidate.get("paper", "")

    # Signal 1: F1 overlap
    pred_bullets = extract_bullet_points(limitations)
    gt_bullets = extract_bullet_points(ground_truth)
    f1 = compute_f1_overlap(pred_bullets, gt_bullets)

    # Signal 2: Semantic similarity
    if embed_model and ground_truth:
        sem_sim = compute_semantic_similarity(embed_model, limitations, ground_truth)
    else:
        sem_sim = 0.5  # neutral default

    # Signal 3: LLM-as-Judge
    full_candidate = f"<think>\n{thinking}\n</think>\n\n{limitations}"
    judge_scores = score_with_llm_judge(client, "", full_candidate, ground_truth, cfg)
    judge_norm = judge_scores["judge_overall"] / 5.0  # normalise to 0-1
    time.sleep(cfg["sleep_sec"])

    # Signal 4: Structural quality
    struct = structural_quality_score(thinking, limitations)

    # Signal 5: Grounding score (RAGEN §3.3 + VAGEN §4.2)
    grounding = compute_grounding_score(thinking, limitations, paper)

    # Composite (rebalanced weights to accommodate grounding)
    composite = (
        0.20 * f1
        + 0.15 * sem_sim
        + 0.30 * judge_norm
        + 0.15 * struct
        + 0.20 * grounding
    )

    candidate.update({
        "f1_score": round(f1, 4),
        "semantic_sim": round(sem_sim, 4),
        "structural_score": round(struct, 4),
        "grounding_score": round(grounding, 4),
        "composite_score": round(composite, 4),
        **judge_scores,
        "weakness_types": tag_weakness_types(limitations),
    })
    return candidate

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — REJECTION SAMPLING (per paper)
# ═══════════════════════════════════════════════════════════════════════════

def rejection_sample(candidates: List[Dict], top_k: int = 1) -> List[Dict]:
    """Keep top_k candidates per paper, ranked by composite_score."""
    ranked = sorted(candidates, key=lambda c: c.get("composite_score", 0), reverse=True)
    return ranked[:top_k]

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — DIVERSITY-AWARE GLOBAL FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def diversity_aware_filter(
    samples: List[Dict], target_n: int, n_clusters: int = 6, seed: int = 42
) -> List[Dict]:
    """
    Cluster samples by weakness type distribution and sample proportionally
    from each cluster to maintain diversity in the final dataset.
    """
    if len(samples) <= target_n:
        return samples

    # Build feature vector: binary presence of each weakness category
    all_cats = sorted(WEAKNESS_CATEGORIES.keys()) + ["other"]
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}

    features = np.zeros((len(samples), len(all_cats)))
    for i, s in enumerate(samples):
        for wt in s.get("weakness_types", ["other"]):
            if wt in cat_to_idx:
                features[i, cat_to_idx[wt]] = 1.0

    # Cluster
    _, KMeans = _load_sklearn()
    n_clusters = min(n_clusters, len(samples))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features)

    # Proportional sampling from each cluster (sorted by score within cluster)
    cluster_groups = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_groups[lab].append(samples[i])

    per_cluster = max(1, target_n / n_clusters)
    remainder = target_n - per_cluster * n_clusters

    selected = []
    for lab in sorted(cluster_groups.keys()):
        group = sorted(cluster_groups[lab], key=lambda x: x.get("composite_score", 0), reverse=True)
        take = per_cluster + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        selected.extend(group[:take])

    # If we still need more, fill from top-scoring remaining
    selected_ids = {id(s) for s in selected}
    remaining = [s for s in samples if id(s) not in selected_ids]
    remaining.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    while len(selected) < target_n and remaining:
        selected.append(remaining.pop(0))

    return selected[:target_n]

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — CONTRASTIVE PAIR GENERATION (optional, for DPO/preference)
# ═══════════════════════════════════════════════════════════════════════════

def build_contrastive_pairs(
    all_candidates: Dict[int, List[Dict]],  # row_idx -> list of scored candidates
) -> List[Dict]:
    """
    For each paper with ≥2 candidates, pair the best vs worst as
    chosen/rejected for DPO-style preference training.
    """
    pairs = []
    for row_idx, cands in all_candidates.items():
        if len(cands) < 2:
            continue
        ranked = sorted(cands, key=lambda c: c.get("composite_score", 0), reverse=True)
        best = ranked[0]
        worst = ranked[-1]
        # Only pair if there's a meaningful quality gap
        if best["composite_score"] - worst["composite_score"] < 0.05:
            continue
        pairs.append({
            "paper": best.get("paper", ""),
            "chosen": f"<think>\n{best['thinking']}\n</think>\n\n{best['limitations']}",
            "rejected": f"<think>\n{worst['thinking']}\n</think>\n\n{worst['limitations']}",
            "chosen_score": best["composite_score"],
            "rejected_score": worst["composite_score"],
            "row_idx": row_idx,
        })
    return pairs

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — CURRICULUM ORDERING
# ═══════════════════════════════════════════════════════════════════════════

def estimate_difficulty(sample: Dict) -> float:
    """
    Estimate paper difficulty for curriculum learning.
    Lower score = easier paper (explicit, surface-level weaknesses).
    Higher score = harder paper (requires deeper inference).

    Heuristic: papers where the model scored high easily are "easy";
    papers where even the best candidate scored low are "hard".
    """
    return 1.0 - sample.get("composite_score", 0.5)

def curriculum_sort(samples: List[Dict]) -> List[Dict]:
    """Sort samples easy → hard for curriculum-style SFT."""
    for s in samples:
        s["_difficulty"] = estimate_difficulty(s)
    return sorted(samples, key=lambda x: x["_difficulty"])

# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_MSG_SFT = (
    "You are an expert scientific reviewer specialised in identifying "
    "limitations in research papers. Think step by step inside <think>...</think> "
    "tags before writing your final structured limitation analysis."
)

def format_sft_record(sample: Dict) -> Dict:
    """Format a scored sample into chat-style SFT record."""
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG_SFT},
            {
                "role": "user",
                "content": (
                    "Identify all key limitations of the following paper "
                    "using 8-step reasoning.\n\n### Paper Content\n"
                    + sample["paper"]
                ),
            },
            {
                "role": "assistant",
                "content": f"<think>\n{sample['thinking']}\n</think>\n\n{sample['limitations']}",
            },
        ],
        "composite_score": sample.get("composite_score", 0),
        "f1_score": sample.get("f1_score", 0),
        "semantic_sim": sample.get("semantic_sim", 0),
        "grounding_score": sample.get("grounding_score", 0),
        "template_risk": sample.get("template_risk", 0),
        "judge_overall": sample.get("judge_overall", 0),
        "weakness_types": sample.get("weakness_types", []),
    }
    if "ground_truth" in sample:
        record["ground_truth"] = sample["ground_truth"]
    return record

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    cfg = {**DEFAULT_CFG}
    for k in cfg:
        if hasattr(args, k) and getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Validate API key
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
    # Validate key exists
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or api_key == "YOUR_NEW_OPENAI_API_KEY":
        raise ValueError("OPENAI_API_KEY environment variable not set (or still placeholder).")

    client = OpenAI(api_key=api_key)

    # Load embedding model
    embed_model = None
    if cfg["use_embeddings"]:
        log.info("Loading sentence-transformer for semantic similarity...")
        embed_model = _load_sentence_transformer()

    # Paths
    scored_path = os.path.join(cfg["output_dir"], "stage2_scored.jsonl")

    # ── Load CSV ────────────────────────────────────────────────────────
    log.info(f"Reading {cfg['num_rows']} rows from {cfg['source_csv']}...")
    df = pd.read_csv(cfg["source_csv"])
    log.info(f"Loaded {len(df)} rows.  Columns: {list(df.columns)}")

    if cfg["text_col"] not in df.columns:
        raise ValueError(f"Column '{cfg['text_col']}' not found. Available: {list(df.columns)}")
    has_gt = cfg["gt_col"] in df.columns
    if not has_gt:
        log.warning(f"Ground truth column '{cfg['gt_col']}' not found. "
                     "Scoring will rely on structural + partial signals only.")

    # ── Resume support ──────────────────────────────────────────────────
    processed_idx = set()
    all_candidates: Dict[int, List[Dict]] = defaultdict(list)  # row_idx -> candidates

    if os.path.exists(scored_path):
        with open(scored_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    ridx = rec.get("row_idx")
                    if ridx is not None:
                        all_candidates[ridx].append(rec)
                        processed_idx.add(ridx)
                except json.JSONDecodeError:
                    continue
        log.info(f"Resume: loaded scored candidates for {len(processed_idx)} papers.")

    # ── STAGE 1+2: Generate & Score ─────────────────────────────────────
    scored_f = open(scored_path, "a")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generate + Score"):
        if idx in processed_idx:
            continue

        paper = str(row.get(cfg["text_col"], "") or "").strip()
        gt = str(row.get(cfg["gt_col"], "") or "").strip() if has_gt else ""

        if len(paper) < 200:
            log.warning(f"Row {idx}: too short ({len(paper)} ch) — skipped.")
            continue

        paper_trunc = paper[: cfg["paper_max_chars"]]

        # Stage 1: Best-of-N generation
        candidates = generate_n_responses(client, paper_trunc, cfg["best_of_n"], cfg)
        if not candidates:
            log.warning(f"Row {idx}: no candidates generated — skipped.")
            continue

        # Stage 2: Score each candidate
        for cand in candidates:
            cand["paper"] = paper_trunc
            cand["ground_truth"] = gt
            cand["row_idx"] = idx
            compute_composite_score(cand, gt, client, cfg, embed_model)

            # Save incrementally
            scored_f.write(json.dumps(cand, default=str) + "\n")
            scored_f.flush()

            all_candidates[idx].append(cand)

        best = max(candidates, key=lambda c: c.get("composite_score", 0))
        log.info(
            f"  Row {idx}: {len(candidates)} candidates | "
            f"best={best['composite_score']:.3f} "
            f"(F1={best['f1_score']:.3f}, sem={best['semantic_sim']:.3f}, "
            f"grnd={best['grounding_score']:.3f}, "
            f"judge={best['judge_overall']:.1f}/5)"
        )

    scored_f.close()
    total_candidates = sum(len(v) for v in all_candidates.values())
    log.info(f"Total candidates scored: {total_candidates} across {len(all_candidates)} papers.")

    # ── Save all scored candidates (no filtering) ───────────────────────
    all_samples = []
    for row_idx, cands in all_candidates.items():
        all_samples.extend(cands)

    output_path = os.path.join(cfg["output_dir"], "sft_all_scored.jsonl")
    stats_path = os.path.join(cfg["output_dir"], "dataset_stats.json")

    sft_records = [format_sft_record(s) for s in all_samples]

    with open(output_path, "w") as f:
        for r in sft_records:
            f.write(json.dumps(r, default=str) + "\n")
    log.info(f"Saved {len(sft_records)} scored records → {output_path}")

    # ── Dataset statistics ─────────────────────────────────────────────
    all_types = []
    for s in all_samples:
        all_types.extend(s.get("weakness_types", []))
    type_dist = dict(Counter(all_types).most_common())

    scores_arr = [s.get("composite_score", 0) for s in all_samples]
    grounding_arr = [s.get("grounding_score", 0) for s in all_samples]

    def _safe(fn, arr, default=0.0):
        return round(float(fn(arr)), 4) if arr else default

    stats = {
        "total_papers": len(all_candidates),
        "total_candidates_generated": total_candidates,
        "best_of_n": cfg["best_of_n"],
        "total_saved": len(sft_records),
        "score_mean": _safe(np.mean, scores_arr),
        "score_std": _safe(np.std, scores_arr),
        "score_min": _safe(np.min, scores_arr),
        "score_max": _safe(np.max, scores_arr),
        "grounding_mean": _safe(np.mean, grounding_arr),
        "grounding_std": _safe(np.std, grounding_arr),
        "weakness_type_distribution": type_dist,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE — ALL RESPONSES SAVED WITH SCORES (NO FILTERING)")
    log.info(f"  Output:      {output_path}  ({len(sft_records)} samples)")
    log.info(f"  Stats:       {stats_path}")
    if scores_arr:
        log.info(f"  Score:       μ={stats['score_mean']:.3f}  σ={stats['score_std']:.3f}  "
                 f"[{stats['score_min']:.3f}, {stats['score_max']:.3f}]")
        log.info(f"  Grounding:   μ={stats['grounding_mean']:.3f}  σ={stats['grounding_std']:.3f}")
    else:
        log.warning("  No samples generated. Check API key and data.")
    log.info(f"  Weakness distribution: {type_dist}")
    log.info("=" * 70)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build high-quality SFT dataset for paper limitation extraction (GRPO-ready)",
    )
    p.add_argument("--source_csv", default=DEFAULT_CFG["source_csv"])
    p.add_argument("--output_dir", default=DEFAULT_CFG["output_dir"])
    p.add_argument("--text_col", default=DEFAULT_CFG["text_col"])
    p.add_argument("--gt_col", default=DEFAULT_CFG["gt_col"])
    p.add_argument("--num_rows", type=int, default=DEFAULT_CFG["num_rows"])
    p.add_argument("--best_of_n", type=int, default=DEFAULT_CFG["best_of_n"])
    p.add_argument("--gpt_model", default=DEFAULT_CFG["gpt_model"])
    p.add_argument("--judge_model", default=DEFAULT_CFG["judge_model"])
    p.add_argument("--temperature", type=float, default=DEFAULT_CFG["temperature"])
    p.add_argument("--no_embeddings", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])

    args = p.parse_args()
    if args.no_embeddings:
        args.use_embeddings = False
    else:
        args.use_embeddings = DEFAULT_CFG["use_embeddings"]

    main(args) 
    