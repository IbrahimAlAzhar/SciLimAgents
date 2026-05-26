"""
SFT Dataset Verifier & DPO Pair Generator
==========================================
Multi-gate filtering pipeline to select the best SFT samples
from the scored candidates, plus contrastive pair generation
for DPO/preference training.

Gates:
  1. Per-paper rejection sampling (top-1 per paper)
  2. Quality floor (composite, judge, grounding thresholds)
  3. Template collapse detection (generic phrase + cross-paper similarity)
  4. Ground-truth alignment via concept coverage (embedding-based)
  5. Diversity balancing (cluster on limitation embeddings, sample proportionally)
  6. Curriculum ordering (easy → hard)

DPO Pairs:
  - For each paper with ≥2 candidates, pair the best vs the candidate
    most dissimilar to the ground truth (maximally wrong but coherent).

Usage:
  python sft_filter_and_dpo.py \
    --scored_jsonl /path/to/sft_all_scored.jsonl \
    --output_dir /path/to/output \
    --use_embeddings

Requirements:
  pip install numpy scikit-learn sentence-transformers tqdm
"""

import os
import re
import json
import logging
import argparse
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = dict(
    scored_jsonl="other_experiments/sft/output/sft_all_scored.jsonl",
    output_dir="other_experiments/sft/output",
    # Gate 2 thresholds
    min_composite=0.45,
    min_judge_overall=3.0,
    min_grounding=0.3,
    # Gate 3 thresholds
    max_generic_phrases=3,          # flag if ≥ this many generic phrases
    template_sim_threshold=0.85,    # cross-paper cosine similarity threshold
    template_penalty=0.3,           # score discount for templated samples
    # Gate 4 thresholds
    gt_concept_coverage_min=0.25,   # minimum fraction of GT concepts covered
    concept_match_threshold=0.55,   # cosine sim to consider a concept "covered"
    # Gate 5 params
    diversity_clusters=8,
    target_frac=0.70,               # keep top 70% after all gates
    # Gate 6
    enable_curriculum=True,
    # DPO
    dpo_min_score_gap=0.05,         # minimum composite gap for DPO pairs
    dpo_max_gt_sim_for_rejected=0.5, # rejected must be dissimilar to GT
    # General
    val_frac=0.15,
    seed=42,
    use_embeddings=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# GENERIC TEMPLATE PHRASES (Gate 3)
# ═══════════════════════════════════════════════════════════════════════════

GENERIC_PHRASES = [
    r"lack(s|ing)?\s+(of\s+)?(comprehensive\s+)?ablation\s+stud(y|ies)",
    r"insufficient\s+statistical\s+(analysis|validation|testing)",
    r"limited\s+generali[sz]ability",
    r"reproducibility\s+concern",
    r"ethical\s+considerations?\s+(are\s+)?not\s+(adequately\s+)?discussed",
    r"lack(s|ing)?\s+(of\s+)?comprehensive\s+proof",
    r"potentially\s+outdated\s+baselines?",
    r"unclear\s+metrics?\s+affecting",
    r"need\s+for\s+more\s+precise\s+details",
    r"lack(s|ing)?\s+(of\s+)?discussion\s+on\s+dataset\s+representativeness",
    r"contribution\s+may\s+be\s+overstated",
    r"incremental\s+rather\s+than\s+groundbreaking",
    r"fairness\s+and\s+privacy",
    r"bias,?\s+fairness,?\s+(and\s+)?privacy",
    r"does\s+not\s+discuss\s+the\s+representativeness",
    r"overlooks?\s+ethical\s+considerations",
    r"may\s+not\s+be\s+statistically\s+robust",
    r"more\s+thorough(ly)?\s+detailed\s+to\s+enable\s+independent\s+verif",
    r"could\s+benefit\s+from\s+greater\s+precision",
    r"not\s+entirely\s+novel",
    r"compute\s+requirements\s+are\s+not\s+discussed\s+in\s+detail",
]

def count_generic_phrases(text: str) -> int:
    """Count how many generic template phrases appear in the text."""
    count = 0
    for pattern in GENERIC_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count

def has_paper_specific_nouns(limitations: str, paper: str) -> bool:
    """
    Check if the limitations reference specific entities from the paper
    (method names, dataset names, specific numbers, etc.).
    """
    # Extract proper nouns / acronyms from paper
    acronyms = set(re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", paper))
    # Filter out common English
    stopwords = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT",
                 "ARE", "NOT", "BUT", "HAS", "WAS", "OUR", "CAN", "ALL"}
    acronyms -= stopwords

    # Check if any appear in limitations
    hits = sum(1 for a in acronyms if a in limitations.upper())
    return hits >= 2  # at least 2 paper-specific references

# ═══════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_scored_samples(path: str) -> List[Dict]:
    """Load all scored samples from JSONL."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                samples.append(rec)
            except json.JSONDecodeError:
                continue
    log.info(f"Loaded {len(samples)} scored samples from {path}")
    return samples

def extract_text_fields(sample: Dict) -> Tuple[str, str, str, str]:
    """Extract paper, thinking, limitations, ground_truth from a sample."""
    # Handle both raw format and SFT-formatted records
    if "messages" in sample:
        # SFT format — extract from messages
        paper = ""
        response = ""
        for msg in sample["messages"]:
            if msg["role"] == "user":
                # Extract paper from user message
                m = re.search(r"### Paper Content\n(.+)", msg["content"], re.DOTALL)
                paper = m.group(1).strip() if m else msg["content"]
            elif msg["role"] == "assistant":
                response = msg["content"]

        # Parse response into thinking + limitations
        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            limitations = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        else:
            thinking = ""
            limitations = response

        gt = sample.get("ground_truth", "")
    else:
        # Raw scored format
        paper = sample.get("paper", "")
        thinking = sample.get("thinking", "")
        limitations = sample.get("limitations", "")
        gt = sample.get("ground_truth", "")

    return paper, thinking, limitations, gt

# ═══════════════════════════════════════════════════════════════════════════
# GATE 1 — Per-Paper Rejection Sampling
# ═══════════════════════════════════════════════════════════════════════════

def gate1_per_paper_topk(samples: List[Dict], top_k: int = 1) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Group by row_idx, keep top-K per paper by composite_score.
    Also returns all candidates grouped by paper (for DPO pair generation).
    """
    grouped = defaultdict(list)
    for s in samples:
        row_idx = s.get("row_idx", id(s))
        grouped[row_idx].append(s)

    selected = []
    for row_idx, cands in grouped.items():
        ranked = sorted(cands, key=lambda c: c.get("composite_score", 0), reverse=True)
        selected.extend(ranked[:top_k])

    log.info(f"Gate 1 (per-paper top-{top_k}): {len(samples)} → {len(selected)} samples "
             f"({len(grouped)} papers)")
    return selected, dict(grouped)

# ═══════════════════════════════════════════════════════════════════════════
# GATE 2 — Quality Floor
# ═══════════════════════════════════════════════════════════════════════════

def gate2_quality_floor(
    samples: List[Dict],
    min_composite: float = 0.45,
    min_judge: float = 3.0,
    min_grounding: float = 0.3,
) -> List[Dict]:
    """Remove samples below quality thresholds."""
    before = len(samples)
    filtered = []
    reasons = Counter()

    for s in samples:
        composite = s.get("composite_score", 0)
        judge = s.get("judge_overall", 0)
        grounding = s.get("grounding_score", 0)

        failed = False
        if composite < min_composite:
            reasons["low_composite"] += 1
            failed = True
        if judge < min_judge:
            reasons["low_judge"] += 1
            failed = True
        if grounding < min_grounding:
            reasons["low_grounding"] += 1
            failed = True

        if not failed:
            filtered.append(s)

    log.info(f"Gate 2 (quality floor): {before} → {len(filtered)} samples")
    if reasons:
        log.info(f"  Removal reasons: {dict(reasons)}")
    return filtered

# ═══════════════════════════════════════════════════════════════════════════
# GATE 3 — Template Collapse Detection
# ═══════════════════════════════════════════════════════════════════════════

def gate3_template_detection(
    samples: List[Dict],
    max_generic: int = 3,
    sim_threshold: float = 0.85,
    penalty: float = 0.3,
    embed_model=None,
) -> List[Dict]:
    """
    Two-pronged template detection:
      (a) Rule-based: count generic phrases in limitations
      (b) Embedding-based: flag samples too similar to other papers' traces
    """
    before = len(samples)
    n_rule_flagged = 0
    n_sim_flagged = 0

    # --- (a) Rule-based generic phrase check ---
    for s in samples:
        _, _, limitations, _ = extract_text_fields(s)
        paper, thinking, _, _ = extract_text_fields(s)

        generic_count = count_generic_phrases(limitations)
        has_specifics = has_paper_specific_nouns(limitations, paper)

        s["generic_phrase_count"] = generic_count
        s["has_paper_specifics"] = has_specifics

        # Penalise if too many generic phrases AND no paper-specific references
        if generic_count >= max_generic and not has_specifics:
            original = s.get("composite_score", 0)
            s["composite_score"] = round(original * (1 - penalty), 4)
            s["template_rule_flag"] = True
            n_rule_flagged += 1
        else:
            s["template_rule_flag"] = False

    log.info(f"  Gate 3a (rule-based): {n_rule_flagged} samples penalised "
             f"(≥{max_generic} generic phrases + no specifics)")

    # --- (b) Cross-paper embedding similarity ---
    if embed_model and len(samples) >= 5:
        from sklearn.metrics.pairwise import cosine_similarity

        traces = []
        for s in samples:
            _, thinking, limitations, _ = extract_text_fields(s)
            traces.append(thinking[:500] + " " + limitations[:500])

        log.info("  Gate 3b: computing cross-paper trace embeddings...")
        embeddings = embed_model.encode(traces, show_progress_bar=True, batch_size=64)
        sim_matrix = cosine_similarity(embeddings)

        row_indices = [s.get("row_idx", i) for i, s in enumerate(samples)]

        for i, s in enumerate(samples):
            cross_sims = []
            for j in range(len(samples)):
                if i != j and row_indices[i] != row_indices[j]:
                    cross_sims.append(sim_matrix[i][j])

            avg_cross = float(np.mean(cross_sims)) if cross_sims else 0.0
            s["cross_paper_sim"] = round(avg_cross, 4)

            if avg_cross > sim_threshold:
                original = s.get("composite_score", 0)
                s["composite_score"] = round(original * (1 - penalty), 4)
                s["template_sim_flag"] = True
                n_sim_flagged += 1
            else:
                s["template_sim_flag"] = False

        log.info(f"  Gate 3b (embedding sim): {n_sim_flagged} samples penalised "
                 f"(cross-paper sim > {sim_threshold})")

    # Re-sort by updated composite and drop bottom tail
    samples.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    log.info(f"Gate 3 (template detection): {before} samples processed, "
             f"{n_rule_flagged + n_sim_flagged} total penalised")
    return samples

# ═══════════════════════════════════════════════════════════════════════════
# GATE 4 — Ground-Truth Concept Coverage
# ═══════════════════════════════════════════════════════════════════════════

def extract_bullet_points(text: str) -> List[str]:
    """Extract individual bullet points."""
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
    if not bullets:
        bullets = [b.strip() for b in text.split("\n") if b.strip()]
    return bullets

def gate4_gt_concept_coverage(
    samples: List[Dict],
    embed_model,
    min_coverage: float = 0.25,
    match_threshold: float = 0.55,
) -> List[Dict]:
    """
    For each sample, check what fraction of ground-truth concepts
    are covered in the generated limitations (via embedding similarity).

    Samples with coverage < min_coverage get penalised; samples with
    high coverage get a bonus.
    """
    if not embed_model:
        log.warning("Gate 4 skipped (no embedding model)")
        return samples

    from sklearn.metrics.pairwise import cosine_similarity

    before = len(samples)
    n_low_coverage = 0

    for s in tqdm(samples, desc="Gate 4: concept coverage"):
        _, _, limitations, gt = extract_text_fields(s)

        if not gt or not limitations:
            s["gt_concept_coverage"] = 0.0
            continue

        gt_bullets = extract_bullet_points(gt)
        pred_bullets = extract_bullet_points(limitations)

        if not gt_bullets or not pred_bullets:
            s["gt_concept_coverage"] = 0.0
            continue

        # Embed all bullets
        gt_embs = embed_model.encode(gt_bullets, show_progress_bar=False)
        pred_embs = embed_model.encode(pred_bullets, show_progress_bar=False)

        # For each GT bullet, check if any pred bullet is close enough
        sim_matrix = cosine_similarity(gt_embs, pred_embs)
        covered = 0
        coverage_details = []
        for gi, gt_b in enumerate(gt_bullets):
            max_sim = float(np.max(sim_matrix[gi]))
            if max_sim >= match_threshold:
                covered += 1
                coverage_details.append((gt_b[:60], round(max_sim, 3)))

        coverage_frac = covered / len(gt_bullets)
        s["gt_concept_coverage"] = round(coverage_frac, 4)
        s["gt_concepts_total"] = len(gt_bullets)
        s["gt_concepts_covered"] = covered

        # Reward high coverage, penalise very low
        if coverage_frac >= 0.5:
            s["composite_score"] = round(s.get("composite_score", 0) * 1.1, 4)
        elif coverage_frac < min_coverage:
            s["composite_score"] = round(s.get("composite_score", 0) * 0.85, 4)
            n_low_coverage += 1

    log.info(f"Gate 4 (GT coverage): {n_low_coverage}/{before} samples penalised "
             f"(coverage < {min_coverage})")

    coverages = [s.get("gt_concept_coverage", 0) for s in samples]
    log.info(f"  Coverage stats: mean={np.mean(coverages):.3f}, "
             f"median={np.median(coverages):.3f}, "
             f"std={np.std(coverages):.3f}")
    return samples

# ═══════════════════════════════════════════════════════════════════════════
# GATE 5 — Diversity-Aware Selection
# ═══════════════════════════════════════════════════════════════════════════

def gate5_diversity_selection(
    samples: List[Dict],
    target_n: int,
    n_clusters: int = 8,
    embed_model=None,
    seed: int = 42,
) -> List[Dict]:
    """
    Cluster samples by limitation text embeddings (not binary weakness tags),
    then sample proportionally from each cluster, preferring higher-scoring
    samples within each cluster.
    """
    if len(samples) <= target_n:
        log.info(f"Gate 5: {len(samples)} ≤ target {target_n}, keeping all")
        return samples

    from sklearn.cluster import KMeans

    # Build embeddings of limitation text
    if embed_model:
        texts = []
        for s in samples:
            _, _, limitations, _ = extract_text_fields(s)
            texts.append(limitations[:500])

        log.info("Gate 5: computing limitation embeddings for clustering...")
        embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    else:
        # Fallback: use weakness type binary features
        CATS = [
            "missing_baseline", "limited_scope", "scalability", "statistical",
            "novelty", "reproducibility", "ablation", "generalization",
            "data_quality", "clarity", "other"
        ]
        cat_idx = {c: i for i, c in enumerate(CATS)}
        embeddings = np.zeros((len(samples), len(CATS)))
        for i, s in enumerate(samples):
            for wt in s.get("weakness_types", ["other"]):
                if wt in cat_idx:
                    embeddings[i, cat_idx[wt]] = 1.0

    n_clusters = min(n_clusters, len(samples))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    # Group by cluster
    cluster_groups = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_groups[lab].append(samples[i])

    # Sort within each cluster by composite score
    for lab in cluster_groups:
        cluster_groups[lab].sort(
            key=lambda x: x.get("composite_score", 0), reverse=True
        )

    # Proportional allocation
    per_cluster = max(1, target_n / n_clusters)
    remainder = target_n - per_cluster * n_clusters

    selected = []
    for lab in sorted(cluster_groups.keys()):
        group = cluster_groups[lab]
        take = per_cluster + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        selected.extend(group[:take])

    # Fill remaining from top-scoring unselected
    selected_ids = {id(s) for s in selected}
    remaining = [s for s in samples if id(s) not in selected_ids]
    remaining.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    while len(selected) < target_n and remaining:
        selected.append(remaining.pop(0))

    selected = selected[:target_n]

    # Log cluster distribution
    cluster_counts = Counter(labels)
    log.info(f"Gate 5 (diversity): {len(samples)} → {len(selected)} samples "
             f"across {n_clusters} clusters")
    log.info(f"  Cluster sizes: {dict(sorted(cluster_counts.items()))}")
    return selected

# ═══════════════════════════════════════════════════════════════════════════
# GATE 6 — Curriculum Ordering
# ═══════════════════════════════════════════════════════════════════════════

def gate6_curriculum_sort(samples: List[Dict]) -> List[Dict]:
    """Sort easy → hard. Difficulty ≈ 1 - composite_score."""
    for s in samples:
        s["difficulty"] = round(1.0 - s.get("composite_score", 0.5), 4)
    sorted_samples = sorted(samples, key=lambda x: x["difficulty"])
    log.info(f"Gate 6 (curriculum): sorted {len(sorted_samples)} samples easy→hard")
    return sorted_samples

# ═══════════════════════════════════════════════════════════════════════════
# DPO PAIR GENERATION (maximally dissimilar to GT for rejected)
# ═══════════════════════════════════════════════════════════════════════════

def generate_dpo_pairs(
    all_candidates: Dict[int, List[Dict]],
    min_score_gap: float = 0.05,
    max_gt_sim_rejected: float = 0.5,
    embed_model=None,
) -> List[Dict]:
    """
    For each paper with ≥2 candidates:
      - chosen  = highest composite score candidate
      - rejected = candidate most dissimilar to ground truth
                   (lowest semantic_sim OR lowest gt_concept_coverage)

    This gives DPO the strongest signal: the chosen is genuinely good,
    the rejected is plausible-looking but misaligned with ground truth.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    pairs = []
    n_skipped_gap = 0
    n_skipped_few = 0

    for row_idx, cands in tqdm(all_candidates.items(), desc="Generating DPO pairs"):
        if len(cands) < 2:
            n_skipped_few += 1
            continue

        # Sort by composite: best first
        ranked = sorted(cands, key=lambda c: c.get("composite_score", 0), reverse=True)
        best = ranked[0]

        # Find the rejected candidate: most dissimilar to GT
        _, _, best_lim, gt = extract_text_fields(best)

        if not gt:
            # No ground truth — fall back to worst composite
            worst = ranked[-1]
        else:
            # Score each non-best candidate by how dissimilar it is to GT
            scored_rejects = []
            for cand in ranked[1:]:
                _, _, cand_lim, _ = extract_text_fields(cand)

                # Use semantic_sim if available, else compute
                sem_sim = cand.get("semantic_sim", None)
                if sem_sim is None and embed_model:
                    embs = embed_model.encode([cand_lim[:500], gt[:500]])
                    sem_sim = float(cos_sim([embs[0]], [embs[1]])[0][0])
                elif sem_sim is None:
                    sem_sim = 0.5

                gt_coverage = cand.get("gt_concept_coverage", 0.5)

                # Dissimilarity = inverse of alignment signals
                dissimilarity = (1 - sem_sim) * 0.5 + (1 - gt_coverage) * 0.5
                scored_rejects.append((cand, dissimilarity, sem_sim))

            # Pick most dissimilar
            scored_rejects.sort(key=lambda x: x[1], reverse=True)
            worst, _, worst_sim = scored_rejects[0]

        # Check minimum quality gap
        gap = best.get("composite_score", 0) - worst.get("composite_score", 0)
        if gap < min_score_gap:
            n_skipped_gap += 1
            continue

        # Build the pair
        _, best_think, best_lim, _ = extract_text_fields(best)
        _, worst_think, worst_lim, _ = extract_text_fields(worst)
        paper, _, _, gt = extract_text_fields(best)

        pair = {
            "row_idx": row_idx,
            "paper": paper,
            "ground_truth": gt,
            # Chosen (high quality, aligned with GT)
            "chosen": f"<think>\n{best_think}\n</think>\n\n{best_lim}",
            "chosen_composite": best.get("composite_score", 0),
            "chosen_judge": best.get("judge_overall", 0),
            "chosen_semantic_sim": best.get("semantic_sim", 0),
            "chosen_grounding": best.get("grounding_score", 0),
            "chosen_gt_coverage": best.get("gt_concept_coverage", 0),
            # Rejected (most dissimilar to GT)
            "rejected": f"<think>\n{worst_think}\n</think>\n\n{worst_lim}",
            "rejected_composite": worst.get("composite_score", 0),
            "rejected_judge": worst.get("judge_overall", 0),
            "rejected_semantic_sim": worst.get("semantic_sim", 0),
            "rejected_grounding": worst.get("grounding_score", 0),
            "rejected_gt_coverage": worst.get("gt_concept_coverage", 0),
            # Metadata
            "score_gap": round(gap, 4),
        }
        pairs.append(pair)

    log.info(f"DPO pairs generated: {len(pairs)}")
    log.info(f"  Skipped (too few candidates): {n_skipped_few}")
    log.info(f"  Skipped (score gap < {min_score_gap}): {n_skipped_gap}")

    if pairs:
        gaps = [p["score_gap"] for p in pairs]
        log.info(f"  Score gap stats: mean={np.mean(gaps):.4f}, "
                 f"min={np.min(gaps):.4f}, max={np.max(gaps):.4f}")
    return pairs

# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING FOR EXPORT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_MSG = (
    "You are an expert scientific reviewer specialised in identifying "
    "limitations in research papers. Think step by step inside <think>...</think> "
    "tags before writing your final structured limitation analysis."
)

def format_sft_record(sample: Dict) -> Dict:
    """Format a filtered sample into chat-style SFT record."""
    paper, thinking, limitations, gt = extract_text_fields(sample)

    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    "Identify all key limitations of the following paper "
                    "using 8-step reasoning.\n\n### Paper Content\n" + paper
                ),
            },
            {
                "role": "assistant",
                "content": f"<think>\n{thinking}\n</think>\n\n{limitations}",
            },
        ],
        # Scores & metadata for inspection (not used in training)
        "composite_score": sample.get("composite_score", 0),
        "f1_score": sample.get("f1_score", 0),
        "semantic_sim": sample.get("semantic_sim", 0),
        "grounding_score": sample.get("grounding_score", 0),
        "gt_concept_coverage": sample.get("gt_concept_coverage", 0),
        "judge_overall": sample.get("judge_overall", 0),
        "template_rule_flag": sample.get("template_rule_flag", False),
        "template_sim_flag": sample.get("template_sim_flag", False),
        "generic_phrase_count": sample.get("generic_phrase_count", 0),
        "cross_paper_sim": sample.get("cross_paper_sim", 0),
        "difficulty": sample.get("difficulty", 0),
        "weakness_types": sample.get("weakness_types", []),
        "row_idx": sample.get("row_idx"),
    }
    if gt:
        record["ground_truth"] = gt
    return record

# ═══════════════════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def train_val_split(
    samples: List[Dict], val_frac: float = 0.15, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by paper (row_idx) so no paper appears in both train and val.
    """
    import random
    rng = random.Random(seed)

    # Group by row_idx
    paper_to_samples = defaultdict(list)
    for s in samples:
        paper_to_samples[s.get("row_idx", id(s))].append(s)

    paper_ids = list(paper_to_samples.keys())
    rng.shuffle(paper_ids)

    val_count = max(1, int(len(paper_ids) * val_frac))
    val_papers = set(paper_ids[:val_count])

    train, val = [], []
    for pid, samps in paper_to_samples.items():
        if pid in val_papers:
            val.extend(samps)
        else:
            train.extend(samps)

    log.info(f"Train/Val split: {len(train)} train, {len(val)} val "
             f"({len(val_papers)} val papers)")
    return train, val

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    cfg = {**DEFAULT_CFG}
    for k in cfg:
        if hasattr(args, k) and getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    np.random.seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load embedding model
    embed_model = None
    if cfg["use_embeddings"]:
        log.info("Loading sentence-transformer...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load all scored samples ────────────────────────────────────────
    all_samples = load_scored_samples(cfg["scored_jsonl"])
    if not all_samples:
        log.error("No samples loaded. Check the input path.")
        return

    # ── GATE 1: Per-paper top-1 ────────────────────────────────────────
    selected, all_grouped = gate1_per_paper_topk(all_samples, top_k=1)

    # ── GATE 2: Quality floor ──────────────────────────────────────────
    selected = gate2_quality_floor(
        selected,
        min_composite=cfg["min_composite"],
        min_judge=cfg["min_judge_overall"],
        min_grounding=cfg["min_grounding"],
    )

    # ── GATE 3: Template collapse ──────────────────────────────────────
    selected = gate3_template_detection(
        selected,
        max_generic=cfg["max_generic_phrases"],
        sim_threshold=cfg["template_sim_threshold"],
        penalty=cfg["template_penalty"],
        embed_model=embed_model,
    )

    # ── GATE 4: GT concept coverage ────────────────────────────────────
    selected = gate4_gt_concept_coverage(
        selected,
        embed_model=embed_model,
        min_coverage=cfg["gt_concept_coverage_min"],
        match_threshold=cfg["concept_match_threshold"],
    )

    # ── GATE 5: Diversity-aware selection ──────────────────────────────
    target_n = int(len(selected) * cfg["target_frac"])
    selected = gate5_diversity_selection(
        selected,
        target_n=target_n,
        n_clusters=cfg["diversity_clusters"],
        embed_model=embed_model,
        seed=cfg["seed"],
    )

    # ── GATE 6: Curriculum ordering ────────────────────────────────────
    if cfg["enable_curriculum"]:
        selected = gate6_curriculum_sort(selected)

    # ── Train / Val split ──────────────────────────────────────────────
    train_samples, val_samples = train_val_split(
        selected, val_frac=cfg["val_frac"], seed=cfg["seed"]
    )

    # ── Format and save SFT records ────────────────────────────────────
    train_records = [format_sft_record(s) for s in train_samples]
    val_records = [format_sft_record(s) for s in val_samples]

    train_path = os.path.join(cfg["output_dir"], "sft_filtered_train.jsonl")
    val_path = os.path.join(cfg["output_dir"], "sft_filtered_val.jsonl")

    for path, records in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        log.info(f"Saved {len(records)} records → {path}")

    # ── DPO Pair Generation ────────────────────────────────────────────
    # First, run Gate 4 on ALL candidates (not just top-1) for coverage scores
    log.info("Computing GT concept coverage for all candidates (for DPO)...")
    all_flat = []
    for cands in all_grouped.values():
        all_flat.extend(cands)

    if embed_model:
        all_flat = gate4_gt_concept_coverage(
            all_flat,
            embed_model=embed_model,
            min_coverage=0.0,  # don't filter, just annotate
            match_threshold=cfg["concept_match_threshold"],
        )

    # Re-group after annotation
    all_grouped_annotated = defaultdict(list)
    for s in all_flat:
        all_grouped_annotated[s.get("row_idx", id(s))].append(s)

    dpo_pairs = generate_dpo_pairs(
        all_grouped_annotated,
        min_score_gap=cfg["dpo_min_score_gap"],
        max_gt_sim_rejected=cfg["dpo_max_gt_sim_for_rejected"],
        embed_model=embed_model,
    )

    dpo_path = os.path.join(cfg["output_dir"], "contrastive_pairs.jsonl")
    with open(dpo_path, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, default=str) + "\n")
    log.info(f"Saved {len(dpo_pairs)} DPO pairs → {dpo_path}")

    # ── Summary Statistics ─────────────────────────────────────────────
    log.info("=" * 70)
    log.info("FILTERING PIPELINE COMPLETE")
    log.info(f"  Input:            {len(all_samples)} scored candidates")
    log.info(f"  After all gates:  {len(selected)} samples")
    log.info(f"  Train:            {len(train_records)} samples → {train_path}")
    log.info(f"  Val:              {len(val_records)} samples → {val_path}")
    log.info(f"  DPO pairs:        {len(dpo_pairs)} pairs → {dpo_path}")

    if selected:
        scores = [s.get("composite_score", 0) for s in selected]
        ground = [s.get("grounding_score", 0) for s in selected]
        coverage = [s.get("gt_concept_coverage", 0) for s in selected]
        generic = [s.get("generic_phrase_count", 0) for s in selected]

        log.info(f"  Composite:  μ={np.mean(scores):.4f} σ={np.std(scores):.4f} "
                 f"[{np.min(scores):.4f}, {np.max(scores):.4f}]")
        log.info(f"  Grounding:  μ={np.mean(ground):.4f} σ={np.std(ground):.4f}")
        log.info(f"  GT coverage: μ={np.mean(coverage):.4f} σ={np.std(coverage):.4f}")
        log.info(f"  Generic phrases: μ={np.mean(generic):.1f} max={np.max(generic)}")

        # Weakness type distribution in final set
        all_types = []
        for s in selected:
            all_types.extend(s.get("weakness_types", []))
        log.info(f"  Weakness dist: {dict(Counter(all_types).most_common())}")

    # Save stats
    stats = {
        "input_total": len(all_samples),
        "after_gate1": len(all_grouped),
        "after_all_gates": len(selected),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "dpo_pairs": len(dpo_pairs),
        "config": {k: v for k, v in cfg.items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))},
    }
    if selected:
        stats["final_score_mean"] = round(float(np.mean(scores)), 4)
        stats["final_score_std"] = round(float(np.std(scores)), 4)
        stats["final_grounding_mean"] = round(float(np.mean(ground)), 4)
        stats["final_gt_coverage_mean"] = round(float(np.mean(coverage)), 4)

    stats_path = os.path.join(cfg["output_dir"], "filtering_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Stats saved → {stats_path}")
    log.info("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Filter SFT dataset and generate DPO pairs for limitation extraction",
    )
    p.add_argument("--scored_jsonl", default=DEFAULT_CFG["scored_jsonl"],
                   help="Path to sft_all_scored.jsonl")
    p.add_argument("--output_dir", default=DEFAULT_CFG["output_dir"])
    p.add_argument("--min_composite", type=float, default=DEFAULT_CFG["min_composite"])
    p.add_argument("--min_judge_overall", type=float, default=DEFAULT_CFG["min_judge_overall"])
    p.add_argument("--min_grounding", type=float, default=DEFAULT_CFG["min_grounding"])
    p.add_argument("--max_generic_phrases", type=int, default=DEFAULT_CFG["max_generic_phrases"])
    p.add_argument("--template_sim_threshold", type=float, default=DEFAULT_CFG["template_sim_threshold"])
    p.add_argument("--gt_concept_coverage_min", type=float, default=DEFAULT_CFG["gt_concept_coverage_min"])
    p.add_argument("--target_frac", type=float, default=DEFAULT_CFG["target_frac"])
    p.add_argument("--diversity_clusters", type=int, default=DEFAULT_CFG["diversity_clusters"])
    p.add_argument("--dpo_min_score_gap", type=float, default=DEFAULT_CFG["dpo_min_score_gap"])
    p.add_argument("--val_frac", type=float, default=DEFAULT_CFG["val_frac"])
    p.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--no_embeddings", action="store_true")
    p.add_argument("--no_curriculum", action="store_true")

    args = p.parse_args()
    if args.no_embeddings:
        args.use_embeddings = False
    else:
        args.use_embeddings = True
    args.enable_curriculum = not args.no_curriculum

    main(args) 
