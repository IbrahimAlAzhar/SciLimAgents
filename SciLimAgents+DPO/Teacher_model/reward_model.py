"""
reward_model.py — Hybrid reward model for multi-agent limitation extraction.

SCORING PHILOSOPHY
------------------
Limitation extraction is OPEN-ENDED SET GENERATION. The correct output is
an unordered set of bullet items. This creates three reward-hacking risks:

  1. VERBOSITY GAMING — generate many vague items to inflate recall.
  2. PARROT / BOILERPLATE — produce generic criticisms that apply to any paper.
  3. REDUNDANCY STUFFING — rephrase the same point N times to pad counts.

The reward model defends against all three via a three-layer scoring stack:

  LAYER 1: LLM-as-Judge (gpt-4o-mini)
    - Bipartite semantic matching: pred ↔ GT  (set-level F-beta)
    - Per-item quality axes: grounded, specific, valid, severity
    - Organization & deduplication for master output

  LAYER 2: NLP Metrics (no LLM needed — cross-validates the judge)
    - ROUGE-L recall/precision vs GT (catches judge hallucinating matches)
    - TF-IDF cosine self-similarity matrix (catches redundancy stuffing)
    - Embedding-free lexical overlap for match verification

  LAYER 3: Rule-Based Guards (deterministic — unhackable)
    - Length penalty: too short (<3 items) or too long (>25 items)
    - Redundancy penalty: pairwise Jaccard within predicted set
    - Boilerplate detector: flags items lacking paper-specific tokens
    - Coverage check: fraction of GT categories addressed

Anti-Reward-Hacking Design:
  - Precision is penalized for ungrounded items (even if they match GT keywords)
  - Novelty bonus requires BOTH validity AND grounding (not just one)
  - Self-similarity penalty directly opposes redundancy stuffing
  - Length penalty is symmetric: too few items is penalized as much as too many
  - NLP metrics cross-validate LLM judge to catch judge-policy collusion

USAGE
-----
  export OPENAI_API_KEY=sk-...
  python reward_model.py \
      --rollout_json path/to/rollout_data_full.json
"""

from __future__ import annotations

import os
import re
import json
import math
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional NLP dependencies — graceful fallback if missing
# ---------------------------------------------------------------------------
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge_score not installed. pip install rouge-score. "
          "ROUGE metrics will be skipped.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed. TF-IDF self-similarity will "
          "use Jaccard fallback.")

# =============================================================================
# CONFIG
# =============================================================================

JUDGE_MODEL       = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_TEMPERATURE = 0.0
CACHE_DIR         = Path(os.environ.get("REWARD_CACHE_DIR", "reward_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAPER_TRUNC_CHARS = 16000     # for grounding/quality prompts
DOC_TRUNC_CHARS   = 6000      # for organization/dedup prompts

# --- Hardcoded output directories ---
SCORE_OUT_DIR   = Path("other_experiments/dpo/output_gpt_score")
SFT_DPO_OUT_DIR = Path("other_experiments/dpo/sft_and_dpo_pairs")

# --- DPO pairing ---
DPO_MIN_SCORE_GAP           = 0.05   # lowered slightly — we filter quality later
DPO_REQUIRE_IDENTICAL_INPUT = True   # strict for correctness

# --- Novelty bonus ---
NOVELTY_WEIGHT = 0.5   # partial credit for valid+grounded items absent from GT

# --- F-beta ---
FBETA = 1.5   # recall-weighted: missing a real limitation > adding a borderline one

# --- Anti-reward-hacking ---
IDEAL_ITEM_COUNT_MIN  = 3
IDEAL_ITEM_COUNT_MAX  = 20
LENGTH_PENALTY_WEIGHT = 0.10   # how much length penalty affects composite
REDUNDANCY_WEIGHT     = 0.10   # how much internal redundancy penalty matters
ROUGE_WEIGHT          = 0.10   # weight of ROUGE cross-validation signal

# --- Composite weights per agent role ---
WEIGHTS = {
    "worker": dict(
        f1=0.40,
        grounding=0.15,
        specificity=0.10,
        severity=0.05,        # NEW: severity of identified limitations
        rouge=ROUGE_WEIGHT,
        length_pen=LENGTH_PENALTY_WEIGHT,
        redundancy_pen=REDUNDANCY_WEIGHT,
    ),
    "leader_feedback": dict(
        delta=0.40,            # causal improvement
        feedback_quality=0.35, # NEW: direct quality of the feedback
        room_adjusted=0.25,    # NEW: delta normalized by room-for-improvement
    ),
    "leader_handoff": dict(
        coverage=0.55,
        structure=0.45,
    ),
    "master": dict(
        f1=0.40,
        grounding=0.10,
        specificity=0.05,
        severity=0.05,
        dedup=0.10,
        organization=0.10,
        rouge=ROUGE_WEIGHT,
        length_pen=LENGTH_PENALTY_WEIGHT,
        redundancy_pen=REDUNDANCY_WEIGHT,
    ),
}

# --- Specialty map (6 workers — matches rollout_gpt.py) ---
SPECIALTY_MAP = {
    "Novelty_Significance_Agent":                     "novelty_significance",
    "Theoretical_Methodological_Agent":               "theoretical_methodological",
    "Experimental_Evaluation_Agent":                  "experimental_evaluation",
    "Generalization_Robustness_Efficiency_Agent":     "generalization_robustness_efficiency",
    "Clarity_Interpretability_Reproducibility_Agent": "clarity_interpretability_reproducibility",
    "Data_Ethics_Agent":                              "data_ethics",
}

# Ground-truth categories (must match SPECIALTY_MAP values + other)
GT_CATEGORIES = [
    "novelty_significance",
    "theoretical_methodological",
    "experimental_evaluation",
    "generalization_robustness_efficiency",
    "clarity_interpretability_reproducibility",
    "data_ethics",
    "other",
]

# =============================================================================
# OPENAI CLIENT
# =============================================================================

_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=key)
    return _client

# =============================================================================
# CACHED JUDGE CALL
# =============================================================================

def _cache_key(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]

def _cached_json_call(tag: str, prompt: str, schema_hint: str,
                      max_retries: int = 2) -> dict:
    """Single OpenAI call with JSON-mode output, cached on disk."""
    key = _cache_key(tag, JUDGE_MODEL, prompt, schema_hint)
    path = CACHE_DIR / f"{tag[:40]}_{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            path.unlink(missing_ok=True)

    for attempt in range(max_retries + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": ("You are a careful, strict scientific reviewer. "
                                 "Output valid JSON only, no prose, no code fences.")},
                    {"role": "user", "content": prompt + "\n\n" + schema_hint},
                ],
                timeout=120,
            )
            out = json.loads(resp.choices[0].message.content)
            path.write_text(json.dumps(out, indent=2))
            return out
        except Exception as e:
            if attempt == max_retries:
                print(f"  [JUDGE ERROR] {tag}: {e}")
                return {}
    return {}

# =============================================================================
# BULLET / GT PARSING
# =============================================================================

_BULLET_RE = re.compile(
    r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]\s|\Z)",
    re.DOTALL | re.MULTILINE,
)

def split_bullets(text: str) -> list[str]:
    """Split a bullet list into individual limitation strings."""
    if not text:
        return []
    matches = _BULLET_RE.findall(text)
    items = [re.sub(r"\s+", " ", m).strip() for m in matches]
    if not items:
        # Fallback: split by double newline or numbered items
        items = re.split(r"\n\s*\d+\.\s+|\n\n+", text)
        items = [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]
    # Drop trivially-short items (stray headers, empty bullets)
    items = [i for i in items if len(i) > 20]
    return items

def split_gt(ground_truth: str) -> list[str]:
    """Ground truth is one limitation per line."""
    if not ground_truth:
        return []
    return [l.strip() for l in ground_truth.splitlines()
            if l.strip() and len(l.strip()) > 10]

# =============================================================================
# LAYER 3: RULE-BASED GUARDS (deterministic, unhackable)
# =============================================================================

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def compute_length_penalty(n_items: int) -> float:
    """
    Symmetric penalty for item counts outside the ideal range.
    Returns 0.0 (no penalty) to 1.0 (maximum penalty).
    """
    if IDEAL_ITEM_COUNT_MIN <= n_items <= IDEAL_ITEM_COUNT_MAX:
        return 0.0
    if n_items < IDEAL_ITEM_COUNT_MIN:
        # Too few: linear penalty
        return min(1.0, (IDEAL_ITEM_COUNT_MIN - n_items) / IDEAL_ITEM_COUNT_MIN)
    # Too many: logarithmic penalty (harsh on extreme counts)
    excess = n_items - IDEAL_ITEM_COUNT_MAX
    return min(1.0, math.log1p(excess) / math.log1p(20))

def compute_redundancy_penalty(items: list[str]) -> float:
    """
    Detect internal redundancy via pairwise similarity.
    Returns 0.0 (no redundancy) to 1.0 (heavy redundancy).

    Uses TF-IDF cosine if sklearn available, else Jaccard.
    """
    if len(items) < 2:
        return 0.0

    if HAS_SKLEARN and len(items) >= 3:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf = vec.fit_transform(items)
            sim_matrix = sk_cosine(tfidf)
            # Zero out diagonal
            for i in range(len(items)):
                sim_matrix[i, i] = 0.0
            # Average of top-k most similar pairs
            max_sims = sim_matrix.max(axis=1)
            high_sim_count = sum(1 for s in max_sims if s > 0.6)
            return min(1.0, high_sim_count / len(items))
        except Exception:
            pass

    # Fallback: pairwise Jaccard
    high_sim = 0
    total_pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _jaccard(items[i], items[j]) > 0.5:
                high_sim += 1
            total_pairs += 1
    return min(1.0, high_sim / max(total_pairs, 1) * 3)  # scale up

def compute_boilerplate_ratio(items: list[str], paper_text: str) -> float:
    """
    Fraction of items that are boilerplate (not paper-specific).
    Heuristic: an item is boilerplate if <10% of its non-stopword tokens
    appear in the paper text.
    Returns 0.0 (all specific) to 1.0 (all boilerplate).
    """
    if not items or not paper_text:
        return 0.5  # uncertain

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "no", "so", "than", "too",
        "very", "just", "about", "up", "down", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "only", "own", "same",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "them", "their", "we", "our", "you", "your", "which", "what", "who",
        "whom", "how", "when", "where", "why", "if", "because", "while",
        "paper", "study", "work", "research", "method", "approach", "model",
        "results", "data", "analysis", "experiment", "also", "however",
        "limitation", "lack", "missing", "insufficient", "limited", "without",
    }

    paper_lower = paper_text.lower()
    boilerplate_count = 0

    for item in items:
        words = [w for w in re.findall(r"[a-z]+", item.lower())
                 if w not in STOPWORDS and len(w) > 3]
        if not words:
            boilerplate_count += 1
            continue
        in_paper = sum(1 for w in words if w in paper_lower)
        if in_paper / len(words) < 0.15:
            boilerplate_count += 1

    return boilerplate_count / len(items)

# =============================================================================
# LAYER 2: NLP METRICS
# =============================================================================

def compute_rouge_scores(pred_text: str, gt_text: str) -> dict:
    """
    ROUGE-L between predicted limitations (joined) and GT (joined).
    Returns dict with rouge_l_precision, rouge_l_recall, rouge_l_f1.
    """
    if not HAS_ROUGE or not pred_text.strip() or not gt_text.strip():
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0,
                "rouge_l_f1": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gt_text, pred_text)
    return {
        "rouge_l_precision": scores["rougeL"].precision,
        "rouge_l_recall":    scores["rougeL"].recall,
        "rouge_l_f1":        scores["rougeL"].fmeasure,
    }

def compute_lexical_match_verification(
    pred: list[str],
    gt: list[str],
    judge_matches: list[dict]
) -> float:
    """
    Cross-validate LLM judge matches with lexical overlap.
    If the judge says P_i matches G_j but they share almost no words,
    flag it as suspicious.

    Returns: fraction of judge matches that pass lexical verification.
    """
    if not judge_matches:
        return 1.0  # no matches to verify

    verified = 0
    for m in judge_matches:
        pi, gi = m.get("pred", -1), m.get("gt", -1)
        if 0 <= pi < len(pred) and 0 <= gi < len(gt):
            jacc = _jaccard(pred[pi], gt[gi])
            # Threshold: semantic matches can have low lexical overlap,
            # but if Jaccard is > 0.05, there's at least SOME word overlap.
            # Very low threshold because the same flaw CAN be phrased
            # completely differently.
            if jacc > 0.03:
                verified += 1
            else:
                # Even zero lexical overlap can be valid for semantic matches,
                # so we give partial credit
                verified += 0.5
        else:
            verified += 0.5  # index out of range — give benefit of doubt

    return verified / len(judge_matches)

# =============================================================================
# LAYER 1: LLM-AS-JUDGE
# =============================================================================

# ---- GT Categorization ----

_GT_CAT_PROMPT = """Assign each limitation below to EXACTLY ONE specialty.

Specialties:
- novelty_significance
- theoretical_methodological
- experimental_evaluation
- generalization_robustness_efficiency
- clarity_interpretability_reproducibility
- data_ethics
- other

LIMITATIONS (one per line, numbered):
{numbered}
"""
_GT_CAT_SCHEMA = (
    'Return JSON: {{"assignments": [{{"index": <1-based int>, '
    '"specialty": "<one of the labels>"}}]}}'
)

def categorize_gt(paper_idx: int, gt_items: list[str]) -> dict[int, str]:
    if not gt_items:
        return {}
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(gt_items))
    out = _cached_json_call(
        f"gt_cat_p{paper_idx}",
        _GT_CAT_PROMPT.format(numbered=numbered),
        _GT_CAT_SCHEMA,
    )
    mapping: dict[int, str] = {}
    for a in out.get("assignments", []):
        idx = a.get("index")
        if isinstance(idx, int) and 1 <= idx <= len(gt_items):
            spec = str(a.get("specialty", "other"))
            if spec in GT_CATEGORIES:
                mapping[idx - 1] = spec
            else:
                mapping[idx - 1] = "other"
    return mapping

def gt_for_specialty(gt_items: list[str],
                     categories: dict[int, str],
                     specialty: str) -> list[str]:
    subset = [gt_items[i] for i in range(len(gt_items))
              if categories.get(i) == specialty]
    # Fallback: if zero GT items in this specialty, use full GT
    # (avoids trivially perfect precision with zero recall)
    return subset if subset else gt_items

# ---- Bipartite Matching ----

_MATCH_PROMPT = """Match predicted limitations to ground-truth limitations.

Two limitations match ONLY if they describe the SAME underlying flaw, even if
phrased differently. Being in the same broad category is NOT enough.
A match means: if a reviewer wrote one, they would NOT also write the other
because they'd consider it the same point.

GROUND TRUTH:
{gt_block}

PREDICTED:
{pred_block}

For each PREDICTED item, identify its best GT match (if any exists).
Be STRICT: marginal or partial overlaps are NOT matches.
"""
_MATCH_SCHEMA = """Return JSON:
{{
  "matches": [{{"pred": <1-based int>, "gt": <1-based int>, "confidence": <float 0..1>}}],
  "unmatched_pred": [<1-based int>],
  "unmatched_gt": [<1-based int>]
}}"""

def bipartite_match(pred: list[str], gt: list[str], tag: str) -> dict:
    if not pred or not gt:
        return {"matches": [],
                "unmatched_pred": list(range(len(pred))),
                "unmatched_gt":   list(range(len(gt)))}

    gt_block   = "\n".join(f"G{i+1}. {s}" for i, s in enumerate(gt))
    pred_block = "\n".join(f"P{i+1}. {s}" for i, s in enumerate(pred))
    out = _cached_json_call(
        tag,
        _MATCH_PROMPT.format(gt_block=gt_block, pred_block=pred_block),
        _MATCH_SCHEMA,
    )

    # Normalize to 0-based; enforce confidence threshold; 1-to-1 greedy.
    raw = []
    for m in out.get("matches", []):
        p, g = m.get("pred"), m.get("gt")
        c = float(m.get("confidence", 0.5))
        if (isinstance(p, int) and isinstance(g, int)
                and 1 <= p <= len(pred) and 1 <= g <= len(gt)
                and c >= 0.55):  # slightly above 0.5 to reduce spurious matches
            raw.append({"pred": p - 1, "gt": g - 1, "confidence": c})

    raw.sort(key=lambda x: -x["confidence"])
    taken_p, taken_g, matches = set(), set(), []
    for m in raw:
        if m["pred"] in taken_p or m["gt"] in taken_g:
            continue
        matches.append(m)
        taken_p.add(m["pred"])
        taken_g.add(m["gt"])

    unmatched_pred = [i for i in range(len(pred)) if i not in taken_p]
    unmatched_gt   = [i for i in range(len(gt))   if i not in taken_g]
    return {"matches": matches,
            "unmatched_pred": unmatched_pred,
            "unmatched_gt":   unmatched_gt}

# ---- Quality + Severity Judge ----

_QUALITY_PROMPT = """Judge each predicted limitation on FOUR axes against the paper.

PAPER CONTENT (truncated):
{paper}

LIMITATIONS:
{items}

For each item rate:
- grounded: 1 if the paper provides direct evidence for this limitation, else 0.
  A limitation about "missing ablation" is grounded only if you verified no
  ablation exists. Generic claims without checking are NOT grounded.
- specific: 1 if it references concrete paper elements (method names, dataset
  names, section numbers, specific numbers, baselines, equations); 0 if it
  could apply verbatim to any ML paper.
- valid: 1 if it identifies a real, substantive flaw; 0 if trivial,
  tautological, factually wrong, or already addressed in the paper.
- severity: 0 (minor/nitpick), 1 (moderate — affects interpretation),
  2 (major — undermines a core claim or finding).
"""
_QUALITY_SCHEMA = (
    'Return JSON: {{"judgments": [{{"index": <1-based int>, '
    '"grounded": 0|1, "specific": 0|1, "valid": 0|1, '
    '"severity": 0|1|2}}]}}'
)

def judge_quality(items: list[str], paper: str, tag: str) -> list[dict]:
    if not items:
        return []
    paper_trim = paper[:PAPER_TRUNC_CHARS]
    numbered = "\n".join(f"L{i+1}. {s}" for i, s in enumerate(items))
    out = _cached_json_call(
        tag,
        _QUALITY_PROMPT.format(paper=paper_trim, items=numbered),
        _QUALITY_SCHEMA,
    )
    by_idx = {j["index"]: j for j in out.get("judgments", [])
              if isinstance(j.get("index"), int)}
    result = []
    for i in range(len(items)):
        j = by_idx.get(i + 1, {})
        result.append({
            "grounded": int(bool(j.get("grounded", 0))),
            "specific": int(bool(j.get("specific", 0))),
            "valid":    int(bool(j.get("valid", 0))),
            "severity": min(2, max(0, int(j.get("severity", 0)))),
        })
    return result

# ---- Leader Feedback Quality Judge ----

_FEEDBACK_QUALITY_PROMPT = """Rate the quality of this feedback given to a specialist worker.

The worker's task was to identify limitations of a scientific paper in the
area of: {specialty}.

WORKER'S INITIAL OUTPUT:
{worker_output}

LEADER'S FEEDBACK:
{feedback}

Rate on three axes (each 0.0 to 1.0):
- actionable: Does the feedback give CONCRETE, specific suggestions the worker
  can act on? (vs vague "be more specific" without saying how)
- accurate: Are the feedback points factually correct about what the worker
  missed or got wrong?
- targeted: Does the feedback address the worker's ACTUAL weaknesses rather
  than generic boilerplate advice?
"""
_FEEDBACK_SCHEMA = (
    'Return JSON: {{"actionable": <float 0..1>, '
    '"accurate": <float 0..1>, "targeted": <float 0..1>}}'
)

def judge_feedback_quality(worker_output: str, feedback: str,
                           specialty: str, tag: str) -> dict:
    out = _cached_json_call(
        tag,
        _FEEDBACK_QUALITY_PROMPT.format(
            specialty=specialty,
            worker_output=worker_output[:DOC_TRUNC_CHARS],
            feedback=feedback[:DOC_TRUNC_CHARS],
        ),
        _FEEDBACK_SCHEMA,
    )
    return {
        "actionable": max(0.0, min(1.0, float(out.get("actionable", 0.5)))),
        "accurate":   max(0.0, min(1.0, float(out.get("accurate", 0.5)))),
        "targeted":   max(0.0, min(1.0, float(out.get("targeted", 0.5)))),
    }

# ---- Organization Judge (master output) ----

_ORG_PROMPT = """Rate the consolidated limitations document.

DOCUMENT:
{doc}

- organization: 0..1 — are limitations grouped by clear, meaningful categories?
- dedup: 0..1 — is each bullet distinct? Penalize near-duplicates or items
  that could be merged without losing information.
- coherence: 0..1 — does each bullet read clearly and stand on its own?
"""
_ORG_SCHEMA = (
    'Return JSON: {{"organization": <float 0..1>, '
    '"dedup": <float 0..1>, "coherence": <float 0..1>}}'
)

# =============================================================================
# CORE SCORING FUNCTIONS
# =============================================================================

def f_beta(precision: float, recall: float, beta: float = FBETA) -> float:
    if precision + recall < 1e-9:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-9)

def score_item_list(pred_text: str,
                    gt_items: list[str],
                    paper: str,
                    tag: str) -> dict:
    """
    Score a bullet list against a GT set.
    Used for worker outputs and master synthesis.
    Returns dict with all sub-scores and composite.
    """
    pred = split_bullets(pred_text)
    n_pred = len(pred)
    n_gt   = len(gt_items)

    if not pred:
        return dict(
            f1=0.0, grounding=0.0, specificity=0.0, severity_avg=0.0,
            precision=0.0, recall=0.0,
            n_pred=0, n_gt=n_gt, n_matched=0, n_novel_valid=0,
            rouge_l_f1=0.0, length_penalty=1.0, redundancy_penalty=0.0,
            boilerplate_ratio=1.0, match_verification=1.0,
        )

    # ---- LLM Judge: matching + quality ----
    match_result = (bipartite_match(pred, gt_items, f"{tag}__match")
                    if gt_items else
                    {"matches": [], "unmatched_pred": list(range(n_pred)),
                     "unmatched_gt": []})
    qual = judge_quality(pred, paper, f"{tag}__qual")

    n_matched = len(match_result["matches"])

    # Novelty bonus: valid + grounded unmatched predictions
    n_novel_valid = sum(
        1 for i in match_result["unmatched_pred"]
        if i < len(qual) and qual[i]["valid"] and qual[i]["grounded"]
    )

    # Precision/Recall with novelty bonus
    precision = (n_matched + NOVELTY_WEIGHT * n_novel_valid) / max(n_pred, 1)
    recall    = (n_matched / max(n_gt, 1)) if n_gt else 0.0
    f1        = f_beta(precision, recall, FBETA) if n_gt else precision

    # Quality averages
    grounding   = sum(q["grounded"] for q in qual) / max(n_pred, 1)
    specificity = sum(q["specific"] for q in qual) / max(n_pred, 1)
    severity_avg = sum(q["severity"] for q in qual) / (2.0 * max(n_pred, 1))
    # severity normalized to [0,1]: max severity=2, so divide by 2*n_pred

    # ---- NLP Metrics ----
    gt_joined   = " ".join(gt_items) if gt_items else ""
    pred_joined = " ".join(pred)
    rouge_scores = compute_rouge_scores(pred_joined, gt_joined)
    rouge_f1     = rouge_scores["rouge_l_f1"]

    # Match verification
    match_verif = compute_lexical_match_verification(
        pred, gt_items, match_result["matches"]
    )

    # ---- Rule-based guards ----
    length_pen     = compute_length_penalty(n_pred)
    redundancy_pen = compute_redundancy_penalty(pred)
    boilerplate    = compute_boilerplate_ratio(pred, paper)

    # Adjust grounding: if boilerplate ratio is high, reduce effective grounding
    effective_grounding = grounding * (1.0 - 0.3 * boilerplate)

    # Adjust F1: if match verification is low, discount matches
    effective_f1 = f1 * (0.5 + 0.5 * match_verif)

    return dict(
        f1=effective_f1,
        grounding=effective_grounding,
        specificity=specificity,
        severity_avg=severity_avg,
        precision=precision,
        recall=recall,
        n_pred=n_pred,
        n_gt=n_gt,
        n_matched=n_matched,
        n_novel_valid=n_novel_valid,
        rouge_l_f1=rouge_f1,
        length_penalty=length_pen,
        redundancy_penalty=redundancy_pen,
        boilerplate_ratio=boilerplate,
        match_verification=match_verif,
        raw_f1=f1,
        raw_grounding=grounding,
    )

def composite_worker(sc: dict) -> float:
    """Compute composite score for a worker sample."""
    w = WEIGHTS["worker"]
    raw = (
        w["f1"]            * sc.get("f1", 0.0)
        + w["grounding"]   * sc.get("grounding", 0.0)
        + w["specificity"] * sc.get("specificity", 0.0)
        + w["severity"]    * sc.get("severity_avg", 0.0)
        + w["rouge"]       * sc.get("rouge_l_f1", 0.0)
        - w["length_pen"]  * sc.get("length_penalty", 0.0)
        - w["redundancy_pen"] * sc.get("redundancy_penalty", 0.0)
    )
    return max(0.0, min(1.0, raw))

def score_leader_feedback(
    initial_sc: dict,
    revised_sc: dict,
    worker_initial_output: str,
    feedback_text: str,
    specialty: str,
    tag: str,
) -> dict:
    """
    Leader feedback reward with three components:
    1. Causal delta (did feedback help?)
    2. Direct feedback quality (is the feedback itself good?)
    3. Room-adjusted delta (normalized by how much room there was to improve)
    """
    c_initial = composite_worker(initial_sc)
    c_revised = composite_worker(revised_sc)
    delta = c_revised - c_initial

    # Room-adjusted: delta / (1 - initial), so good feedback on already-good
    # output is still credited
    room = max(1.0 - c_initial, 0.05)  # floor to avoid div-by-zero
    room_adjusted_delta = max(0.0, min(1.0, delta / room))

    # Squash raw delta to [0, 1]
    delta_score = max(0.0, min(1.0, 0.5 + 2.0 * delta))

    # Direct feedback quality from LLM judge
    fq = judge_feedback_quality(
        worker_initial_output, feedback_text, specialty, tag
    )
    feedback_quality = (fq["actionable"] + fq["accurate"] + fq["targeted"]) / 3.0

    # Composite
    w = WEIGHTS["leader_feedback"]
    composite = (
        w["delta"]         * delta_score
        + w["feedback_quality"] * feedback_quality
        + w["room_adjusted"]    * room_adjusted_delta
    )
    composite = max(0.0, min(1.0, composite))

    return dict(
        delta=delta,
        delta_score=delta_score,
        room_adjusted_delta=room_adjusted_delta,
        feedback_quality=feedback_quality,
        feedback_detail=fq,
        composite=composite,
        initial_composite=c_initial,
        revised_composite=c_revised,
    )

def score_leader_handoff(handoff_text: str,
                         worker_outputs: dict[str, str]) -> dict:
    """Score leader handoff to master."""
    # Coverage: fraction of workers mentioned
    present = sum(1 for name in worker_outputs if name in handoff_text)
    coverage = present / max(len(worker_outputs), 1)

    # Structure checks
    has_master = "master" in handoff_text.lower()
    n_bullets  = len(_BULLET_RE.findall(handoff_text))
    has_sections = sum(1 for name in worker_outputs
                       if name in handoff_text and "###" in handoff_text)

    structure = 0.0
    if has_master:
        structure += 0.3
    if n_bullets >= 6:
        structure += 0.3
    elif n_bullets >= 3:
        structure += 0.15
    if has_sections >= len(worker_outputs) * 0.5:
        structure += 0.4
    elif has_sections > 0:
        structure += 0.2
    structure = min(1.0, structure)

    w = WEIGHTS["leader_handoff"]
    composite = w["coverage"] * coverage + w["structure"] * structure
    return dict(coverage=coverage, structure=structure, composite=composite)

def score_master(master_text: str,
                 gt_items: list[str],
                 paper: str,
                 tag: str) -> dict:
    """Score master synthesis output."""
    item_sc = score_item_list(master_text, gt_items, paper, tag)

    # Organization & dedup from LLM judge
    org_out = _cached_json_call(
        f"{tag}__org",
        _ORG_PROMPT.format(doc=master_text[:DOC_TRUNC_CHARS]),
        _ORG_SCHEMA,
    )
    organization = max(0.0, min(1.0, float(org_out.get("organization", 0.5))))
    dedup        = max(0.0, min(1.0, float(org_out.get("dedup", 0.5))))
    coherence    = max(0.0, min(1.0, float(org_out.get("coherence", 0.5))))

    w = WEIGHTS["master"]
    composite = (
        w["f1"]           * item_sc["f1"]
        + w["grounding"]  * item_sc["grounding"]
        + w["specificity"] * item_sc["specificity"]
        + w["severity"]   * item_sc["severity_avg"]
        + w["dedup"]      * dedup
        + w["organization"] * organization
        + w["rouge"]      * item_sc["rouge_l_f1"]
        - w["length_pen"] * item_sc["length_penalty"]
        - w["redundancy_pen"] * item_sc["redundancy_penalty"]
    )
    composite = max(0.0, min(1.0, composite))

    return {
        **item_sc,
        "organization": organization,
        "dedup": dedup,
        "coherence": coherence,
        "composite": composite,
    }

# =============================================================================
# SCORE ENTIRE DATASET
# =============================================================================

def score_samples(paper_records: list[dict]) -> dict:
    """
    Score every sample across all papers and rollouts.

    Returns: dict keyed by
      (paper_idx, rollout_id, agent_role, turn_type, agent_name_or_target)
      -> {"score": {...}, "sample": <original dict>}
    """
    scores: dict = {}

    # Pre-compute GT categories per paper
    gt_cache: dict[int, tuple[list[str], dict[int, str]]] = {}
    for p in paper_records:
        pid = p["paper_idx"]
        if pid not in gt_cache:
            gt_items = split_gt(p.get("ground_truth", ""))
            gt_cats  = categorize_gt(pid, gt_items) if gt_items else {}
            gt_cache[pid] = (gt_items, gt_cats)

    for p in tqdm(paper_records, desc="Scoring papers"):
        pid        = p["paper_idx"]
        paper_text = p.get("paper_text", "")
        gt_items, gt_cats = gt_cache[pid]

        for rollout in p.get("rollouts", []):
            if rollout.get("error"):
                continue
            rid     = rollout["rollout_id"]
            samples = rollout.get("samples", [])

            # ---- Index samples by (agent_name, turn_type) for cross-referencing ----
            sample_index: dict[tuple[str, str], dict] = {}
            for s in samples:
                sample_index[(s["agent_name"], s["turn_type"])] = s

            # ---- Pass 1: Score all worker samples ----
            worker_scores: dict[tuple[str, str], dict] = {}
            worker_revised_outputs: dict[str, str] = {}

            for s in samples:
                if s["agent_role"] != "worker":
                    continue
                name      = s["agent_name"]
                turn      = s["turn_type"]
                specialty = SPECIALTY_MAP.get(name, "other")
                gt_sub    = gt_for_specialty(gt_items, gt_cats, specialty)

                tag = f"p{pid}_r{rid}_{name}_{turn}"
                item_sc = score_item_list(s["output"], gt_sub, paper_text, tag)
                item_sc["composite"] = composite_worker(item_sc)

                key = (pid, rid, "worker", turn, name)
                scores[key] = {"score": item_sc, "sample": s}
                worker_scores[(name, turn)] = item_sc

                if turn == "revised":
                    worker_revised_outputs[name] = s["output"]

            # ---- Pass 2: Score leader turns ----
            for s in samples:
                if s["agent_role"] != "leader":
                    continue

                if s["turn_type"] == "feedback_to_worker":
                    target_name = s.get("target_worker", "")
                    specialty = SPECIALTY_MAP.get(target_name, "other")
                    init_sc = worker_scores.get((target_name, "initial"))
                    rev_sc  = worker_scores.get((target_name, "revised"))

                    # Get the worker's initial output for feedback quality judge
                    worker_init_sample = sample_index.get(
                        (target_name, "initial"), {}
                    )
                    worker_init_output = worker_init_sample.get("output", "")

                    if init_sc and rev_sc:
                        tag = f"p{pid}_r{rid}_leader_fb_{target_name}"
                        sc = score_leader_feedback(
                            init_sc, rev_sc,
                            worker_init_output, s["output"],
                            specialty, tag,
                        )
                    else:
                        sc = dict(delta=0.0, delta_score=0.5,
                                  room_adjusted_delta=0.0,
                                  feedback_quality=0.0, composite=0.0)

                    key = (pid, rid, "leader", "feedback_to_worker", target_name)
                    scores[key] = {"score": sc, "sample": s}

                elif s["turn_type"] == "handoff_to_master":
                    sc = score_leader_handoff(s["output"], worker_revised_outputs)
                    key = (pid, rid, "leader", "handoff_to_master", None)
                    scores[key] = {"score": sc, "sample": s}

            # ---- Pass 3: Score master ----
            for s in samples:
                if s["agent_role"] != "master":
                    continue
                tag = f"p{pid}_r{rid}_master"
                sc  = score_master(s["output"], gt_items, paper_text, tag)
                key = (pid, rid, "master", "synthesis", None)
                scores[key] = {"score": sc, "sample": s}

    return scores

# =============================================================================
# SFT / DPO SELECTION
# =============================================================================

def _input_hash(input_messages: list[dict]) -> str:
    """Deterministic hash of input messages for DPO identity check."""
    blob = json.dumps(input_messages, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def select_sft_and_dpo(scores: dict) -> tuple[list[dict], list[dict]]:
    """
    SFT: For each (paper, agent_role, turn_type, target), pick the single
         highest-scoring rollout. No input-identity constraint needed.

    DPO: For each group, find best-worst pair with:
         1. Identical inputs (byte-level)
         2. Score gap >= DPO_MIN_SCORE_GAP
         3. The "chosen" sample must have composite >= 0.3 (quality floor)
            to prevent both chosen and rejected from being terrible.
    """
    # Group by (paper, role, turn, target)
    grouped: dict = defaultdict(list)
    for (pid, rid, role, turn, tgt), entry in scores.items():
        grouped[(pid, role, turn, tgt)].append((rid, entry))

    sft_samples: list[dict] = []
    dpo_pairs:   list[dict] = []

    for (pid, role, turn, tgt), items in grouped.items():
        # Sort by composite descending
        items.sort(
            key=lambda x: x[1]["score"].get("composite", 0.0),
            reverse=True,
        )
        best_rid, best_entry = items[0]
        best_composite = best_entry["score"].get("composite", 0.0)

        # ---- SFT: always pick the best ----
        sft_record = {
            "paper_idx":      pid,
            "agent_role":     role,
            "turn_type":      turn,
            "target_worker":  tgt,
            "rollout_id":     best_rid,
            "composite_score": best_composite,
            "score_detail":   best_entry["score"],
            "input_messages": best_entry["sample"]["input_messages"],
            "output":         best_entry["sample"]["output"],
            "ground_truth":   best_entry["sample"].get("ground_truth", ""),
            # HuggingFace chat-template-ready format
            "messages": (
                best_entry["sample"]["input_messages"]
                + [{"role": "assistant",
                    "content": best_entry["sample"]["output"]}]
            ),
        }
        sft_samples.append(sft_record)

        # ---- DPO: best vs worst with constraints ----
        if len(items) < 2:
            continue

        # Quality floor: chosen must be at least decent
        if best_composite < 0.3:
            continue

        # Try pairs from best down to worst, looking for valid DPO pair
        best_input_hash = _input_hash(
            best_entry["sample"]["input_messages"]
        )

        for worst_rid, worst_entry in reversed(items):
            if worst_rid == best_rid:
                continue

            worst_composite = worst_entry["score"].get("composite", 0.0)
            gap = best_composite - worst_composite

            if gap < DPO_MIN_SCORE_GAP:
                continue

            # Input identity check
            if DPO_REQUIRE_IDENTICAL_INPUT:
                worst_input_hash = _input_hash(
                    worst_entry["sample"]["input_messages"]
                )
                if best_input_hash != worst_input_hash:
                    continue

            dpo_pairs.append({
                "paper_idx":      pid,
                "agent_role":     role,
                "turn_type":      turn,
                "target_worker":  tgt,
                "input_messages": best_entry["sample"]["input_messages"],
                "chosen":         best_entry["sample"]["output"],
                "rejected":       worst_entry["sample"]["output"],
                "chosen_score":   best_composite,
                "rejected_score": worst_composite,
                "score_gap":      gap,
                "chosen_rollout_id":   best_rid,
                "rejected_rollout_id": worst_rid,
                "chosen_score_detail":  best_entry["score"],
                "rejected_score_detail": worst_entry["score"],
                "ground_truth":  best_entry["sample"].get("ground_truth", ""),
            })
            break  # only one DPO pair per group

    return sft_samples, dpo_pairs

# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(scores: dict, sft: list, dpo: list):
    """Print summary statistics to help tune weights and debug."""
    print("\n" + "=" * 70)
    print("REWARD MODEL DIAGNOSTICS")
    print("=" * 70)

    # Composite score distribution by role+turn
    by_role_turn: dict = defaultdict(list)
    for (pid, rid, role, turn, tgt), entry in scores.items():
        c = entry["score"].get("composite", 0.0)
        by_role_turn[(role, turn)].append(c)

    print("\nComposite Score Distribution:")
    print(f"  {'Role/Turn':<40} {'Count':>6} {'Mean':>6} {'Std':>6} "
          f"{'Min':>6} {'Max':>6}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for (role, turn), vals in sorted(by_role_turn.items()):
        import statistics
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {f'{role}/{turn}':<40} {len(vals):>6} {mean:>6.3f} "
              f"{std:>6.3f} {min(vals):>6.3f} {max(vals):>6.3f}")

    # SFT counts
    sft_counts: dict = defaultdict(int)
    for s in sft:
        sft_counts[(s["agent_role"], s["turn_type"])] += 1

    print(f"\nSFT Samples: {len(sft)}")
    for k, v in sorted(sft_counts.items()):
        print(f"  {k}: {v}")

    # DPO counts
    dpo_counts: dict = defaultdict(int)
    dpo_gaps: list = []
    for d in dpo:
        dpo_counts[(d["agent_role"], d["turn_type"])] += 1
        dpo_gaps.append(d["score_gap"])

    print(f"\nDPO Pairs: {len(dpo)}")
    for k, v in sorted(dpo_counts.items()):
        print(f"  {k}: {v}")

    if dpo_gaps:
        import statistics
        print(f"\nDPO Score Gaps: mean={statistics.mean(dpo_gaps):.3f}, "
              f"min={min(dpo_gaps):.3f}, max={max(dpo_gaps):.3f}")

    # Anti-hacking metric averages (worker samples only)
    worker_metrics = defaultdict(list)
    for (pid, rid, role, turn, tgt), entry in scores.items():
        if role == "worker":
            sc = entry["score"]
            for metric in ["length_penalty", "redundancy_penalty",
                           "boilerplate_ratio", "match_verification"]:
                if metric in sc:
                    worker_metrics[metric].append(sc[metric])

    if worker_metrics:
        print("\nAnti-Hacking Metrics (worker samples):")
        for metric, vals in sorted(worker_metrics.items()):
            import statistics
            print(f"  {metric}: mean={statistics.mean(vals):.3f}, "
                  f"max={max(vals):.3f}")

    print("=" * 70)

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Score rollout trajectories and produce SFT/DPO datasets."
    )
    ap.add_argument(
        "--rollout_json", required=True,
        help="Path to rollout_data_full.json from the rollout generation script.",
    )
    ap.add_argument(
        "--cache_dir", default=None,
        help="Directory for caching LLM judge calls. Default: ./reward_cache",
    )
    args = ap.parse_args()

    # Override cache dir if provided
    if args.cache_dir:
        global CACHE_DIR
        CACHE_DIR = Path(args.cache_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Hardcoded output directories ---
    score_out_dir   = SCORE_OUT_DIR
    sft_dpo_out_dir = SFT_DPO_OUT_DIR
    score_out_dir.mkdir(parents=True, exist_ok=True)
    sft_dpo_out_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout data
    papers = json.loads(Path(args.rollout_json).read_text()) 
    
    print(f"Loaded {len(papers)} papers from {args.rollout_json}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Score output dir: {score_out_dir}")
    print(f"SFT/DPO output dir: {sft_dpo_out_dir}")

    # Validate structure
    total_rollouts = sum(len(p.get("rollouts", [])) for p in papers)
    total_samples  = sum(
        len(r.get("samples", []))
        for p in papers for r in p.get("rollouts", [])
    )
    print(f"Total rollouts: {total_rollouts}, total samples: {total_samples}")

    if total_samples == 0:
        print("[ERROR] No samples found. Check rollout_data_full.json format.")
        return

    # Score all samples
    print("\nScoring samples...")
    scores = score_samples(papers)
    print(f"Scored {len(scores)} sample entries.")

    # ================================================================
    # Save per-sample scores SPLIT BY ROLE (worker / leader / master)
    # ================================================================
    flat_by_role = {"worker": [], "leader": [], "master": []}
    for (pid, rid, role, turn, tgt), entry in scores.items():
        record = {
            "paper_idx":      pid,
            "rollout_id":     rid,
            "agent_role":     role,
            "turn_type":      turn,
            "target_worker":  tgt,
            "agent_name":     entry["sample"].get("agent_name", ""),
            "input_messages": entry["sample"].get("input_messages", []),
            "output":         entry["sample"].get("output", ""),
            "ground_truth":   entry["sample"].get("ground_truth", ""),
            "score":          entry["score"],
        }
        flat_by_role[role].append(record)

    # Also save the combined file for convenience
    all_flat = flat_by_role["worker"] + flat_by_role["leader"] + flat_by_role["master"]
    scores_all_path = score_out_dir / "per_sample_scores.json"
    scores_all_path.write_text(json.dumps(all_flat, indent=2))
    print(f"Per-sample scores (all): {scores_all_path} ({len(all_flat)} entries)")

    for role_name, role_records in flat_by_role.items():
        role_path = score_out_dir / f"per_sample_scores_{role_name}.json"
        role_path.write_text(json.dumps(role_records, indent=2))
        print(f"Per-sample scores ({role_name}): {role_path} ({len(role_records)} entries)")

    # ================================================================
    # Select SFT and DPO datasets, then SPLIT BY ROLE
    # ================================================================
    print("\nSelecting SFT and DPO datasets...")
    sft, dpo = select_sft_and_dpo(scores)

    # --- Split SFT by role ---
    sft_by_role = {"worker": [], "leader": [], "master": []}
    for s in sft:
        sft_by_role[s["agent_role"]].append(s)

    # Save combined SFT
    sft_all_path = sft_dpo_out_dir / "sft_dataset.json"
    sft_all_path.write_text(json.dumps(sft, indent=2))
    print(f"\nSFT (all): {sft_all_path} ({len(sft)} samples)")

    # Save per-role SFT
    for role_name, role_sft in sft_by_role.items():
        sft_path = sft_dpo_out_dir / f"sft_dataset_{role_name}.json"
        sft_path.write_text(json.dumps(role_sft, indent=2))
        print(f"SFT ({role_name}): {sft_path} ({len(role_sft)} samples)")

    # --- Split DPO by role ---
    dpo_by_role = {"worker": [], "leader": [], "master": []}
    for d in dpo:
        dpo_by_role[d["agent_role"]].append(d)

    # Save combined DPO
    dpo_all_path = sft_dpo_out_dir / "dpo_dataset.json"
    dpo_all_path.write_text(json.dumps(dpo, indent=2))
    print(f"\nDPO (all): {dpo_all_path} ({len(dpo)} pairs)")

    # Save per-role DPO
    for role_name, role_dpo in dpo_by_role.items():
        dpo_path = sft_dpo_out_dir / f"dpo_dataset_{role_name}.json"
        dpo_path.write_text(json.dumps(role_dpo, indent=2))
        print(f"DPO ({role_name}): {dpo_path} ({len(role_dpo)} pairs)")

    # Print diagnostics
    print_diagnostics(scores, sft, dpo)

    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    print(f"\nScore dir:   {score_out_dir}")
    print(f"  per_sample_scores.json          ({len(all_flat)} total)")
    for role_name in ["worker", "leader", "master"]:
        print(f"  per_sample_scores_{role_name}.json  ({len(flat_by_role[role_name])})")
    print(f"\nSFT/DPO dir: {sft_dpo_out_dir}")
    print(f"  sft_dataset.json                ({len(sft)} total)")
    for role_name in ["worker", "leader", "master"]:
        print(f"  sft_dataset_{role_name}.json        ({len(sft_by_role[role_name])})")
    print(f"  dpo_dataset.json                ({len(dpo)} total)")
    for role_name in ["worker", "leader", "master"]:
        print(f"  dpo_dataset_{role_name}.json        ({len(dpo_by_role[role_name])})")
    print("\nDone.")

if __name__ == "__main__":
    main()