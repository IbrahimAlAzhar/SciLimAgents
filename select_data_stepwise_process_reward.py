"""
select_data_stepwise_process_reward.py
=========================

Score multi-hop / multi-mode rollouts with a STEPWISE (process) reward model,
then build:

    * SFT dataset : the single BEST candidate per (paper, agent) group.
    * DPO dataset : the BEST vs WORST candidate per group (a preference pair).

REWARD MODEL: **GPT-4o-mini** (unchanged from your original pipeline) is used as
the LLM alignment judge for workers. The process (stepwise) reward itself is
API-free and rule-based. Rollouts are now produced by Qwen3-4B via vLLM, but
this file does not care who generated them — it only reads the JSON schema.

HOP -> STEP MAPPING
-------------------
    hop_search              ->  SEARCH   step   (weight 0.22)
    hop_query               ->  RETRIEVE step   (weight 0.30)
    hop_draft / revised     ->  SYNTHESIZE step (weight 0.48)

    worker_total = 0.22*search + 0.30*retrieve + 0.48*synthesize
    master_total = synthesize-only

SELECTION
---------
    Group workers by (paper_idx, worker_name); master/leader by (paper_idx).
      SFT : argmax(final score)
      DPO : argmax vs argmin (workers only, if gap >= DPO_MIN_GAP)
    Worker chosen/rejected are stored under a CANONICAL, mode-free prompt so the
    exploration-mode text never leaks into training.

USAGE
-----
    python select_data_stepwise_process_reward.py \
        --rollout_json $DIR/rollouts/rollout_data_full.json \
        --out_dir      $DIR/select
"""

from __future__ import annotations

import os
import re
import csv
import json
import hashlib
import argparse
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Any

from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except Exception:
    _ROUGE = None
    print("[WARN] rouge_score not installed (pip install rouge-score); "
          "GT similarity will use bullet-match only.")

# =============================================================================
# CONFIG  (env-overridable)
# =============================================================================



# =============================================================================
# ENVIRONMENT CONFIGURATION
# All input/output locations are supplied at run time. No paths, dataset sizes
# or credentials are stored in this file.
# =============================================================================
def _require_env(name, hint=""):
    """Return a mandatory environment variable, or exit with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Required environment variable {name} is not set."
            + (f"  Expected: {hint}" if hint else "")
        )
    return value


def _optional_int(name):
    """Return an int env var, or None when unset/blank (meaning 'no limit')."""
    raw = os.environ.get(name, "").strip()
    if raw in ("", "none", "None", "null"):
        return None
    return int(raw)


DEFAULT_ROLLOUT_JSON = os.environ.get("ROLLOUT_JSON")   # or pass --rollout_json
DEFAULT_OUT_DIR      = os.environ.get("SELECT_DIR")     # or pass --out_dir

STEP_WEIGHTS = {"search": 0.22, "retrieve": 0.30, "synthesize": 0.48}

DPO_MIN_GAP   = float(os.environ.get("DPO_MIN_GAP", 0.05))
SFT_MIN_SCORE = float(os.environ.get("SFT_MIN_SCORE", 0.30))
# Master merges are abstract paraphrases -> lower token-overlap grounding.
# Keep a 0 floor so the master group is never dropped.
MASTER_SFT_MIN_SCORE = float(os.environ.get("MASTER_SFT_MIN_SCORE", 0.0))
SOURCE_TRUNC        = 16000   # chars of source text used for grounding checks
CANON_PROMPT_TOKENS = 20000   # chars of paper kept in the canonical worker prompt

# --- Ground-truth blending ---
STEPWISE_WEIGHT  = float(os.environ.get("STEPWISE_WEIGHT", 0.70))
GT_WEIGHT        = float(os.environ.get("GT_WEIGHT", 0.30))
GT_MATCH_JACCARD = 0.18

WRITE_CSV = True

# --- Worker GT alignment (GPT-4o-mini judge + ROUGE-L + lexical) ---
USE_LLM_GT_JUDGE  = os.environ.get("USE_LLM_GT_JUDGE", "1") == "1"
LLM_JUDGE_MODEL   = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")   # <-- reward model
JUDGE_BASE_URL    = os.environ.get("JUDGE_BASE_URL")               # None => real OpenAI
WORKER_GT_WEIGHTS = {"llm_align": 0.70, "rouge_l": 0.15, "lexical": 0.15}
REWARD_CACHE_DIR  = Path(os.environ.get("REWARD_CACHE_DIR", ".reward_cache"))

WORKER_SPECIALTY = {
    "Novelty_Significance_Agent":                     "novelty_significance",
    "Theoretical_Methodological_Agent":               "theoretical_methodological",
    "Experimental_Evaluation_Agent":                  "experimental_evaluation",
    "Generalization_Robustness_Efficiency_Agent":     "generalization_robustness_efficiency",
    "Clarity_Interpretability_Reproducibility_Agent": "clarity_interpretability_reproducibility",
    "Data_Ethics_Agent":                              "data_ethics",
}

SPECIALTY_KEYWORDS = {
    "novelty_significance": ["novel", "novelty", "significance", "contribution", "incremental",
                             "original", "prior work", "impact", "motivation"],
    "theoretical_methodological": ["theoretical", "theory", "assumption", "proof", "derivation",
                                   "method", "methodolog", "ablation", "component", "formulation"],
    "experimental_evaluation": ["experiment", "evaluation", "baseline", "metric", "statistical",
                                "significance test", "comparison", "benchmark", "result", "error bar"],
    "generalization_robustness_efficiency": ["generaliz", "robust", "efficien", "latency", "compute",
                                             "cost", "out-of-distribution", "ood", "scalab",
                                             "deployment", "resource"],
    "clarity_interpretability_reproducibility": ["clarity", "clear", "interpret", "explain", "reproduc",
                                                 "replicat", "code", "seed", "hyperparameter", "document"],
    "data_ethics": ["data", "dataset", "bias", "fairness", "ethic", "privacy", "label", "annotation",
                    "representative", "societal"],
}

# =============================================================================
# CANONICAL (mode-free) PROMPTS — identical to the rollout script's base prompts
# =============================================================================

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

WORKER_PROMPT_MAP = {
    "Novelty_Significance_Agent":                     (get_novelty_significance_prompt,                     "novelty and significance"),
    "Theoretical_Methodological_Agent":               (get_theoretical_methodological_prompt,               "theoretical and methodological soundness (including ablations)"),
    "Experimental_Evaluation_Agent":                  (get_experimental_evaluation_prompt,                  "experimental evaluation, baselines, and metrics"),
    "Generalization_Robustness_Efficiency_Agent":     (get_generalization_robustness_efficiency_prompt,     "generalization, robustness, efficiency, and applicability"),
    "Clarity_Interpretability_Reproducibility_Agent": (get_clarity_interpretability_reproducibility_prompt, "clarity, interpretability, and reproducibility"),
    "Data_Ethics_Agent":                              (get_data_ethics_prompt,                              "data integrity, bias, fairness, and ethics"),
}

# =============================================================================
# LOW-LEVEL TEXT HELPERS
# =============================================================================

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "paper", "study",
    "method", "model", "results", "limitation", "limitations",
}

def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) > 2 and t not in _STOP}

def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _contains_quote(quote: str, source_text: str, min_chars: int = 18) -> bool:
    quote = re.sub(r"\s+", " ", str(quote or "")).strip().lower()
    if len(quote) < min_chars:
        return False
    haystack = re.sub(r"\s+", " ", source_text).lower()
    return quote[:180] in haystack

def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", " ", str(text), flags=re.DOTALL | re.IGNORECASE)

_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]\s|\Z)", re.DOTALL | re.MULTILINE)

def split_bullets(text: str) -> list[str]:
    if not text:
        return []
    text = strip_think_blocks(text)
    items = [re.sub(r"\s+", " ", m).strip() for m in _BULLET_RE.findall(text)]
    if not items:
        items = re.split(r"\n\s*\d+\.\s+|\n\n+", text)
        items = [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]
    return [i for i in items if len(i) > 20]

def build_units(text: str) -> list[dict[str, Any]]:
    units = []
    for b in split_bullets(text):
        quotes = re.findall(r'["“‘\']([^"”’\']{12,200})["”’\']', b)
        pointers = [{"quote": q} for q in quotes]
        units.append({
            "statement": b, "evidence_pointers": pointers,
            "impact": "", "suggested_fix": "", "severity": "", "title": "",
        })
    return units

# =============================================================================
# STEP SCORERS
# =============================================================================

def score_search(search_text: str) -> float:
    search = strip_think_blocks(search_text).strip()
    if not search:
        return 0.0
    lines = [l for l in search.splitlines() if l.strip()]
    has_sources = any(w in search.lower() for w in
                      ("paper", "cited", "rag", "abstract", "introduction", "baseline"))
    specificity = min(1.0, len(_tokens(search)) / 35.0)
    structure = min(1.0, len(lines) / 3.0)
    total = 0.35 + 0.35 * specificity + 0.20 * structure + 0.10 * float(has_sources)
    return round(_clamp(total), 4)

def score_retrieve(retrieve_text: str, source_text: str) -> float:
    retrieve = strip_think_blocks(retrieve_text).strip()
    if not retrieve:
        return 0.0
    snippets = re.findall(r"(?:Evidence|Quote|Span)\s*:?\s*([^.\n]{18,220}(?:\.|$))",
                          retrieve, flags=re.IGNORECASE)
    if not snippets:
        snippets = [l.strip(" -0123456789.") for l in retrieve.splitlines()
                    if len(l.strip()) > 18]
    grounded = 0
    for snippet in snippets[:6]:
        if _contains_quote(snippet, source_text) or _overlap(snippet, source_text[:12000]) > 0.05:
            grounded += 1
    grounded_score = grounded / max(1, min(len(snippets), 6))
    density = min(1.0, len(snippets) / 3.0)
    total = 0.25 + 0.55 * grounded_score + 0.20 * density
    return round(_clamp(total), 4)

def _score_specificity(unit: dict[str, Any]) -> float:
    text = " ".join(str(unit.get(k) or "") for k in ("title", "statement", "impact", "suggested_fix"))
    words = str(unit.get("statement") or "").split()
    score = 0.0
    if 8 <= len(words) <= 55:
        score += 0.30
    elif 5 <= len(words) <= 80:
        score += 0.16
    if any(term in text.lower() for term in
           ("baseline", "ablation", "dataset", "metric", "seed", "split", "compute",
            "code", "reproduc", "rag", "cited", "robust")):
        score += 0.35
    if re.search(r"\b\d+(?:\.\d+)?%?|\b[A-Z][A-Za-z0-9_-]{2,}\b", text):
        score += 0.15
    if str(unit.get("impact") or "").strip():
        score += 0.10
    if str(unit.get("suggested_fix") or "").strip():
        score += 0.10
    generic = len(re.findall(r"\b(some|various|several|many|things|issues|aspects)\b", text.lower()))
    score -= min(0.25, 0.05 * generic)
    return _clamp(score)

def _redundancy(statement: str, others: list[str]) -> float:
    return max((_overlap(statement, o) for o in others if o != statement), default=0.0)

def _score_unit(unit: dict[str, Any], units: list[dict[str, Any]],
                source_text: str, index: int) -> dict[str, Any]:
    statement = str(unit.get("statement") or "")
    pointers = unit.get("evidence_pointers") or []
    quotes = [str(p.get("quote") or "") for p in pointers if isinstance(p, dict)]
    grounded_scores = []
    for q in quotes:
        if _contains_quote(q, source_text):
            grounded_scores.append(1.0)
        else:
            grounded_scores.append(min(1.0, 3.0 * _overlap(q, source_text[:SOURCE_TRUNC])))
    groundedness = max(grounded_scores,
                       default=min(1.0, 2.2 * _overlap(statement, source_text[:SOURCE_TRUNC])))
    evidence_quality = _clamp(0.35 * bool(quotes) + 0.65 * groundedness)
    specificity = _score_specificity(unit)
    actionability = _clamp(
        0.45 * bool(str(unit.get("impact") or "").strip())
        + 0.45 * bool(str(unit.get("suggested_fix") or "").strip())
        + 0.10 * bool(str(unit.get("severity") or "").strip())
    )
    statements = [str(o.get("statement") or "") for o in units]
    redundancy = _clamp(_redundancy(statement, statements))
    score = _clamp(
        0.42 * groundedness
        + 0.24 * specificity
        + 0.20 * evidence_quality
        + 0.10 * actionability
        + 0.04 * (1.0 - redundancy)
    )
    return {
        "index": index, "statement": statement[:220], "score": round(score, 4),
        "groundedness": round(groundedness, 4), "specificity": round(specificity, 4),
        "evidence_quality": round(evidence_quality, 4),
        "actionability": round(actionability, 4),
        "redundancy_penalty": round(redundancy, 4),
    }

def score_synthesize(units: list[dict[str, Any]], source_text: str) -> tuple[float, list[dict]]:
    if not units:
        return 0.0, []
    unit_scores = [_score_unit(u, units, source_text, i) for i, u in enumerate(units)]
    mean_unit = sum(u["score"] for u in unit_scores) / len(unit_scores)
    n = len(units)
    count_factor = 1.0 if 2 <= n <= 7 else (0.80 if n == 1 or n <= 10 else 0.55)
    synth = _clamp(0.90 * mean_unit + 0.10 * count_factor)
    return round(synth, 4), unit_scores

# =============================================================================
# TRAJECTORY-LEVEL SCORING
# =============================================================================

def score_worker(search_text: str, retrieve_text: str, synth_text: str,
                 source_text: str) -> dict[str, Any]:
    s = score_search(search_text)
    r = score_retrieve(retrieve_text, source_text)
    syn, unit_scores = score_synthesize(build_units(synth_text), source_text)
    total = _clamp(STEP_WEIGHTS["search"] * s
                   + STEP_WEIGHTS["retrieve"] * r
                   + STEP_WEIGHTS["synthesize"] * syn)
    return {"total": round(total, 4),
            "steps": {"search": s, "retrieve": r, "synthesize": syn},
            "n_units": len(unit_scores)}

def score_master(synth_text: str, source_text: str) -> dict[str, Any]:
    syn, unit_scores = score_synthesize(build_units(synth_text), source_text)
    return {"total": round(syn, 4),
            "steps": {"search": None, "retrieve": None, "synthesize": syn},
            "n_units": len(unit_scores)}

# =============================================================================
# GROUND-TRUTH SIMILARITY + BLENDING
# =============================================================================

def split_gt(gt_text: str) -> list[str]:
    if not gt_text:
        return []
    items = [re.sub(r"^\s*[-*•\d.]+\s*", "", l).strip() for l in str(gt_text).splitlines()]
    return [i for i in items if len(i) > 10]

def _rouge_l_f1(pred: str, gt: str) -> float:
    if not _ROUGE or not pred.strip() or not gt.strip():
        return 0.0
    return float(_ROUGE.score(gt, pred)["rougeL"].fmeasure)

def _bullet_match_f1(pred_items: list[str], gt_items: list[str]) -> float:
    if not pred_items or not gt_items:
        return 0.0
    pairs = []
    for pi, p in enumerate(pred_items):
        for gi, g in enumerate(gt_items):
            j = _overlap(p, g)
            if j >= GT_MATCH_JACCARD:
                pairs.append((j, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for _, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    precision = len(used_p) / max(len(pred_items), 1)
    recall    = len(used_g) / max(len(gt_items), 1)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

def gt_similarity(pred_text: str, gt_text: str):
    """Full-GT similarity (MASTER/LEADER), in [0,1]. None if no ground truth."""
    gt_items = split_gt(gt_text)
    if not gt_items:
        return None
    pred_items = split_bullets(pred_text)
    if not pred_items:
        return 0.0
    match_f1 = _bullet_match_f1(pred_items, gt_items)
    rouge_f1 = _rouge_l_f1(" ".join(pred_items), " ".join(gt_items))
    return _clamp(0.5 * match_f1 + 0.5 * rouge_f1)

def route_gt_to_specialties(gt_items: list[str]) -> dict[str, list[str]]:
    routes: dict[str, list[str]] = defaultdict(list)
    for item in gt_items:
        low = item.lower()
        best_spec, best_hits = "other", 0
        for spec, kws in SPECIALTY_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in low)
            if hits > best_hits:
                best_hits, best_spec = hits, spec
        routes[best_spec].append(item)
    return routes

# --- GPT-4o-mini judge -------------------------------------------------------
_OAI = None
def _oai_client():
    global _OAI
    if _OAI is None:
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                return None
            kw = {"api_key": key, "timeout": 90}
            if JUDGE_BASE_URL:
                kw["base_url"] = JUDGE_BASE_URL
            _OAI = OpenAI(**kw)
        except Exception:
            return None
    return _OAI

def llm_gt_align(worker_output: str, gt_slice_text: str, specialty: str):
    """GPT-4o-mini alignment score in [0,1], or None if unavailable. Disk-cached."""
    client = _oai_client()
    if client is None:
        return None
    REWARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((LLM_JUDGE_MODEL + specialty + worker_output[:4000]
                          + gt_slice_text[:2000]).encode()).hexdigest()[:24]
    cpath = REWARD_CACHE_DIR / f"align_{key}.json"
    if cpath.exists():
        try:
            return float(json.loads(cpath.read_text()).get("alignment"))
        except Exception:
            pass
    prompt = (
        f"You are judging a specialist reviewer whose area is: {specialty}.\n\n"
        f"GOLD limitations (reference, for this area):\n{gt_slice_text[:2000]}\n\n"
        f"REVIEWER's limitations:\n{worker_output[:4000]}\n\n"
        "Rate how well the reviewer's limitations ALIGN with the gold ones — do they "
        "capture the same substantive flaws (regardless of wording)? Penalize misses "
        "and off-target/unsupported points. Return ONLY JSON: "
        '{"alignment": <float 0..1>}'
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_JUDGE_MODEL, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict, fair scientific reviewer. JSON only."},
                {"role": "user", "content": prompt},
            ],
            timeout=90,
        )
        obj = json.loads(resp.choices[0].message.content)
        a = _clamp(float(obj.get("alignment", 0.0)))
        cpath.write_text(json.dumps({"alignment": a}))
        return a
    except Exception as e:
        print(f"  [LLM align error] {specialty}: {e}")
        return None

def worker_gt_alignment(worker_output: str, gt_slice_items: list[str], specialty: str):
    if not gt_slice_items:
        return None
    pred_items = split_bullets(worker_output)
    gt_text = "\n".join(gt_slice_items)
    rouge   = _rouge_l_f1(" ".join(pred_items), gt_text)
    lexical = _bullet_match_f1(pred_items, gt_slice_items)
    llm = llm_gt_align(worker_output, gt_text, specialty) if USE_LLM_GT_JUDGE else None

    w = WORKER_GT_WEIGHTS
    if llm is None:
        denom = w["rouge_l"] + w["lexical"]
        return _clamp((w["rouge_l"] * rouge + w["lexical"] * lexical) / denom) if denom else lexical
    total = w["llm_align"] * llm + w["rouge_l"] * rouge + w["lexical"] * lexical
    return _clamp(total / (w["llm_align"] + w["rouge_l"] + w["lexical"]))

def blend_final(stepwise_total: float, gt_sim):
    if gt_sim is None:
        return round(stepwise_total, 4)
    total = (STEPWISE_WEIGHT * stepwise_total + GT_WEIGHT * gt_sim) / (STEPWISE_WEIGHT + GT_WEIGHT)
    return round(_clamp(total), 4)

# =============================================================================
# ROLLOUT -> CANDIDATE EXTRACTION
# =============================================================================

def extract_worker_final(rollout: dict) -> dict[str, dict]:
    hops = defaultdict(dict)
    revised = defaultdict(list)
    for s in rollout.get("samples", []):
        if s.get("agent_role") != "worker":
            continue
        name, tt = s.get("agent_name"), s.get("turn_type")
        if tt in ("hop_search", "hop_query", "hop_draft"):
            hops[name][tt] = s
        elif tt == "revised":
            revised[name].append(s)

    out = {}
    for name, hop_samples in hops.items():
        search_s = hop_samples.get("hop_search")
        query_s  = hop_samples.get("hop_query")
        draft_s  = hop_samples.get("hop_draft")
        if revised.get(name):
            final_s = max(revised[name], key=lambda x: x.get("critique_round", 0))
        else:
            final_s = draft_s
        out[name] = {
            "search":     (search_s or {}).get("output", ""),
            "retrieve":   (query_s or {}).get("output", ""),
            "synthesize": (final_s or draft_s or {}).get("output", ""),
            "final_sample": final_s or draft_s,
        }
    return out

# =============================================================================
# SCORE + SELECT
# =============================================================================

def _canonical_worker_prompt(worker_name: str, paper_text: str) -> list[dict]:
    fn, spec = WORKER_PROMPT_MAP[worker_name]
    return [
        {"role": "system", "content": fn(paper_text[:CANON_PROMPT_TOKENS])},
        {"role": "user", "content": f"Identify limitations focused on {spec}. "
                                    "Return only an evidence-grounded bullet list."},
    ]

def process_dataset(papers: list[dict], progress: bool = True):
    all_candidates: list[dict] = []
    sft: list[dict] = []
    dpo: list[dict] = []

    for p in (tqdm(papers, desc="Scoring papers (stepwise)") if progress else papers):
        pid = p.get("paper_idx")
        source_text = (p.get("paper_text", "") or "")
        if p.get("citation_text"):
            source_text = source_text + "\n\n" + p["citation_text"]
        gt_text = p.get("ground_truth", "") or ""
        gt_items_all = split_gt(gt_text)
        gt_routes = route_gt_to_specialties(gt_items_all) if gt_items_all else {}

        worker_cands: dict[str, list[dict]] = defaultdict(list)
        master_cands: list[dict] = []
        leader_cands: list[dict] = []

        for rollout in p.get("rollouts", []):
            if rollout.get("error") or not rollout.get("samples"):
                continue
            mode = rollout.get("mode")
            rid  = rollout.get("rollout_id")

            # ---- workers ----
            wf = extract_worker_final(rollout)
            for name, tr in wf.items():
                sc = score_worker(tr["search"], tr["retrieve"], tr["synthesize"], source_text)
                specialty = WORKER_SPECIALTY.get(name, "other")
                gt_slice = gt_routes.get(specialty) or gt_items_all
                gt_sim = worker_gt_alignment(tr["synthesize"], gt_slice, specialty)
                cand = {
                    "paper_idx": pid, "rollout_id": rid, "mode": mode,
                    "agent_role": "worker", "agent_name": name,
                    "output": tr["synthesize"],
                    "stepwise_score": sc["total"], "gt_similarity": gt_sim,
                    "score": blend_final(sc["total"], gt_sim), "score_detail": sc,
                }
                worker_cands[name].append(cand)
                all_candidates.append(cand)

            # ---- master ----
            msample = next((s for s in rollout.get("samples", [])
                            if s.get("agent_role") == "master"
                            and s.get("turn_type") == "synthesis"), None)
            if msample and msample.get("output", "").strip():
                sc = score_master(msample["output"], source_text)
                gt_sim = gt_similarity(msample["output"], gt_text)
                cand = {
                    "paper_idx": pid, "rollout_id": rid, "mode": mode,
                    "agent_role": "master", "agent_name": "Master_Agent",
                    "output": msample["output"],
                    "input_messages": msample.get("input_messages", []),
                    "stepwise_score": sc["total"], "gt_similarity": gt_sim,
                    "score": blend_final(sc["total"], gt_sim), "score_detail": sc,
                }
                master_cands.append(cand)
                all_candidates.append(cand)

            # ---- leader handoff ----
            lsample = next((s for s in rollout.get("samples", [])
                            if s.get("agent_role") == "leader"
                            and s.get("turn_type") == "handoff_to_master"), None)
            if lsample and lsample.get("output", "").strip():
                sc = score_master(lsample["output"], source_text)
                gt_sim = gt_similarity(lsample["output"], gt_text)
                cand = {
                    "paper_idx": pid, "rollout_id": rid, "mode": mode,
                    "agent_role": "leader", "agent_name": "Leader_Agent",
                    "output": lsample["output"],
                    "input_messages": lsample.get("input_messages", []),
                    "stepwise_score": sc["total"], "gt_similarity": gt_sim,
                    "score": blend_final(sc["total"], gt_sim), "score_detail": sc,
                }
                leader_cands.append(cand)
                all_candidates.append(cand)

        # ---- WORKERS: SFT (best) + DPO (best/worst), canonical mode-free prompt ----
        for name, cands in worker_cands.items():
            prompt = _canonical_worker_prompt(name, source_text)
            _select(cands, prompt, "worker", name, pid, sft, dpo,
                    min_score=SFT_MIN_SCORE, min_gap=DPO_MIN_GAP)

        # ---- MASTER & LEADER: SFT ONLY, using their REAL merger input ----
        if master_cands:
            _sft_best(master_cands, "master", "Master_Agent", pid, sft,
                      min_score=MASTER_SFT_MIN_SCORE)
        if leader_cands:
            _sft_best(leader_cands, "leader", "Leader_Agent", pid, sft, min_score=0.0)

    return all_candidates, sft, dpo


def _sft_best(cands: list[dict], role: str, name: str, pid,
              sft: list[dict], min_score: float = 0.0):
    """SFT ONLY: highest-scoring candidate, conditioned on ITS OWN upstream input."""
    cands = [c for c in cands if c.get("output", "").strip()]
    if not cands:
        return
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    if best["score"] < min_score:
        return
    prompt = best.get("input_messages") or []
    sft.append({
        "paper_idx": pid, "agent_role": role, "agent_name": name,
        "mode": best["mode"], "rollout_id": best["rollout_id"],
        "final_score": best["score"],
        "stepwise_score": best.get("stepwise_score"),
        "gt_similarity": best.get("gt_similarity"),
        "score_detail": best["score_detail"],
        "prompt": prompt,
        "output": best["output"],
        "messages": prompt + [{"role": "assistant", "content": best["output"]}],
    })


def _select(cands: list[dict], prompt: list[dict], role: str, name: str,
            pid, sft: list[dict], dpo: list[dict],
            min_score: float = 0.0, min_gap: float = DPO_MIN_GAP):
    cands = [c for c in cands if c.get("output", "").strip()]
    if not cands:
        return
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    if best["score"] < min_score:
        return

    sft.append({
        "paper_idx": pid, "agent_role": role, "agent_name": name,
        "mode": best["mode"], "rollout_id": best["rollout_id"],
        "final_score": best["score"],
        "stepwise_score": best.get("stepwise_score"),
        "gt_similarity": best.get("gt_similarity"),
        "score_detail": best["score_detail"],
        "prompt": prompt,
        "output": best["output"],
        "messages": prompt + [{"role": "assistant", "content": best["output"]}],
    })

    if len(cands) < 2:
        return
    worst = cands[-1]
    gap = best["score"] - worst["score"]
    if gap < min_gap:
        return
    dpo.append({
        "paper_idx": pid, "agent_role": role, "agent_name": name,
        "prompt": prompt,
        "chosen": best["output"], "rejected": worst["output"],
        "chosen_mode": best["mode"], "rejected_mode": worst["mode"],
        "chosen_score": best["score"], "rejected_score": worst["score"],
        "chosen_stepwise": best.get("stepwise_score"),
        "rejected_stepwise": worst.get("stepwise_score"),
        "chosen_gt_similarity": best.get("gt_similarity"),
        "rejected_gt_similarity": worst.get("gt_similarity"),
        "chosen_rollout_id": best["rollout_id"], "rejected_rollout_id": worst["rollout_id"],
        "chosen_score_detail": best["score_detail"],
        "rejected_score_detail": worst["score_detail"],
        "score_gap": round(gap, 4),
    })

# =============================================================================
# RESUME  — per-paper shards + backfill from previously written aggregates
# =============================================================================
#
# Every scored paper is written to  <out_dir>/per_paper/paper_00300.json  holding
# that paper's candidates / sft rows / dpo rows. On the next run those papers are
# skipped and their shards are merged straight into the final aggregates.
#
# If you already have aggregate files in <out_dir> from earlier runs (before
# sharding existed), `backfill_shards_from_aggregates()` reads them, splits them
# per paper, and writes the shards — so nothing already scored is recomputed.

SHARD_SUBDIR = "per_paper"

# (filename, section) — first match wins per section, so we never double-count
# sft_dataset.json against sft_dataset_worker.json.
_AGGREGATE_SOURCES = {
    "candidates": ["stepwise_candidate_scores.json"],
    "sft":        ["sft_dataset.json",
                   "sft_dataset_worker.json", "sft_dataset_leader.json",
                   "sft_dataset_master.json"],
    "dpo":        ["dpo_dataset.json", "dpo_dataset_worker.json"],
}

_DEDUP_KEYS = {
    "candidates": ("paper_idx", "agent_role", "agent_name", "rollout_id", "mode"),
    "sft":        ("paper_idx", "agent_role", "agent_name"),
    "dpo":        ("paper_idx", "agent_role", "agent_name"),
}


def _dedup(rows: list[dict], section: str) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = tuple(r.get(f) for f in _DEDUP_KEYS[section])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def shard_path(shard_dir: Path, pid) -> Path:
    try:
        return shard_dir / f"paper_{int(pid):05d}.json"
    except (TypeError, ValueError):
        return shard_dir / f"paper_{str(pid)}.json"


def write_shard(shard_dir: Path, pid, cands, sft_rows, dpo_rows):
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp = shard_path(shard_dir, pid).with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"paper_idx": pid, "candidates": cands, "sft": sft_rows, "dpo": dpo_rows},
        indent=2))
    tmp.replace(shard_path(shard_dir, pid))     # atomic: no half-written shards


def load_shards(shard_dir: Path) -> dict:
    """paper_idx -> {"candidates": [...], "sft": [...], "dpo": [...]}"""
    out = {}
    if not shard_dir.exists():
        return out
    for f in sorted(shard_dir.glob("paper_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  [resume] corrupt shard skipped: {f.name} ({e})")
            continue
        pid = d.get("paper_idx")
        out[pid] = {"candidates": d.get("candidates", []),
                    "sft": d.get("sft", []), "dpo": d.get("dpo", [])}
    return out


def backfill_shards_from_aggregates(out_dir: Path, shard_dir: Path,
                                    known: set, extra_globs: list[str] | None = None) -> int:
    """Turn previously written aggregate JSONs into per-paper shards.

    Only papers NOT already in `known` (i.e. not already sharded) are created.
    Returns the number of shards written.
    """
    by_paper: dict = defaultdict(lambda: {"candidates": [], "sft": [], "dpo": []})

    files_used = []
    for section, names in _AGGREGATE_SOURCES.items():
        # prefer the combined file if it exists; otherwise take the per-role ones
        chosen = None
        for n in names:
            if (out_dir / n).exists():
                chosen = [n] if n in ("stepwise_candidate_scores.json",
                                      "sft_dataset.json", "dpo_dataset.json") else None
                break
        picked = chosen if chosen else [n for n in names if (out_dir / n).exists()]
        for n in picked:
            try:
                rows = json.loads((out_dir / n).read_text())
            except Exception as e:
                print(f"  [resume] could not read {n}: {e}")
                continue
            if not isinstance(rows, list):
                continue
            files_used.append(f"{n}({len(rows)})")
            for r in rows:
                if isinstance(r, dict) and "paper_idx" in r:
                    by_paper[r["paper_idx"]][section].append(r)

    for pat in (extra_globs or []):
        for f in sorted(out_dir.glob(pat)):
            if f.name in sum(_AGGREGATE_SOURCES.values(), []):
                continue
            try:
                rows = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                continue
            # classify by the fields present
            section = ("dpo" if "chosen" in rows[0] and "rejected" in rows[0]
                       else "sft" if "messages" in rows[0]
                       else "candidates" if "stepwise_score" in rows[0] else None)
            if not section:
                continue
            files_used.append(f"{f.name}({len(rows)})")
            for r in rows:
                if "paper_idx" in r:
                    by_paper[r["paper_idx"]][section].append(r)

    if files_used:
        print(f"  [resume] ingesting: {', '.join(files_used)}")

    written = 0
    for pid, sec in by_paper.items():
        if pid in known:
            continue
        if not (sec["candidates"] or sec["sft"] or sec["dpo"]):
            continue
        write_shard(shard_dir, pid,
                    _dedup(sec["candidates"], "candidates"),
                    _dedup(sec["sft"], "sft"),
                    _dedup(sec["dpo"], "dpo"))
        written += 1
    return written


def merge_shards(shards: dict):
    """Flatten all shards into (candidates, sft, dpo), ordered by paper_idx."""
    cands, sft, dpo = [], [], []

    def _key(pid):
        try:
            return (0, int(pid))
        except (TypeError, ValueError):
            return (1, str(pid))

    for pid in sorted(shards, key=_key):
        cands.extend(shards[pid]["candidates"])
        sft.extend(shards[pid]["sft"])
        dpo.extend(shards[pid]["dpo"])
    return cands, sft, dpo


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def write_candidates_csv(all_candidates, path):
    cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id",
            "stepwise_score", "gt_similarity", "final_score",
            "search", "retrieve", "synthesize", "n_units"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in all_candidates:
            steps = (c.get("score_detail") or {}).get("steps", {})
            w.writerow([
                c.get("paper_idx"), c.get("agent_role"), c.get("agent_name"),
                c.get("mode"), c.get("rollout_id"),
                c.get("stepwise_score"), c.get("gt_similarity"), c.get("score"),
                steps.get("search"), steps.get("retrieve"), steps.get("synthesize"),
                (c.get("score_detail") or {}).get("n_units"),
            ])


def diagnostics(all_candidates, sft, dpo):
    print("\n" + "=" * 66)
    print("STEPWISE REWARD DIAGNOSTICS")
    print("=" * 66)

    by_role = defaultdict(list)
    for c in all_candidates:
        by_role[c["agent_role"]].append(c["score"])
    print(f"\n{'Role':<10}{'Cand':>7}{'Mean':>8}{'Std':>8}{'Min':>7}{'Max':>7}")
    for role, vals in sorted(by_role.items()):
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{role:<10}{len(vals):>7}{statistics.mean(vals):>8.3f}{std:>8.3f}"
              f"{min(vals):>7.3f}{max(vals):>7.3f}")

    sft_by = defaultdict(int)
    for s in sft:
        sft_by[s["agent_role"]] += 1
    dpo_by = defaultdict(int)
    gaps = []
    for d in dpo:
        dpo_by[d["agent_role"]] += 1
        gaps.append(d["score_gap"])

    print(f"\nSFT samples: {len(sft)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(sft_by.items())))
    print(f"DPO pairs:   {len(dpo)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(dpo_by.items())))
    if gaps:
        print(f"DPO gaps: mean={statistics.mean(gaps):.3f} min={min(gaps):.3f} max={max(gaps):.3f}")
    print("=" * 66)

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stepwise-reward scoring + SFT/DPO selection.")
    ap.add_argument("--rollout_json", default=DEFAULT_ROLLOUT_JSON,
                    help="Path to rollout_data_full.json.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                    help="Directory for the scored candidates + SFT/DPO datasets.")
    ap.add_argument("--judge_model", default=None,
                    help="Override the reward/judge model (default gpt-4o-mini).")
    ap.add_argument("--no_llm_judge", action="store_true",
                    help="Disable the GPT-4o-mini judge (ROUGE-L + lexical only).")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="Skip papers already scored (default ON).")
    ap.add_argument("--no_resume", dest="resume", action="store_false",
                    help="Re-score every paper from scratch.")
    ap.add_argument("--shard_dir", default=None,
                    help=f"Per-paper checkpoints (default <out_dir>/{SHARD_SUBDIR}).")
    ap.add_argument("--ingest_glob", action="append", default=None,
                    help="Extra JSON files in out_dir to ingest when backfilling "
                         "(repeatable, e.g. --ingest_glob 'sft_dataset_*_old.json').")
    args = ap.parse_args()

    global LLM_JUDGE_MODEL, USE_LLM_GT_JUDGE
    if args.judge_model:
        LLM_JUDGE_MODEL = args.judge_model
    if args.no_llm_judge:
        USE_LLM_GT_JUDGE = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reward/judge model: {LLM_JUDGE_MODEL} "
          f"(enabled={USE_LLM_GT_JUDGE and bool(os.environ.get('OPENAI_API_KEY'))})")

    papers = json.loads(Path(args.rollout_json).read_text())
    print(f"Loaded {len(papers)} papers from {args.rollout_json}")

    shard_dir = Path(args.shard_dir) if args.shard_dir else (out_dir / SHARD_SUBDIR)

    # ---------------- RESUME ----------------
    shards: dict = {}
    if args.resume:
        shards = load_shards(shard_dir)
        print(f"[resume] {len(shards)} papers already in {shard_dir}")
        n_back = backfill_shards_from_aggregates(
            out_dir, shard_dir, known=set(shards), extra_globs=args.ingest_glob)
        if n_back:
            print(f"[resume] backfilled {n_back} papers from existing aggregate files")
            shards = load_shards(shard_dir)
        done = set(shards)
        todo = [p for p in papers if p.get("paper_idx") not in done]
        skipped = len(papers) - len(todo)
        print(f"[resume] skipping {skipped} scored papers, scoring {len(todo)} new")
        if not todo:
            print("[resume] nothing new to score — rebuilding aggregates from shards only.")
    else:
        print("[resume] DISABLED (--no_resume): scoring every paper")
        todo = papers

    # ---------------- SCORE (checkpoint after every paper) ----------------
    for p in tqdm(todo, desc="Scoring papers (stepwise)"):
        pid = p.get("paper_idx")
        c_rows, s_rows, d_rows = process_dataset([p], progress=False)
        write_shard(shard_dir, pid, c_rows, s_rows, d_rows)
        shards[pid] = {"candidates": c_rows, "sft": s_rows, "dpo": d_rows}

    all_candidates, sft, dpo = merge_shards(shards)
    print(f"\nmerged {len(shards)} papers -> {len(all_candidates)} candidates, "
          f"{len(sft)} sft, {len(dpo)} dpo")

    (out_dir / "stepwise_candidate_scores.json").write_text(json.dumps(all_candidates, indent=2))
    if WRITE_CSV:
        write_candidates_csv(all_candidates, out_dir / "stepwise_candidate_scores.csv")

    (out_dir / "sft_dataset.json").write_text(json.dumps(sft, indent=2))
    for role in ("worker", "master", "leader"):
        rows = [s for s in sft if s["agent_role"] == role]
        (out_dir / f"sft_dataset_{role}.json").write_text(json.dumps(rows, indent=2))

    (out_dir / "dpo_dataset.json").write_text(json.dumps(dpo, indent=2))
    for role in ("worker",):
        rows = [d for d in dpo if d["agent_role"] == role]
        (out_dir / f"dpo_dataset_{role}.json").write_text(json.dumps(rows, indent=2))

    diagnostics(all_candidates, sft, dpo)

    print(f"\nOutputs written to: {out_dir}")
    print(f"  stepwise_candidate_scores.json  ({len(all_candidates)} candidates)")
    print(f"  sft_dataset.json                ({len(sft)})")
    print(f"  dpo_dataset.json                ({len(dpo)})")


if __name__ == "__main__":
    main()