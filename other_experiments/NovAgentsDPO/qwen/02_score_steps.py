"""
score_and_pair.py
=================
End-to-end:
  1. Load rollout JSONs from output/rollouts (paper_<row>_<strong|weak>.json)
  2. Look up the corresponding paper from df_updated_with_retrieval.csv
     (parse 'pdf_text_without_gt' with ast, get refs + ground_truth_lim_peer)
  3. Score each rollout step-wise -> composite score
  4. Save updated JSONs (preserving all original keys) to output/rollout_score
  5. Save best rollout per paper to output/sft_data
  6. Build DPO pairs (best vs worst) to output/dpo_pair
"""
import os
import re
import ast
import json
import glob
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
ROLLOUTS_DIR = "other_experiments/dpo_novagents/output/rollouts"
PAPER_CSV    = "data/balanced_data/df_updated_with_retrieval.csv"

OUT_BASE         = "other_experiments/dpo_novagents/output"
ROLLOUT_SCORE_DIR = os.path.join(OUT_BASE, "rollout_score")
SFT_DATA_DIR      = os.path.join(OUT_BASE, "sft_data")
DPO_PAIR_DIR      = os.path.join(OUT_BASE, "dpo_pair")

for d in [ROLLOUT_SCORE_DIR, SFT_DATA_DIR, DPO_PAIR_DIR]:
    os.makedirs(d, exist_ok=True)

# Aggregation weights for the four steps -> composite score
STEP_WEIGHTS = {
    "claim_extraction":     0.15,
    "novelty_technical":    0.25,
    "experimental_scope":   0.25,
    "limitation_synthesis": 0.35,
}

JUDGE_MODEL     = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_CACHE        = "hf_cache"

USE_LLM_JUDGE  = True   # set False to skip OpenAI calls
MIN_REWARD_GAP = 0.05   # min gap between chosen/rejected to keep a DPO pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# DATA LOADING HELPERS
# ============================================================
def chunk_text(text: str, n: int = 150) -> List[str]:
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]

def parse_pdf_text(raw: str) -> Dict:
    """Convert pdf_text_without_gt string -> dict via ast (fallback to json)."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except Exception:
            return {}

def extract_paper_data(row: pd.Series) -> Dict:
    """Pull text/sections/references/ground_truth from a CSV row."""
    raw = None
    for col in ["pdf_text_without_gt", "pdf_text"]:
        if col in row and pd.notna(row[col]):
            raw = str(row[col])
            break
    if raw is None:
        return {}

    parsed = parse_pdf_text(raw)
    if not isinstance(parsed, dict):
        return {}

    abstract_text = parsed.get("abstractText", "") or ""
    raw_sections  = parsed.get("sections", []) or []
    sections, parts = {}, []

    if abstract_text:
        sections["abstract"] = abstract_text
        parts.append(abstract_text)
    for sec in raw_sections:
        if isinstance(sec, dict):
            h = (sec.get("heading", "unknown") or "unknown").strip()
            t = sec.get("text", "") or ""
            if t:
                sections[h.lower()] = t
                parts.append(f"{h}\n{t}")
    full_text = "\n\n".join(parts)

    refs = []
    for r in parsed.get("references", []) or []:
        if isinstance(r, dict) and r.get("title"):
            refs.append(r["title"])

    gt_lim = ""
    for col in ["ground_truth_lim_peer", "limitations", "gt_limitations", "ground_truth"]:
        if col in row and pd.notna(row[col]):
            gt_lim = str(row[col])
            break

    return {
        "title":            parsed.get("title", ""),
        "text":             full_text,
        "abstract":         abstract_text,
        "sections":         sections,
        "references":       refs,
        "ground_truth_lim": gt_lim,
    }

# ============================================================
# STEP-WISE REWARD SCORER
# ============================================================
class StepRewardScorer:
    def __init__(self):
        self._emb = None
        self._oai = None

    @property
    def emb(self):
        if self._emb is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._emb = SentenceTransformer(EMBEDDING_MODEL, cache_folder=HF_CACHE)
        return self._emb

    @property
    def oai(self):
        if self._oai is None:
            import openai
            self._oai = openai.OpenAI()
        return self._oai

    @staticmethod
    def _cos(a, b):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(a, b)

    # ---------- Step 1: Claim Extraction ----------
    def score_claim_extraction(self, claims, paper_text, sections):
        keys = ["format", "coverage", "faithfulness", "specificity", "total"]
        if not claims or not isinstance(claims, list):
            return {k: 0.0 for k in keys}

        required = {"claim", "section", "type", "evidence_quote"}
        fmt = [1.0 if isinstance(c, dict) and required.issubset(c.keys()) else 0.0 for c in claims]
        format_score = float(np.mean(fmt))

        mentioned = {str(c.get("section", "")).lower() for c in claims if isinstance(c, dict)}
        key_secs = {"abstract", "introduction", "method", "methodology", "results",
                    "experiment", "discussion", "conclusion", "related work"}
        avail = {s.lower() for s in sections} if sections else set()
        target = key_secs & avail if avail else key_secs
        coverage = min(len(mentioned & target) / max(len(target), 1), 1.0)

        claim_texts = [str(c.get("claim", "")) for c in claims
                       if isinstance(c, dict) and c.get("claim")]
        if claim_texts and paper_text:
            chunks = chunk_text(paper_text)[:20]
            sims = self._cos(self.emb.encode(claim_texts), self.emb.encode(chunks)).max(axis=1)
            faithfulness = float(np.mean(sims))
        else:
            faithfulness = 0.0

        spec = []
        for c in claims:
            if not isinstance(c, dict):
                continue
            t = str(c.get("claim", ""))
            n = bool(re.search(r'\d+\.?\d*\s*%?', t))
            m = bool(re.search(r'(algorithm|model|framework|approach|network|transformer|encoder|decoder|architecture)', t, re.I))
            cmp_ = bool(re.search(r'(outperform|improve|better|faster|state-of-the-art|SOTA|baseline|benchmark|accuracy|F1)', t, re.I))
            spec.append(0.4 * n + 0.3 * m + 0.3 * cmp_)
        specificity = float(np.mean(spec)) if spec else 0.0

        total = 0.10*format_score + 0.15*coverage + 0.50*faithfulness + 0.25*specificity
        return {"format": format_score, "coverage": coverage,
                "faithfulness": faithfulness, "specificity": specificity, "total": total}

    # ---------- Step 2: Novelty / Technical ----------
    def score_novelty_technical(self, parsed, paper_text, references):
        keys = ["format", "grounding", "reference_overlap", "specificity", "total"]
        text = parsed if isinstance(parsed, str) else str(parsed or "")
        if len(text) < 50:
            return {k: 0.0 for k in keys}

        has_nov = bool(re.search(r'Novelty Score[:\s]*\d', text, re.I))
        has_rig = bool(re.search(r'Rigor Score[:\s]*\d', text, re.I))
        has_evd = bool(re.search(r'Evidence', text, re.I))
        format_score = (has_nov + has_rig + has_evd) / 3.0

        if paper_text:
            chunks = chunk_text(paper_text)[:20]
            sim = self._cos(self.emb.encode([text[:2000]]), self.emb.encode(chunks)).max(axis=1)
            grounding = float(np.mean(sim))
        else:
            grounding = 0.0

        if references:
            tl = text.lower()
            hits = sum(1 for r in references[:30] if r and len(r) > 5 and r.lower()[:30] in tl)
            ref_overlap = min(hits / 3.0, 1.0)
        else:
            ref_overlap = 0.5

        has_concrete = bool(re.search(r'(specifically|concretely|in particular|for example|e\.g\.)', text, re.I))
        has_compare  = bool(re.search(r'(compared to|in contrast|whereas|while|differs from|similar to)', text, re.I))
        length_score = min(len(text.split()) / 200.0, 1.0)
        specificity  = 0.3*has_concrete + 0.3*has_compare + 0.4*length_score

        total = 0.20*format_score + 0.30*grounding + 0.25*ref_overlap + 0.25*specificity
        return {"format": format_score, "grounding": grounding,
                "reference_overlap": ref_overlap, "specificity": specificity, "total": total}

    # ---------- Step 3: Experimental Scope ----------
    def score_experimental_scope(self, parsed, paper_text):
        keys = ["format", "grounding", "completeness", "specificity", "total"]
        text = parsed if isinstance(parsed, str) else str(parsed or "")
        if len(text) < 50:
            return {k: 0.0 for k in keys}

        scores_found = len(re.findall(
            r'(?:Validation|Contextualization|Generalizability) Score[:\s]*\d', text, re.I))
        format_score = min(scores_found / 3.0, 1.0)

        if paper_text:
            chunks = chunk_text(paper_text)[:20]
            sim = self._cos(self.emb.encode([text[:2000]]), self.emb.encode(chunks)).max(axis=1)
            grounding = float(np.mean(sim))
        else:
            grounding = 0.0

        dims = ["validation", "contextualization", "generalizability", "scope"]
        completeness = sum(1 for d in dims if d in text.lower()) / len(dims)

        has_evd = bool(re.search(r'Evidence', text, re.I))
        has_lim = bool(re.search(r'limitation', text, re.I))
        length_score = min(len(text.split()) / 200.0, 1.0)
        specificity = 0.3*has_evd + 0.3*has_lim + 0.4*length_score

        total = 0.20*format_score + 0.25*grounding + 0.30*completeness + 0.25*specificity
        return {"format": format_score, "grounding": grounding,
                "completeness": completeness, "specificity": specificity, "total": total}

    # ---------- Step 4: Limitation Synthesis ----------
    def score_limitation_synthesis(self, lims, paper_text, gt_lim, use_judge=True):
        keys = ["specificity", "grounding", "non_redundancy", "gt_coverage", "soundness", "total"]
        if not lims or not isinstance(lims, list):
            return {k: 0.0 for k in keys}

        spec = []
        for l in lims:
            if not isinstance(l, dict):
                continue
            t = str(l.get("limitation", ""))
            has_concrete = bool(re.search(
                r'(specific|particular|e\.g\.|for example|such as|Table|Figure|Section|equation|\d+\.?\d*%?|dataset|benchmark|baseline|ablation)',
                t, re.I))
            has_cat = bool(l.get("category"))
            has_sev = bool(l.get("severity"))
            ls = min(len(t.split()) / 35.0, 1.0)
            spec.append(0.35*has_concrete + 0.15*has_cat + 0.10*has_sev + 0.40*ls)
        specificity = float(np.mean(spec)) if spec else 0.0

        lim_texts = [str(l.get("limitation", "")) for l in lims if isinstance(l, dict)]

        if lim_texts and paper_text:
            chunks = chunk_text(paper_text)[:20]
            sim = self._cos(self.emb.encode(lim_texts[:10]), self.emb.encode(chunks)).max(axis=1)
            grounding = float(np.mean(sim))
        else:
            grounding = 0.0

        if len(lim_texts) > 1:
            le = self.emb.encode(lim_texts[:10])
            pw = self._cos(le, le)
            np.fill_diagonal(pw, 0)
            non_redundancy = float(1.0 - pw.mean())
        else:
            non_redundancy = 0.5

        gt_coverage = 0.0
        if gt_lim and lim_texts:
            gt_items = [s.strip(" -*\t") for s in re.split(r'\n+', gt_lim) if len(s.strip()) > 10]
            if gt_items:
                le = self.emb.encode(lim_texts[:10])
                ge = self.emb.encode(gt_items[:15])
                # for each GT item, max sim to any generated limitation
                gt_coverage = float(np.mean(self._cos(le, ge).max(axis=0)))

        soundness = self._llm_judge(lims, paper_text, gt_lim) if use_judge else 0.5

        total = (0.20*specificity + 0.20*grounding + 0.15*non_redundancy +
                 0.20*gt_coverage + 0.25*soundness)
        return {"specificity": specificity, "grounding": grounding,
                "non_redundancy": non_redundancy, "gt_coverage": gt_coverage,
                "soundness": soundness, "total": total}

    def _llm_judge(self, lims, paper_text, gt_lim):
        prompt = f"""Rate these scientific limitations on a 1-5 scale.

Paper excerpt: {paper_text[:500]}

Ground-truth peer-review limitations:
{gt_lim[:800] if gt_lim else "(none provided)"}

Generated Limitations:
{json.dumps(lims[:5], indent=1, default=str)[:1500]}

For EACH limitation, rate (1-5):
- soundness: logical validity
- specificity: paper-specific?
- actionability: addressable?
- gt_alignment: matches ground-truth concerns?

Return ONLY JSON:
{{"scores": [{{"idx": 0, "soundness": X, "specificity": X, "actionability": X, "gt_alignment": X}}]}}"""
        try:
            resp = self.oai.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=500,
            )
            raw = re.sub(r'```json|```', '', resp.choices[0].message.content.strip()).strip()
            scores = json.loads(raw).get("scores", [])
            if scores:
                avg = np.mean([
                    np.mean([s.get("soundness", 3), s.get("specificity", 3),
                             s.get("actionability", 3), s.get("gt_alignment", 3)])
                    for s in scores
                ])
                return float(avg / 5.0)
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
        return 0.5

# ============================================================
# TRAJECTORY SCORING
# ============================================================
def score_trajectory(traj: Dict, paper: Dict, scorer: StepRewardScorer,
                     use_judge: bool = True) -> Dict:
    steps = traj.get("steps", {})
    claims = steps.get("claim_extraction", {}).get("parsed", [])
    nt     = steps.get("novelty_technical", {}).get("parsed", "")
    es     = steps.get("experimental_scope", {}).get("parsed", "")
    lims   = steps.get("limitation_synthesis", {}).get("parsed", [])

    s1 = scorer.score_claim_extraction(claims, paper["text"], paper["sections"])
    s2 = scorer.score_novelty_technical(nt, paper["text"], paper["references"])
    s3 = scorer.score_experimental_scope(es, paper["text"])
    s4 = scorer.score_limitation_synthesis(lims, paper["text"],
                                            paper["ground_truth_lim"], use_judge)

    step_scores = {
        "claim_extraction":     s1,
        "novelty_technical":    s2,
        "experimental_scope":   s3,
        "limitation_synthesis": s4,
    }
    composite = sum(STEP_WEIGHTS[k] * step_scores[k]["total"] for k in STEP_WEIGHTS)

    out = dict(traj)                      # preserve every original key
    out["step_scores"]     = step_scores
    out["composite_score"] = float(composite)
    return out

def parse_filename(fp: str) -> Tuple[str, str]:
    """paper_435_strong.json -> ('435', 'strong')"""
    base = os.path.basename(fp).replace(".json", "")
    parts = base.split("_")
    return (parts[1], parts[2]) if len(parts) >= 3 else (None, None)

def build_prompt(traj: Dict) -> str:
    return traj.get("steps", {}).get("claim_extraction", {}).get("prompt", "")

def build_response(traj: Dict) -> str:
    parts = []
    for s in ["claim_extraction", "novelty_technical",
              "experimental_scope", "limitation_synthesis"]:
        out = traj.get("steps", {}).get(s, {}).get("raw_output", "")
        if out:
            parts.append(f"=== {s} ===\n{out}")
    return "\n\n".join(parts)

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"Loading paper CSV: {PAPER_CSV}")
    df = pd.read_csv(PAPER_CSV)
    logger.info(f"  {len(df)} rows")

    all_files = sorted(glob.glob(os.path.join(ROLLOUTS_DIR, "paper_*.json")))
    logger.info(f"Found {len(all_files)} rollout files in {ROLLOUTS_DIR}")

    # Group files by paper id (strong + weak share the same id)
    by_paper: Dict[str, List[str]] = {}
    for fp in all_files:
        pid, _ = parse_filename(fp)
        if pid is not None:
            by_paper.setdefault(pid, []).append(fp)

    paper_ids = sorted(by_paper.keys(), key=lambda x: int(x) if x.isdigit() else 0)

    # ===== TESTING: process only the first paper =====
    # Comment out the next line to process ALL papers:
    # paper_ids = paper_ids[435:436]
    # ==================================================

    logger.info(f"Processing {len(paper_ids)} paper(s)")

    scorer = StepRewardScorer()

    for pid in paper_ids:
        try:
            row_idx = int(pid)
        except ValueError:
            logger.warning(f"  Bad paper id: {pid}")
            continue
        if row_idx >= len(df):
            logger.warning(f"  paper id {pid} out of CSV range")
            continue

        paper = extract_paper_data(df.iloc[row_idx])
        if not paper.get("text"):
            logger.warning(f"  No text for paper {pid}, skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"PAPER {pid}: {paper.get('title', '')[:80]}")
        logger.info(f"  refs={len(paper['references'])} | gt_lim={'yes' if paper['ground_truth_lim'] else 'no'}")
        logger.info("=" * 60)

        all_trajs: List[Dict] = []

        # ---- Step 1: score every rollout in every file for this paper ----
        for fp in by_paper[pid]:
            logger.info(f"  Reading {os.path.basename(fp)}")
            with open(fp) as f:
                rollouts = json.load(f)

            scored_list = []
            for traj in rollouts:
                logger.info(f"    Scoring {traj.get('model_type')} r{traj.get('rollout_idx')}")
                scored = score_trajectory(traj, paper, scorer, USE_LLM_JUDGE)
                logger.info(
                    f"      S1={scored['step_scores']['claim_extraction']['total']:.3f}  "
                    f"S2={scored['step_scores']['novelty_technical']['total']:.3f}  "
                    f"S3={scored['step_scores']['experimental_scope']['total']:.3f}  "
                    f"S4={scored['step_scores']['limitation_synthesis']['total']:.3f}  "
                    f"=> COMPOSITE={scored['composite_score']:.4f}"
                )
                scored_list.append(scored)
                all_trajs.append(scored)

            # Save updated json (preserves all original keys + adds scores)
            out_path = os.path.join(ROLLOUT_SCORE_DIR, os.path.basename(fp))
            with open(out_path, "w") as f:
                json.dump(scored_list, f, indent=2, default=str)
            logger.info(f"  -> Saved {out_path}")

        if not all_trajs:
            continue

        # ---- Step 2: best (SFT) and worst (for DPO) ----
        all_trajs.sort(key=lambda t: t["composite_score"], reverse=True)
        best, worst = all_trajs[0], all_trajs[-1]

        sft_path = os.path.join(SFT_DATA_DIR, f"paper_{pid}_best.json")
        with open(sft_path, "w") as f:
            json.dump(best, f, indent=2, default=str)
        logger.info(f"  SFT best -> {sft_path} (score={best['composite_score']:.4f})")

        gap = best["composite_score"] - worst["composite_score"]
        if gap >= MIN_REWARD_GAP:
            pair = {
                "paper_id":       pid,
                "prompt":         build_prompt(best),
                "chosen":         build_response(best),
                "rejected":       build_response(worst),
                "chosen_score":   best["composite_score"],
                "rejected_score": worst["composite_score"],
                "score_gap":      gap,
                "chosen_meta":  {"model_type": best.get("model_type"),
                                 "rollout_idx": best.get("rollout_idx"),
                                 "step_scores": best.get("step_scores")},
                "rejected_meta": {"model_type": worst.get("model_type"),
                                  "rollout_idx": worst.get("rollout_idx"),
                                  "step_scores": worst.get("step_scores")},
            }
            dpo_path = os.path.join(DPO_PAIR_DIR, f"paper_{pid}_pair.json")
            with open(dpo_path, "w") as f:
                json.dump(pair, f, indent=2, default=str)
            logger.info(f"  DPO pair -> {dpo_path} (gap={gap:.4f})")
        else:
            logger.info(f"  Skipping DPO pair: gap {gap:.4f} < {MIN_REWARD_GAP}")

    logger.info("\nDONE")

if __name__ == "__main__":
    main()