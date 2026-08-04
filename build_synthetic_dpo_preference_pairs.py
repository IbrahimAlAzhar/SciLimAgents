"""
build_synthetic_dpo_preference_pairs.py
======================

Builds a SECOND worker DPO dataset ("augmented") from the one that
`select_data_stepwise_process_reward.py` already produced. That script is NOT modified or
re-run; this one only consumes its output.

    dpo_dataset_worker.json            <- TYPE A (natural)   : existing, untouched
    dpo_dataset_worker_augmented.json  <- TYPE B (this script)

RECIPE
------
1. POOL = every pair in `dpo_dataset_worker.json`. Those already cleared the
   reward model's `DPO_MIN_GAP` (0.05) and `SFT_MIN_SCORE` (0.30) filters, so the
   pool is exactly the set of groups with a real, non-noise preference signal.
   (`--min_gap` can tighten it further without re-running the scorer.)
2. Sort the pool by `score_gap` descending.
3. **Top 70%**  -> keep the natural pair unchanged (chosen = best real candidate,
   rejected = worst real candidate).
4. **Bottom 30%** -> keep the SAME chosen sample, but REPLACE the rejected half
   with a gpt-4o-mini generation that is on-topic and fluent yet ungrounded:
   generic claims, fabricated datasets/numbers/baselines, invented evidence IDs.
   These are the pairs whose natural rejection was barely worse than the chosen
   one, so the natural preference carried the least information.

WHY THIS IS A CLEAN ABLATION
----------------------------
TYPE A and TYPE B end up with the SAME number of pairs, over the SAME groups,
with the SAME chosen samples and the SAME prompts. The only difference is the
rejected half of the bottom 30%. So a difference between the two DPO models is
attributable to the synthetic rejections and not to dataset size or composition
— no separate size-matched control is needed.

If a synthetic generation fails the quality gate, that group falls back to its
natural pair (`pair_type: "natural_fallback"`) so the two datasets stay
identical in size. The count is reported.

QUALITY GATE (every generation)
-------------------------------
  * parses into >= MIN_BULLETS bullets
  * not a near-copy of chosen (token Jaccard < MAX_COPY_JACCARD)
  * `score_synthesize` grounding score at least MIN_SYNTH_MARGIN BELOW chosen
Up to MAX_GEN_RETRIES attempts at rising temperature. This stops the generator
from accidentally producing something *good* and inverting the preference.

Grounding/prompt helpers are imported from `stepwise_reward_select`, so "grounded"
has one definition across the reward model and this gate.

USAGE
-----
    python build_synthetic_dpo_preference_pairs.py \
        --natural_dpo  $DIR/select/dpo_dataset_worker.json \
        --rollout_json $DIR/rollouts/rollout_data_full.json \
        --out          $DIR/select/dpo_dataset_worker_augmented.json \
        --natural_frac 0.70

    # offline plumbing test (no API):
    python build_synthetic_dpo_preference_pairs.py --no_api --allow_fallback ...
"""

from __future__ import annotations

import os
import re
import csv
import json
import random
import hashlib
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# Single source of truth for prompts + grounding scores.
import stepwise_reward_select as S

# =============================================================================
# CONFIG
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


ROOT = _require_env("SELECT_DIR", "directory holding the selected preference datasets")
DEFAULT_NATURAL_DPO = os.path.join(ROOT, "dpo_dataset_worker.json")
DEFAULT_OUT         = os.path.join(ROOT, "dpo_dataset_worker_augmented.json")
DEFAULT_CANDIDATES  = os.path.join(ROOT, "stepwise_candidate_scores.json")  # reporting only
DEFAULT_ROLLOUT_JSON = os.environ.get("ROLLOUT_JSON")   # or pass --rollout_json

# Fraction of the pool (ranked by gap, descending) that keeps its natural pair.
NATURAL_FRAC = float(os.environ.get("NATURAL_FRAC", 0.70))
# Optional extra floor on top of whatever DPO_MIN_GAP the scorer used.
MIN_GAP = os.environ.get("MIN_GAP")
MIN_GAP = float(MIN_GAP) if MIN_GAP not in (None, "", "none", "None") else None

GEN_MODEL       = os.environ.get("GEN_MODEL", "gpt-4o-mini")
GEN_TEMP        = float(os.environ.get("GEN_TEMP", 0.95))
GEN_MAX_TOKENS  = int(os.environ.get("GEN_MAX_TOKENS", 900))
MAX_GEN_RETRIES = int(os.environ.get("MAX_GEN_RETRIES", 3))

MIN_BULLETS         = int(os.environ.get("MIN_BULLETS", 3))
MAX_COPY_JACCARD    = float(os.environ.get("MAX_COPY_JACCARD", 0.80))
MIN_SYNTH_MARGIN    = float(os.environ.get("MIN_SYNTH_MARGIN", 0.05))
PAPER_EXCERPT_CHARS = int(os.environ.get("PAPER_EXCERPT_CHARS", 6000))

GEN_CACHE_DIR = Path(os.environ.get(
    "GEN_CACHE_DIR", str(Path(ROOT).parent / "synth_cache")))

WORKER_SPECIALTY_LABEL = {
    "Novelty_Significance_Agent":                     "novelty and significance",
    "Theoretical_Methodological_Agent":               "theoretical and methodological soundness (including ablations)",
    "Experimental_Evaluation_Agent":                  "experimental evaluation, baselines, and metrics",
    "Generalization_Robustness_Efficiency_Agent":     "generalization, robustness, efficiency, and applicability",
    "Clarity_Interpretability_Reproducibility_Agent": "clarity, interpretability, and reproducibility",
    "Data_Ethics_Agent":                              "data integrity, bias, fairness, and ethics",
}

# =============================================================================
# SYNTHETIC "PLAUSIBLE BUT UNGROUNDED" REJECTION
# =============================================================================

_SYS_PROMPT = (
    "You produce deliberately WEAK peer-review limitation lists. They are used as "
    "the rejected half of a preference-learning pair, so a model can learn to "
    "prefer evidence-grounded criticism over confident-sounding but unsupported "
    "criticism. Never add disclaimers, meta-commentary, or any hint that the "
    "output is intentionally flawed. Return only the bullet list."
)

def _gen_user_prompt(paper_excerpt: str, specialty: str, chosen: str,
                     n_bullets: int, uses_evidence_ids: bool) -> str:
    style = (
        "Cite invented evidence IDs in the form [E##] to match the good answer's "
        "style — the numbers must NOT correspond to anything real in the paper.\n"
        if uses_evidence_ids else
        "Do not use evidence-ID markers.\n"
    )
    return (
        f"SPECIALTY AREA: {specialty}\n\n"
        f"PAPER EXCERPT (the only real source material):\n{paper_excerpt}\n\n"
        f"A STRONG, WELL-GROUNDED review of this paper in that area looks like:\n{chosen}\n\n"
        "Now write a WEAK version: it must read as fluent, confident, on-topic "
        "peer review, but must fail on substance. Requirements:\n"
        f"- Exactly {n_bullets} bullets, same markdown bullet format, similar length "
        "per bullet as the strong version.\n"
        "- Stay inside the specialty area above so it looks like the right kind of critique.\n"
        "- Make the criticisms GENERIC — they should apply to almost any paper in the "
        "field and not to this paper's actual content.\n"
        "- FABRICATE specifics: dataset names, benchmark names, baseline methods, "
        "percentages, and table/section references that do NOT appear in the excerpt.\n"
        "- Where you do reference the paper, misattribute it: claim it did something it "
        "did not, or omitted something it actually reported.\n"
        "- Provide no real quotes or verifiable anchors from the excerpt.\n"
        f"{style}"
        "Return ONLY the bullet list."
    )

_OAI = None
def _client():
    global _OAI
    if _OAI is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        kw = {"api_key": key, "timeout": 120}
        base = os.environ.get("JUDGE_BASE_URL")
        if base:
            kw["base_url"] = base
        _OAI = OpenAI(**kw)
    return _OAI


def _fallback_bad_sample(chosen: str, specialty: str) -> str:
    """Offline, API-free degradation — plumbing tests only, never for training."""
    fake = ["the CIFAR-Bench suite", "the OpenEval-2M corpus", "the NeuroBase-7 baseline",
            "Table 9", "Appendix D.4", "a 12.7% drop", "the DistilCompare baseline"]
    out = []
    for i, b in enumerate(S.split_bullets(chosen)):
        b = re.sub(r"\[E\d+\]", f"[E{random.randint(400, 999)}]", b)
        b = re.sub(r'["“][^"”]{10,}["”]', "", b)
        out.append(f"- The work does not adequately address {specialty}; results on "
                   f"{random.choice(fake)} are omitted and {random.choice(fake)} "
                   f"is not discussed, which several related efforts consider standard. "
                   f"{b[:120]}")
        if i >= 6:
            break
    return "\n".join(out)


def generate_bad_sample(paper_excerpt: str, specialty: str, chosen: str,
                        source_text: str, chosen_synth_score: float,
                        use_api: bool, allow_fallback: bool):
    """Returns (rejected_text, meta) or (None, meta) if every attempt failed."""
    chosen_bullets = S.split_bullets(chosen)
    n_bullets = max(MIN_BULLETS, min(8, len(chosen_bullets) or 5))
    uses_ids = bool(re.search(r"\[E\d+\]", chosen))
    meta = {"attempts": 0, "gen_method": None, "reject_reasons": []}

    GEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = _client() if use_api else None

    for attempt in range(MAX_GEN_RETRIES if client else 1):
        meta["attempts"] = attempt + 1
        temp = min(1.15, GEN_TEMP + 0.1 * attempt)

        if client:
            key = hashlib.sha256(
                (GEN_MODEL + specialty + chosen[:3000] + paper_excerpt[:1500]
                 + str(attempt)).encode()).hexdigest()[:24]
            cpath = GEN_CACHE_DIR / f"bad_{key}.json"
            cand = None
            if cpath.exists():
                try:
                    cand = json.loads(cpath.read_text())["text"]
                except Exception:
                    cand = None
            if cand is None:
                try:
                    resp = client.chat.completions.create(
                        model=GEN_MODEL, temperature=temp, max_tokens=GEN_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": _SYS_PROMPT},
                            {"role": "user", "content": _gen_user_prompt(
                                paper_excerpt, specialty, chosen, n_bullets, uses_ids)},
                        ],
                    )
                    cand = (resp.choices[0].message.content or "").strip()
                    cpath.write_text(json.dumps({"text": cand}))
                except Exception as e:
                    meta["reject_reasons"].append(f"api_error:{type(e).__name__}")
                    continue
            method = f"{GEN_MODEL}@T{temp:.2f}"
        elif allow_fallback:
            cand = _fallback_bad_sample(chosen, specialty)
            method = "fallback_template"
        else:
            meta["reject_reasons"].append("no_api_and_no_fallback")
            return None, meta

        # ---- quality gate ----
        cand = re.sub(r"<think>.*?</think>", " ", cand,
                      flags=re.DOTALL | re.IGNORECASE).strip()
        bullets = S.split_bullets(cand)
        if len(bullets) < MIN_BULLETS:
            meta["reject_reasons"].append(f"too_few_bullets:{len(bullets)}")
            continue
        copy_j = S._overlap(cand, chosen)
        if copy_j >= MAX_COPY_JACCARD:
            meta["reject_reasons"].append(f"near_copy:{copy_j:.2f}")
            continue
        synth_score, _ = S.score_synthesize(S.build_units(cand), source_text)
        if synth_score > chosen_synth_score - MIN_SYNTH_MARGIN:
            meta["reject_reasons"].append(
                f"not_worse:{synth_score:.3f}vs{chosen_synth_score:.3f}")
            continue

        meta.update({"gen_method": method, "rejected_synth_score": synth_score,
                     "chosen_synth_score": chosen_synth_score,
                     "copy_jaccard": round(copy_j, 4), "n_bullets": len(bullets)})
        return cand, meta

    return None, meta

# =============================================================================
# IO HELPERS
# =============================================================================

def _steps(row: dict, side: str):
    """Per-component breakdown the reward model stored on this pair, if any."""
    d = row.get(f"{side}_score_detail") or {}
    return d.get("steps") or {}


def write_pairs_csv(rows: list[dict], path: Path):
    """Flat audit trail: every component of every score, both sides of the pair."""
    cols = ["paper_idx", "agent_name", "pair_type", "gap_rank",
            "chosen_mode", "rejected_mode",
            "chosen_score", "rejected_score", "score_gap",
            "chosen_jsd", "chosen_judge", "chosen_grounding",
            "rejected_jsd", "rejected_judge", "rejected_grounding",
            "chosen_score_basis", "rejected_score_basis",
            "synth_score_gap", "copy_jaccard", "gen_method", "gen_attempts"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            cs, rs = _steps(r, "chosen"), _steps(r, "rejected")
            w.writerow([
                r.get("paper_idx"), r.get("agent_name"), r.get("pair_type"),
                r.get("gap_rank"), r.get("chosen_mode"), r.get("rejected_mode"),
                r.get("chosen_score"), r.get("rejected_score"), r.get("score_gap"),
                cs.get("stage1_jsd_reward", cs.get("jsd")), cs.get("stage2_judge"),
                cs.get("grounding"),
                rs.get("stage1_jsd_reward", rs.get("jsd")), rs.get("stage2_judge"),
                rs.get("grounding"),
                r.get("chosen_score_basis"), r.get("rejected_score_basis"),
                r.get("synth_score_gap"), r.get("copy_jaccard"),
                r.get("gen_method"), r.get("gen_attempts"),
            ])

# =============================================================================
# MAIN
# =============================================================================

def main():
    global GEN_MODEL
    ap = argparse.ArgumentParser(
        description="TYPE B worker DPO set: top-X%% natural pairs + synthetic rejections "
                    "for the rest, same size/groups/chosen as TYPE A.")
    ap.add_argument("--natural_dpo", default=DEFAULT_NATURAL_DPO,
                    help="TYPE A dataset from select_data_stepwise_process_reward.py — this IS the pool.")
    ap.add_argument("--rollout_json", default=DEFAULT_ROLLOUT_JSON,
                    help="Needed for paper text (the generator's source material).")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--natural_frac", type=float, default=NATURAL_FRAC,
                    help="Top fraction by gap that keeps its natural rejection (default 0.70).")
    ap.add_argument("--min_gap", type=float, default=MIN_GAP,
                    help="Optional extra gap floor on the pool (scorer already applied 0.05).")
    ap.add_argument("--gen_model", default=GEN_MODEL)
    ap.add_argument("--no_api", action="store_true", help="Do not call the generator model.")
    ap.add_argument("--allow_fallback", action="store_true",
                    help="Permit the offline template generator (plumbing tests only).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    GEN_MODEL = args.gen_model

    use_api = not args.no_api and bool(os.environ.get("OPENAI_API_KEY"))
    if not use_api and not args.allow_fallback:
        raise SystemExit(
            "No generator available: set OPENAI_API_KEY, or pass --allow_fallback "
            "to use the offline template degradation (plumbing tests only).")

    # ---------- load the pool ----------
    pool = json.loads(Path(args.natural_dpo).read_text())
    pool = [r for r in pool
            if r.get("agent_role", "worker") == "worker"
            and (r.get("chosen") or "").strip()
            and (r.get("rejected") or "").strip()]
    if not pool:
        raise SystemExit(f"No usable worker pairs in {args.natural_dpo}")

    gaps_in = [r.get("score_gap", 0.0) for r in pool]
    if args.min_gap is not None:
        before = len(pool)
        pool = [r for r in pool if r.get("score_gap", 0.0) >= args.min_gap]
        print(f"extra --min_gap {args.min_gap}: {before} -> {len(pool)} pairs")

    papers = json.loads(Path(args.rollout_json).read_text())
    src_by_pid = {}
    for p in papers:
        src = p.get("paper_text", "") or ""
        if p.get("citation_text"):
            src = src + "\n\n" + p["citation_text"]
        src_by_pid[p.get("paper_idx")] = src

    # ---------- rank by gap, split 70 / 30 ----------
    pool.sort(key=lambda r: r.get("score_gap", 0.0), reverse=True)
    n_total = len(pool)
    n_natural = int(round(args.natural_frac * n_total))
    n_natural = max(0, min(n_total, n_natural))
    top, bottom = pool[:n_natural], pool[n_natural:]

    print("=" * 70)
    print("AUGMENTED WORKER DPO BUILDER  (paired 70/30 design)")
    print("=" * 70)
    print(f"pool (TYPE A)  : {n_total} pairs   {args.natural_dpo}")
    print(f"gap in pool    : min={min(gaps_in):.3f} max={max(gaps_in):.3f} "
          f"mean={statistics.mean(gaps_in):.3f}")
    print(f"split          : top {args.natural_frac:.0%} natural = {len(top)}   |   "
          f"bottom {1-args.natural_frac:.0%} synthetic = {len(bottom)}")
    if bottom:
        print(f"boundary gap   : {top[-1].get('score_gap') if top else float('nan'):.3f} "
              f"(last natural) / {bottom[0].get('score_gap'):.3f} (first synthetic)")
    print(f"generator      : {GEN_MODEL if use_api else 'OFFLINE TEMPLATE (not for training)'}")
    print(f"out            : {args.out}\n")

    out_rows: list[dict] = []

    # ---------- top 70%: natural, unchanged ----------
    for rank, r in enumerate(top, 1):
        row = dict(r)
        row["pair_type"] = "natural"
        row["gap_rank"] = rank
        # both sides were scored by the same reward model, on the same scale
        row["chosen_score_basis"] = "reward_model"
        row["rejected_score_basis"] = "reward_model"
        out_rows.append(row)

    # ---------- bottom 30%: same chosen, synthetic rejected ----------
    n_synth, n_fallback = 0, 0
    fail_reasons = defaultdict(int)

    for rank, r in enumerate(tqdm(bottom, desc="Synthetic rejections"), len(top) + 1):
        pid = r.get("paper_idx")
        name = r.get("agent_name")
        chosen = r["chosen"]
        source_text = src_by_pid.get(pid, "")
        specialty = WORKER_SPECIALTY_LABEL.get(name, "the paper's limitations")

        rejected, meta = (None, {"reject_reasons": ["no_source_text"], "attempts": 0})
        if source_text:
            chosen_synth, _ = S.score_synthesize(S.build_units(chosen), source_text)
            rejected, meta = generate_bad_sample(
                paper_excerpt=source_text[:PAPER_EXCERPT_CHARS],
                specialty=specialty, chosen=chosen, source_text=source_text,
                chosen_synth_score=chosen_synth,
                use_api=use_api, allow_fallback=args.allow_fallback,
            )

        row = dict(r)
        row["gap_rank"] = rank
        row["chosen_score_basis"] = "reward_model"
        if rejected is None:
            # Keep the natural pair so both datasets stay the same size.
            row["pair_type"] = "natural_fallback"
            row["rejected_score_basis"] = "reward_model"
            row["synth_failed_reasons"] = meta.get("reject_reasons", [])
            n_fallback += 1
            for reason in meta.get("reject_reasons", []):
                fail_reasons[reason.split(":")[0]] += 1
        else:
            row["pair_type"] = "synthetic"
            # The synthetic rejection was never seen by the reward model, so its
            # score is a grounding score, NOT a combined reward-model score.
            # Flagged so the two are never compared as if they were on one scale.
            row["rejected_score_basis"] = "synthesize_grounding"
            row["natural_rejected"] = r["rejected"]          # keep for auditing
            row["natural_rejected_mode"] = r.get("rejected_mode")
            row["natural_score_gap"] = r.get("score_gap")
            row["rejected"] = rejected
            row["rejected_mode"] = "synthetic_hallucinated"
            row["rejected_score"] = meta.get("rejected_synth_score")
            row["rejected_stepwise"] = meta.get("rejected_synth_score")
            row["rejected_gt_similarity"] = None
            row["rejected_rollout_id"] = None
            row["chosen_synth_score"] = meta.get("chosen_synth_score")
            row["rejected_synth_score"] = meta.get("rejected_synth_score")
            row["synth_score_gap"] = round((meta.get("chosen_synth_score") or 0)
                                           - (meta.get("rejected_synth_score") or 0), 4)
            row["copy_jaccard"] = meta.get("copy_jaccard")
            row["gen_method"] = meta.get("gen_method")
            row["gen_attempts"] = meta.get("attempts")
            n_synth += 1
        out_rows.append(row)

    # ---------- write ----------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, indent=2))
    csv_path = out_path.with_suffix(".csv")
    write_pairs_csv(out_rows, csv_path)

    # ---------- report ----------
    syn = [r for r in out_rows if r["pair_type"] == "synthetic"]
    nat = [r for r in out_rows if r["pair_type"] == "natural"]

    print("\n" + "=" * 70)
    print("TYPE B SUMMARY")
    print("=" * 70)
    print(f"total pairs        : {len(out_rows)}  (must equal TYPE A = {n_total})")
    print(f"  natural (top)    : {len(nat)}")
    print(f"  synthetic        : {n_synth}")
    print(f"  natural_fallback : {n_fallback}   <- generation failed the gate")
    if nat:
        g = [r["score_gap"] for r in nat]
        print(f"natural gap        : mean={statistics.mean(g):.3f} "
              f"min={min(g):.3f} max={max(g):.3f}")
    if syn:
        sg = [r["synth_score_gap"] for r in syn if r.get("synth_score_gap") is not None]
        ng = [r["natural_score_gap"] for r in syn if r.get("natural_score_gap") is not None]
        att = [r["gen_attempts"] for r in syn]
        print(f"synth score gap    : mean={statistics.mean(sg):.3f} "
              f"min={min(sg):.3f} max={max(sg):.3f}")
        print(f"gap they replaced  : mean={statistics.mean(ng):.3f} max={max(ng):.3f}")
        print(f"gen attempts       : mean={statistics.mean(att):.2f} max={max(att)}")
        by_agent = defaultdict(int)
        for r in syn:
            by_agent[r["agent_name"]] += 1
        print("synthetic by agent : " + ", ".join(f"{k.split('_')[0]}:{v}"
                                                 for k, v in sorted(by_agent.items())))
    if fail_reasons:
        print(f"gate rejections    : {dict(fail_reasons)}")
    if len(out_rows) != n_total:
        print(f"\n[WARN] size mismatch vs TYPE A ({len(out_rows)} vs {n_total}) — "
              "the paired comparison is no longer exact.")
    if syn and syn[0].get("gen_method") == "fallback_template":
        print("\n*** WARNING: offline template generator was used. These rejections are")
        print("*** for plumbing tests only — do NOT train a reported model on them.")

    print("\n" + "-" * 70)
    print("PAIRED ABLATION — same groups, same chosen samples, same size:")
    print(f"  TYPE A : {n_total:>4} pairs  all-natural rejections   {args.natural_dpo}")
    print(f"  TYPE B : {len(out_rows):>4} pairs  bottom {1-args.natural_frac:.0%} "
          f"synthetic         {out_path}")
    print(f"  audit  : {csv_path}")
    print("\nTrain both from the SAME worker_sft/final:")
    print(f"  DPO_DATA={args.natural_dpo}\n      OUT_DIR=$DIR/worker_dpo_natural")
    print(f"  DPO_DATA={out_path}\n      OUT_DIR=$DIR/worker_dpo_synth")


if __name__ == "__main__":
    main()