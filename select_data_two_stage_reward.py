"""
select_data_two_stage_reward.py
===================

TYPE D reward: a **two-stage cascade** producing one final score per candidate.

    STAGE 1   Qwen3-32B-AWQ token JSD          (local GPU, no API cost)
              -> how closely the student's token distribution tracks the teacher's
    STAGE 2   gpt-4o-mini judge                (API, costs money)
              -> how well the limitations align with the gold peer review
    FINAL     weighted combination of the normalized stage scores

This script computes NO model forwards itself. It CONSUMES the per-candidate
scores the two earlier passes already wrote, matches them per candidate, and
produces the combined datasets:

    stage 1 input : <select_jsd>/stepwise_candidate_scores.json   (jsd_reward)
    stage 2 input : <select>/stepwise_candidate_scores.json       (gt_similarity,
                                                                   stepwise_score)

If a candidate has no stage-2 score on file, the gpt-4o-mini judge is called for
it directly (disk-cached in REWARD_CACHE_DIR, so re-runs are free).

WHY NORMALIZE BEFORE COMBINING
------------------------------
JSD rewards cluster tightly (often 0.7-0.9, sd ~0.02) while judge scores spread
across most of [0,1]. A raw weighted sum would let the wider-spread signal
dominate regardless of the weights you set. Each component is therefore
min-max (or z-) normalized across all candidates first, so the weights mean what
you think they mean. `--normalize none` reproduces the naive behaviour.

CASCADE MODE (optional, saves API spend)
---------------------------------------
`--cascade_top_k 2` sends only the 2 best-by-JSD candidates per group to the
paid judge. The rest keep a stage-1-only score, renormalized over the components
they actually have. With 4 candidates per group that halves judge calls.

OUTPUT (its own directory, nothing else is touched)
    <out_dir>/two_stage_candidate_scores.json / .csv
    <out_dir>/sft_dataset_{worker,leader,master}.json
    <out_dir>/dpo_dataset_worker.json
    <out_dir>/per_paper/                        resume shards

USAGE
    python select_data_two_stage_reward.py \
        --jsd_scores      $DIR/select_jsd/stepwise_candidate_scores.json \
        --stepwise_scores $DIR/select/stepwise_candidate_scores.json \
        --rollout_json    $DIR/rollouts/rollout_data_full.json \
        --out_dir         $DIR/select_two_stage
"""

from __future__ import annotations

import os
import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

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


ROOT_DEFAULT = _require_env("ROOT", "pipeline output root containing rollouts/ and select*/")

DEFAULT_JSD      = os.environ.get("JSD_SCORES",
                                  f"{ROOT_DEFAULT}/select_jsd/stepwise_candidate_scores.json")
DEFAULT_STEPWISE = os.environ.get("STEPWISE_SCORES",
                                  f"{ROOT_DEFAULT}/select/stepwise_candidate_scores.json")
DEFAULT_ROLLOUT  = os.environ.get("ROLLOUT_JSON",
                                  f"{ROOT_DEFAULT}/rollouts/rollout_data_full.json")
DEFAULT_OUT      = os.environ.get("SELECT_TWO_STAGE_DIR",
                                  f"{ROOT_DEFAULT}/select_two_stage")

# Component weights. All three signals contribute by default:
#   0.25 JSD  +  0.50 gpt-4o-mini judge  +  0.25 stepwise grounding
# Each is min-max normalized first (see build_normalizer), so these weights mean
# what they look like rather than being swamped by whichever signal has the
# widest raw spread.
W_JSD    = float(os.environ.get("W_JSD", 0.25))     # stage 1: Qwen3-32B token JSD
W_JUDGE  = float(os.environ.get("W_JUDGE", 0.50))   # stage 2: gpt-4o-mini vs gold
W_GROUND = float(os.environ.get("W_GROUND", 0.25))  # rule-based stepwise grounding

NORMALIZE     = os.environ.get("NORMALIZE", "minmax")     # minmax | zscore | none
CASCADE_TOP_K = os.environ.get("CASCADE_TOP_K")
CASCADE_TOP_K = int(CASCADE_TOP_K) if CASCADE_TOP_K not in (None, "", "0") else None

DPO_MIN_GAP   = float(os.environ.get("TS_DPO_MIN_GAP", 0.05))
SFT_MIN_SCORE = float(os.environ.get("TS_SFT_MIN_SCORE", 0.0))

KEY_FIELDS = ("paper_idx", "agent_role", "agent_name", "rollout_id")


def ckey(r: dict):
    return tuple(r.get(f) for f in KEY_FIELDS)

# =============================================================================
# NORMALIZATION
# =============================================================================

def build_normalizer(values: list[float], mode: str):
    """Return f(x) -> [0,1]-ish, plus a description for the log."""
    vals = [v for v in values if v is not None]
    if not vals or mode == "none":
        return (lambda x: x), "identity"
    if mode == "zscore":
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if sd == 0:
            return (lambda x: 0.5), "zscore(sd=0 -> 0.5)"
        # squash to [0,1] with a logistic so outliers can't dominate the sum
        return (lambda x: 1.0 / (1.0 + math.exp(-(x - mu) / sd))), \
               f"zscore(mu={mu:.4f}, sd={sd:.4f}) -> logistic"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return (lambda x: 0.5), "minmax(flat -> 0.5)"
    return (lambda x: (x - lo) / (hi - lo)), f"minmax(lo={lo:.4f}, hi={hi:.4f})"


def combine(jsd_n, judge_n, ground_n):
    """Weighted mean over the components that are actually present."""
    num = den = 0.0
    for w, v in ((W_JSD, jsd_n), (W_JUDGE, judge_n), (W_GROUND, ground_n)):
        if v is not None and w > 0:
            num += w * v
            den += w
    if den == 0:
        return None
    return round(num / den, 6)

# =============================================================================
# STAGE 2 ON DEMAND
# =============================================================================

def judge_for(cand: dict, gt_text: str, gt_routes: dict, gt_items_all: list):
    """gpt-4o-mini score for a candidate that stage 2 never saw. Disk-cached."""
    role = cand.get("agent_role")
    out = cand.get("output", "")
    if not out.strip():
        return None
    if role == "worker":
        spec = S.WORKER_SPECIALTY.get(cand.get("agent_name"), "other")
        gt_slice = gt_routes.get(spec) or gt_items_all
        return S.worker_gt_alignment(out, gt_slice, spec)
    return S.gt_similarity(out, gt_text)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global W_JSD, W_JUDGE, W_GROUND
    ap = argparse.ArgumentParser(
        description="Two-stage reward: Qwen3-32B JSD then gpt-4o-mini, combined.")
    ap.add_argument("--jsd_scores", default=DEFAULT_JSD)
    ap.add_argument("--stepwise_scores", default=DEFAULT_STEPWISE)
    ap.add_argument("--rollout_json", default=DEFAULT_ROLLOUT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--w_jsd", type=float, default=W_JSD)
    ap.add_argument("--w_judge", type=float, default=W_JUDGE)
    ap.add_argument("--w_ground", type=float, default=W_GROUND)
    ap.add_argument("--normalize", default=NORMALIZE, choices=["minmax", "zscore", "none"])
    ap.add_argument("--cascade_top_k", type=int, default=CASCADE_TOP_K,
                    help="Judge only the top-K by JSD per group (saves API spend).")
    ap.add_argument("--min_gap", type=float, default=DPO_MIN_GAP)
    ap.add_argument("--no_judge_calls", action="store_true",
                    help="Never call the API; use only stage-2 scores already on file.")
    args = ap.parse_args()

    W_JSD, W_JUDGE, W_GROUND = args.w_jsd, args.w_judge, args.w_ground

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / S.SHARD_SUBDIR

    print("=" * 72)
    print("TWO-STAGE REWARD  (stage 1: Qwen3-32B JSD -> stage 2: gpt-4o-mini)")
    print("=" * 72)
    print(f"  stage 1 scores : {args.jsd_scores}")
    print(f"  stage 2 scores : {args.stepwise_scores}")
    print(f"  out            : {out_dir}")
    print(f"  weights        : jsd={W_JSD} judge={W_JUDGE} ground={W_GROUND}")
    print(f"  normalize      : {args.normalize}")
    print(f"  cascade_top_k  : {args.cascade_top_k or 'off (judge everything)'}")

    # ---------------- load stage 1 ----------------
    jsd_rows = json.loads(Path(args.jsd_scores).read_text())
    jsd_by = {ckey(r): r for r in jsd_rows}
    print(f"\nstage 1: {len(jsd_by)} candidates with JSD scores")

    # ---------------- load stage 2 (may be partial) ----------------
    step_by = {}
    if Path(args.stepwise_scores).exists():
        try:
            rows = json.loads(Path(args.stepwise_scores).read_text())
            if not isinstance(rows, list):
                raise ValueError("not a JSON list")
            for r in rows:
                step_by[ckey(r)] = r
            print(f"stage 2: {len(step_by)} candidates with gpt-4o-mini scores on file")
        except Exception as e:
            print(f"stage 2: could not parse {args.stepwise_scores} "
                  f"({type(e).__name__}: {e}) — treating as empty, will call the judge")
    else:
        print(f"stage 2: {args.stepwise_scores} not found — will call the judge")

    overlap = len(set(jsd_by) & set(step_by))
    print(f"overlap : {overlap} candidates have BOTH stages")
    if overlap == 0 and step_by:
        print("[WARN] zero overlap — the two score files may come from different runs.")

    # sanity: the same candidate key must refer to the same text in both files
    mismatch = 0
    for k in list(set(jsd_by) & set(step_by))[:500]:
        a = (jsd_by[k].get("output") or "").strip()
        b = (step_by[k].get("output") or "").strip()
        if a and b and a != b:
            mismatch += 1
    if mismatch:
        print(f"[WARN] {mismatch}/500 sampled candidates have DIFFERENT text between "
              "the two score files — check that both were scored from the same "
              "rollout_data_full.json.")
    else:
        print("text consistency across stages: ok")

    # ---------------- paper-level ground truth (for on-demand judging) ----------------
    papers = json.loads(Path(args.rollout_json).read_text())
    gt_by_pid, routes_by_pid, items_by_pid = {}, {}, {}
    for p in papers:
        pid = p.get("paper_idx")
        gt = p.get("ground_truth", "") or ""
        items = S.split_gt(gt)
        gt_by_pid[pid] = gt
        items_by_pid[pid] = items
        routes_by_pid[pid] = S.route_gt_to_specialties(items) if items else {}

    # ---------------- assemble candidates, group by (paper, role, agent) ----------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for k, jr in jsd_by.items():
        pid, role, name, rid = k
        sr = step_by.get(k, {})
        groups[(pid, role, name)].append({
            "paper_idx": pid, "agent_role": role, "agent_name": name,
            "rollout_id": rid, "mode": jr.get("mode"),
            "output": jr.get("output", ""),
            "input_messages": jr.get("input_messages") or sr.get("input_messages") or [],
            "jsd": jr.get("jsd"),
            "stage1_jsd_reward": jr.get("jsd_reward"),
            "teacher_nll": jr.get("teacher_nll"),
            "student_nll": jr.get("student_nll"),
            "stage2_judge": sr.get("gt_similarity"),
            "grounding": sr.get("stepwise_score"),
        })

    # ---------------- cascade: decide who gets stage 2 ----------------
    n_judged_file = sum(1 for g in groups.values() for c in g
                        if c["stage2_judge"] is not None)
    to_judge = []
    for key, cands in groups.items():
        cands.sort(key=lambda c: (c["stage1_jsd_reward"] is None,
                                  -(c["stage1_jsd_reward"] or 0)))
        eligible = cands if args.cascade_top_k is None else cands[:args.cascade_top_k]
        for c in cands:
            c["cascade_selected"] = c in eligible
        for c in eligible:
            if c["stage2_judge"] is None:
                to_judge.append(c)

    print(f"\nstage 2 coverage: {n_judged_file} from file, {len(to_judge)} to compute"
          f"{' (SKIPPED: --no_judge_calls)' if args.no_judge_calls else ''}")

    if to_judge and not args.no_judge_calls:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY unset — judge scores will fall back to "
                  "ROUGE-L + lexical inside stepwise_reward_select.")
        for c in tqdm(to_judge, desc="stage 2 (gpt-4o-mini)"):
            pid = c["paper_idx"]
            c["stage2_judge"] = judge_for(c, gt_by_pid.get(pid, ""),
                                         routes_by_pid.get(pid, {}),
                                         items_by_pid.get(pid, []))

    # ---------------- normalize each component globally ----------------
    all_c = [c for g in groups.values() for c in g]
    f_jsd, d_jsd = build_normalizer([c["stage1_jsd_reward"] for c in all_c], args.normalize)
    f_jud, d_jud = build_normalizer([c["stage2_judge"] for c in all_c], args.normalize)
    f_grd, d_grd = build_normalizer([c["grounding"] for c in all_c], args.normalize)
    print(f"\nnormalizers:\n  jsd      {d_jsd}\n  judge    {d_jud}\n  grounding {d_grd}")

    for c in all_c:
        c["jsd_norm"]   = f_jsd(c["stage1_jsd_reward"]) if c["stage1_jsd_reward"] is not None else None
        c["judge_norm"] = f_jud(c["stage2_judge"])      if c["stage2_judge"] is not None else None
        c["ground_norm"] = f_grd(c["grounding"])        if c["grounding"] is not None else None
        c["score"] = combine(c["jsd_norm"], c["judge_norm"], c["ground_norm"])
        c["stages_used"] = "+".join(
            n for n, v in (("jsd", c["jsd_norm"]), ("judge", c["judge_norm"]),
                           ("ground", c["ground_norm"])) if v is not None)
        # score_detail keeps the selection helpers and downstream CSVs happy
        c["score_detail"] = {"total": c["score"],
                             "steps": {"jsd": c["jsd"],
                                       "stage1_jsd_reward": c["stage1_jsd_reward"],
                                       "stage2_judge": c["stage2_judge"],
                                       "grounding": c["grounding"]},
                             "n_units": len(S.split_bullets(c.get("output", "")))}

    all_c = [c for c in all_c if c["score"] is not None]
    if not all_c:
        raise SystemExit("No candidate ended up with a usable score — check inputs.")

    # ---------------- select, per paper, into resume shards ----------------
    src_by_pid = {}
    for p in papers:
        s = p.get("paper_text", "") or ""
        if p.get("citation_text"):
            s += "\n\n" + p["citation_text"]
        src_by_pid[p.get("paper_idx")] = s

    by_paper: dict = defaultdict(lambda: defaultdict(list))
    for c in all_c:
        by_paper[c["paper_idx"]][(c["agent_role"], c["agent_name"])].append(c)

    shards = {}
    for pid, gm in tqdm(sorted(by_paper.items(), key=lambda kv: (kv[0] is None, kv[0])),
                        desc="selecting"):
        sft, dpo, cands = [], [], []
        for (role, name), group in gm.items():
            cands.extend(group)
            if role == "worker":
                prompt = S._canonical_worker_prompt(name, src_by_pid.get(pid, ""))
                S._select(group, prompt, "worker", name, pid, sft, dpo,
                          min_score=SFT_MIN_SCORE, min_gap=args.min_gap)
            else:
                S._sft_best(group, role, name, pid, sft, min_score=0.0)
        S.write_shard(shard_dir, pid, cands, sft, dpo)
        shards[pid] = {"candidates": cands, "sft": sft, "dpo": dpo}

    cands, sft, dpo = S.merge_shards(shards)

    # ---------------- write ----------------
    (out_dir / "two_stage_candidate_scores.json").write_text(json.dumps(cands, indent=2))
    with open(out_dir / "two_stage_candidate_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_idx", "agent_role", "agent_name", "mode", "rollout_id",
                    "jsd", "stage1_jsd_reward", "stage2_judge", "grounding",
                    "jsd_norm", "judge_norm", "ground_norm",
                    "final_score", "stages_used", "cascade_selected"])
        for c in cands:
            w.writerow([c["paper_idx"], c["agent_role"], c["agent_name"], c["mode"],
                        c["rollout_id"], c["jsd"], c["stage1_jsd_reward"],
                        c["stage2_judge"], c["grounding"], c["jsd_norm"],
                        c["judge_norm"], c["ground_norm"], c["score"],
                        c["stages_used"], c.get("cascade_selected")])

    (out_dir / "sft_dataset.json").write_text(json.dumps(sft, indent=2))
    for role in ("worker", "master", "leader"):
        (out_dir / f"sft_dataset_{role}.json").write_text(
            json.dumps([x for x in sft if x["agent_role"] == role], indent=2))
    (out_dir / "dpo_dataset.json").write_text(json.dumps(dpo, indent=2))
    (out_dir / "dpo_dataset_worker.json").write_text(
        json.dumps([d for d in dpo if d["agent_role"] == "worker"], indent=2))

    # ---------------- report ----------------
    print("\n" + "=" * 72)
    print("TWO-STAGE DIAGNOSTICS")
    print("=" * 72)
    by_role = defaultdict(list)
    for c in cands:
        by_role[c["agent_role"]].append(c["score"])
    print(f"\n{'Role':<10}{'Cand':>7}{'Mean':>9}{'Std':>9}{'Min':>8}{'Max':>8}")
    for role, v in sorted(by_role.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{role:<10}{len(v):>7}{statistics.mean(v):>9.4f}{sd:>9.4f}"
              f"{min(v):>8.4f}{max(v):>8.4f}")

    combos = defaultdict(int)
    for c in cands:
        combos[c["stages_used"]] += 1
    print("\ncomponents present: " + ", ".join(f"{k}:{v}" for k, v in sorted(combos.items())))

    def pearson(xs, ys):
        pts = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
        if len(pts) < 3:
            return float("nan")
        xs2, ys2 = zip(*pts)
        mx, my = statistics.mean(xs2), statistics.mean(ys2)
        num = sum((a - mx) * (b - my) for a, b in pts)
        den = math.sqrt(sum((a - mx) ** 2 for a in xs2) * sum((b - my) ** 2 for b in ys2))
        return num / den if den else float("nan")

    r_stages = pearson([c["stage1_jsd_reward"] for c in cands],
                       [c["stage2_judge"] for c in cands])
    print(f"\nPearson r(stage 1 JSD, stage 2 judge) = {r_stages:+.3f}")
    print("  near 0  -> the stages are complementary; combining them adds information")
    print("  near 1  -> they agree, so the cheap stage alone would do")
    print("  negative-> they actively disagree; inspect before trusting the blend")

    sb, db, gaps = defaultdict(int), defaultdict(int), []
    for s in sft:
        sb[s["agent_role"]] += 1
    for d in dpo:
        db[d["agent_role"]] += 1
        gaps.append(d["score_gap"])
    print(f"\nSFT samples: {len(sft)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(sb.items())))
    print(f"DPO pairs:   {len(dpo)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(db.items())))
    if gaps:
        print(f"DPO gaps: mean={statistics.mean(gaps):.4f} min={min(gaps):.4f} "
              f"max={max(gaps):.4f}  (threshold {args.min_gap})")
    else:
        print(f"[WARN] no DPO pairs cleared --min_gap {args.min_gap}. Lower it; "
              "normalized blends compress the spread.")
    print("=" * 72)
    print(f"\nOutputs: {out_dir}")
    print("Train with:")
    print(f"  CATEGORY=two_stage SELECT_SRC={out_dir} \\")
    print(f"  DPO_FILE={out_dir}/dpo_dataset_worker.json bash train_category.sh")


if __name__ == "__main__":
    main()