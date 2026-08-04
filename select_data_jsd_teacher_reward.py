"""
select_data_jsd_teacher_reward.py
====================

TYPE C reward model: **Qwen3-32B-AWQ as a distributional judge.**

Instead of asking an LLM to grade the text (TYPE A/B use gpt-4o-mini), this
scores each student candidate by how far the STUDENT's next-token distribution
is from the TEACHER's, measured by Jensen-Shannon divergence under
teacher forcing.

    reward = 1 - mean_t JSD( P_teacher(. | x_<t) || P_student(. | x_<t) )

WHAT THIS DOES AND DOES NOT REQUIRE
-----------------------------------
* Student rollouts: REUSED. Nothing is regenerated. The Qwen3-4B rollouts you
  already have in rollout_data_full.json are the input.
* Teacher rollouts: NOT NEEDED. The teacher never samples. It does a single
  teacher-forced forward pass over `prompt + student_completion` and returns
  per-position distributions. Same for the student. JSD is then computed
  position-by-position over the completion only.
* Both models must share a tokenizer/vocab — true for Qwen3-4B and Qwen3-32B
  (checked at startup).
* No gradients, no optimizer: this is inference only, so it is far lighter than
  train_gkd.py. AWQ teacher (~19GB) + bf16 student (~8GB) fits a 40GB card.

Full-vocab JSD needs real logits, so this runs through `transformers`, not the
vLLM OpenAI API (which caps logprobs at top-20 — far too truncated for a 152k
vocab).

OUTPUT
------
Same schema as select_data_stepwise_process_reward.py, in its own directory, so
train_sft_lora_by_role.py / train_dpo_lora_worker.py consume it unchanged:

    select_jsd/stepwise_candidate_scores.json   (here: JSD scores per candidate)
    select_jsd/sft_dataset_{worker,leader,master}.json
    select_jsd/dpo_dataset_worker.json          <- TYPE C preference pairs
    select_jsd/per_paper/paper_00300.json       <- resume shards

RESUME: on by default, same per-paper shard mechanism as the stepwise scorer.

CAVEAT WORTH STATING IN THE PAPER
---------------------------------
JSD measures agreement with the teacher's *distribution*, not factual
correctness. A fluent, generic, high-probability answer scores well even if it
is unsupported; a correct but unusually-worded answer scores badly. Use
--blend_stepwise to mix in the grounding score if you want both signals.

USAGE
    python select_data_jsd_teacher_reward.py \
        --rollout_json $DIR/rollouts/rollout_data_full.json \
        --out_dir      $DIR/select_jsd
    python select_data_jsd_teacher_reward.py --selftest      # verify the JSD math, no models
"""

from __future__ import annotations

import os
import gc
import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# Canonical prompts, bullet parsing, shard/resume helpers, grounding scores.
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


STUDENT_BASE    = _require_env("STUDENT_BASE", "student checkpoint directory")
TEACHER_PATH    = _require_env("TEACHER_PATH", "teacher checkpoint directory (AWQ-quantized)")
STUDENT_ADAPTER = os.environ.get("STUDENT_ADAPTER", "")   # optional LoRA on the student

DEFAULT_ROLLOUT_JSON = os.environ.get("ROLLOUT_JSON")      # or pass --rollout_json
DEFAULT_OUT_DIR      = os.environ.get("SELECT_JSD_DIR")    # or pass --out_dir

# Context budget. The teacher forward is dominated by prompt length, and BOTH
# models see the identical truncated prompt, so the comparison stays fair.
MAX_PROMPT_TOKENS     = int(os.environ.get("MAX_PROMPT_TOKENS", 3072))
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", 512))
JSD_CHUNK             = int(os.environ.get("JSD_CHUNK", 64))   # positions per JSD chunk

# Selection thresholds (mirror the stepwise scorer's semantics).
DPO_MIN_GAP   = float(os.environ.get("JSD_DPO_MIN_GAP", 0.02))
SFT_MIN_SCORE = float(os.environ.get("JSD_SFT_MIN_SCORE", 0.0))
BLEND_STEPWISE = float(os.environ.get("BLEND_STEPWISE", 0.0))  # 0 = pure JSD reward

SHARD_SUBDIR = S.SHARD_SUBDIR

# =============================================================================
# JSD MATH  (importable + self-testable without any model)
# =============================================================================

def jsd_from_logits(teacher_logits, student_logits, chunk: int = JSD_CHUNK) -> float:
    """Mean per-position Jensen-Shannon divergence, log base 2 -> value in [0,1].

    Both tensors are (T, V) logits for the SAME positions. Computed in fp32 and
    chunked over positions so a 152k vocab never materializes all at once.
    """
    import torch
    import torch.nn.functional as F

    T = teacher_logits.shape[0]
    if T == 0:
        return float("nan")
    total = 0.0
    ln2 = math.log(2.0)
    for i in range(0, T, chunk):
        tl = teacher_logits[i:i + chunk].float()
        sl = student_logits[i:i + chunk].float()
        log_p = F.log_softmax(tl, dim=-1)
        log_q = F.log_softmax(sl, dim=-1)
        p, q = log_p.exp(), log_q.exp()
        m = 0.5 * (p + q)
        log_m = m.clamp_min(1e-12).log()
        kl_pm = (p * (log_p - log_m)).sum(-1)
        kl_qm = (q * (log_q - log_m)).sum(-1)
        jsd = 0.5 * (kl_pm + kl_qm) / ln2          # nats -> bits, bounded [0,1]
        total += jsd.clamp(0.0, 1.0).sum().item()
        del tl, sl, log_p, log_q, p, q, m, log_m, kl_pm, kl_qm, jsd
    return total / T


def mean_nll(logits, target_ids) -> float:
    """Mean negative log-likelihood (nats) of `target_ids` under `logits`."""
    import torch
    import torch.nn.functional as F
    if logits.shape[0] == 0:
        return float("nan")
    return F.cross_entropy(logits.float(), target_ids, reduction="mean").item()


def _selftest():
    """Validate the JSD implementation without loading any model."""
    import torch
    V = 5000
    torch.manual_seed(0)

    a = torch.randn(8, V)
    print(f"identical distributions      JSD = {jsd_from_logits(a, a):.6f}   (expect 0.0)")

    # disjoint one-hot supports -> maximum divergence (1 bit)
    big = 50.0
    p = torch.full((4, V), -big); p[:, 0] = big
    q = torch.full((4, V), -big); q[:, 1] = big
    print(f"disjoint one-hot supports    JSD = {jsd_from_logits(p, q):.6f}   (expect 1.0)")

    b = torch.randn(8, V)
    j1 = jsd_from_logits(a, b)
    j2 = jsd_from_logits(b, a)
    print(f"symmetry                     {j1:.6f} vs {j2:.6f}  (expect equal)")
    assert abs(j1 - j2) < 1e-6

    # chunking must not change the result
    j_full = jsd_from_logits(a, b, chunk=1000)
    print(f"chunk-invariance             {j1:.6f} vs {j_full:.6f}")
    assert abs(j1 - j_full) < 1e-5

    # sharper disagreement -> larger JSD
    near = a + 0.05 * torch.randn(8, V)
    far  = a + 5.0 * torch.randn(8, V)
    jn, jf = jsd_from_logits(a, near), jsd_from_logits(a, far)
    print(f"monotonicity                 near={jn:.4f} < far={jf:.4f}")
    assert jn < jf
    assert 0.0 <= jn <= 1.0 and 0.0 <= jf <= 1.0
    print("\nJSD SELFTEST PASSED")

# =============================================================================
# MODEL WRAPPER
# =============================================================================

class Scorer:
    """Holds teacher + student and returns JSD/NLL for a (prompt, completion)."""

    def __init__(self, student_base, teacher_path, student_adapter="",
                 device="cuda", dtype=None):
        # AutoAWQ 0.2.x breaks on transformers >= ~4.52: transformers' own AWQ
        # loader calls `from awq.modules.linear.gemm import WQLinear_GEMM`, which
        # executes awq/__init__.py, which imports activation classes that newer
        # transformers removed. awq_transformers_compat_shim.py restores them. This MUST run before
        # from_pretrained touches an AWQ checkpoint.
        try:
            import awq_compat
            awq_compat.patch_activations(verbose=True)
        except ImportError:
            print("[WARN] awq_transformers_compat_shim.py not found next to this script.")
            print("       If the teacher is an AWQ checkpoint and the load fails with")
            print("       \"cannot import name 'PytorchGELUTanh' from transformers.activations\",")
            print("       copy awq_transformers_compat_shim.py into this directory and re-run.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        print(f"[load] student tokenizer: {student_base}")
        self.tok = AutoTokenizer.from_pretrained(student_base, trust_remote_code=True)

        from transformers import AutoTokenizer as AT
        t_tok = AT.from_pretrained(teacher_path, trust_remote_code=True)
        if t_tok.vocab_size != self.tok.vocab_size:
            raise SystemExit(
                f"Tokenizer mismatch: student vocab {self.tok.vocab_size} != "
                f"teacher vocab {t_tok.vocab_size}. JSD requires a shared vocabulary.")
        print(f"[load] shared vocab confirmed: {self.tok.vocab_size}")

        # Some id slots can map to DIFFERENT tokens in the two models even when the
        # vocab size matches — e.g. Mistral-Small-2409 repurposes reserved control
        # slots (ids ~10-35) as [IMG]/[PREFIX]/[MIDDLE] where Mistral-7B still has
        # [control_N]. Comparing probabilities at such a slot is meaningless, so we
        # mask them and let JSD renormalize over the shared support.
        self.bad_ids = []
        try:
            sv, tv = self.tok.get_vocab(), t_tok.get_vocab()
            if sv != tv:
                inv_s = {v: k for k, v in sv.items()}
                inv_t = {v: k for k, v in tv.items()}
                self.bad_ids = [i for i in range(self.tok.vocab_size)
                                if inv_s.get(i) != inv_t.get(i)]
        except Exception as e:
            print(f"[load] could not compare vocab mappings ({e}); no masking applied")

        if self.bad_ids:
            frac = 100.0 * len(self.bad_ids) / self.tok.vocab_size
            preview = ", ".join(
                f"{i}:{self.tok.convert_ids_to_tokens([i])[0]!r}" for i in self.bad_ids[:4])
            print(f"[load] {len(self.bad_ids)} id slots ({frac:.2f}%) map to different "
                  f"tokens; masking them from JSD.  e.g. {preview}")
            if frac > 5.0:
                raise SystemExit(
                    f"{frac:.1f}% of the vocabulary disagrees — too much to mask. "
                    "These tokenizers are not interchangeable; do not run JSD.")

        # TEACHER_4BIT=1 quantizes an UNQUANTIZED teacher on the fly with
        # bitsandbytes NF4. Use it when the teacher ships only in bf16/fp16 and
        # is too large to hold alongside the student (e.g. Mistral-Small-2409 is
        # ~44GB in bf16, ~13GB in NF4). It is transformers-native, so it avoids
        # the AutoAWQ/transformers version conflict entirely.
        # Do NOT set it for an already-AWQ/GPTQ checkpoint — those carry their
        # own quantization_config and transformers handles them directly.
        t_kwargs = {"dtype": torch.float16, "device_map": device,
                    "trust_remote_code": True}
        if os.environ.get("TEACHER_4BIT", "0") == "1":
            from transformers import BitsAndBytesConfig
            print("[load] teacher: bitsandbytes NF4 (TEACHER_4BIT=1)")
            t_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        print(f"[load] teacher: {teacher_path}")
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_path, **t_kwargs)
        self.teacher.eval()

        print(f"[load] student: {student_base}")
        self.student = AutoModelForCausalLM.from_pretrained(
            student_base, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True)
        if student_adapter:
            from peft import PeftModel
            print(f"[load] student adapter: {student_adapter}")
            self.student = PeftModel.from_pretrained(self.student, student_adapter)
            self.student = self.student.merge_and_unload()
        self.student.eval()

        if torch.cuda.is_available():
            print(f"[load] GPU memory after load: "
                  f"{torch.cuda.memory_allocated()/2**30:.1f} GiB allocated")

    # ---- tokenisation ----
    @staticmethod
    def _merge_system(messages):
        """Fold a system turn into the first user turn (templates without a
        system role, e.g. Mistral-7B-Instruct v0.x)."""
        if not messages or messages[0].get("role") != "system":
            return messages
        sys_txt, rest = messages[0]["content"], messages[1:]
        if rest and rest[0].get("role") == "user":
            return ([{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}]
                    + rest[1:])
        return [{"role": "user", "content": sys_txt}] + rest

    def build_ids(self, messages, completion):
        """(prompt_ids, completion_ids). Tolerates templates without a system role
        or without the Qwen-only enable_thinking kwarg."""
        merge = os.environ.get("MERGE_SYSTEM", "0") == "1"
        first = self._merge_system(messages) if merge else messages
        ptxt = None
        for attempt in (first, self._merge_system(messages)):
            for kw in ({"enable_thinking": False}, {}):
                try:
                    ptxt = self.tok.apply_chat_template(
                        attempt, tokenize=False, add_generation_prompt=True, **kw)
                    break
                except Exception:
                    continue
            if ptxt is not None:
                break
        if ptxt is None:
            raise RuntimeError("apply_chat_template failed for every fallback")
        p_ids = self.tok(ptxt, add_special_tokens=False)["input_ids"]
        c_ids = self.tok(completion.strip(), add_special_tokens=False)["input_ids"]
        if len(p_ids) > MAX_PROMPT_TOKENS:          # keep the tail: task + instructions
            p_ids = p_ids[:1] + p_ids[-(MAX_PROMPT_TOKENS - 1):]
        c_ids = c_ids[:MAX_COMPLETION_TOKENS]
        return p_ids, c_ids

    def _logits_for_completion(self, model, input_ids, n_completion):
        """Logits predicting the completion tokens, shape (n_completion, V)."""
        torch = self.torch
        with torch.no_grad():
            try:
                out = model(input_ids, logits_to_keep=n_completion + 1)
                lg = out.logits[0]
                lg = lg[-(n_completion + 1):-1, :]
            except TypeError:                        # older transformers
                out = model(input_ids)
                lg = out.logits[0][-(n_completion + 1):-1, :]
        del out
        return lg

    def score(self, messages, completion):
        """-> dict(jsd, reward, teacher_nll, student_nll, n_tokens)"""
        torch = self.torch
        p_ids, c_ids = self.build_ids(messages, completion)
        if len(c_ids) < 5:
            return {"jsd": None, "reward": None, "teacher_nll": None,
                    "student_nll": None, "n_tokens": len(c_ids)}

        ids = torch.tensor([p_ids + c_ids], device=self.teacher.device)
        n = len(c_ids)
        targets = torch.tensor(c_ids, device=self.teacher.device)

        t_lg = self._logits_for_completion(self.teacher, ids, n)
        s_ids = ids.to(self.student.device)
        s_lg = self._logits_for_completion(self.student, s_ids, n).to(t_lg.device)

        # NLL of the actual (real-text) tokens is computed BEFORE masking.
        t_nll = mean_nll(t_lg, targets)
        s_nll = mean_nll(s_lg, targets.to(s_lg.device))

        # Mask id slots whose token identity differs between the two tokenizers, so
        # the softmax renormalizes over the shared support and JSD compares like
        # with like. Reserved control slots carry negligible mass on real text, so
        # this changes almost nothing numerically — it just removes the ambiguity.
        if self.bad_ids:
            if not hasattr(self, "_bad_idx"):
                self._bad_idx = torch.tensor(self.bad_ids, device=t_lg.device,
                                             dtype=torch.long)
            neg = torch.finfo(t_lg.dtype).min
            t_lg[:, self._bad_idx] = neg
            s_lg[:, self._bad_idx] = neg

        jsd = jsd_from_logits(t_lg, s_lg)

        del t_lg, s_lg, ids, s_ids, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"jsd": round(jsd, 6), "reward": round(1.0 - jsd, 6),
                "teacher_nll": round(t_nll, 4), "student_nll": round(s_nll, 4),
                "n_tokens": n}

# =============================================================================
# PER-PAPER SCORING
# =============================================================================

def _blend(reward, stepwise):
    if BLEND_STEPWISE <= 0 or stepwise is None:
        return reward
    return round((1 - BLEND_STEPWISE) * reward + BLEND_STEPWISE * stepwise, 6)


def score_paper(paper: dict, scorer: Scorer, stepwise_lookup: dict):
    """Returns (candidates, sft_rows, dpo_rows) for one paper."""
    pid = paper.get("paper_idx")
    source_text = paper.get("paper_text", "") or ""
    if paper.get("citation_text"):
        source_text = source_text + "\n\n" + paper["citation_text"]

    cands: list[dict] = []
    sft: list[dict] = []
    dpo: list[dict] = []

    worker_groups: dict[str, list[dict]] = defaultdict(list)
    master_cands: list[dict] = []
    leader_cands: list[dict] = []

    for rollout in paper.get("rollouts", []):
        if rollout.get("error") or not rollout.get("samples"):
            continue
        mode, rid = rollout.get("mode"), rollout.get("rollout_id")

        # ---- workers: score the final synthesize output ----
        for name, tr in S.extract_worker_final(rollout).items():
            text = (tr.get("synthesize") or "").strip()
            if not text:
                continue
            prompt = S._canonical_worker_prompt(name, source_text)
            m = scorer.score(prompt, text)
            if m["reward"] is None:
                continue
            sw = stepwise_lookup.get((pid, "worker", name, rid))
            cand = {
                "paper_idx": pid, "rollout_id": rid, "mode": mode,
                "agent_role": "worker", "agent_name": name,
                "output": text,
                "jsd": m["jsd"], "jsd_reward": m["reward"],
                "teacher_nll": m["teacher_nll"], "student_nll": m["student_nll"],
                "n_tokens": m["n_tokens"],
                "stepwise_score": sw,
                "score": _blend(m["reward"], sw),
                "score_detail": {"total": m["reward"],
                                 "steps": {"jsd": m["jsd"],
                                           "teacher_nll": m["teacher_nll"],
                                           "student_nll": m["student_nll"]},
                                 "n_units": len(S.split_bullets(text))},
            }
            worker_groups[name].append(cand)
            cands.append(cand)

        # ---- leader / master: score against their REAL upstream input ----
        for role, turn, agent in (("master", "synthesis", "Master_Agent"),
                                  ("leader", "handoff_to_master", "Leader_Agent")):
            s = next((x for x in rollout.get("samples", [])
                      if x.get("agent_role") == role and x.get("turn_type") == turn), None)
            if not s or not (s.get("output") or "").strip():
                continue
            msgs = s.get("input_messages") or []
            if not msgs:
                continue
            m = scorer.score(msgs, s["output"])
            if m["reward"] is None:
                continue
            cand = {
                "paper_idx": pid, "rollout_id": rid, "mode": mode,
                "agent_role": role, "agent_name": agent,
                "output": s["output"], "input_messages": msgs,
                "jsd": m["jsd"], "jsd_reward": m["reward"],
                "teacher_nll": m["teacher_nll"], "student_nll": m["student_nll"],
                "n_tokens": m["n_tokens"],
                "stepwise_score": None, "score": m["reward"],
                "score_detail": {"total": m["reward"],
                                 "steps": {"jsd": m["jsd"],
                                           "teacher_nll": m["teacher_nll"],
                                           "student_nll": m["student_nll"]},
                                 "n_units": len(S.split_bullets(s["output"]))},
            }
            (master_cands if role == "master" else leader_cands).append(cand)
            cands.append(cand)

    # ---- selection: identical semantics to the stepwise scorer ----
    for name, group in worker_groups.items():
        prompt = S._canonical_worker_prompt(name, source_text)
        S._select(group, prompt, "worker", name, pid, sft, dpo,
                  min_score=SFT_MIN_SCORE, min_gap=DPO_MIN_GAP)
    if master_cands:
        S._sft_best(master_cands, "master", "Master_Agent", pid, sft, min_score=0.0)
    if leader_cands:
        S._sft_best(leader_cands, "leader", "Leader_Agent", pid, sft, min_score=0.0)

    return cands, sft, dpo

# =============================================================================
# REPORTING
# =============================================================================

def write_csv(cands, path):
    cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id",
            "jsd", "jsd_reward", "teacher_nll", "student_nll", "n_tokens",
            "stepwise_score", "final_score"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cands:
            w.writerow([c.get("paper_idx"), c.get("agent_role"), c.get("agent_name"),
                        c.get("mode"), c.get("rollout_id"), c.get("jsd"),
                        c.get("jsd_reward"), c.get("teacher_nll"), c.get("student_nll"),
                        c.get("n_tokens"), c.get("stepwise_score"), c.get("score")])


def diagnostics(cands, sft, dpo):
    print("\n" + "=" * 70)
    print("JSD REWARD DIAGNOSTICS  (reward = 1 - JSD, higher is closer to teacher)")
    print("=" * 70)
    by_role = defaultdict(list)
    for c in cands:
        by_role[c["agent_role"]].append(c["score"])
    print(f"\n{'Role':<10}{'Cand':>7}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}")
    for role, v in sorted(by_role.items()):
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{role:<10}{len(v):>7}{statistics.mean(v):>8.4f}{std:>8.4f}"
              f"{min(v):>8.4f}{max(v):>8.4f}")

    by_mode = defaultdict(list)
    for c in cands:
        if c["agent_role"] == "worker":
            by_mode[c["mode"]].append(c["jsd"])
    if by_mode:
        print(f"\n{'Mode':<20}{'N':>6}{'mean JSD':>11}")
        for m, v in sorted(by_mode.items(), key=lambda kv: statistics.mean(kv[1])):
            print(f"{m:<20}{len(v):>6}{statistics.mean(v):>11.4f}")

    sb, db, gaps = defaultdict(int), defaultdict(int), []
    for s in sft:
        sb[s["agent_role"]] += 1
    for d in dpo:
        db[d["agent_role"]] += 1
        gaps.append(d["score_gap"])
    print(f"\nSFT samples: {len(sft)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(sb.items())))
    print(f"DPO pairs:   {len(dpo)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(db.items())))
    if gaps:
        print(f"DPO gaps: mean={statistics.mean(gaps):.4f} "
              f"min={min(gaps):.4f} max={max(gaps):.4f}")
    print("=" * 70)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global BLEND_STEPWISE
    ap = argparse.ArgumentParser(
        description="TYPE C reward: teacher/student JSD over existing rollouts.")
    ap.add_argument("--rollout_json", default=DEFAULT_ROLLOUT_JSON)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--student", default=STUDENT_BASE)
    ap.add_argument("--teacher", default=TEACHER_PATH)
    ap.add_argument("--student_adapter", default=STUDENT_ADAPTER)
    ap.add_argument("--stepwise_scores", default=None,
                    help="stepwise_candidate_scores.json, for --blend_stepwise "
                         "and for side-by-side reporting.")
    ap.add_argument("--blend_stepwise", type=float, default=BLEND_STEPWISE,
                    help="0 = pure JSD reward (default); 0.3 = 70%% JSD + 30%% grounding.")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N papers.")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate the JSD math and exit (no models loaded).")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    BLEND_STEPWISE = args.blend_stepwise

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / SHARD_SUBDIR

    papers = json.loads(Path(args.rollout_json).read_text())
    if args.limit:
        papers = papers[:args.limit]
    print(f"Loaded {len(papers)} papers from {args.rollout_json}")

    stepwise_lookup = {}
    if args.stepwise_scores and Path(args.stepwise_scores).exists():
        for c in json.loads(Path(args.stepwise_scores).read_text()):
            stepwise_lookup[(c.get("paper_idx"), c.get("agent_role"),
                             c.get("agent_name"), c.get("rollout_id"))] = c.get("stepwise_score")
        print(f"Loaded {len(stepwise_lookup)} stepwise scores for comparison "
              f"(blend={BLEND_STEPWISE})")

    # ---- resume ----
    shards = {}
    if args.resume:
        shards = S.load_shards(shard_dir)
        print(f"[resume] {len(shards)} papers already scored in {shard_dir}")
        n_back = S.backfill_shards_from_aggregates(out_dir, shard_dir, known=set(shards))
        if n_back:
            print(f"[resume] backfilled {n_back} papers from existing aggregates")
            shards = S.load_shards(shard_dir)
        todo = [p for p in papers if p.get("paper_idx") not in set(shards)]
        print(f"[resume] skipping {len(papers) - len(todo)}, scoring {len(todo)} new")
    else:
        print("[resume] DISABLED: scoring every paper")
        todo = papers

    if todo:
        print(f"\nBudget: <= {MAX_PROMPT_TOKENS} prompt + {MAX_COMPLETION_TOKENS} "
              f"completion tokens per forward, 2 models per candidate.\n")
        scorer = Scorer(args.student, args.teacher, args.student_adapter)
        for p in tqdm(todo, desc="JSD scoring papers"):
            pid = p.get("paper_idx")
            c, s, d = score_paper(p, scorer, stepwise_lookup)
            S.write_shard(shard_dir, pid, c, s, d)
            shards[pid] = {"candidates": c, "sft": s, "dpo": d}
        del scorer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    cands, sft, dpo = S.merge_shards(shards)
    print(f"\nmerged {len(shards)} papers -> {len(cands)} candidates, "
          f"{len(sft)} sft, {len(dpo)} dpo")

    (out_dir / "stepwise_candidate_scores.json").write_text(json.dumps(cands, indent=2))
    write_csv(cands, out_dir / "stepwise_candidate_scores.csv")
    (out_dir / "sft_dataset.json").write_text(json.dumps(sft, indent=2))
    for role in ("worker", "master", "leader"):
        (out_dir / f"sft_dataset_{role}.json").write_text(
            json.dumps([x for x in sft if x["agent_role"] == role], indent=2))
    (out_dir / "dpo_dataset.json").write_text(json.dumps(dpo, indent=2))
    (out_dir / "dpo_dataset_worker.json").write_text(
        json.dumps([d for d in dpo if d["agent_role"] == "worker"], indent=2))

    diagnostics(cands, sft, dpo)

    if stepwise_lookup:
        pairs = [(c["stepwise_score"], c["jsd_reward"]) for c in cands
                 if c.get("stepwise_score") is not None and c.get("jsd_reward") is not None]
        if len(pairs) > 2:
            xs, ys = zip(*pairs)
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in pairs)
            den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            r = num / den if den else float("nan")
            print(f"\nPearson r(stepwise grounding, JSD reward) = {r:+.3f} over {len(pairs)} candidates")
            print("  near 0 -> the two reward models rank candidates differently,")
            print("            so TYPE C is a genuinely different signal, not a re-run of TYPE A.")

    print(f"\nOutputs: {out_dir}")
    print(f"  dpo_dataset_worker.json  ({len([d for d in dpo if d['agent_role']=='worker'])} pairs)")
    print("\nTrain TYPE C from the SAME worker_sft/final:")
    print(f"  DPO_DATA={out_dir}/dpo_dataset_worker.json  OUT_DIR=$DIR/worker_dpo_jsd")


if __name__ == "__main__":
    main()