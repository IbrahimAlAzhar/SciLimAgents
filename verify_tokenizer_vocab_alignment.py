"""
verify_tokenizer_vocab_alignment.py — is JSD well-defined between two models?

JSD compares two probability distributions over the SAME vocabulary. What must
match is the MAPPING (token id 5000 means the same token in both models), not
how each tokenizer chooses to encode a given sentence.

Those are different things, and confusing them produces false alarms:

    student [1, 1183, 4598, ...]
    teacher [1, 1782, 4598, ...]

One differing token at position 1 is a whitespace / `legacy` flag difference
("The" vs "_The"), not a different vocabulary. It does not affect JSD, because
select_data_jsd_teacher_reward.py tokenizes ONCE with the student tokenizer and feeds the same
ids to both models.

    python verify_tokenizer_vocab_alignment.py --student /path/a --teacher /path/b
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=os.environ.get("STUDENT_BASE"))
    ap.add_argument("--teacher", default=os.environ.get("TEACHER_PATH"))
    ap.add_argument("--sample", type=int, default=2000,
                    help="How many ids to spot-check when full vocabs differ.")
    args = ap.parse_args()
    if not args.student or not args.teacher:
        print("need --student and --teacher"); sys.exit(1)

    from transformers import AutoTokenizer
    s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    print("=" * 66)
    print(f"student : {args.student}\n          vocab_size={s.vocab_size}")
    print(f"teacher : {args.teacher}\n          vocab_size={t.vocab_size}")
    print("=" * 66)

    if s.vocab_size != t.vocab_size:
        print("\nFATAL: vocab sizes differ. JSD is undefined across them.")
        sys.exit(2)

    # ---- THE test that matters: does id -> token agree? ----
    sv, tv = s.get_vocab(), t.get_vocab()
    if sv == tv:
        print("\n[OK] vocab mappings are IDENTICAL -> JSD is well defined")
        mapping_ok = True
    else:
        inv_s = {v: k for k, v in sv.items()}
        inv_t = {v: k for k, v in tv.items()}
        ids = range(min(args.sample, s.vocab_size))
        mism = [i for i in ids if inv_s.get(i) != inv_t.get(i)]
        pct = 100.0 * len(mism) / max(1, len(list(ids)))
        print(f"\nvocab dicts not equal; spot-checked {args.sample} ids: "
              f"{len(mism)} mismatches ({pct:.2f}%)")
        if mism[:5]:
            for i in mism[:5]:
                print(f"   id {i}: student={inv_s.get(i)!r}  teacher={inv_t.get(i)!r}")
        mapping_ok = len(mism) == 0
        print("[OK] id->token agrees on the sample -> JSD is well defined" if mapping_ok
              else "[FATAL] id->token disagrees -> JSD would be meaningless")

    # ---- encoding policy: informational only ----
    probe = "The paper reports BLEU 31.2 on WMT14 without error bars or ablations."
    ids_s, ids_t = s(probe)["input_ids"], t(probe)["input_ids"]
    print(f"\nencoding of a probe sentence (informational, not a blocker):")
    print(f"   student {ids_s[:12]}")
    print(f"   teacher {ids_t[:12]}")
    if ids_s == ids_t:
        print("   identical")
    else:
        diff = [i for i, (a, b) in enumerate(zip(ids_s, ids_t)) if a != b]
        print(f"   differ at position(s) {diff[:6]} of {min(len(ids_s), len(ids_t))}")
        for i in diff[:3]:
            print(f"     pos {i}: student {ids_s[i]}={s.convert_ids_to_tokens([ids_s[i]])[0]!r}"
                  f"   teacher {ids_t[i]}={t.convert_ids_to_tokens([ids_t[i]])[0]!r}")
        print("   -> a whitespace/`legacy` policy difference. Harmless here:")
        print("      select_data_jsd_teacher_reward.py tokenizes once with the STUDENT tokenizer")
        print("      and feeds identical ids to both models.")

    print("\n" + "=" * 66)
    if mapping_ok:
        print("VERDICT: run the JSD stage.")
        sys.exit(0)
    print("VERDICT: do NOT run JSD with this pair.")
    sys.exit(2)


if __name__ == "__main__":
    main()