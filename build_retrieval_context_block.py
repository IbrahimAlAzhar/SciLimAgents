"""
build_retrieval_context_block.py — build cited-paper context from the CSV for inference.

Adds `retrieved_abs_int` and `ret_abs_int_cit` to the prompt, formatted EXACTLY
the way generate_multi_agent_rollouts_vllm.py formatted it during retrieval_heavy rollouts, so the
model sees a familiar layout rather than a new one.

    from retrieval_context import build_retrieval_block

    ctx = build_retrieval_block(row, tokenizer=tok, max_tokens=4000)
    if ctx:
        system_prompt = system_prompt + ctx

IMPORTANT — TRAIN/TEST MISMATCH
    The SFT and DPO prompts contain the paper ONLY. Retrieved context was used
    during retrieval_heavy rollout generation but stripped from the canonical
    prompts used for training. Injecting it at inference is therefore a change of
    input distribution. Run both conditions (USE_RETRIEVAL=0 and =1) and compare;
    do not assume more context is better.
"""

import ast
import os
import re

# Columns to pull, in order. Both are parsed the same way; whichever is present
# and non-empty is used, and if both exist they are concatenated (deduplicated by
# paper title so the same reference is not shown twice).
DEFAULT_COLS = [c.strip() for c in os.environ.get(
    "RETRIEVAL_COLS", "retrieved_abs_int,ret_abs_int_cit").split(",") if c.strip()]

TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", 8))


def clean_text_detailed(text):
    """Same normalization the rollout script applied to cited-paper fields."""
    if text is None:
        return ""
    try:
        import pandas as pd
        if isinstance(text, float) and pd.isna(text):
            return ""
    except Exception:
        pass
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse(cell):
    """Parse a CSV cell into a python object. Handles dict, list, JSON string,
    python-repr string, and returns None when nothing usable is found."""
    if cell is None:
        return None
    try:
        import pandas as pd
        if isinstance(cell, float) and pd.isna(cell):
            return None
    except Exception:
        pass
    if isinstance(cell, (dict, list)):
        return cell
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if not s or s.lower() in {"nan", "none", "null", "{}", "[]"}:
        return None
    import json
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except Exception:
            continue
    return s          # plain text: usable as-is


# Field names seen across different dump formats.
_TITLE_KEYS = ("title", "paper_title", "Title", "name")
_ABS_KEYS = ("abstractText", "abstract", "Abstract", "abstract_text", "summary")
_INTRO_KEYS = ("introduction", "Introduction", "intro", "intro_text")
_SECTION_KEYS = ("sections", "section", "body", "paragraphs")


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return ""


def _one_record(data):
    """dict -> (title, abstract, intro), tolerating several key spellings."""
    if not isinstance(data, dict):
        return None
    intro = _first(data, _INTRO_KEYS)
    if not intro:
        secs = None
        for k in _SECTION_KEYS:
            if isinstance(data.get(k), list):
                secs = data[k]
                break
        for sec in secs or []:
            if isinstance(sec, dict):
                head = str(sec.get("heading") or sec.get("section") or
                           sec.get("title") or "").lower()
                if "introduction" in head or "intro" in head:
                    intro = sec.get("text") or sec.get("content") or ""
                    break
    t = clean_text_detailed(_first(data, _TITLE_KEYS))
    a = clean_text_detailed(_first(data, _ABS_KEYS))
    i = clean_text_detailed(intro)
    return (t, a, i) if (t or a or i) else None


def _entries(cell):
    """-> [(title, abstract, introduction), ...] from one column's cell.

    Accepts:
      {"id": {...}, "id2": {...}}      dict of records   (original format)
      [{...}, {...}]                   list of records
      {"title": ..., "abstract": ...}  a single record
      "free text"                      one pseudo-record with the text as abstract
    """
    parsed = _parse(cell)
    if parsed is None:
        return []
    out = []

    if isinstance(parsed, str):
        txt = clean_text_detailed(parsed)
        return [("", txt, "")] if len(txt) > 40 else []

    if isinstance(parsed, dict):
        rec = _one_record(parsed)          # a single record, not a container?
        if rec:
            return [rec]
        values = parsed.values()
    elif isinstance(parsed, list):
        values = parsed
    else:
        return []

    for data in values:
        if isinstance(data, str):
            txt = clean_text_detailed(data)
            if len(txt) > 40:
                out.append(("", txt, ""))
            continue
        rec = _one_record(data)
        if rec:
            out.append(rec)
    return out


def build_retrieval_block(row, cols=None, top_k=TOP_K, tokenizer=None,
                          max_tokens=None, max_chars=None):
    """Formatted context block, or '' when there is nothing to add.

    row        a pandas Series (or dict) for one paper
    cols       column names to read; defaults to retrieved_abs_int, ret_abs_int_cit
    top_k      max number of cited papers to include
    tokenizer  optional HF tokenizer, used with max_tokens to truncate exactly
    max_chars  fallback truncation when no tokenizer is given
    """
    cols = cols or DEFAULT_COLS
    seen, entries = set(), []
    for col in cols:
        cell = row.get(col) if hasattr(row, "get") else None
        if cell is None:
            continue
        for t, a, i in _entries(cell):
            key = (t or "")[:120].lower()
            if key and key in seen:
                continue          # same reference already added from the other column
            seen.add(key)
            entries.append((t, a, i))
            if len(entries) >= top_k:
                break
        if len(entries) >= top_k:
            break

    if not entries:
        return ""

    parts = [
        f"Paper{n}_Title: {t}\nPaper{n}_Abstract: {a}\nPaper{n}_Introduction: {i}"
        for n, (t, a, i) in enumerate(entries, 1)
    ]
    body = "\n\n".join(parts)

    if tokenizer is not None and max_tokens:
        ids = tokenizer(body, add_special_tokens=False)["input_ids"]
        if len(ids) > max_tokens:
            body = tokenizer.decode(ids[:max_tokens]) + " ... [TRUNCATED]"
    elif max_chars and len(body) > max_chars:
        body = body[:max_chars] + " ... [TRUNCATED]"

    # Same header the rollout script used, so the layout is familiar to the model.
    return f"\n\n=== RETRIEVED / CITED CONTEXT ===\n{body}"


# =============================================================================
# RAW MODE — use the columns verbatim, no parsing
# =============================================================================

# Light cleanup of python/JSON dump punctuation so the model reads prose rather
# than syntax. Set RAW_CLEAN=0 to pass the cells through completely untouched.
RAW_CLEAN = os.environ.get("RAW_CLEAN", "1") != "0"

_NOISE = [
    (re.compile(r"\\+n"), " "),            # literal \n sequences
    (re.compile(r"\\+[\"']"), '"'),        # escaped quotes
    (re.compile(r"[{}\[\]]"), " "),        # dict/list punctuation
    (re.compile(r"'\s*:\s*'"), ": "),      # 'key': 'value'
    (re.compile(r"'\s*,\s*'"), ", "),
    (re.compile(r"\s{2,}"), " "),
]


def _raw_clean(text):
    if not RAW_CLEAN:
        return text
    for pat, rep in _NOISE:
        text = pat.sub(rep, text)
    return text.strip()


def build_raw_block(row, cols=None, tokenizer=None, max_tokens=None, max_chars=None):
    """Concatenate the raw column text. Never empty when the cells hold anything.

    Costs more tokens than the parsed form (dump syntax survives even after
    cleanup) but is immune to schema surprises, which is the point.
    """
    cols = cols or DEFAULT_COLS
    parts = []
    for col in cols:
        cell = row.get(col) if hasattr(row, "get") else None
        if cell is None:
            continue
        try:
            import pandas as pd
            if isinstance(cell, float) and pd.isna(cell):
                continue
        except Exception:
            pass
        text = _raw_clean(str(cell))
        if len(text) > 40 and text.lower() not in {"nan", "none", "null"}:
            parts.append(f"--- {col} ---\n{text}")
    if not parts:
        return ""

    body = "\n\n".join(parts)
    if tokenizer is not None and max_tokens:
        ids = tokenizer(body, add_special_tokens=False)["input_ids"]
        if len(ids) > max_tokens:
            body = tokenizer.decode(ids[:max_tokens]) + " ... [TRUNCATED]"
    elif max_chars and len(body) > max_chars:
        body = body[:max_chars] + " ... [TRUNCATED]"
    return f"\n\n=== RETRIEVED / CITED CONTEXT ===\n{body}"


def build_shared_memory(row, mode="auto", cols=None, top_k=TOP_K,
                        tokenizer=None, max_tokens=None, max_chars=None):
    """One entry point for both strategies.

        mode="parsed"  structured Paper1_Title/Abstract/Introduction layout
        mode="raw"     the two columns verbatim (lightly de-punctuated)
        mode="auto"    parsed when it yields records, otherwise raw  [default]

    Returns (block, used_mode) so the caller can record which path ran.
    """
    mode = (mode or "auto").lower()
    if mode in ("auto", "parsed"):
        block = build_retrieval_block(row, cols=cols, top_k=top_k,
                                      tokenizer=tokenizer, max_tokens=max_tokens,
                                      max_chars=max_chars)
        if block:
            return block, "parsed"
        if mode == "parsed":
            return "", "parsed_empty"
    block = build_raw_block(row, cols=cols, tokenizer=tokenizer,
                            max_tokens=max_tokens, max_chars=max_chars)
    return block, ("raw" if block else "empty")


if __name__ == "__main__":
    import sys
    import pandas as pd

    csv = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("INPUT_CSV")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    df = pd.read_csv(csv)

    present = [c for c in DEFAULT_COLS if c in df.columns]
    missing = [c for c in DEFAULT_COLS if c not in df.columns]
    print("columns present:", present)
    if missing:
        print("MISSING columns:", missing)

    # ---- diagnose the RAW cell so an unparseable format is identifiable ----
    for col in present:
        cell = df.iloc[n][col]
        print("\n" + "=" * 66)
        print(f"RAW  {col}   (row {n})")
        print("=" * 66)
        print(f"  python type : {type(cell).__name__}")
        try:
            print(f"  length      : {len(cell)}")
        except Exception:
            print(f"  value       : {cell!r}")
        if isinstance(cell, str):
            print(f"  first 300   : {cell[:300]!r}")
            print(f"  last 120    : {cell[-120:]!r}")
        parsed = _parse(cell)
        print(f"  parsed type : {type(parsed).__name__ if parsed is not None else 'None (unparseable)'}")
        if isinstance(parsed, dict):
            ks = list(parsed.keys())[:5]
            print(f"  dict keys   : {ks}{' ...' if len(parsed) > 5 else ''}")
            if ks:
                v = parsed[ks[0]]
                print(f"  first value : type={type(v).__name__}")
                if isinstance(v, dict):
                    print(f"                keys={list(v.keys())[:12]}")
        elif isinstance(parsed, list):
            print(f"  list length : {len(parsed)}")
            if parsed:
                v = parsed[0]
                print(f"  first item  : type={type(v).__name__}")
                if isinstance(v, dict):
                    print(f"                keys={list(v.keys())[:12]}")
        recs = _entries(cell)
        print(f"  -> extracted {len(recs)} record(s)")
        if recs:
            t, a, i = recs[0]
            print(f"     title: {t[:80]!r}")
            print(f"     abstr: {a[:80]!r}")
            print(f"     intro: {i[:80]!r}")

    for m in ("parsed", "raw", "auto"):
        block, used = build_shared_memory(df.iloc[n], mode=m, max_chars=900)
        print("\n" + "=" * 66)
        print(f"MODE={m}  ->  used={used}, {len(block)} chars")
        print("=" * 66)
        print(block[:900] if block else "(empty)")

    if len(sys.argv) > 3 and sys.argv[3] == "--scan":
        k = min(len(df), int(os.environ.get("SCAN_ROWS", len(df))))
        print("\nscanning the requested rows:")
        for m in ("parsed", "raw", "auto"):
            nz = sum(1 for j in range(k) if build_shared_memory(df.iloc[j], mode=m)[0])
            print(f"  mode={m:7s} non-empty on {nz}/{k} rows")