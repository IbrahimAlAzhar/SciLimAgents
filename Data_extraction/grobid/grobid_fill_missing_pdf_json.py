#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET

# ============================================================
# CONFIG
# ============================================================

# ✅ Read the already-generated CSV that has NaNs
INPUT_CSV = "df_with_pdf_json_final.csv"

BASE_DIR = Path("grobid_new")

TMP_PDF_DIR = BASE_DIR / "tmp_pdfs"
TMP_TEI_DIR = BASE_DIR / "tmp_tei"
TMP_JSON_DIR = BASE_DIR / "tmp_json"

OUT_DIR = BASE_DIR / "outputs"

# separate log for this "fill missing" job
STATUS_LOG = OUT_DIR / "status_log_fill_missing.jsonl"

# GROBID server (must be started by PBS)
GROBID_BASE = "http:/127.0.0.1:8070"

# Save checkpoint after every N attempts (attempt = we tried a missing row)
CHECKPOINT_EVERY = 10

# Only process first N missing rows? set None for all missing
LIMIT_MISSING = None

# If true: only process rows with missing pdf_json_file
ONLY_FILL_MISSING = True

# ============================================================

TEI_NS = {"tei": "http:/www.tei-c.org/ns/1.0"}

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    for d in [BASE_DIR, TMP_PDF_DIR, TMP_TEI_DIR, TMP_JSON_DIR, OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def safe_parse_submission(submission_raw: Any) -> Optional[Dict[str, Any]]:
    """
    submission column is often a string repr of a dict.
    """
    if isinstance(submission_raw, dict):
        return submission_raw
    if not isinstance(submission_raw, str) or not submission_raw.strip():
        return None
    try:
        obj = ast.literal_eval(submission_raw.strip())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def safe_filename(name: str, default: str = "paper") -> str:
    if not isinstance(name, str) or not name.strip():
        return default
    base = os.path.basename(name.strip())
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^A-Za-z0-9._-]+", "", base)
    return base or default

def normalize_pdf_url(url: str) -> str:
    """
    If pdf_url starts with '/pdf/' or '/' -> prefix 'https:/openreview.net'
    If pdf_url doesn't start with http -> also prefix
    """
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if u.startswith("https:/openreview.net/") or u.startswith("http:/openreview.net/"):
        return u
    if u.startswith("/pdf/") or u.startswith("/"):
        return "https:/openreview.net" + u
    if not u.startswith("http"):
        return "https:/openreview.net/" + u.lstrip("/")
    return u

def download_pdf(url: str, dst_pdf: Path, timeout: int = 180, retries: int = 3) -> Tuple[bool, str]:
    """
    Download into dst_pdf via .part then rename.
    Validate magic header %PDF-.
    """
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            tmp = dst_pdf.with_suffix(dst_pdf.suffix + ".part")

            wrote_any = False
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    wrote_any = True
                    f.write(chunk)
            r.close()

            if (not wrote_any) or (not tmp.exists()) or tmp.stat().st_size < 1024:
                tmp.unlink(missing_ok=True)
                return False, "file_too_small_or_empty"

            with tmp.open("rb") as f:
                magic = f.read(5)
            if magic != b"%PDF-":
                tmp.unlink(missing_ok=True)
                return False, f"not_pdf_magic({magic!r})"

            tmp.replace(dst_pdf)
            return True, "ok"

        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)

    return False, f"download_error::{type(last_err).__name__}::{last_err}"

def grobid_process_fulltext(pdf_path: Path, timeout: int = 600) -> str:
    """
    POST /api/processFulltextDocument with multipart 'input'
    """
    url = GROBID_BASE.rstrip("/") + "/api/processFulltextDocument"
    with pdf_path.open("rb") as fp:
        files = {"input": fp}
        r = requests.post(url, files=files, timeout=timeout)
        r.raise_for_status()
        return r.text

def text_of(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def extract_title(root: ET.Element) -> str:
    t = root.find("./tei:teiHeader/tei:fileDesc/tei:titleStmt//tei:title", TEI_NS)
    return text_of(t)

def extract_abstract(root: ET.Element) -> str:
    a = root.find("./tei:teiHeader/tei:profileDesc/tei:abstract", TEI_NS)
    if a is not None:
        return text_of(a)

    # fallback: body section called Abstract
    for div in root.findall("./tei:text/tei:body/tei:div", TEI_NS):
        head = text_of(div.find("./tei:head", TEI_NS)).lower()
        if head == "abstract":
            paras = [text_of(p) for p in div.findall("./tei:p", TEI_NS)]
            paras = [p for p in paras if p]
            return "\n".join(paras).strip()

    return ""

def extract_authors(root: ET.Element) -> List[Dict[str, str]]:
    authors: List[Dict[str, str]] = []
    for a in root.findall("./tei:teiHeader/tei:fileDesc/tei:titleStmt//tei:author", TEI_NS):
        pers = a.find("./tei:persName", TEI_NS)
        if pers is not None:
            forename = text_of(pers.find("./tei:forename", TEI_NS))
            surname = text_of(pers.find("./tei:surname", TEI_NS))
            name = (forename + " " + surname).strip() or text_of(pers)
        else:
            name = text_of(a)
        if name:
            authors.append({"name": name})
    return authors

def extract_sections(root: ET.Element) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    body = root.find("./tei:text/tei:body", TEI_NS)
    if body is None:
        return sections

    for div in body.findall("./tei:div", TEI_NS):
        head = text_of(div.find("./tei:head", TEI_NS))
        paras: List[str] = []
        for p in div.findall("./tei:p", TEI_NS):
            t = text_of(p)
            if t:
                paras.append(t)
        if head or paras:
            sections.append({"heading": head, "paragraphs": paras})
    return sections

def extract_references(root: ET.Element) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for bibl in root.findall("./tei:listBibl/tei:biblStruct", TEI_NS):
        analytic_title = bibl.find("./tei:analytic/tei:title", TEI_NS)
        monogr_title = bibl.find("./tei:monogr/tei:title", TEI_NS)
        title = text_of(analytic_title) or text_of(monogr_title)

        year = ""
        date_el = bibl.find("./tei:monogr/tei:imprint/tei:date", TEI_NS)
        if date_el is not None:
            year = date_el.attrib.get("when", "") or text_of(date_el)

        doi = ""
        for idno in bibl.findall("./tei:idno", TEI_NS):
            if idno.attrib.get("type", "").lower() == "doi":
                doi = text_of(idno)
                break

        rec = {"title": title, "year": year, "doi": doi}
        if any(v for v in rec.values()):
            refs.append(rec)
    return refs

def tei_xml_to_json_dict(tei_xml: str, source_file: str, pdf_url: str) -> Dict[str, Any]:
    root = ET.fromstring(tei_xml.encode("utf-8", errors="ignore"))
    return {
        "_source_file": source_file,
        "pdf_url": pdf_url,
        "title": extract_title(root),
        "authors": extract_authors(root),
        "abstract": extract_abstract(root),
        "sections": extract_sections(root),
        "references": extract_references(root),
        "source": "grobid-tei",
    }

def is_missing(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s == "" or s == "nan"

def save_checkpoint(df: pd.DataFrame, tag: str):
    out_path = OUT_DIR / f"df_with_pdf_json_checkpoint_fill_missing_{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"[CHECKPOINT] wrote {out_path}")

def main():
    ensure_dirs()

    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Basic checks
    if "submission" not in df.columns:
        raise ValueError("CSV missing column: submission")
    if "_source_file" not in df.columns:
        raise ValueError("CSV missing column: _source_file")
    if "pdf_json_file" not in df.columns:
        df["pdf_json_file"] = ""

    # ---- Find missing rows
    missing_mask = df["pdf_json_file"].apply(is_missing)
    missing_idx = df.index[missing_mask].tolist()

    print("=======================================")
    print("[INFO] INPUT_CSV :", INPUT_CSV)
    print("[INFO] Missing pdf_json_file rows:", len(missing_idx))
    if len(missing_idx) > 0:
        print("[INFO] First 20 missing indices:", missing_idx[:20])
    print("[INFO] GROBID    :", GROBID_BASE)
    print("[INFO] CHECKPOINT_EVERY :", CHECKPOINT_EVERY)
    print("=======================================")

    # Decide which indices to process
    if ONLY_FILL_MISSING:
        targets = missing_idx
    else:
        targets = df.index.tolist()

    if LIMIT_MISSING is not None:
        targets = targets[: min(len(targets), LIMIT_MISSING)]

    attempted = 0
    ok_count = 0

    with STATUS_LOG.open("a", encoding="utf-8") as logf:
        for i in targets:
            src_raw = str(df.at[i, "_source_file"])
            src_safe = safe_filename(src_raw, default=f"row_{i}")

            # if for some reason it is not missing anymore, skip
            if ONLY_FILL_MISSING and (not is_missing(df.at[i, "pdf_json_file"])):
                continue

            attempted += 1

            sub = safe_parse_submission(df.at[i, "submission"])
            pdf_url_raw = sub.get("pdf_url") if isinstance(sub, dict) else ""
            pdf_url = normalize_pdf_url(pdf_url_raw)

            if not pdf_url:
                logf.write(json.dumps({"row": int(i), "_source_file": src_raw, "status": "no_pdf_url"}) + "\n")
                continue

            pdf_tmp = TMP_PDF_DIR / (src_safe + ".pdf")
            tei_tmp = TMP_TEI_DIR / (src_safe + ".tei.xml")
            json_tmp = TMP_JSON_DIR / (src_safe + ".json")

            try:
                ok, status = download_pdf(pdf_url, pdf_tmp)
                if not ok:
                    logf.write(json.dumps({
                        "row": int(i), "_source_file": src_raw, "pdf_url": pdf_url,
                        "status": status
                    }) + "\n")
                    continue

                tei_xml = grobid_process_fulltext(pdf_tmp)
                tei_tmp.write_text(tei_xml, encoding="utf-8", errors="ignore")

                data = tei_xml_to_json_dict(tei_xml, source_file=src_raw, pdf_url=pdf_url)
                json_tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

                # store json dict in CSV-safe way
                try:
                    df.at[i, "pdf_json_file"] = json.dumps(data, ensure_ascii=False)
                except Exception:
                    df.at[i, "pdf_json_file"] = str(data)

                ok_count += 1
                logf.write(json.dumps({"row": int(i), "_source_file": src_raw, "status": "ok"}) + "\n")

            except Exception as e:
                logf.write(json.dumps({
                    "row": int(i), "_source_file": src_raw, "pdf_url": pdf_url,
                    "status": f"error::{type(e).__name__}::{e}"
                }) + "\n")

            finally:
                # cleanup per-row temp files
                try:
                    pdf_tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    tei_tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    json_tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            # checkpoint every N attempts
            if attempted % CHECKPOINT_EVERY == 0:
                save_checkpoint(df, str(attempted))
                latest_path = OUT_DIR / "df_with_pdf_json_latest_fill_missing.csv"
                df.to_csv(latest_path, index=False)

    # Final save
    final_path = OUT_DIR / "df_with_pdf_json_final_filled.csv"
    df.to_csv(final_path, index=False)

    print(f"[DONE] attempted_missing={attempted}, ok_filled={ok_count}")
    print(f"[DONE] wrote_final={final_path}")
    print("[DONE] Temp files cleaned (per-row).")

if __name__ == "__main__":
    main()
