# =============================================================================
# pipeline.py
# -----------------------------------------------------------------------------
# End-to-end "ReviewRL-style" inference pipeline for LIMITATION GENERATION.
#
# This file orchestrates exactly the three steps that ReviewRL performs at
# inference time (Section 3 of the paper):
#
#     Step 1 (Section 3.2): generate 3 query questions for the paper
#                           (Table 6 prompt).
#     Step 2 (Section 3.2): retrieve / assemble related-work context
#                           (here we substitute ArXiv-MCP with the user's
#                            pre-retrieved `cited_in_text` + `cited_in_ret`).
#     Step 3 (Section 3.3 + 3.4):
#                           run the policy model with a CoT prompt that
#                           wraps thinking in <think>...</think> and emits
#                           a structured "## Limitations" section.
#
# All intermediate artifacts (queries, context, raw output, parsed
# limitations, format reward) are written back into the dataframe so the
# user can inspect / cite each agent step individually.
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd
from tqdm import tqdm

from data_utils import ensure_output_columns, save_checkpoint
from prompts import (
    GENERATE_QUERIES_PROMPT,
    LIMITATION_GENERATION_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
)
from retrieval import build_retrieval_context
from reward import extract_limitations_section, format_reward

# Quick truncation helper so the paper text never blows up the context window.
def _truncate_paper(text: str, max_chars: int = 14000) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + " [...truncated]"

# Cleans up the LLM's "1. ... 2. ... 3. ..." answer to the query-generation step
# into a single newline-separated string of three questions.
_QUERY_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$", re.MULTILINE)

def _parse_three_queries(text: str) -> str:
    matches = _QUERY_LINE_RE.findall(text)
    if not matches:
        # Fallback: just keep first 3 non-empty lines.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
        return "\n".join(lines)
    queries = [m[1].strip() for m in matches[:3]]
    return "\n".join(queries)

def run_pipeline(
    df: pd.DataFrame,
    generator,                     # QwenGenerator | OpenAIGenerator
    text_column: str,              # e.g. "input_text_cleaned"
    cited_in_text_column: Optional[str],
    cited_in_ret_column: Optional[str],
    use_citations: bool,
    output_dir: str,
    output_name: str,
    save_every: int = 10,
    skip_query_step: bool = False,
    max_new_tokens_review: int = 1024,
    max_new_tokens_query: int = 256,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Run the ReviewRL-style limitation-generation pipeline over `df`.

    Parameters
    ----------
    df : DataFrame already sliced to [start:end] by data_utils.load_input_csv.
    generator : a QwenGenerator or OpenAIGenerator instance.
    text_column : column with the paper body.
    cited_in_text_column / cited_in_ret_column :
        columns providing pre-retrieved citation context.  Either or both
        may be empty; build_retrieval_context() handles missing values.
    use_citations : bool
        If False, the retrieval step is skipped entirely (the
        "w/o Retrieval" ablation from Figure 1 of the paper).
    output_dir / output_name :
        path used for incremental checkpointing every `save_every` rows.
    skip_query_step :
        If True, we skip the 3-query generation (Step 1).  Useful when
        the citations are already curated upstream.

    Returns
    -------
    DataFrame with the new ReviewRL columns populated.
    """
    df = ensure_output_columns(df)

    pbar = tqdm(range(len(df)), desc="ReviewRL limitations")
    for i in pbar:
        paper_text = _truncate_paper(df.at[i, text_column])

        # --------------------------------------------------------------
        # Step 1 (Section 3.2): generate three retrieval queries.
        # --------------------------------------------------------------
        queries_str = ""
        if not skip_query_step:
            try:
                q_prompt = GENERATE_QUERIES_PROMPT.format(paper=paper_text)
                raw_q = generator.generate(
                    q_prompt,
                    system=RETRIEVAL_SYSTEM_PROMPT,
                    max_new_tokens=max_new_tokens_query,
                    temperature=temperature,
                )
                queries_str = _parse_three_queries(raw_q)
            except Exception as e:
                print(f"[row {i}] query-generation failed: {e}")
                queries_str = ""
        df.at[i, "reviewrl_queries"] = queries_str

        # --------------------------------------------------------------
        # Step 2 (Section 3.2): assemble retrieval context.
        # We replace the live ArXiv-MCP call with a deterministic merge
        # of the user's `cited_in_text` + `cited_in_ret` columns.
        # --------------------------------------------------------------
        cited_in_text_val = (
            df.at[i, cited_in_text_column] if cited_in_text_column else None
        )
        cited_in_ret_val = (
            df.at[i, cited_in_ret_column] if cited_in_ret_column else None
        )
        context = build_retrieval_context(
            cited_in_text=cited_in_text_val,
            cited_in_ret=cited_in_ret_val,
            use_citations=use_citations,
        )
        df.at[i, "reviewrl_context"] = context

        # --------------------------------------------------------------
        # Step 3 (Section 3.3 / Table 8 prompt): run the policy model.
        # --------------------------------------------------------------
        try:
            review_prompt = LIMITATION_GENERATION_PROMPT.format(
                paper=paper_text,
                context=context,
            )
            raw_response = generator.generate(
                review_prompt,
                system=None,                # the prompt itself is self-contained
                max_new_tokens=max_new_tokens_review,
                temperature=temperature,
            )
        except Exception as e:
            print(f"[row {i}] generation failed: {e}")
            raw_response = ""

        # --------------------------------------------------------------
        # Parse & score the response.
        # --------------------------------------------------------------
        df.at[i, "reviewrl_raw_response"] = raw_response

        # Extract the <think> block (if any) for transparency / debugging.
        if "<think>" in raw_response and "</think>" in raw_response:
            think = raw_response.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        else:
            think = ""
        df.at[i, "reviewrl_thinking"] = think

        # Extract just the "## Limitations" section.
        limitations = extract_limitations_section(raw_response)
        df.at[i, "reviewrl_limitations"] = limitations

        # Format / structural reward (informational; mirrors review_eval.py).
        rew = format_reward(raw_response)
        df.at[i, "reviewrl_format_reward"] = json.dumps(rew)
        df.at[i, "reviewrl_total_reward"] = float(rew["total_reward"])

        pbar.set_postfix({"reward": f"{rew['total_reward']:.2f}"})

        # --------------------------------------------------------------
        # Incremental checkpoint every `save_every` rows.
        # --------------------------------------------------------------
        if save_every and (i + 1) % save_every == 0:
            path = save_checkpoint(df, output_dir, output_name)
            tqdm.write(f"[checkpoint] saved {i+1} rows -> {path}")

    # Final save (catches the trailing rows that didn't hit the modulo).
    final_path = save_checkpoint(df, output_dir, output_name)
    print(f"[done] final dataframe saved to {final_path}")
    return df