#!/usr/bin/env python3
"""
Multi-adapter inference for limitation extraction.

Loads the base model once, attaches:
- worker_dpo
- leader_sft
- master_sft

Then switches adapters during inference without reloading the model.

Defaults are conservative for official Llama-3-8B-Instruct, whose context
window is 8192 tokens.
"""

import argparse
import ast
import gc
import os
import re
import time

import pandas as pd
import tiktoken
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser(description="Multi-adapter limitation extraction inference.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    p.add_argument("--worker-dpo-adapter", required=True)
    p.add_argument(
        "--worker-sft-adapter",
        default=None,
        help="Optional Worker SFT LoRA adapter. Use this with worker DPO if DPO was trained on an SFT-merged base.",
    )
    p.add_argument("--leader-sft-adapter", required=True)
    p.add_argument("--master-sft-adapter", required=True)

    p.add_argument("--input-csv", required=True)
    p.add_argument("--text-column", default="input_text_cleaned")
    p.add_argument("--cited-column", default="cited_in")
    p.add_argument("--cited-ret-column", default="cited_in_ret")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-suffix", default="llama3_improved")
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)

    p.add_argument("--max-paper-tokens", type=int, default=5200)
    p.add_argument("--max-citation-tokens", type=int, default=900)
    p.add_argument("--max-leader-input-tokens", type=int, default=5200)
    p.add_argument("--max-master-input-tokens", type=int, default=5200)
    p.add_argument("--context-window", type=int, default=8192)

    p.add_argument("--max-new-tokens-worker", type=int, default=1024)
    p.add_argument("--max-new-tokens-leader", type=int, default=1024)
    p.add_argument("--max-new-tokens-master", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--do-sample", action="store_true", default=False)
    return p.parse_args()

class MultiAdapterModel:
    def __init__(self, args):
        self.args = args
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

        if args.worker_sft_adapter:
            print(f"Attaching worker_sft adapter: {args.worker_sft_adapter}")
            self.model = PeftModel.from_pretrained(
                base_model,
                args.worker_sft_adapter,
                adapter_name="worker_sft",
                torch_dtype=dtype,
            )
            print(f"Attaching worker_dpo adapter: {args.worker_dpo_adapter}")
            self.model.load_adapter(args.worker_dpo_adapter, adapter_name="worker_dpo")
            self.worker_adapter = "worker_sft_dpo"
            if not hasattr(self.model, "add_weighted_adapter"):
                raise RuntimeError(
                    "This PEFT version cannot activate multiple adapters together and "
                    "does not provide add_weighted_adapter(). Please upgrade PEFT or use "
                    "the fallback script that saves an SFT-merged worker base model."
                )
            print("Creating combined worker adapter: worker_sft_dpo = worker_sft + worker_dpo")
            try:
                self.model.add_weighted_adapter(
                    adapters=["worker_sft", "worker_dpo"],
                    weights=[1.0, 1.0],
                    adapter_name=self.worker_adapter,
                    combination_type="cat",
                )
            except TypeError:
                self.model.add_weighted_adapter(
                    adapters=["worker_sft", "worker_dpo"],
                    weights=[1.0, 1.0],
                    adapter_name=self.worker_adapter,
                )
        else:
            print(f"Attaching worker_dpo adapter: {args.worker_dpo_adapter}")
            self.model = PeftModel.from_pretrained(
                base_model,
                args.worker_dpo_adapter,
                adapter_name="worker_dpo",
                torch_dtype=dtype,
            )
            self.worker_adapter = "worker_dpo"

        print(f"Attaching leader_sft adapter: {args.leader_sft_adapter}")
        self.model.load_adapter(args.leader_sft_adapter, adapter_name="leader_sft")
        print(f"Attaching master_sft adapter: {args.master_sft_adapter}")
        self.model.load_adapter(args.master_sft_adapter, adapter_name="master_sft")

        self.model.eval()
        self.model.set_adapter(self.worker_adapter)
        self.active_adapter = self.worker_adapter

    def switch_adapter(self, adapter_name: str):
        target_adapter = self.worker_adapter if adapter_name == "worker_dpo" else adapter_name
        if self.active_adapter != target_adapter:
            self.model.set_adapter(target_adapter)
            self.active_adapter = target_adapter

    def generate(self, system_prompt: str, user_message: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            input_budget = max(self.args.context_window - max_new_tokens, 512)
            old_len = inputs["input_ids"].shape[1]
            if old_len > input_budget:
                print(f"    WARNING: truncating input {old_len} -> {input_budget}, keeping end")
                inputs["input_ids"] = inputs["input_ids"][:, -input_budget:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -input_budget:]

            input_len = inputs["input_ids"].shape[1]
            print(f"    [{self.active_adapter}] input tokens: {input_len}")

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": self.args.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.args.do_sample:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                    }
                )
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            generated = outputs[0][input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            del inputs, outputs, generated
            torch.cuda.empty_cache()
            return response

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return "ERROR: CUDA out of memory"
        except Exception as e:
            return f"ERROR: {e}"

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is exclusively in novelty and significance.

Identify limitations related to novelty, significance, weak motivation, incremental contribution, overstated importance, and limited impact.

OUTPUT FORMAT:
Provide a concise bullet list. Each bullet must include a limitation and brief evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the Citation Agent in a multi-agent system for identifying limitations in scientific papers.

Compare the main article against the cited/retrieved paper information. Identify missing, weak, selective, or misinterpreted citation usage.

OUTPUT FORMAT:
Provide a concise bullet list. Each bullet should follow:
- [Limitation]: Explanation (Ref: Paper X when available)

MAIN ARTICLE:
{paper_content}

CITED/RETRIEVED PAPERS:
{citation_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent for theoretical and methodological soundness, including ablations and component analysis.

Identify limitations involving assumptions, incomplete theory, restricted method conditions, missing ablations, weak component attribution, and logical gaps.

OUTPUT FORMAT:
Provide a concise bullet list with evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent for experimental evaluation.

Identify limitations involving insufficient runs, missing significance testing, weak/outdated baselines, unfair comparisons, missing variance, misleading metrics, or narrow experiments.

OUTPUT FORMAT:
Provide a concise bullet list with evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent for generalization, robustness, computational efficiency, and real-world applicability.

Identify limitations involving benchmark overfitting, missing OOD tests, sensitivity, resource demands, synthetic-only validation, deployment constraints, or missing real-world tests.

OUTPUT FORMAT:
Provide a concise bullet list with evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent for clarity, interpretability, and reproducibility.

Identify limitations involving unclear method descriptions, ambiguity, black-box behavior, missing code/data/hyperparameters, missing seeds, or hard-to-reproduce procedures.

OUTPUT FORMAT:
Provide a concise bullet list with evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent for data integrity, bias, fairness, and ethical considerations.

Identify limitations involving dataset quality/diversity, leakage, bias, fairness metrics, privacy, misuse, or missing ethical discussion.

OUTPUT FORMAT:
Provide a concise bullet list with evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str) -> str:
    return f"""You are the Leader Agent in a sequential multi-agent pipeline for identifying limitations in scientific papers.

Review specialist outputs for quality, evidence, redundancy, and completeness. Consolidate them into categories. Mark repeated claims as [CROSS-VALIDATED] and weak claims as [NEEDS EVIDENCE].

PAPER CONTENT FOR REFERENCE:
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the Master Agent in a sequential multi-agent pipeline for identifying limitations in scientific papers.

Integrate the Leader summary. Remove redundancies, prioritize severe well-supported limitations, preserve evidence, and do not introduce new limitations.

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then provide a bulleted list:
- **Category:** Specific limitation statement with brief explanation and evidence.

PAPER CONTENT FOR REFERENCE:
{paper_content}"""

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry):
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    out = []
    for idx, (_, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        title = clean_text_detailed(data.get("title", ""))
        abstract = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        intro = clean_text_detailed(intro)
        if title or abstract or intro:
            out.append(
                f"Paper{idx}_Title: {title}\n"
                f"Paper{idx}_Abstract: {abstract}\n"
                f"Paper{idx}_Introduction: {intro}"
            )
    return "\n\n".join(out)

_ENC = None

def get_encoding():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    enc = get_encoding()
    toks = enc.encode(str(text))
    if len(toks) <= max_tokens:
        return str(text)
    print(f"  Truncating text: {len(toks)} -> {max_tokens} tokens")
    return enc.decode(toks[:max_tokens]) + "\n[TRUNCATED]"

def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f"df_inference_{args.output_suffix}.csv")

    df = pd.read_csv(args.input_csv)
    end = args.end if args.end >= 0 else len(df)
    df = df.iloc[args.start:end].reset_index(drop=True)

    agent_columns = [
        "lim_novelty_significance",
        "lim_theoretical_methodological",
        "lim_experimental_evaluation",
        "lim_generalization_robustness_efficiency",
        "lim_clarity_interpretability_reproducibility",
        "lim_data_ethics",
        "lim_citation",
        "leader_consolidated",
        "final_limitations_master",
    ]

    start_idx = 0
    if os.path.exists(output_csv):
        ckpt = pd.read_csv(output_csv)
        if len(ckpt) == len(df):
            for col in agent_columns:
                if col in ckpt.columns:
                    df[col] = ckpt[col]
            for idx in range(len(df)):
                value = str(df.iloc[idx].get("final_limitations_master", "PENDING"))
                if value in ("PENDING", "nan", "", "None"):
                    start_idx = idx
                    break
            else:
                start_idx = len(df)

    for col in agent_columns:
        if col not in df.columns:
            df[col] = "PENDING"

    if start_idx >= len(df):
        print("All rows already processed.")
        return

    multi_model = MultiAdapterModel(args)

    specialist_config = [
        ("lim_novelty_significance", get_novelty_significance_prompt),
        ("lim_theoretical_methodological", get_theoretical_methodological_prompt),
        ("lim_experimental_evaluation", get_experimental_evaluation_prompt),
        ("lim_generalization_robustness_efficiency", get_generalization_robustness_efficiency_prompt),
        ("lim_clarity_interpretability_reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("lim_data_ethics", get_data_ethics_prompt),
    ]
    worker_user_msg = (
        "Analyze the paper provided in your instructions. Identify limitations in your domain. "
        "Return concise bullets with explanations and evidence."
    )

    t0 = time.time()
    total_rows = len(df)
    for i in tqdm(range(start_idx, total_rows), initial=start_idx, total=total_rows):
        row = df.iloc[i]
        current = str(row.get("final_limitations_master", "PENDING"))
        if current not in ("PENDING", "nan", "", "None"):
            continue

        paper_text = truncate_text_to_tokens(str(row.get(args.text_column, "")), args.max_paper_tokens)

        citation_parts = []
        citation_value = row.get(args.cited_column, "")
        citation_ret_value = row.get(args.cited_ret_column, "")
        extracted_citations = extract_intro_and_abstract(citation_value)
        if extracted_citations:
            citation_parts.append(f"=== extracted {args.cited_column} ===\n{extracted_citations}")
        elif not pd.isna(citation_value) and str(citation_value).strip():
            citation_parts.append(f"=== raw {args.cited_column} ===\n{citation_value}")
        if not pd.isna(citation_ret_value) and str(citation_ret_value).strip():
            citation_parts.append(f"=== {args.cited_ret_column} ===\n{citation_ret_value}")
        citation_text = truncate_text_to_tokens("\n\n".join(citation_parts), args.max_citation_tokens)

        if len(paper_text) < 100:
            for col in agent_columns:
                df.iat[i, df.columns.get_loc(col)] = "SKIPPED_SHORT_TEXT"
            continue

        try:
            print(f"\n{'=' * 40} Row {i}/{total_rows - 1} {'=' * 40}")
            multi_model.switch_adapter("worker_dpo")
            specialist_outputs = []

            for col_name, prompt_func in specialist_config:
                label = col_name.replace("lim_", "").replace("_", " ").title()
                print(f"  [Worker] {label}")
                out = multi_model.generate(
                    prompt_func(paper_text),
                    worker_user_msg,
                    max_new_tokens=args.max_new_tokens_worker,
                )
                df.iat[i, df.columns.get_loc(col_name)] = out
                specialist_outputs.append(f"=== {label} ===\n{out}")

            print("  [Worker] Citation")
            cite_out = multi_model.generate(
                get_citation_agent_prompt(paper_text, citation_text),
                "Analyze citation-related limitations.",
                max_new_tokens=args.max_new_tokens_worker,
            )
            df.iat[i, df.columns.get_loc("lim_citation")] = cite_out
            specialist_outputs.append(f"=== Citation Agent ===\n{cite_out}")

            multi_model.switch_adapter("leader_sft")
            combined = truncate_text_to_tokens(
                "\n\n".join(specialist_outputs), args.max_leader_input_tokens
            )
            print("  [Leader] Consolidating")
            leader_out = multi_model.generate(
                get_leader_agent_prompt(paper_text),
                "Specialist outputs:\n\n"
                f"{combined}\n\n"
                "Review, organize, flag weak claims, and produce a consolidated summary.",
                max_new_tokens=args.max_new_tokens_leader,
            )
            df.iat[i, df.columns.get_loc("leader_consolidated")] = leader_out

            multi_model.switch_adapter("master_sft")
            leader_trunc = truncate_text_to_tokens(leader_out, args.max_master_input_tokens)
            print("  [Master] Final synthesis")
            final_out = multi_model.generate(
                get_master_agent_prompt(paper_text),
                "Leader consolidated summary:\n\n"
                f"{leader_trunc}\n\n"
                "Produce the final non-redundant limitation list.",
                max_new_tokens=args.max_new_tokens_master,
            )
            if len(final_out.strip()) < 50 or final_out.startswith("ERROR"):
                final_out = "NO_OUTPUT_FROM_MASTER"
            df.iat[i, df.columns.get_loc("final_limitations_master")] = final_out

        except Exception as e:
            print(f"ERROR on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_limitations_master")] = f"ERROR: {e}"
            torch.cuda.empty_cache()

        if (i + 1) % args.checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            elapsed = time.time() - t0
            done = i - start_idx + 1
            rate = elapsed / max(done, 1)
            eta = (total_rows - i - 1) * rate
            print(
                f"Checkpoint row {i}: {done} rows in {elapsed / 60:.1f} min, "
                f"{rate:.1f}s/row, ETA {eta / 3600:.1f}h"
            )

    df.to_csv(output_csv, index=False)
    multi_model.cleanup()
    print(f"Saved final output to {output_csv}")

if __name__ == "__main__":
    run_pipeline(parse_args())
