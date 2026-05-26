import os
import pandas as pd
import time
import ast
import re
import gc
from tqdm import tqdm
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_MODEL_PATH = "qwen2_5_3b_instruct"

# LoRA adapter paths
WORKER_DPO_PATH = "other_experiments/dpo/train/worker_dpo/final"
WORKER_SFT_PATH = "other_experiments/dpo/train/worker_sft/final"
LEADER_SFT_PATH = "other_experiments/dpo/train/leader_sft/final"
MASTER_SFT_PATH = "other_experiments/dpo/train/master_sft/final"

SAFE_INPUT_LIMIT = 28000

INPUT_CSV = "data/balanced_data/df_updated_with_retrieval.csv"
OUTPUT_DIR = "other_experiments/dpo/inference"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "df_inference_dpo_worker_sft_leader_master.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. MULTI-ADAPTER MODEL (load once, switch instantly)
# ==========================================

class MultiAdapterModel:
    """
    Loads the base model once and attaches ALL LoRA adapters simultaneously.
    Switching between adapters is instant via set_adapter() — no loading/unloading,
    no memory fragmentation, no repeated disk I/O.

    Memory cost: base model (~6GB) + all adapters combined (~200-400MB) ≈ ~6.5GB
    This is negligible compared to the del/reload approach which fragments VRAM over time.
    """

    def __init__(self):
        print("=" * 50)
        print("Loading base model and all LoRA adapters...")
        print("=" * 50)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, trust_remote_code=True
        )

        # Load base model
        print("[1/4] Loading base Qwen model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load first adapter (worker_dpo) — this creates the PeftModel
        print("[2/4] Attaching worker_dpo adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            WORKER_DPO_PATH,
            adapter_name="worker_dpo",
            torch_dtype=torch.float16,
        )

        # Load additional adapters into the SAME PeftModel
        print("[3/4] Attaching leader_sft adapter...")
        self.model.load_adapter(LEADER_SFT_PATH, adapter_name="leader_sft")

        print("[4/4] Attaching master_sft adapter...")
        self.model.load_adapter(MASTER_SFT_PATH, adapter_name="master_sft")

        self.model.eval()
        self.active_adapter = "worker_dpo"

        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nGPU memory — allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

        print("All adapters loaded. Ready for inference.\n")

    def switch_adapter(self, adapter_name: str):
        """Switch active adapter — instant, no memory overhead."""
        if self.active_adapter != adapter_name:
            self.model.set_adapter(adapter_name)
            self.active_adapter = adapter_name

    def generate(self, system_prompt: str, user_message: str,
                 max_new_tokens: int = 2048) -> str:
        """Run inference with the currently active adapter."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            input_len = inputs["input_ids"].shape[1]
            print(f"    [{self.active_adapter}] Input tokens: {input_len}")

            # Guard against exceeding model context
            if input_len > 30000:
                print(f"    WARNING: Truncating {input_len} -> 30000 tokens")
                inputs["input_ids"] = inputs["input_ids"][:, :30000]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :30000]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )

            generated = outputs[0][input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Free generation tensors
            del inputs, outputs, generated
            torch.cuda.empty_cache()

            return response

        except torch.cuda.OutOfMemoryError:
            print("    CUDA OOM! Clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            return "ERROR: CUDA out of memory"
        except Exception as e:
            print(f"    LLM call error: {e}")
            return f"ERROR: {e}"

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# 3. PROMPT DEFINITIONS (sequential-aligned)
# ==========================================

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is exclusively in **novelty and significance**.

Your task: Analyze the paper below and identify all limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.

Look for issues such as:
- Rebranding existing ideas without substantial improvement
- Lack of clear differentiation from prior work
- Exaggerated claims of breakthrough
- Narrow scope that limits broader significance
- Failure to articulate why the work matters beyond a niche setting
- Unaddressed alternatives or ignored related problems that diminish perceived impact

OUTPUT FORMAT:
Provide a concise bullet list of novelty- and significance-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent** in a multi-agent system for identifying limitations in scientific papers.

Your task: Compare the main article against the cited papers' information below. Identify limitations related to how the paper uses, interprets, or fails to engage with its citations.

Specifically check:
- Did the article fail to address key insights from its citations?
- Does the paper misinterpret or selectively cite prior work to make its own contribution look stronger?
- Are important related works missing from the citation landscape?

OUTPUT FORMAT:
Provide a bullet list of citation-related limitations. Each bullet should follow this format:
- [Limitation]: Explanation (Ref: Paper X)

=== MAIN ARTICLE ===
{paper_content}

=== CITED PAPERS INFO ===
{citation_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **theoretical and methodological soundness**, including ablations and component analysis.

Your task: Analyze the paper below and identify all limitations related to theoretical foundations, methodology, and component analysis.

Look for issues such as:
- Unstated or overly strong assumptions
- Incomplete theoretical analysis or errors in derivations
- Methods that only work under restricted conditions not clearly acknowledged
- Missing ablations or lack of isolation of individual contributions
- Ablations that do not convincingly attribute performance gains
- Logical gaps between claims and supporting evidence

OUTPUT FORMAT:
Provide a concise bullet list of theoretical, methodological, and ablation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **experimental evaluation**, including validation, rigor, comparisons, baselines, and metrics.

Your task: Analyze the paper below and identify all limitations in the empirical evaluation.

Look for issues such as:
- Insufficient runs or lack of statistical significance testing
- Cherry-picked results or narrow experimental conditions
- Outdated, weak, or missing baselines and key competitors
- Unfair hyperparameter tuning or comparison setups
- Missing error bars, confidence intervals, or variance reporting
- Reliance on misleading metrics or missing standard metrics
- Overemphasis on minor gains without practical or statistical significance

OUTPUT FORMAT:
Provide a concise bullet list of experimental evaluation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise covers **generalization, robustness, computational efficiency, and real-world applicability**.

Your task: Analyze the paper below and identify all limitations related to how well the method generalizes, how robust it is, how efficient it is, and how applicable it is to real-world scenarios.

Look for issues such as:
- Overfitting to benchmarks or lack of out-of-distribution testing
- Sensitivity to hyperparameters or poor performance under distribution shifts
- Excessive training/inference demands or high resource requirements
- Reliance on synthetic data without real-world validation
- Ignoring deployment constraints like cost, latency, or hardware limitations
- Missing user studies or field tests

OUTPUT FORMAT:
Provide a concise bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **clarity, interpretability, and reproducibility**.

Your task: Analyze the paper below and identify all limitations related to how clearly the work is presented, how interpretable the method/results are, and whether the work can be reproduced.

Look for issues such as:
- Unclear explanations of methods, settings, or key concepts
- Ambiguities or unstated assumptions that hinder comprehension
- Black-box behavior without explanations or mechanistic understanding
- Missing code, data, or hyperparameter details needed for replication
- Unreported random seeds or ambiguous experimental procedures

OUTPUT FORMAT:
Provide a concise bullet list of clarity-, interpretability-, and reproducibility-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **data integrity, bias, fairness, and ethical considerations**.

Your task: Analyze the paper below and identify all limitations related to data quality, potential biases, fairness issues, and ethical concerns.

Look for issues such as:
- Small or non-diverse datasets, labeling errors, or undocumented preprocessing
- Data leakage or reliance on flawed/biased datasets without validation
- Biased outcomes that could lead to discrimination
- Lack of fairness metrics or unreported subgroup performance
- Privacy risks, dual-use concerns, or failure to discuss misuse potential

OUTPUT FORMAT:
Provide a concise bullet list of data integrity-, bias-, fairness-, and ethics-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Leader Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

CONTEXT: Multiple specialist worker agents have independently analyzed the paper below, each focusing on a different aspect (novelty, methodology, experiments, generalization, clarity, data/ethics, citations). Their outputs have been collected and will be provided to you.

YOUR TASK:
1. Read all specialist outputs carefully.
2. Review each for quality, relevance, and completeness.
3. Flag any weak or unsupported claims that lack evidence from the paper.
4. Note redundancies across specialists (similar limitations raised by multiple agents).
5. Organize all limitations into a well-structured consolidated summary, grouped by category.
6. Preserve the specificity and evidence from the original analyses.

OUTPUT FORMAT:
Produce a structured summary organized by category (e.g., Novelty & Significance, Methodology, Experimental Evaluation, etc.). For each category:
- List the relevant limitations with their supporting evidence
- Mark any limitations that appear across multiple specialists as [CROSS-VALIDATED]
- Mark any limitations that seem weakly supported as [NEEDS EVIDENCE]

This consolidated summary will be passed to the Master Agent for final synthesis.

PAPER CONTENT (for reference):
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

CONTEXT: The Leader Agent has reviewed and consolidated limitation analyses from multiple specialist workers. The Leader Agent's consolidated summary will be provided to you.

YOUR TASK:
- Carefully read and integrate the Leader Agent's consolidated summary.
- Remove redundancies by merging similar limitations into single, well-stated points.
- Prioritize the most severe and well-justified limitations.
- Preserve specificity and evidence from the original analyses.
- Organize the final list logically, grouped by category.
- Ensure each limitation is clearly stated, concise, and grounded in the paper.
- Do NOT introduce new limitations not raised by the specialists.
- Aim for 10-20 strong limitations (adjust based on paper quality).

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then provide a bulleted list:
- **Category:** Specific limitation statement (with brief explanation and evidence reference if it adds value).

If no major limitations were found, state: "The paper appears methodologically sound with only minor limitations: [list them]."

PAPER CONTENT (for reference):
{paper_content}"""

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry):
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except:
        return ""
    if not isinstance(parsed, dict):
        return ""
    processed = []
    for idx, (pid, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)
        if t_clean or a_clean or i_clean:
            processed.append(
                f"'Paper{idx}_Title: {t_clean}', 'Paper{idx}_Abstract': '{a_clean}', "
                f"'Paper{idx}_Introduction': '{i_clean}'."
            )
    return "\n".join(processed)

def truncate_text_to_tokens(text: str, max_tokens: int = SAFE_INPUT_LIMIT) -> str:
    if not text:
        return ""
    try:
        encoding = tiktoken.get_encoding("o200k_base")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    print(f"  Warning: Truncating input: {len(tokens)} -> {max_tokens} tokens.")
    return encoding.decode(tokens[:max_tokens]) + "... [TRUNCATED]"

def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  [GPU {label}] allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

# ==========================================
# 5. SEQUENTIAL PIPELINE
# ==========================================

def run_pipeline():
    print("=" * 60)
    print("INFERENCE PIPELINE: DPO Worker + SFT Leader + SFT Master")
    print("Using multi-adapter loading (all adapters in memory)")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading CSV file...")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} rows total.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- Resume from checkpoint if exists ---
    start_idx = 0
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing checkpoint at {OUTPUT_CSV}, loading...")
        df_checkpoint = pd.read_csv(OUTPUT_CSV)

        # Use checkpoint columns if they exist
        if "final_limitations_master" in df_checkpoint.columns:
            # Merge checkpoint results back into full df
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
            for col in agent_columns:
                if col in df_checkpoint.columns:
                    df[col] = df_checkpoint[col]

            # Find resume point
            for idx in range(len(df)):
                val = str(df.iloc[idx].get("final_limitations_master", "PENDING"))
                if val in ("PENDING", "nan", "", "None"):
                    start_idx = idx
                    break
            else:
                start_idx = len(df)  # All done

        del df_checkpoint
        print(f"Resuming from row {start_idx}")

    # --- Initialize output columns ---
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
    for col in agent_columns:
        if col not in df.columns:
            df[col] = "PENDING"

    if start_idx >= len(df):
        print("All rows already processed!")
        return

    # --- Load model with all adapters ---
    multi_model = MultiAdapterModel()
    log_gpu_memory("after loading all adapters")

    # --- Define specialist agents (all use worker_dpo adapter) ---
    specialist_config = [
        ("lim_novelty_significance", get_novelty_significance_prompt),
        ("lim_theoretical_methodological", get_theoretical_methodological_prompt),
        ("lim_experimental_evaluation", get_experimental_evaluation_prompt),
        ("lim_generalization_robustness_efficiency", get_generalization_robustness_efficiency_prompt),
        ("lim_clarity_interpretability_reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("lim_data_ethics", get_data_ethics_prompt),
    ]

    worker_user_msg = (
        "Analyze the paper provided in your instructions. "
        "Identify all limitations in your domain of expertise. "
        "Provide a concise bullet list of limitations with explanations and evidence."
    )

    # --- Main loop ---
    total_rows = len(df)
    print(f"\nProcessing rows {start_idx} to {total_rows - 1}...")
    start_time = time.time()

    for i in tqdm(range(start_idx, total_rows), initial=start_idx, total=total_rows):
        row = df.iloc[i]

        # Skip already processed rows
        current_val = str(row.get("final_limitations_master", "PENDING"))
        if current_val not in ("PENDING", "nan", "", "None"):
            continue

        paper_text = str(row.get("input_text_cleaned", ""))
        citation_text = extract_intro_and_abstract(row.get("cited_in", ""))

        paper_text = truncate_text_to_tokens(paper_text, max_tokens=20000)
        citation_text = truncate_text_to_tokens(citation_text, max_tokens=5000)

        if len(paper_text) < 100:
            for col in agent_columns:
                df.iat[i, df.columns.get_loc(col)] = "SKIPPED_SHORT_TEXT"
            continue

        row_start = time.time()
        print(f"\n{'='*40} Row {i}/{total_rows-1} {'='*40}")

        try:
            # ============================================
            # STAGE 1: Worker DPO — all 7 specialist agents
            # ============================================
            multi_model.switch_adapter("worker_dpo")
            all_specialist_outputs = []

            for col_name, prompt_func in specialist_config:
                agent_label = col_name.replace("lim_", "").replace("_", " ").title()
                print(f"  [Worker] {agent_label}...")

                system_prompt = prompt_func(paper_text)
                output = multi_model.generate(system_prompt, worker_user_msg, max_new_tokens=2048)
                df.iat[i, df.columns.get_loc(col_name)] = output
                all_specialist_outputs.append(f"=== {agent_label} ===\n{output}")

            # Citation agent (also worker_dpo)
            print("  [Worker] Citation Agent...")
            citation_system = get_citation_agent_prompt(paper_text, citation_text)
            citation_output = multi_model.generate(
                citation_system,
                "Analyze citation-related limitations as described in your instructions.",
                max_new_tokens=2048
            )
            df.iat[i, df.columns.get_loc("lim_citation")] = citation_output
            all_specialist_outputs.append(f"=== Citation Agent ===\n{citation_output}")

            # ============================================
            # STAGE 2: Leader SFT — consolidate
            # ============================================
            multi_model.switch_adapter("leader_sft")  # Instant switch!

            combined_specialist = "\n\n".join(all_specialist_outputs)
            combined_specialist = truncate_text_to_tokens(combined_specialist, max_tokens=12000)

            print("  [Leader] Consolidating...")
            leader_system = get_leader_agent_prompt(paper_text)
            leader_input = (
                f"Here are the limitation analyses from the specialist worker agents:\n\n"
                f"{combined_specialist}\n\n"
                f"Please review, organize, flag any weak claims, and produce a consolidated "
                f"summary of all limitations grouped by category for the Master Agent."
            )
            leader_output = multi_model.generate(leader_system, leader_input, max_new_tokens=2048)
            df.iat[i, df.columns.get_loc("leader_consolidated")] = leader_output

            # ============================================
            # STAGE 3: Master SFT — final synthesis
            # ============================================
            multi_model.switch_adapter("master_sft")  # Instant switch!

            leader_output_truncated = truncate_text_to_tokens(leader_output, max_tokens=12000)

            print("  [Master] Final synthesis...")
            master_system = get_master_agent_prompt(paper_text)
            master_input = (
                f"The Leader Agent has reviewed and consolidated the specialist analyses. "
                f"Here is the Leader Agent's consolidated summary:\n\n"
                f"{leader_output_truncated}\n\n"
                f"Please synthesize this into a single, final, high-quality, non-redundant "
                f"list of limitations, grouped by category."
            )
            final_output = multi_model.generate(master_system, master_input, max_new_tokens=3000)

            if len(final_output.strip()) < 50 or final_output.startswith("ERROR"):
                final_output = "NO_OUTPUT_FROM_MASTER"

            df.iat[i, df.columns.get_loc("final_limitations_master")] = final_output

            row_elapsed = time.time() - row_start
            print(f"  Row {i} done in {row_elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_limitations_master")] = f"ERROR: {e}"
            torch.cuda.empty_cache()

        # --- Checkpoint ---
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            elapsed = time.time() - start_time
            rows_done = i - start_idx + 1
            rate = elapsed / rows_done if rows_done > 0 else 0
            remaining = (total_rows - i - 1) * rate
            print(f"  >>> Checkpoint at row {i} | "
                  f"{rows_done} rows in {elapsed/60:.1f}min | "
                  f"~{rate:.1f}s/row | "
                  f"ETA: {remaining/3600:.1f}h")

        time.sleep(0.2)

    # --- Final save ---
    df.to_csv(OUTPUT_CSV, index=False)
    total_elapsed = time.time() - start_time
    print(f"\nDone! Saved to: {OUTPUT_CSV}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")

    multi_model.cleanup()

if __name__ == "__main__":
    run_pipeline() 
    