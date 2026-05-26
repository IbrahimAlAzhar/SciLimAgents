#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import ast
import json
import pandas as pd
from tqdm import tqdm

import autogen
from openai import OpenAI
import tiktoken

# ============================================================
# API KEY SETUP (REVOKE YOUR COMPROMISED KEY)
# ============================================================
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or api_key == "YOUR_NEW_OPENAI_API_KEY_HERE":
    raise ValueError("OPENAI_API_KEY environment variable not set (or still placeholder).")

client = OpenAI(api_key=api_key)

# ============================================================
# 0) Global guardrails & Worker Appendages
# ============================================================
HARSH_REVIEWER_POLICY = """
You are an extremely strict, harsh peer reviewer.

STRICT REVIEW MODE:
- Default assumption: the paper has limitations unless it provides clear, concrete evidence.
- Prefer flagging possible weaknesses rather than giving benefit of the doubt.
- Focus on limitations that undermine novelty, significance, credibility, rigor, or generalizability.

WHAT TO PRODUCE:
- Only limitations-focused content (no scores, no ratings).
- Evidence pointers from Paper A / Paper B excerpts when possible.
""".strip()

NO_TOOL_NO_JSON = """
CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- Do NOT output JSON, braces {}, or any {"name": ...} structures.
- Do NOT mention "parameters", "function_call", "tool", or "tools".
- Do NOT call tools or pretend to call tools.
""".strip()

WORKER_FEEDBACK_INSTRUCTION = """
The Leader_Agent will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the Leader_Agent and deliver a bullet list of theoretical, methodological, and ablation-related limitations (tailored to your domain) with supporting evidence.
If the Leader_Agent tells you that your status is CONVERGED, simply reply with "CONVERGED".
""".strip()

# ============================================================
# 1) Leader and Verifier Prompts 
# ============================================================
def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    return f"""You are the **Leader_Agent**. Coordinate specialist agents to produce strong, evidence-based limitations.

Available specialist agents: {agent_names}

PROTOCOL (follow strictly):
1) Ask each active specialist agent to analyze the paper.
2) Read the feedback from the Verifier_Agent.
3) If the Verifier_Agent marks a specialist's score difference as < 10% and labels them CONVERGED, explicitly tell that agent to stop analyzing and output "CONVERGED".
4) If an agent is not converged, ask them follow-up questions to improve specificity and groundedness.
5) When ALL agents are CONVERGED, PRODUCE A FINAL HANDOFF in this exact format:

=== MASTER_HANDOFF_START ===
[Literature_Review_and_Data_Analysis_Agent]
<final bullet list from that agent>

[Hypothesis_Refinement_and_Critical_Reflection_Agent]
<final bullet list from that agent>

[Methodological_Novelty_Agent]
<final bullet list from that agent>

[Experimental_Novelty_Agent]
<final bullet list from that agent>

[Problem_Formulation_Novelty_Agent]
<final bullet list from that agent>

[Writing_Claim_Novelty_Agent]
<final bullet list from that agent>
=== MASTER_HANDOFF_END ===

PAPER CONTENT:
{paper_content}
"""

VERIFIER_AGENT_PROMPT = """
You are the **Verifier_Agent**.
Your job is to evaluate the latest outputs of the specialist agents based on:
1. Groundedness (clear references to Paper A and B)
2. Specificity (exact, detailed limitations)
3. Verbosity (concise but thorough, no fluff)

For each active agent, output a Score (0-100).
You MUST calculate the difference between their new score and their previous score. 
If it is round 2 or later, and the difference is less than 10%, mark them as CONVERGED.

OUTPUT FORMAT:
[Agent_Name]
- Score: X/100
- Previous Score: Y/100
- Difference: Z%
- Status: [ACTIVE or CONVERGED]
"""

# ============================================================
# 2) Worker Prompts
# ============================================================
PROMPT_1_NOVELTY_TECH = """
Prompt 1: Technical Contributions & Incremental Nature
Identify limitations in the paper’s technical contributions that undermine claims of novelty or significance.
Focus on whether ideas are rebranded existing methods, minor tweaks, combinations of known components, or 
lack substantive advancement beyond prior work.
Output format (STRICT):
Limitations in Technical Contributions (A vs B):
- <bullet 1 limitation with explanation; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_2_EXPERIMENTS = """
Prompt 2: Experimental Validation & Comparative Analysis
Identify limitations in experimental design, benchmarking, or comparative analysis that weaken the paper’s 
novelty or claimed significance (e.g., missing strong baselines, inadequate datasets, no ablation, overstated
 improvements).

Output format (STRICT):
Limitations in Experimental Validation (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_3_LIT_REVIEW = """
Prompt 3: Literature Review & Contextualization
Identify limitations in the literature review or positioning that undermine perceived novelty or significance 
(overlooking key prior work, vague differentiation, failure to explain why the gap matters).
Output format (STRICT):
Limitations in Literature Review & Contextualization (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_4_SCOPE_GENERALIZABILITY = """
Prompt 4: Scope of Analysis & Generalizability
Identify limitations in scope, datasets, tasks, or discussed implications that restrict the work’s broader 
significance or generalizability (narrow domain, toy settings, ignored real-world constraints).
Output format (STRICT):
Limitations in Scope & Generalizability (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_5_CLAIMS_OVERCLAIMING = """
Prompt 5: Claim Accuracy & Overclaiming
Identify limitations stemming from overstated novelty, impact, effectiveness, or importance claims that lack 
supporting evidence or ignore caveats.
Output format (STRICT):
Limitations in Claims & Overclaiming (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

PROMPT_6_METHOD_CLARITY_RIGOR = """
Prompt 6: Methodological Clarity & Rigor
Identify limitations in methodological description, reproducibility, or rigor that erode confidence in the 
claimed novelty or significance (missing details, ambiguous setups, unverifiable experiments).

Output format (STRICT):
Limitations in Methodological Clarity & Rigor (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
Evidence (A and B):
- A: <pointer from Paper A>
- B: <pointer from Paper B>
"""

literature_review_and_data_analysis_agent = PROMPT_3_LIT_REVIEW + "\n" + WORKER_FEEDBACK_INSTRUCTION
hypothesis_refinement_and_critical_reflection_agent = PROMPT_1_NOVELTY_TECH + "\n" + WORKER_FEEDBACK_INSTRUCTION
methodological_novelty_agent = PROMPT_6_METHOD_CLARITY_RIGOR + "\n" + WORKER_FEEDBACK_INSTRUCTION
experimental_novelty_agent = PROMPT_2_EXPERIMENTS + "\n" + WORKER_FEEDBACK_INSTRUCTION
problem_Formulation_novelty_Agent = PROMPT_4_SCOPE_GENERALIZABILITY + "\n" + WORKER_FEEDBACK_INSTRUCTION
writing_claim_novelty_agent = PROMPT_5_CLAIMS_OVERCLAIMING + "\n" + WORKER_FEEDBACK_INSTRUCTION

# ============================================================
# 3) Master Agent prompt (UPDATED: NO PAPER A/B)
# ============================================================
master_agent_prompt = """
You are the Master Agent.
Wait until the Leader_Agent outputs the block starting with "=== MASTER_HANDOFF_START ===". 
If you do not see that block in the most recent messages, reply ONLY with the word "WAITING".

Once the Leader_Agent provides the handoff block, synthesize an overall limitations summary focused on the novelty & significance of the evaluated paper.

CRITICAL FINAL REPORT RULES:
1. The final report MUST state the limitations directly regarding the evaluated paper only.
2. You are STRICTLY FORBIDDEN from using the phrases "Paper A", "Paper B", "the main paper", or "the retrieved papers". 
3. Transform comparative statements from the specialists into objective weaknesses. 
   - INSTEAD OF: "Paper A lacks robust baselines compared to Paper B."
   - WRITE: "The experimental validation lacks robust baselines, failing to account for contemporary state-of-the-art standards."
4. Remove redundancies and ensure the tone is professional and objective.

OUTPUT FORMAT (STRICT):
**Technical Contributions:**
- <bullet 1 limitation>
- <bullet 2 limitation>

**Experimental Validation:**
- <bullet 1 limitation>

**Literature Review & Contextualization:**
- <bullet 1 limitation>

**Scope & Generalizability:**
- <bullet 1 limitation>

**Claims & Overclaiming:**
- <bullet 1 limitation>

**Methodological Clarity & Rigor:**
- <bullet 1 limitation>

End your final consolidated summary with: TERMINATE
""".strip()

# ============================================================
# 4) Config & Settings
# ============================================================
INPUT_CSV = "novelty_llm_agents_rag_based/df_with_retrieved_sections.csv"
OUTPUT_DIR = "llm_autogen_7_agents/gpt_outside_master_agent_and_tool/with_novelty_lim"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "df_gpt4o_mini_novelty_agents_leader.csv")


TEXT_COL_MAIN = "input_text_cleaned"
RELATED_COL = "relevant_papers_list"
MODEL_ID = "gpt-4o-mini"
MAX_CTX = 120_000 
RESERVED_FOR_CONVO = min(30_000, MAX_CTX / 2)
RESERVED_FOR_GEN = min(2_000, MAX_CTX / 20)

AVAILABLE_FOR_INPUT = max(4_000, MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN)
PAPER_TOK_BUDGET = min(50_000, max(2_000, int(AVAILABLE_FOR_INPUT * 0.82)))
CITATION_TOK_BUDGET = min(7_000, max(1_000, AVAILABLE_FOR_INPUT - PAPER_TOK_BUDGET))

MAX_TOKENS_PER_REPLY = 800
MAX_ROUND = 40 
TEMPERATURE = 0.2
TIMEOUT = 600
PER_RETR_PAPER_INPUT_TOK = 3000
PER_RETR_PAPER_SUMMARY_TOK = 700

tokenizer = tiktoken.encoding_for_model(MODEL_ID)

# ============================================================
# 5) LLM config for AutoGen
# ============================================================
llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "timeout": TIMEOUT,
    "temperature": TEMPERATURE,
}

def is_terminate_msg(msg) -> bool:
    c = (msg.get("content") or "").strip()
    return c == "TERMINATE" or c.endswith("\nTERMINATE") or c.endswith("TERMINATE")

# ============================================================
# 6) Helpers
# ============================================================
def tok_len(text: str) -> int:
    return len(tokenizer.encode(text or ""))

def truncate_to_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    if not text: return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens: return text
    ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
    return tokenizer.decode(ids)

def normalize_any_to_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    if isinstance(x, str): return x.strip()
    return str(x)

def parse_relevant_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x.strip())
            return parsed if isinstance(parsed, list) else [parsed]
        except: return [x.strip()]
    return [str(x)]

def build_dual_paper_input(main_text: str, b_summaries_text: str) -> str:
    main_text = truncate_to_tokens(main_text, PAPER_TOK_BUDGET, keep="head")
    b_summaries_text = truncate_to_tokens(b_summaries_text, CITATION_TOK_BUDGET, keep="head")
    return f"=== MAIN PAPER (A) ===\n{main_text}\n\n=== RELEVANT PAPERS (B) ===\n{b_summaries_text}".strip()

# ============================================================
# 7) Open AI Direct Sum Helper
# ============================================================
def openai_chat_completion(content: str, max_tokens: int, temperature: float = 0.2) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI chat/completions failed: {e}")

def build_b_summaries_from_list(lst, k=3) -> tuple[str, list[str]]:
    summaries = []
    for idx in range(k):
        if idx < len(lst):
            item_text = truncate_to_tokens(normalize_any_to_text(lst[idx]), PER_RETR_PAPER_INPUT_TOK)
            prompt = f"Summarize Paper B for limitations comparison:\n{item_text}"
            summaries.append(openai_chat_completion(prompt, PER_RETR_PAPER_SUMMARY_TOK).replace("{", ""))
        else:
            summaries.append("Paper B Summary:\n(Missing item.)")
    combined = "\n\n".join([f"--- Retrieved Paper #{i+1} ---\n{summaries[i]}" for i in range(k)]).strip()
    return combined, summaries

# ============================================================
# 8) Main pipeline
# ============================================================ 

def run_pipeline():
    df = pd.read_csv(INPUT_CSV)
    needed_cols = ["relevant_paper_sum1", "relevant_paper_sum2", "relevant_paper_sum3", "novelty_report", "full_chat_history"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = "PENDING" 
    
    agent_names_str = """
    Literature_Review_and_Data_Analysis_Agent, 
    Hypothesis_Refinement_and_Critical_Reflection_Agent, 
    Methodological_Novelty_Agent, 
    Experimental_Novelty_Agent, 
    Problem_Formulation_Novelty_Agent, 
    Writing_Claim_Novelty_Agent
    """

    for i in tqdm(range(len(df)), desc="AutoGen Limitations (GPT-4o-mini)"):
        row = df.iloc[i]
        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, ""))
        rel_list = parse_relevant_list(row.get(RELATED_COL, ""))

        if len(main_text_raw) < 200: 
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue
            
        b_combined, b_summaries = build_b_summaries_from_list(rel_list, k=3)
        combined_input = build_dual_paper_input(main_text_raw, b_combined)

        df.at[df.index[i], "relevant_paper_sum1"] = b_summaries[0]
        df.at[df.index[i], "relevant_paper_sum2"] = b_summaries[1] if len(b_summaries) > 1 else ""
        df.at[df.index[i], "relevant_paper_sum3"] = b_summaries[2] if len(b_summaries) > 2 else ""

        user_proxy = autogen.UserProxyAgent(
            name="User_Proxy", 
            human_input_mode="NEVER", 
            max_consecutive_auto_reply=999,
            code_execution_config=False,
            is_termination_msg=is_terminate_msg 
        )

        leader = autogen.AssistantAgent("Leader_Agent", system_message=HARSH_REVIEWER_POLICY + "\n\n" + get_leader_agent_prompt(combined_input, agent_names_str) + "\n\n" + NO_TOOL_NO_JSON, llm_config=llm_config)
        verifier = autogen.AssistantAgent("Verifier_Agent", system_message=VERIFIER_AGENT_PROMPT + "\n\n" + NO_TOOL_NO_JSON, llm_config=llm_config)
        lit = autogen.AssistantAgent("Literature_Review_and_Data_Analysis_Agent", system_message=literature_review_and_data_analysis_agent, llm_config=llm_config)
        hyp = autogen.AssistantAgent("Hypothesis_Refinement_and_Critical_Reflection_Agent", system_message=hypothesis_refinement_and_critical_reflection_agent, llm_config=llm_config)
        meth = autogen.AssistantAgent("Methodological_Novelty_Agent", system_message=methodological_novelty_agent, llm_config=llm_config)
        exp = autogen.AssistantAgent("Experimental_Novelty_Agent", system_message=experimental_novelty_agent, llm_config=llm_config)
        prob = autogen.AssistantAgent("Problem_Formulation_Novelty_Agent", system_message=problem_Formulation_novelty_Agent, llm_config=llm_config)
        write = autogen.AssistantAgent("Writing_Claim_Novelty_Agent", system_message=writing_claim_novelty_agent, llm_config=llm_config)
        
        master = autogen.AssistantAgent("Master_Agent", system_message=master_agent_prompt, llm_config=llm_config, is_termination_msg=is_terminate_msg)

        agents_list = [user_proxy, leader, lit, hyp, meth, exp, prob, write, verifier, master]
        
        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=MAX_ROUND,
            speaker_selection_method="round_robin",
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        task_msg = "Leader_Agent, begin coordinating the specialists to review the paper."
        
        try:
            chat_result = user_proxy.initiate_chat(manager, message=task_msg, clear_history=True)
            
            def get_master_report(chat_history) -> str:
                for m in reversed(chat_history):
                    if m.get("name") == "Master_Agent" and m.get("content"):
                        content = m.get("content").strip()
                        if content != "WAITING":
                            return content
                return "NO_REPORT_GENERATED"

            df.at[df.index[i], "novelty_report"] = get_master_report(chat_result.chat_history)
            df.at[df.index[i], "full_chat_history"] = str(chat_result.chat_history)
        except Exception as e:
            df.at[df.index[i], "novelty_report"] = f"ERROR: {repr(e)}"

        df.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    run_pipeline()