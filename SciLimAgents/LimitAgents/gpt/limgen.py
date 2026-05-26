import os
import pandas as pd
import autogen
import sys
import time
import ast
import re
from tqdm import tqdm
import tiktoken

# ==========================================
# 1. CONFIGURATION
# ==========================================
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"
# Safe limit for text truncation helper
SAFE_INPUT_LIMIT = 48000 

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2, 
    "timeout": 120, 
    "cache_seed": None 
}

INPUT_CSV = "data/balanced_data/df_updated_with_retrieval.csv"
# Paths
OUTPUT_SLICE = "llm_agents/gpt/limagents_master_outside/data/df_autogen_gpt_7_agents_bal_data_output.csv"

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. PROMPT DEFINITIONS (EXACTLY AS PROVIDED)
# ==========================================

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
The review_leader will ask for your feedback; respond thoroughly and ask clarifying questions if needed. When finished, inform the review_leader and provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}""" 

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
Identify issues like unstated or overly strong assumptions, incomplete theoretical analysis, errors in derivations, methods that only work under restricted conditions not clearly acknowledged, missing ablations, lack of isolation of individual contributions, or ablations that do not convincingly attribute performance gains.
The review_leader will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the review_leader and deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
Highlight issues like small-scale experiments, missing error bars or confidence intervals, unreported failed experiments, outdated or weak baselines, missing key competitors, unfair hyperparameter tuning, reliance on misleading metrics, missing standard metrics, or overemphasis on minor gains without practical or statistical significance.
The review_leader will interact with you; respond critically and seek clarification if needed. When finished, inform the review_leader and provide a bullet list of experimental evaluation-related limitations, including validation, comparisons, baselines, and metrics.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings (e.g., different datasets, domains, noise, adversarial conditions), is practical in terms of resources (time, memory, hardware, scalability), and addresses genuine deployment needs without ignoring real-world constraints.
Point out limitations like overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, poor performance under shifts, excessive training/inference demands, high resource needs restricting deployment, reliance on synthetic data, ignoring constraints like cost or latency, lack of user studies or field tests, or over-optimistic assumptions about environments.
The review_leader will seek your input; respond thoroughly and clarify ambiguities. When finished, inform the review_leader and provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations of methods, settings, concepts, or organization hindering understanding; lack of explainability or insights into decisions; and insufficient details for replication, such as code, data, hyperparameters, or protocols.
Identify issues like ambiguities, unstated assumptions, vague terms undermining comprehension, black-box behavior without explanations, missing feature importance or mechanistic understanding, poorly organized sections, missing code/data release, unreported seeds, ambiguous procedures, or lack of open science practices.
The review_leader will ask questions; respond and ask follow-up questions if needed. When done, inform the review_leader and provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, including suggestions for improvement where relevant.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for issues in collection, labeling, cleaning, representativeness, or documentation; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Point out limitations such as small or non-diverse data, labeling errors, undocumented preprocessing, data leakage, reliance on flawed datasets without validation, biased outcomes leading to discrimination, lack of fairness metrics, unreported subgroup performance, ethical oversights, or failure to discuss misuse potential.
The review_leader will consult you; provide evidence-based critique and ask clarifying questions. When done, inform the review_leader and provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}""" 

def get_leader_agent_prompt(paper_content: str, agent_names: str = "Novelty_Significance_Agent, Citation_Agent, Theoretical_Methodological_Agent, Experimental_Evaluation_Agent, Generalization_Robustness_Efficiency_Agent, Clarity_Interpretability_Reproducibility_Agent, Data_Ethics_Agent") -> str:
    return f"""You are the **Leader Agent**. Your role is to coordinate a team of specialist agents to produce a comprehensive, high-quality, and well-justified list of limitations for the provided scientific paper.
Available specialist agents: {agent_names}
PROTOCOL (follow strictly):
1. **Start**: Begin by instructing each specialist agent (one by one or in small groups) to analyze the paper for limitations in their specific domain. Phrase your request clearly.
   - Note: The **Citation_Agent** has access to cited papers; ask them to check for citation-related limitations specifically.
2. **Evaluate Responses**: When a specialist agent responds:
   - Check if their limitations are specific, evidence-based, and directly tied to the paper content.
   - Verify that they avoid vague or generic statements.
   - Ensure they provide a final bullet-point summary list when they indicate they are finished.
3. **Refine if Needed**: If an agent's output is weak, vague, incomplete, or not sufficiently grounded, ask targeted follow-up questions.
4. **Collect All Outputs**: Continue until you have strong, detailed limitation lists from all relevant specialist agents.
5. **Finalize**: Once satisfied with all individual analyses, instruct the **Master Agent** with:
   "Master Agent, here are the limitation analyses from the team: [paste or summarize all specialist outputs]. Please synthesize them into a single, consolidated, non-redundant, high-quality list of limitations, grouped by category where appropriate."
6. **Terminate**: After the Master Agent delivers the final consolidated list, respond only with "TERMINATE" and nothing else.
GLOBAL INSTRUCTIONS:
- Be strict but constructive.
- Push for specificity and evidence.
- Eliminate redundancy across agents.
- Do not generate limitations yourself — only orchestrate and refine.
PAPER CONTENT:
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent**. Your role is to receive limitation analyses from multiple specialist agents (via the Leader Agent) and produce a single, final, high-quality, consolidated list of limitations for the scientific paper.
TASK:
- Carefully read and integrate all provided specialist outputs.
- Remove redundancies (merge similar limitations).
- Prioritize the most severe and well-justified limitations.
- Preserve specificity and evidence from the original analyses.
- Organize the final list logically (e.g., group by category: Novelty & Significance, Citation Analysis, Clarity, Experimental Rigor, etc., or rank by importance).
- Ensure each limitation is clearly stated, concise, and grounded in the paper.
- Avoid introducing new limitations not raised by the specialists.
- Aim for 10–20 strong limitations (adjust based on paper quality).
OUTPUT FORMAT:
Start with a brief introductory sentence: "Here is the consolidated list of key limitations identified in the paper:"
Then provide a bulleted list:
- **Category:** Specific limitation statement (with brief explanation and evidence reference if it adds value).
If no major limitations were found across agents, state: "The paper appears methodologically sound with only minor limitations: [list them]."
Do NOT add commentary, scores, or recommendations unless explicitly present in specialist inputs.
PAPER CONTENT (for context):
{paper_content}
Wait for the Leader Agent to provide the collected specialist analyses, then generate the final list."""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None: return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry): return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except: return ""
    if not isinstance(parsed, dict): return ""

    processed = []
    for idx, (pid, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict): continue
        # Intro scan
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)

        if t_clean or a_clean or i_clean:
            processed.append(f"'Paper{idx}_Title: {t_clean}', 'Paper{idx}_Abstract': '{a_clean}', 'Paper{idx}_Introduction': '{i_clean}'.")
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
    
    print(f"  ⚠️ Truncating input: {len(tokens)} tokens -> {max_tokens} tokens.")
    return encoding.decode(tokens[:max_tokens]) + "... [TRUNCATED]"

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df1 = pd.read_csv(INPUT_CSV) 
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Initialize Columns
    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    # User Proxy (Constant)
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # --- LIST OF STANDARD SPECIALISTS (Single Input: Paper Text) ---
    standard_specialist_config = [
        ("Novelty_Significance", get_novelty_significance_prompt),
        ("Theoretical_Methodological", get_theoretical_methodological_prompt),
        ("Experimental_Evaluation", get_experimental_evaluation_prompt),
        ("Generalization_Robustness_Efficiency", get_generalization_robustness_efficiency_prompt),
        ("Clarity_Interpretability_Reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("Data_Ethics", get_data_ethics_prompt)
    ]

    # Names for the Leader to know
    all_agent_names = [name + "_Agent" for name, _ in standard_specialist_config]
    all_agent_names.append("Citation_Agent")
    agent_names_str = ", ".join(all_agent_names)

    for i in tqdm(range(0, len(df))):
        
        row = df.iloc[i]
        
        # Data Prep
        paper_text = str(row.get("input_text_cleaned", "")) 
        citation_text = extract_intro_and_abstract(row.get("cited_in", ""))
        
        # === TOKEN SAFEGUARDS ===
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=40000) 
        citation_text = truncate_text_to_tokens(citation_text, max_tokens=10000)

        # Skip short text
        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue
            
        combined_content = f"""
=== MAIN PAPER CONTENT ===
{paper_text}

=== CITED PAPERS CONTEXT ===
{citation_text}
"""

        # ==========================================
        # DYNAMIC AGENT CREATION
        # ==========================================
        
        # Notice: Master Agent is NO LONGER appended to this list
        agents_list = [user_proxy]

        # 1. Leader Agent (Needs Combined Content)
        leader_prompt = get_leader_agent_prompt(combined_content, agent_names=agent_names_str)
        leader_agent = autogen.AssistantAgent(
            name="Leader_Agent",
            system_message=leader_prompt,
            llm_config=llm_config
        )
        agents_list.append(leader_agent)

        # 2. Master Agent (Needs Combined Content) -> Instantiated but kept outside the group
        master_prompt = get_master_agent_prompt(combined_content)
        master_agent = autogen.AssistantAgent(
            name="Master_Agent",
            system_message=master_prompt,
            llm_config=llm_config
        )

        # 3. Citation Agent (Needs Paper + Citation Content)
        citation_sys_msg = get_citation_agent_prompt(paper_text, citation_text)
        citation_agent = autogen.AssistantAgent(
            name="Citation_Agent",
            system_message=citation_sys_msg,
            llm_config=llm_config
        )
        agents_list.append(citation_agent)

        # 4. Standard Specialists (Need Paper Content Only)
        for agent_name_base, prompt_func in standard_specialist_config:
            sys_msg = prompt_func(paper_text)
            
            specialist = autogen.AssistantAgent(
                name=f"{agent_name_base}_Agent",
                system_message=sys_msg,
                llm_config=llm_config
            )
            agents_list.append(specialist)

        # 5. Group Chat (Master Agent excluded)
        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=45, 
            speaker_selection_method="auto" 
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # 6. Initiate Main Group Chat
        task_msg = f"""Leader_Agent, please start the limitation analysis process for the provided paper content."""

        try:
            # RUN PHASE 1: The Group Chat
            group_chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            # EXTRACT HAND-OFF: Find the Leader's final message intended for the Master
            handoff_message = ""
            for msg in group_chat_result.chat_history:
                if msg.get("name") == "Leader_Agent" and "Master Agent" in str(msg.get("content")):
                    handoff_message = msg.get("content")
            
            # Fallback in case the Leader failed to use the exact phrasing dictated by its prompt
            if not handoff_message:
                handoff_message = f"Master Agent, here are the limitation analyses from the team:\n{str(group_chat_result.chat_history)}"

            # RUN PHASE 2: Independent Chat with Master Agent
            master_chat_result = user_proxy.initiate_chat(
                master_agent,
                message=handoff_message,
                clear_history=True
            )

            # Capture Output from Master Agent's separate chat
            master_messages = []
            for msg in master_chat_result.chat_history:
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    clean_content = content.replace("TERMINATE", "").strip()
                    
                    if len(clean_content) > 50:
                        master_messages.append(clean_content)
            
            if master_messages:
                final_output = master_messages[-1] 
            else:
                final_output = "NO_OUTPUT_FROM_MASTER"

            col_idx_lim = df.columns.get_loc("final_merged_limitations")
            col_idx_hist = df.columns.get_loc("full_chat_history")
            
            df.iat[i, col_idx_lim] = final_output
            
            # Save the combined history of both phases for review
            combined_history = {
                "group_chat": group_chat_result.chat_history,
                "master_chat": master_chat_result.chat_history
            }
            df.iat[i, col_idx_hist] = str(combined_history)
            
        except Exception as e:
            print(f"Error on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = f"ERROR: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_SLICE, index=False)
        time.sleep(1)

    df.to_csv(OUTPUT_SLICE, index=False)

if __name__ == "__main__":
    run_pipeline()