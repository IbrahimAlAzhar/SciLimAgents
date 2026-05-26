import os
import pandas as pd
import autogen
import sys
import time
import ast
import re
from tqdm import tqdm
import tiktoken  # Required for accurate token counting

# ==========================================
# 1. CONFIGURATION
# ==========================================

os.environ['OPENAI_API_KEY'] = ''
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 128000
# We leave buffer for system prompts and generation (approx 18k buffer)
SAFE_INPUT_LIMIT = 110000 

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2, 
    "timeout": 120, 
    "cache_seed": None 
}

# Paths
INPUT_CSV = "df.csv"
OUTPUT_SLICE = "df1.csv"

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. DATA CLEANING & TOKEN HELPERS
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
    """
    Checks token count for gpt-4o-mini and truncates if it exceeds the limit.
    """
    if not text:
        return ""
    
    try:
        # GPT-4o and 4o-mini use the 'o200k_base' encoding
        encoding = tiktoken.get_encoding("o200k_base")
    except:
        # Fallback if specific encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # 
    print(f"  ⚠️ Truncating input: {len(tokens)} tokens -> {max_tokens} tokens.")
    
    # Slice the tokens and decode back to string
    truncated_text = encoding.decode(tokens[:max_tokens])
    return truncated_text + "... [TRUNCATED]"

# ==========================================
# 3. PROMPTS
# ==========================================

GLOBAL_CONTEXT_NOTE = """
[SYSTEM NOTE]: The MAIN ARTICLE and CITED PAPERS are provided in the first message.
Do not use external tools. Analyze only the provided text.
"""

def get_agent_prompts():
    return {
        "Extractor": f"""You are the **Extractor Agent**. 
Task: Extract explicitly stated limitations from the Main Article.
- Quote verbatim where possible.
- If none, state "No explicit limitations."
{GLOBAL_CONTEXT_NOTE}""",

        "Analyzer": f"""You are the **Analyzer Agent**. 
Task: Infer limitations NOT explicitly stated.
- Focus on methodology, sample size, and statistical flaws.
{GLOBAL_CONTEXT_NOTE}""",

        "Reviewer": f"""You are the **Reviewer Agent**.
Task: Evaluate Technical Correctness, Novelty, and Rigor (ICLR style).
- Identify weak baselines or overclaimed novelty.
{GLOBAL_CONTEXT_NOTE}""",

        "Citation": f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
{GLOBAL_CONTEXT_NOTE}""",

        "Master": f"""You are the **Master Agent**.
Task: Synthesize reports from Extractor, Analyzer, Reviewer, and Citation agents.
- Merge into a single, cohesive, Numbered List.
- Remove redundancies.
- Format: "1. [Statement]: Justification [Source Agent]."
{GLOBAL_CONTEXT_NOTE}""",

        "Leader": f"""You are the **Leader Agent**.
PROTOCOL:
1. **Instruct**: Ask Extractor, Analyzer, Reviewer, and Citation agents for reports.
2. **Wait**: Let them generate responses.
3. **Merge**: Instruct **Master Agent** to merge everything.
4. **Terminate**: Once Master Agent responds, reply "TERMINATE".
{GLOBAL_CONTEXT_NOTE}"""
    }

# ==========================================
# 4. SWARM SETUP
# ==========================================

def create_swarm():
    prompts = get_agent_prompts()
    
    leader = autogen.AssistantAgent(name="Leader_Agent", system_message=prompts["Leader"], llm_config=llm_config)
    master = autogen.AssistantAgent(name="Master_Agent", system_message=prompts["Master"], llm_config=llm_config)
    
    specialists = []
    for name in ["Extractor", "Analyzer", "Reviewer", "Citation"]:
        agent = autogen.AssistantAgent(
            name=f"{name}_Agent",
            system_message=prompts[name],
            llm_config=llm_config
        )
        specialists.append(agent)

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    all_agents = [user_proxy, leader, master] + specialists
    
    groupchat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=25, 
        speaker_selection_method="auto"
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    return user_proxy, manager

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df1 = pd.read_csv(INPUT_CSV) 
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 1. Initialize Columns (if not exist)
    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    # Initialize agents once
    user_proxy, manager = create_swarm()

    # 3. Iterate over RANGE directly to ensure we modify the main dataframe
    for i in tqdm(range(0, len(df))):
        
        # Access the row safely
        row = df.iloc[i]
        
        # Prepare Data
        paper_text = str(row.get("input_text_cleaned", "")) 
        print('paper text',paper_text)
        citation_text = extract_intro_and_abstract(row.get("cited_in", ""))
        print('citation text',citation_text)

        # Check Total Length & Truncate if necessary
        # We combine them to check the total load on the context window
        combined_text = paper_text + "\n" + citation_text
        
        # If combined is too large, we truncate the components proportionally
        # (Here we prioritize paper_text, so we truncate citation first if needed, or simply truncate the final combined string)
        
        # Apply Truncation
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=80000) # Give bulk to paper
        citation_text = truncate_text_to_tokens(citation_text, max_tokens=30000) # Give rest to citations

        # Skip short text
        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue

        # Initial Message
        task_msg = f"""
        Here is the data for analysis.
        
        === MAIN ARTICLE BEGIN ===
        {paper_text}
        === MAIN ARTICLE END ===

        === CITED PAPERS INFO BEGIN ===
        {citation_text}
        === CITED PAPERS INFO END ===

        Leader Agent, please coordinate the team. Ensure the Master Agent merges the final list.
        """

        try:
            # Start Chat
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            # --- FIXED EXTRACTION LOGIC ---
            master_messages = []
            
            for msg in chat_result.chat_history:
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    # Capture content even if it has TERMINATE, just clean it
                    clean_content = content.replace("TERMINATE", "").strip()
                    
                    if len(clean_content) > 50:
                        master_messages.append(clean_content)
            
            if master_messages:
                final_output = "\n\n".join(master_messages)
            else:
                final_output = "NO_OUTPUT_FROM_MASTER"

            # Update Main DataFrame using Integer Location (.iat)
            # This is the safest way to ensure the value sticks
            col_idx_lim = df.columns.get_loc("final_merged_limitations")
            col_idx_hist = df.columns.get_loc("full_chat_history")
            
            df.iat[i, col_idx_lim] = final_output
            df.iat[i, col_idx_hist] = str(chat_result.chat_history)
            
        except Exception as e:
            print(f"Error on row {i}: {e}")
            # Ensure we write the error so it's not "PENDING"
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = f"ERROR: {e}"

        # Periodic Save
        if i % 5 == 0:
            # Save Full
            df.to_csv(OUTPUT_SLICE, index=False)
        time.sleep(1)

    # Final Save
    df.to_csv(OUTPUT_SLICE, index=False)

if __name__ == "__main__":
    run_pipeline()