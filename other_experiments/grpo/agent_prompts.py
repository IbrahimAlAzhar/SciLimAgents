"""
Agent Prompts (v2)
===================
Key v2 changes:
  - UNIFIED WORKER PROMPT: single function with role parameter.
    One model serves all 7 roles; the role is injected into the prompt.
  - ENHANCED LEADER: decides which workers to call, how many rounds,
    provides structured feedback, assesses coverage gaps.
  - Master: same consolidation role.
"""

from typing import List, Optional
from config import ROLE_DESCRIPTIONS, WORKER_ROLES

# ================================================================
# ROLE DETAIL BLOCKS (injected into unified worker prompt)
# ================================================================

ROLE_INSTRUCTIONS = {
    "novelty_significance": """Focus on NOVELTY and SIGNIFICANCE limitations:
- Are contributions truly novel or merely incremental?
- Are claims of importance overstated?
- Is motivation or real-world relevance weak?
- Is this rebranding existing ideas without substantial improvement?
- Is differentiation from prior work clear and honest?
- Are there ignored alternatives that diminish perceived impact?""",

    "theoretical_methodological": """Focus on THEORETICAL and METHODOLOGICAL limitations:
- Are there flaws in the core method or unrealistic assumptions?
- Are proofs missing, incomplete, or contain logical gaps?
- Are ablation studies insufficient or missing?
- Is the approach oversimplified for the problem complexity?
- Is there a failure to explain why the method works?
- Are there missing components in the theoretical framework?""",

    "experimental_evaluation": """Focus on EXPERIMENTAL EVALUATION limitations:
- Are there insufficient experimental runs or lack of statistical significance?
- Are results cherry-picked or evaluated under narrow conditions?
- Are baselines inappropriate, outdated, or missing?
- Are metrics misleading or comparisons incomplete?
- Is the analysis of results superficial?
- Are error bars, confidence intervals, or variance reports absent?""",

    "generalization_robustness_efficiency": """Focus on GENERALIZATION, ROBUSTNESS, and EFFICIENCY limitations:
- Does the method perform well beyond tested settings?
- Is robustness to noise, distribution shift, or adversarial inputs analyzed?
- Are computational costs or resource requirements impractical?
- Is scalability to larger problems demonstrated?
- Are deployment constraints addressed?
- Is there analysis of failure modes?""",

    "clarity_interpretability_reproducibility": """Focus on CLARITY, INTERPRETABILITY, and REPRODUCIBILITY limitations:
- Are explanations unclear or notation inconsistent?
- Are implementation details sufficient for reproducibility?
- Is model behavior or decision-making interpretable?
- Are hyperparameters, seeds, and setup fully documented?
- Is code or data publicly available?
- Are figures and tables clear and well-labeled?""",

    "data_ethics": """Focus on DATA INTEGRITY, BIAS, FAIRNESS, and ETHICS limitations:
- Are there data quality issues or potential biases?
- Is fairness across demographic groups analyzed?
- Are privacy or ethical considerations addressed?
- Is annotation quality and inter-annotator agreement reported?
- Are potential societal impacts discussed?
- Is data provenance and licensing clear?""",

    "citation": """Focus on CITATION and RELATED WORK limitations:
- Does the paper fail to cite or address key related work?
- Is prior work misinterpreted or selectively cited?
- Are important baselines from related work missing?
- Does the paper exaggerate its novelty relative to existing literature?
- Are concurrent or very recent relevant works acknowledged?
- Is the positioning within the broader field accurate?""",
}

# ================================================================
# UNIFIED WORKER PROMPT (single model, multiple roles)
# ================================================================

def get_unified_worker_prompt(paper_content: str, role: str) -> str:
    """
    Build a worker prompt for any role.
    The SAME model handles all roles; the role is specified in the prompt.
    This is the key design for GRPO training of one unified worker model.
    """
    role_desc = ROLE_DESCRIPTIONS.get(role, role)
    role_instructions = ROLE_INSTRUCTIONS.get(role, f"Focus on {role} limitations.")

    return f"""You are an expert scientific paper reviewer.
Your assigned review role is: **{role_desc}**

{role_instructions}

INSTRUCTIONS:
1. Read the paper carefully.
2. Identify 3-8 specific limitations within your assigned role.
3. For each limitation, provide:
   - A clear, specific statement of the limitation
   - Evidence or reference to the specific section/table/figure in the paper
   - Why this matters for the paper's contributions
4. Be critical but fair. Avoid vague or generic criticisms.
5. Format as a numbered list.

PAPER:
{paper_content}"""

# ================================================================
# ENHANCED LEADER PROMPT (decision-making + feedback)
# ================================================================

def get_leader_prompt(paper_content: str) -> str:
    """
    Enhanced leader prompt that makes three decisions:
    1. Which worker roles to activate (based on paper type/content)
    2. What specific guidance to give each worker
    3. After reviewing worker outputs, provide structured feedback

    The leader's output is parsed to extract its decisions.
    """
    roles_menu = "\n".join(
        f"  - {role}: {desc}"
        for role, desc in ROLE_DESCRIPTIONS.items()
    )

    return f"""You are the **Leader Agent** coordinating a team of specialist reviewers to identify limitations in a scientific paper.

AVAILABLE SPECIALIST ROLES:
{roles_menu}

YOUR TASKS:

**PHASE 1 — PLANNING (output this now):**
Analyze the paper and decide:

1. **SELECTED_WORKERS**: Choose which specialist roles are most relevant for this paper.
   Not all papers need all 7 roles. Select 3-7 roles based on the paper's content.
   Output as: SELECTED_WORKERS: [role1, role2, ...]

2. **WORKER_GUIDANCE**: For each selected worker, provide 1-2 sentences of specific guidance.
   What should they focus on given THIS paper's particular content?
   Output as:
   GUIDANCE_FOR_<role>: <specific instructions>

3. **NUM_ROUNDS**: How many rounds of review are needed? (1 for straightforward papers, 2 for complex ones)
   Output as: NUM_ROUNDS: <1 or 2>

4. **COVERAGE_ASSESSMENT**: What are the most critical limitation categories for this paper type?
   Output as: CRITICAL_CATEGORIES: [category1, category2, ...]

PAPER:
{paper_content}"""

def get_leader_feedback_prompt(
    paper_content: str,
    worker_outputs: str,
    round_number: int = 1,
) -> str:
    """
    Leader reviews worker outputs and provides structured feedback.
    Used after workers produce their first round of limitations.
    """
    return f"""You are the **Leader Agent**. You have received analyses from your specialist workers.

Review their outputs and provide:

1. **COVERAGE_GAPS**: Which important limitation categories are missing or underrepresented?
   Output as: COVERAGE_GAPS: [gap1, gap2, ...]

2. **QUALITY_FEEDBACK**: For each worker, rate quality (good/needs_improvement) and give specific feedback.
   Output as:
   FEEDBACK_FOR_<role>: <quality_rating> — <specific feedback>

3. **ADDITIONAL_WORKERS**: Should any additional specialist roles be activated for round {round_number + 1}?
   Output as: ADDITIONAL_WORKERS: [role1, role2, ...] or ADDITIONAL_WORKERS: none

4. **PRIORITY_GUIDANCE**: What should the Master Agent prioritize when consolidating?
   Output as: PRIORITY_GUIDANCE: <instructions for master>

5. **FINAL_ASSESSMENT**: Overall quality score (1-10) and summary.
   Output as: FINAL_ASSESSMENT: <score>/10 — <summary>

WORKER OUTPUTS:
{worker_outputs}

PAPER (for reference):
{paper_content}"""

# ================================================================
# MASTER PROMPT (same role, takes leader feedback too)
# ================================================================

def get_master_prompt(
    paper_content: str,
    worker_outputs: str,
    leader_feedback: str = "",
) -> str:
    """
    Master consolidates worker outputs into final limitation list.
    Now also receives leader feedback/priority guidance.
    """
    leader_section = ""
    if leader_feedback:
        leader_section = f"""
LEADER FEEDBACK & PRIORITY GUIDANCE:
{leader_feedback}
"""

    return f"""You are the **Master Agent**. Produce ONE final consolidated limitation list.

Rules:
- Integrate all specialist outputs into a coherent list
- Remove redundancy (merge similar limitations)
- Keep specificity and evidence references
- Do NOT invent new limitations beyond what specialists identified
- Group by category
- Rank by severity/importance
- Follow the Leader's priority guidance if provided
{leader_section}
SPECIALIST ANALYSES:
{worker_outputs}

PAPER (for context):
{paper_content}

Output format:
Start with: "Here is the consolidated list of key limitations:"
Then provide numbered limitations grouped by category."""

# ================================================================
# LEADER OUTPUT PARSER
# ================================================================

def parse_leader_planning(leader_output: str) -> dict:
    """
    Parse leader's planning output to extract decisions.
    Returns structured decisions dict.
    """
    import re

    decisions = {
        "selected_workers": [],
        "worker_guidance": {},
        "num_rounds": 1,
        "critical_categories": [],
        "raw": leader_output,
    }

    # Parse SELECTED_WORKERS
    match = re.search(
        r"SELECTED_WORKERS:\s*\[([^\]]+)\]", leader_output, re.IGNORECASE
    )
    if match:
        roles_str = match.group(1)
        # Clean and split
        roles = [r.strip().strip("'\"") for r in roles_str.split(",")]
        # Fuzzy match to valid role names
        valid_roles = []
        for r in roles:
            r_lower = r.lower().replace(" ", "_")
            for valid in WORKER_ROLES:
                if r_lower in valid or valid in r_lower:
                    if valid not in valid_roles:
                        valid_roles.append(valid)
                    break
        decisions["selected_workers"] = valid_roles if valid_roles else list(WORKER_ROLES[:5])
    else:
        # Default: use all workers
        decisions["selected_workers"] = list(WORKER_ROLES)

    # Parse GUIDANCE_FOR_<role>
    for role in WORKER_ROLES:
        pattern = rf"GUIDANCE_FOR_{role.upper()}:\s*(.+?)(?=\n(?:GUIDANCE_FOR|NUM_ROUNDS|CRITICAL|SELECTED|$))"
        match = re.search(pattern, leader_output, re.IGNORECASE | re.DOTALL)
        if match:
            decisions["worker_guidance"][role] = match.group(1).strip()

    # Parse NUM_ROUNDS
    match = re.search(r"NUM_ROUNDS:\s*(\d+)", leader_output, re.IGNORECASE)
    if match:
        decisions["num_rounds"] = min(int(match.group(1)), 2)

    # Parse CRITICAL_CATEGORIES
    match = re.search(
        r"CRITICAL_CATEGORIES:\s*\[([^\]]+)\]", leader_output, re.IGNORECASE
    )
    if match:
        decisions["critical_categories"] = [
            c.strip().strip("'\"") for c in match.group(1).split(",")
        ]

    return decisions

def parse_leader_feedback(feedback_output: str) -> dict:
    """
    Parse leader's feedback output.
    """
    import re

    feedback = {
        "coverage_gaps": [],
        "worker_feedback": {},
        "additional_workers": [],
        "priority_guidance": "",
        "final_score": 0,
        "final_summary": "",
        "raw": feedback_output,
    }

    # Coverage gaps
    match = re.search(
        r"COVERAGE_GAPS:\s*\[([^\]]+)\]", feedback_output, re.IGNORECASE
    )
    if match:
        feedback["coverage_gaps"] = [
            g.strip().strip("'\"") for g in match.group(1).split(",")
        ]

    # Worker feedback
    for role in WORKER_ROLES:
        pattern = rf"FEEDBACK_FOR_{role.upper()}:\s*(.+?)(?=\n(?:FEEDBACK_FOR|ADDITIONAL|PRIORITY|FINAL|$))"
        match = re.search(pattern, feedback_output, re.IGNORECASE | re.DOTALL)
        if match:
            feedback["worker_feedback"][role] = match.group(1).strip()

    # Additional workers
    match = re.search(
        r"ADDITIONAL_WORKERS:\s*\[([^\]]+)\]", feedback_output, re.IGNORECASE
    )
    if match and "none" not in match.group(1).lower():
        feedback["additional_workers"] = [
            w.strip().strip("'\"") for w in match.group(1).split(",")
        ]

    # Priority guidance
    match = re.search(
        r"PRIORITY_GUIDANCE:\s*(.+?)(?=\n(?:FINAL_ASSESSMENT|$))",
        feedback_output, re.IGNORECASE | re.DOTALL,
    )
    if match:
        feedback["priority_guidance"] = match.group(1).strip()

    # Final assessment
    match = re.search(
        r"FINAL_ASSESSMENT:\s*(\d+)/10\s*[—-]\s*(.+)",
        feedback_output, re.IGNORECASE,
    )
    if match:
        feedback["final_score"] = int(match.group(1))
        feedback["final_summary"] = match.group(2).strip()

    return feedback