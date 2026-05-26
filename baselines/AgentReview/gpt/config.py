"""
config.py
=========
Configuration for AgentReview-Limitations: an adaptation of the AgentReview
framework (Jin et al., EMNLP 2024) for the task of generating limitations of
scientific documents.

Original paper:
    Jin, Y., Zhao, Q., Wang, Y., Chen, H., Zhu, K., Xiao, Y., Wang, J.
    "AgentReview: Exploring Peer Review Dynamics with LLM Agents." EMNLP 2024.
    https:/arxiv.org/abs/2406.12708
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ----------------------------------------------------------------------------
# LLM backend configuration
# ----------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for the LLM backend.

    The original AgentReview paper uses gpt-4-1106-preview. We default to a
    similar GPT-4 model but allow easy swapping.
    """
    # Provider: "openai" or "azure"
    provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI / Azure model identifier
    model: str = os.getenv("LLM_MODEL", "gpt-4o")

    # Generation hyper-parameters
    temperature: float = 0.2
    max_tokens: int = 1500
    top_p: float = 1.0

    # API credentials (read from environment by default)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "")
    azure_deployment: str = os.getenv("AZURE_DEPLOYMENT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    # Robustness
    max_retries: int = 4
    retry_backoff_seconds: float = 5.0
    request_timeout: int = 90

# ----------------------------------------------------------------------------
# Agent (reviewer / area-chair) characteristics
# ----------------------------------------------------------------------------
# These mirror the three latent dimensions identified in Jin et al. (2024):
#   * commitment      -> responsible / irresponsible
#   * intention       -> benign / malicious
#   * knowledgeability-> knowledgeable / unknowledgeable
# and the three AC styles: authoritarian / conformist / inclusive.

@dataclass
class ReviewerProfile:
    """A single reviewer's latent characteristics."""
    name: str
    knowledgeability: str = "knowledgeable"   # {knowledgeable, unknowledgeable}
    commitment: str = "responsible"           # {responsible, irresponsible}
    intention: str = "benign"                 # {benign, malicious}

@dataclass
class AreaChairProfile:
    """Area-chair style."""
    style: str = "inclusive"   # {inclusive, authoritarian, conformist}

# ----------------------------------------------------------------------------
# Pipeline-level experiment configuration
# ----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """High level configuration for one run of the limitation pipeline."""

    # Dataset I/O ----------------------------------------------------------
    input_csv: str = (
        "df_updated_with_retrieval.csv"
    )
    text_column: str = "input_text_cleaned"
    citations_column: str = "cited_in_text"   # may be missing for some rows
    id_column: Optional[str] = None           # auto-uses dataframe index if None
    output_dir: str = "./outputs/limitations_run"

    # Pipeline behaviour ---------------------------------------------------
    num_reviewers: int = 3
    enable_rebuttal: bool = True              # Phase II + III on/off
    use_citation_context: bool = True         # feed `cited_in_text` to agents

    # Truncation (LLM context window protection) ---------------------------
    max_paper_chars: int = 18000
    max_citation_chars: int = 4000

    # Reviewer & AC defaults -----------------------------------------------
    reviewers: List[ReviewerProfile] = field(default_factory=lambda: [
        ReviewerProfile(name="R1",
                        knowledgeability="knowledgeable",
                        commitment="responsible",
                        intention="benign"),
        ReviewerProfile(name="R2",
                        knowledgeability="knowledgeable",
                        commitment="responsible",
                        intention="benign"),
        ReviewerProfile(name="R3",
                        knowledgeability="knowledgeable",
                        commitment="responsible",
                        intention="benign"),
    ])
    area_chair: AreaChairProfile = field(
        default_factory=lambda: AreaChairProfile(style="inclusive")
    )

    # Run options ----------------------------------------------------------
    start_index: int = 0
    end_index: Optional[int] = None     # None -> run to the end of the CSV
    save_every: int = 5                 # checkpoint frequency
    verbose: bool = True

# Convenient global default ---------------------------------------------------
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig() 
