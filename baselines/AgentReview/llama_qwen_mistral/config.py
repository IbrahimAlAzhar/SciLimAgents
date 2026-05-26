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

Extended in this version to support local Hugging Face models
(Llama-3 / Mistral / Qwen) in addition to OpenAI / Azure OpenAI.
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

    `provider` selects the backend:
        "openai"  -> OpenAI HTTP API
        "azure"   -> Azure OpenAI HTTP API
        "hf"      -> local Hugging Face model via `transformers`
    """
    # Provider: "openai" | "azure" | "hf"
    provider: str = os.getenv("LLM_PROVIDER", "openai")

    # For OpenAI/Azure this is the model/deployment name.
    # For HF this can be left empty (use hf_model_id) or set to the HF id.
    model: str = os.getenv("LLM_MODEL", "gpt-4o")

    # Generation hyper-parameters (apply to all backends)
    temperature: float = 0.2
    max_tokens: int = 1500          # reused as max_new_tokens for HF
    top_p: float = 1.0

    # ----- OpenAI -----
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ----- Azure OpenAI -----
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "")
    azure_deployment: str = os.getenv("AZURE_DEPLOYMENT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    # ----- Hugging Face local -----
    # HF hub id (e.g. "meta-llama/Meta-Llama-3-8B-Instruct") OR a local path.
    hf_model_id: str = os.getenv("HF_MODEL_ID", "")
    # Cache directory for HF downloads.
    hf_cache_dir: str = os.getenv("HF_CACHE_DIR", "")
    # Numeric precision: "bf16" (recommended), "fp16", or "fp32".
    hf_dtype: str = os.getenv("HF_DTYPE", "bf16")
    # Device map: "auto" | "cuda" | "cuda:0" | "cpu".
    hf_device: str = os.getenv("HF_DEVICE", "auto")

    # ----- Robustness -----
    max_retries: int = 4
    retry_backoff_seconds: float = 5.0
    request_timeout: int = 90       # only used by OpenAI/Azure

# ----------------------------------------------------------------------------
# Agent (reviewer / area-chair) characteristics
# ----------------------------------------------------------------------------

@dataclass
class ReviewerProfile:
    """A single reviewer's latent characteristics (Jin et al. 2024)."""
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
    """High-level configuration for one run of the limitation pipeline."""

    # Dataset I/O ----------------------------------------------------------
    input_csv: str = (
        ""
        "data/balanced_data/df_updated_with_retrieval.csv"
    )
    text_column: str = "input_text_cleaned"
    citations_column: str = "cited_in_text"
    id_column: Optional[str] = None
    output_dir: str = "./outputs/limitations_run"

    # Pipeline behaviour ---------------------------------------------------
    num_reviewers: int = 3
    enable_rebuttal: bool = True
    use_citation_context: bool = True

    # Truncation (LLM context-window protection) ---------------------------
    # NOTE: Llama-3-8B-Instruct has only 8K tokens of context, so for that
    # model we recommend dropping max_paper_chars to ~12000.
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
    end_index: Optional[int] = None
    save_every: int = 5
    verbose: bool = True

DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig() 