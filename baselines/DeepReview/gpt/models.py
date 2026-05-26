"""
models.py
=========
Backbone LLM wrappers used by the DeepReview-baseline limitation pipeline.

We expose two interchangeable classes that share the same `chat(...)` API:

    QwenLLM        - local Qwen-2.5-3B-Instruct served by vLLM (fits in 40 GB).
    OpenAIChatLLM  - OpenAI Chat Completions, default model = gpt-4o-mini.

Both classes accept a list of "message dicts" (OpenAI style) and return a
plain-text completion string.  The pipeline never has to care which backend
is in use.

The Qwen wrapper mirrors the original DeepReview code which used vLLM with
`temperature=0.4` and `top_p=0.95`.  We keep those defaults.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Qwen / vLLM backend
# ---------------------------------------------------------------------------
class QwenLLM:
    """Local Qwen-2.5-3B-Instruct served by vLLM.

    Mirrors the way DeepReview's `deep_reviewer.py` initializes its model.
    Designed to fit inside a single 40 GB GPU:
        - bf16 weights (~6 GB)
        - vLLM KV-cache governed by gpu_memory_utilization (default 0.85)
        - max_model_len defaults to 12288 (enough for our truncated papers).
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 12288,
        tensor_parallel_size: int = 1,
        seed: int = 42,
    ):
        # vLLM and HuggingFace tokenizer are imported here so that the
        # OpenAI-only path does not require a CUDA + vLLM install.
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams  # noqa: F401  (re-exported)

        self._SamplingParams = SamplingParams

        # Tokenizer is needed for `apply_chat_template` (Qwen has its own
        # ChatML-like template).  Using the same path keeps tokenizer + weights
        # in sync.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )

        # vLLM engine.  `dtype="bfloat16"` is the safe default for Qwen 2.5.
        self.llm = LLM(
            model=model_id,
            download_dir=cache_dir,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            trust_remote_code=True,
        )
        self.max_model_len = max_model_len

    # ---------------------------------------------------------------- chat
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 800,
        temperature: float = 0.4,
        top_p: float = 0.95,
    ) -> str:
        """Run one chat completion using Qwen's chat template.

        `messages` follows the OpenAI convention:
            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."},
             ...]
        """
        # Apply the chat template -> single string that vLLM tokenizes
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

# ---------------------------------------------------------------------------
# OpenAI / gpt-4o-mini backend
# ---------------------------------------------------------------------------
class OpenAIChatLLM:
    """Thin wrapper around the OpenAI Chat Completions API.

    We use `gpt-4o-mini` by default — fast, cheap, and good enough to
    serve as a baseline closed-model reviewer.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 4,
        retry_delay: float = 3.0,
    ):
        # Lazy import so that machines without `openai` installed can still
        # use the Qwen path.
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided. Pass --openai-api-key or set the "
                "OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ---------------------------------------------------------------- chat
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 800,
        temperature: float = 0.4,
        top_p: float = 0.95,
    ) -> str:
        """Call OpenAI Chat Completions with simple exponential backoff."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:                 # broad: API errors vary
                last_exc = e
                wait = self.retry_delay * (2 ** attempt)
                print(f"[OpenAIChatLLM] retry {attempt+1}/{self.max_retries} "
                      f"after error: {e!r}.  Sleeping {wait:.1f}s ...")
                time.sleep(wait)
        # Give up — return an empty string so the pipeline can keep going.
        print(f"[OpenAIChatLLM] giving up after {self.max_retries} retries: "
              f"{last_exc!r}")
        return ""

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_llm(args) -> "QwenLLM | OpenAIChatLLM":
    """Build the LLM backend selected on the command line.

    `args` is the Namespace returned by `config.parse_args()`.
    """
    if args.model == "qwen":
        return QwenLLM(
            model_id=args.qwen_model_id,
            cache_dir=args.qwen_cache_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )
    elif args.model == "gpt-4o-mini":
        return OpenAIChatLLM(
            model=args.openai_model,
            api_key=args.openai_api_key,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}") 