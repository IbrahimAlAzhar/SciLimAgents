"""
llm_client.py
-------------
Unified LLM interface that supports BOTH closed (OpenAI / GPT-4o-mini) and
open (Qwen 2.5 Instruct via HuggingFace transformers) backends.

The two public helpers below mirror the API used inside AI-Scientist's
`ai_scientist/llm.py`:

    get_response_from_llm(...)       -> single-shot generation w/ a system msg
    get_batch_responses_from_llm(...) -> n parallel completions for ensembling

For OpenAI we use the `openai` Python SDK.  For Qwen we instantiate a single
HuggingFace pipeline and re-use it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy imports so that environments without one of the backends still work.
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

# ---------------------------------------------------------------------------
# Identifiers --------------------------------------------------------------
# ---------------------------------------------------------------------------
OPENAI_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
}

def is_openai_model(model: str) -> bool:
    return model.lower() in OPENAI_MODELS or model.lower().startswith("gpt-")

# ===========================================================================
#  Backend wrappers
# ===========================================================================
@dataclass
class _OpenAIBackend:
    """Thin wrapper around the OpenAI chat completions API."""

    model: str
    api_key: Optional[str] = None
    max_retries: int = 5

    def __post_init__(self) -> None:
        if openai is None:
            raise RuntimeError(
                "`openai` package is not installed. `pip install openai`."
            )
        # OpenAI v1 SDK: use the Client class.
        self._client = openai.OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
        )

    def chat(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> List[str]:
        """Return a list of `n` completion strings."""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [c.message.content or "" for c in resp.choices]
            except Exception as e:  # noqa: BLE001  (broad catch on purpose)
                wait = 2 ** attempt
                print(f"[OpenAI] retry {attempt+1}/{self.max_retries} after {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError("OpenAI request failed after retries.")

@dataclass
class _QwenBackend:
    """Local HuggingFace causal-LM backend, defaults tuned for Qwen 2.5."""

    model_path: str
    cache_dir: Optional[str] = None
    dtype: str = "bf16"  # bf16 fits the 3B model in <8GB and is the Qwen default

    def __post_init__(self) -> None:
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "`transformers` and `torch` are required for Qwen. `pip install transformers torch`."
            )
        torch_dtype = (
            torch.bfloat16 if self.dtype == "bf16"
            else torch.float16 if self.dtype == "fp16"
            else torch.float32
        )
        print(f"[Qwen] Loading model from {self.model_path} ({self.dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad() if torch is not None else (lambda f: f)
    def chat(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> List[str]:
        # Qwen uses the standard chat template baked into the tokenizer.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs: List[str] = []
        for _ in range(n):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Strip the prompt; only keep the newly generated tokens.
            new_tokens = gen[0, inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
        return outputs

# ===========================================================================
#  Unified client
# ===========================================================================
class LLMClient:
    """One object that the rest of the pipeline talks to, regardless of backend."""

    def __init__(
        self,
        model: str,
        qwen_model_path: Optional[str] = None,
        qwen_cache_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        dtype: str = "bf16",
    ):
        self.model = model
        if is_openai_model(model):
            self.backend = _OpenAIBackend(model=model, api_key=openai_api_key)
        else:
            # Treat anything else as a local HuggingFace path/id (Qwen).
            if not qwen_model_path:
                raise ValueError(
                    "When using a local model you must pass --qwen-model-path."
                )
            self.backend = _QwenBackend(
                model_path=qwen_model_path,
                cache_dir=qwen_cache_dir,
                dtype=dtype,
            )

    # ------------------------------------------------------------------
    #  Mirrors AI-Scientist's `get_response_from_llm`
    # ------------------------------------------------------------------
    def get_response_from_llm(
        self,
        prompt: str,
        system_message: str,
        msg_history: Optional[List[dict]] = None,
        temperature: float = 0.75,
        max_tokens: int = 4096,
    ) -> Tuple[str, List[dict]]:
        """
        Send `prompt` (the new user turn) to the model.  Returns the assistant
        reply and the updated message history (so the caller can keep
        multi-turn reflection state, exactly like AI-Scientist does).
        """
        msg_history = list(msg_history) if msg_history else []
        messages = (
            [{"role": "system", "content": system_message}]
            + msg_history
            + [{"role": "user", "content": prompt}]
        )
        completions = self.backend.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        reply = completions[0]
        new_history = msg_history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
        return reply, new_history

    # ------------------------------------------------------------------
    #  Mirrors AI-Scientist's `get_batch_responses_from_llm`
    # ------------------------------------------------------------------
    def get_batch_responses_from_llm(
        self,
        prompt: str,
        system_message: str,
        n_responses: int,
        temperature: float = 0.75,
        max_tokens: int = 4096,
    ) -> Tuple[List[str], List[List[dict]]]:
        """Sample `n_responses` independent completions from the same prompt."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        completions = self.backend.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n_responses,
        )
        histories = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": c},
            ]
            for c in completions
        ]
        return completions, histories 
    