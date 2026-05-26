"""
llm_client.py
-------------

Two LLM client implementations + a factory:

  * GPTClient      -- OpenAI Chat Completions API, with retries.
  * HFLocalClient  -- Local HuggingFace `transformers` inference for
                      Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3,
                      and Qwen-2.5-3B-Instruct (any chat-templated model).
  * make_client(...)  -- factory that picks one based on `--backend`.

Both clients share the same `chat(system, user, temperature, max_tokens)`
interface, so agents.py treats them identically.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ===========================================================================
# OpenAI client (gpt-4o-mini, gpt-4o, ...)
# ===========================================================================

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _HAS_OPENAI = False

class GPTClient:
    """Minimal OpenAI Chat Completions client with retry-on-transient logic."""

    _RETRIABLE_SUBSTRINGS = (
        "rate limit", "rate_limit", "429",
        "timeout", "timed out",
        "500", "502", "503", "504",
        "connection", "connect",
        "internal server error", "server error",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: int = 120,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        retry_initial_delay: float = 1.0,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError(
                "The 'openai' package (>=1.0.0) is required for --backend openai. "
                "Install it with: pip install 'openai>=1.0.0'"
            )
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Either pass api_key=... or "
                "export OPENAI_API_KEY before running."
            )
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_initial_delay = retry_initial_delay

    def _is_retriable(self, exc: Exception) -> bool:
        s = str(exc).lower()
        return any(sub in s for sub in self._RETRIABLE_SUBSTRINGS)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries - 1 and self._is_retriable(e):
                    delay = self.retry_initial_delay * (
                        self.retry_backoff ** attempt
                    )
                    delay += random.uniform(0, delay * 0.1)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
                    continue
                raise
        raise last_err if last_err else RuntimeError("LLM call failed without exception")

# ===========================================================================
# Local HuggingFace client (Llama / Mistral / Qwen / ...)
# ===========================================================================

class HFLocalClient:
    """Run any chat-templated HuggingFace causal LM locally on a single GPU.

    Tested on:
      * meta-llama/Meta-Llama-3-8B-Instruct
      * mistralai/Mistral-7B-Instruct-v0.3
      * Qwen/Qwen2.5-3B-Instruct (HF id) or a local path

    Notes:
      - Loads the model once at init time. Reuse the same instance across all
        agents to avoid reloading weights.
      - Calls are NOT thread-safe -- the GPU and the model state are shared.
        main.py auto-disables --parallel-grounding when --backend hf.
      - Uses the model's bundled chat template via tokenizer.apply_chat_template.
        For models that don't accept a 'system' role (some Mistral builds),
        we transparently merge the system message into the user turn.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        hf_token: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        # Lazy-import torch / transformers so the openai-only path doesn't
        # require these heavy dependencies.
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers + torch are required for --backend hf. "
                "Install with: pip install 'transformers>=4.42' 'torch>=2.0' accelerate"
            ) from e

        self._torch = torch
        logger.info("Loading HF tokenizer + model: %s (cache_dir=%s)", model_id, cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        # Many causal LMs ship without a pad token. Use eos to keep
        # `generate()` happy with attention masks.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        load_kwargs = dict(
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,  # 'auto' lets accelerate handle placement
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        if attn_implementation:
            # e.g. 'flash_attention_2' if installed; safe to omit.
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.device = device
        self.model_id = model_id
        logger.info("HF model ready: %s", model_id)

    # ----------------------------------------------------------------------

    def _apply_chat_template(self, system: str, user: str) -> str:
        """Render a chat template, falling back if the model rejects 'system'."""
        msgs_with_system = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                msgs_with_system, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:  # noqa: BLE001
            # Some Mistral / older Llama templates only allow [user, assistant].
            logger.debug(
                "Chat template rejected 'system' role (%s); merging into user turn.", e,
            )
            merged = (system.strip() + "\n\n" + user.strip()) if system else user
            msgs = [{"role": "user", "content": merged}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        prompt_text = self._apply_chat_template(system, user)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        # Sampling vs greedy. We pass top_p only when sampling.
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with self._torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        # Strip the prompt prefix so we only decode the new tokens.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

# ===========================================================================
# Factory
# ===========================================================================

def make_client(
    backend: str,
    model: str,
    *,
    # OpenAI args
    api_key: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 4,
    # HF args
    cache_dir: Optional[str] = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    attn_implementation: Optional[str] = None,
):
    """Build a chat-compatible client based on `backend`.

    Args:
        backend: "openai" or "hf".
        model:   Model name / id / local path.
        ...:     See GPTClient / HFLocalClient docstrings.
    """
    if backend == "openai":
        return GPTClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    if backend == "hf":
        return HFLocalClient(
            model_id=model,
            cache_dir=cache_dir,
            dtype=dtype,
            device=device,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            attn_implementation=attn_implementation,
        )
    raise ValueError(f"Unknown backend: {backend!r} (expected 'openai' or 'hf')")