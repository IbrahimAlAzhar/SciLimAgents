"""
llm_client.py
=============
Retry-safe wrapper around three chat-completion backends:

  * OpenAI               (provider="openai")
  * Azure OpenAI         (provider="azure")
  * Hugging Face local   (provider="hf")    <-- new

The HF backend loads a causal-LM via `transformers`, applies the model's chat
template, and generates with `model.generate(...)`. Tested with:

  * meta-llama/Meta-Llama-3-8B-Instruct
  * mistralai/Mistral-7B-Instruct-v0.3
  * Qwen/Qwen2.5-3B-Instruct  (or a local checkpoint path)

JSON enforcement on local models is *prompt-based* (no native
response_format support), so we also ship a robust JSON parser that:
  - strips ```json ... ``` code fences,
  - extracts the first balanced {...} block, and
  - removes trailing commas before re-parsing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Dict, Optional

from config import LLMConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Client
# ============================================================================

class LLMClient:
    """Chat-completion client supporting OpenAI, Azure, and local HF models."""

    _HF_PROVIDERS = {"hf", "huggingface", "transformers", "local"}

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._provider = cfg.provider.lower()

        # OpenAI / Azure
        self._client = None
        # HF state
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_device = None

        self._build_client()

    # ------------------------------------------------------------------
    # Backend bootstrap
    # ------------------------------------------------------------------
    def _build_client(self):
        if self._provider == "azure":
            self._client = self._build_azure()
        elif self._provider == "openai":
            self._client = self._build_openai()
        elif self._provider in self._HF_PROVIDERS:
            self._build_hf()
        else:
            raise ValueError(f"Unknown provider: {self.cfg.provider}")

    def _build_openai(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Please `pip install openai>=1.0.0` to use the OpenAI backend."
            ) from e
        if not self.cfg.openai_api_key:
            raise ValueError("OpenAI backend selected but OPENAI_API_KEY is not set.")
        return OpenAI(api_key=self.cfg.openai_api_key)

    def _build_azure(self):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError(
                "Please `pip install openai>=1.0.0` to use the Azure backend."
            ) from e
        if not (self.cfg.azure_endpoint and self.cfg.azure_api_key):
            raise ValueError(
                "Azure backend selected but AZURE_ENDPOINT / AZURE_OPENAI_KEY "
                "are not set."
            )
        return AzureOpenAI(
            azure_endpoint=self.cfg.azure_endpoint,
            api_key=self.cfg.azure_api_key,
            api_version=self.cfg.azure_api_version,
        )

    def _build_hf(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Please `pip install transformers torch accelerate` for the HF backend."
            ) from e

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.cfg.hf_dtype.lower(), torch.bfloat16)

        model_id = self.cfg.hf_model_id or self.cfg.model
        cache_dir = self.cfg.hf_cache_dir or None
        if not model_id:
            raise ValueError(
                "HF backend requires --hf-model-id (or --model) to be set."
            )

        logger.info("Loading HF tokenizer: %s (cache=%s)", model_id, cache_dir)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self._hf_tokenizer.pad_token is None:
            # Llama 3 / Mistral often have no pad token by default
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self._hf_tokenizer.pad_token_id = self._hf_tokenizer.eos_token_id

        logger.info("Loading HF model: %s (dtype=%s, device=%s)",
                    model_id, torch_dtype, self.cfg.hf_device)
        device_map = self.cfg.hf_device  # "auto" | "cuda" | "cuda:0" | "cpu"
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._hf_model.eval()
        self._hf_device = next(self._hf_model.parameters()).device
        logger.info("HF model loaded on %s", self._hf_device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format_json: bool = False,
    ) -> str:
        """Issue a chat-completion request and return the text content."""
        if self._provider in self._HF_PROVIDERS:
            return self._chat_hf(messages, temperature, max_tokens, response_format_json)
        return self._chat_openai_like(messages, temperature, max_tokens, response_format_json)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Same as `chat` but tries to enforce a JSON object response.
        On OpenAI/Azure: uses native response_format json_object.
        On HF local: appends a strict instruction and parses defensively.
        """
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        return _safe_json_loads(raw)

    # ------------------------------------------------------------------
    # OpenAI / Azure path
    # ------------------------------------------------------------------
    def _chat_openai_like(self, messages, temperature, max_tokens, response_format_json):
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens

        model = (
            self.cfg.azure_deployment
            if self._provider == "azure"
            else self.cfg.model
        )

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.cfg.top_p,
            timeout=self.cfg.request_timeout,
        )
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                wait = self.cfg.retry_backoff_seconds * attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ------------------------------------------------------------------
    # Hugging Face local path
    # ------------------------------------------------------------------
    def _chat_hf(self, messages, temperature, max_tokens, response_format_json):
        import torch
        temperature = self.cfg.temperature if temperature is None else temperature
        max_new_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens

        # If JSON requested, append a strict instruction (no native enforcement here)
        if response_format_json:
            messages = self._inject_json_instruction(messages)

        # Apply the model's chat template
        try:
            prompt = self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning(
                "apply_chat_template failed (%s). Falling back to naive concatenation.",
                exc,
            )
            prompt = "\n\n".join(
                f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
                for m in messages
            )
            prompt += "\n\n[ASSISTANT]\n"

        inputs = self._hf_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self._hf_device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self._hf_tokenizer.pad_token_id,
            eos_token_id=self._hf_tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(self.cfg.top_p)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                with torch.no_grad():
                    out = self._hf_model.generate(**inputs, **gen_kwargs)
                # Strip the prompt prefix
                gen_tokens = out[0][inputs["input_ids"].shape[1]:]
                text = self._hf_tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return text.strip()
            except torch.cuda.OutOfMemoryError as exc:
                logger.error("CUDA OOM during generation: %s", exc)
                torch.cuda.empty_cache()
                # Halve max_new_tokens and try again
                gen_kwargs["max_new_tokens"] = max(256, gen_kwargs["max_new_tokens"] / 2)
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                wait = self.cfg.retry_backoff_seconds * attempt
                logger.warning(
                    "HF generation failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"HF generation failed after retries: {last_err}")

    # ------------------------------------------------------------------
    @staticmethod
    def _inject_json_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Append a strict JSON-only instruction to the last user message."""
        instr = (
            "\n\nIMPORTANT FORMATTING REQUIREMENT:\n"
            "Respond with a SINGLE valid JSON object only. "
            "Begin your response with `{` and end with `}`. "
            "Do not include any explanation, preamble, markdown code fences, "
            "or any text outside the JSON object. "
            "Use double quotes for all keys and string values, and "
            "do not put trailing commas before `}` or `]`."
        )
        new_msgs = [dict(m) for m in messages]
        for m in reversed(new_msgs):
            if m.get("role") == "user":
                m["content"] = (m.get("content", "") or "") + instr
                return new_msgs
        new_msgs.append({"role": "user", "content": instr.strip()})
        return new_msgs

# ============================================================================
# JSON parsing helpers
# ============================================================================

def _safe_json_loads(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return {}

    cleaned = text.strip()

    # 1. Strip leading/trailing markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Try first balanced {...} block (with trailing-comma cleanup)
    obj = _extract_first_json_object(cleaned)
    if obj is not None:
        return obj

    logger.warning(
        "Could not parse LLM JSON response. First 200 chars: %r",
        text[:200],
    )
    return {"_raw_text": text}

def _extract_first_json_object(text: str):
    """Find the first balanced JSON object in `text` and parse it."""
    n = len(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, n):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Remove trailing commas before } or ]
                            cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
                            try:
                                return json.loads(cleaned)
                            except json.JSONDecodeError:
                                break
        start = text.find("{", start + 1)
    return None 
