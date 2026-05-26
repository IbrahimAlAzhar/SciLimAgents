"""
models.py
=========
Backbone LLM wrappers used by the DeepReview-baseline limitation pipeline.

Three open-weight backbones (all run on a single 40 GB GPU) and one closed:

    HFTransformerLLM   - generic HuggingFace `transformers` backend that
                         loads any chat-template model: Qwen, Llama-3,
                         Mistral, etc.  Default backend (no vLLM needed).
    QwenLLM            - faster Qwen-2.5-3B served by vLLM (optional;
                         only imported if --backend=vllm).
    OpenAIChatLLM      - OpenAI Chat Completions, default = gpt-4o-mini.

All classes expose the same `chat(messages, ...)` API so the pipeline
never needs to know which backend is in use.

Memory budget on a 40 GB GPU (bf16 weights):
    * Qwen-2.5-3B-Instruct       ~6 GB weights  -> easy
    * Mistral-7B-Instruct-v0.3   ~14 GB weights -> safe
    * Meta-Llama-3-8B-Instruct   ~16 GB weights -> safe (leaves ~22 GB
                                                  for KV-cache & activations)
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

# ===========================================================================
# 1. HuggingFace transformers backend (works for Qwen, Llama, Mistral, ...)
# ===========================================================================
class HFTransformerLLM:
    """Generic chat-template HuggingFace backend.

    Works with any *Instruct / *Chat model that ships a chat_template
    (Qwen-2.5-Instruct, Meta-Llama-3-Instruct, Mistral-Instruct-v0.3, ...).
    No vLLM dependency.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        dtype: str = "bfloat16",          # "bfloat16" | "float16" | "float32"
        max_model_len: int = 8192,
        device: str = "cuda",
        seed: int = 42,
    ):
        # Imports are local so users without GPU/transformers installed can
        # still use the OpenAI path.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }.get(dtype, torch.bfloat16)

        # ---- Tokenizer ---------------------------------------------------
        # Some models (Mistral-Instruct, Llama-2, ...) ship sentencepiece
        # tokenizers.  The HF *fast* tokenizer needs the `sentencepiece`
        # python package to convert them.  If the fast tokenizer fails
        # because sentencepiece is missing, fall back to the slow tokenizer
        # — but only after telling the user what to install.
        self.tokenizer = self._load_tokenizer_robust(
            AutoTokenizer, model_id, cache_dir
        )
        # Llama-3 ships without a pad token — fall back to EOS so batched /
        # padded generation does not crash.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---- Weights -----------------------------------------------------
        # Some local checkpoints have a `quantization_config` baked into
        # their config.json (e.g. a previous bitsandbytes 4-bit save).  If
        # bitsandbytes is not installed we strip that config and try a
        # full-precision load.  If the *weights* themselves are quantized
        # we cannot recover and report a clear install hint.
        self.model = self._load_model_robust(
            AutoModelForCausalLM, model_id, cache_dir, torch_dtype, device,
        )
        self.model.eval()

        self.device = device
        self.max_model_len = max_model_len
        self.model_id = model_id

        # Some chat templates emit special end-of-turn tokens that are NOT
        # the standard EOS (Llama-3 uses <|eot_id|>).  We add them so the
        # model stops cleanly.
        self._eos_token_ids = self._collect_eos_ids()

        # Reproducible-ish sampling
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ tokenizer loader
    @staticmethod
    def _load_tokenizer_robust(AutoTokenizer, model_id, cache_dir):
        """Load tokenizer; fall back to slow if fast needs sentencepiece.

        Triggers seen on the user's cluster:
            ValueError: Cannot instantiate this tokenizer from a slow
            version. If it's based on sentencepiece, make sure you have
            sentencepiece installed.
        """
        try:
            return AutoTokenizer.from_pretrained(
                model_id, cache_dir=cache_dir, trust_remote_code=True,
                use_fast=True,
            )
        except Exception as e:
            msg = str(e).lower()
            looks_like_sp = (
                "sentencepiece" in msg
                or "slow version" in msg
                or "tokenizer.json" in msg
            )
            if not looks_like_sp:
                raise

            print("[HFTransformerLLM] Fast tokenizer failed because the "
                  "`sentencepiece` package is not installed.")
            print("[HFTransformerLLM] Trying slow tokenizer fallback. For "
                  "best results please run:\n"
                  "    pip install sentencepiece protobuf")
            try:
                return AutoTokenizer.from_pretrained(
                    model_id, cache_dir=cache_dir, trust_remote_code=True,
                    use_fast=False,
                )
            except Exception as e2:
                raise RuntimeError(
                    "Could not load tokenizer for "
                    f"{model_id!r}.  Most likely fix:\n"
                    "    pip install sentencepiece protobuf\n"
                    f"Original error: {e2!r}"
                ) from e2

    # ------------------------------------------------------------------ model loader
    @staticmethod
    def _load_model_robust(AutoModelForCausalLM, model_id, cache_dir,
                           torch_dtype, device):
        """Load model; if config has stale `quantization_config` and
        `bitsandbytes` is missing, strip the quantization config and retry
        at full precision.
        """
        from transformers import AutoConfig

        # ---- attempt 1: as-is
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = str(e).lower()
            is_bnb_issue = (
                "bitsandbytes" in msg
                or "quantization" in msg
                or "load_in_4bit" in msg
                or "load_in_8bit" in msg
            )
            if not is_bnb_issue:
                raise
            print("[HFTransformerLLM] WARNING: the checkpoint config "
                  "contains a `quantization_config` (e.g. saved as "
                  "load_in_4bit) but `bitsandbytes` is not installed.")
            print("[HFTransformerLLM] Stripping quantization_config and "
                  "retrying at full precision ...")
            print("[HFTransformerLLM] If this still fails, the WEIGHTS are "
                  "actually pre-quantized and you must install bitsandbytes:")
            print("    pip install bitsandbytes")

        # ---- attempt 2: strip quantization_config
        cfg = AutoConfig.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True,
        )
        # Remove ANY quantization-related field if present
        for attr in ("quantization_config", "quantization", "_quantization_config"):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, None)
                except Exception:
                    pass
        if hasattr(cfg, "to_dict"):
            d = cfg.to_dict()
            d.pop("quantization_config", None)

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                config=cfg,
                quantization_config=None,
            )
        except Exception as e2:
            raise RuntimeError(
                "Failed to load the model after stripping its "
                "`quantization_config`.  This usually means the saved "
                "weights are actually pre-quantized.  Most likely fix:\n"
                "    pip install bitsandbytes\n"
                "Alternatively, re-download a non-quantized copy of "
                f"{model_id!r}.\n"
                f"Original error: {e2!r}"
            ) from e2

    # ------------------------------------------------------------------ utils
    def _collect_eos_ids(self) -> list[int]:
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        # Llama-3 specific
        for tok in ("<|eot_id|>", "<|end_of_text|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid not in (None, self.tokenizer.unk_token_id):
                if tid >= 0:
                    ids.add(tid)
        return sorted(ids)

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 800,
        temperature: float = 0.4,
        top_p: float = 0.95,
    ) -> str:
        """Run one chat completion with the model's own chat template."""
        torch = self._torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Truncate from the LEFT so the most recent messages are kept whole.
        # We reserve room for the requested generation budget.
        max_prompt_len = max(self.max_model_len - max_new_tokens, 1024)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._eos_token_ids if self._eos_token_ids else None,
                repetition_penalty=1.05,
            )

        # Strip the prompt prefix and decode only the new tokens.
        new_tokens = out_ids[0, enc.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ===========================================================================
# 2. Optional vLLM backend (faster but heavier dependency)
# ===========================================================================
class QwenLLM:
    """Local Qwen-2.5-3B-Instruct served by vLLM.

    Same public API as `HFTransformerLLM`.  Use only if `vllm` is installed
    and you want the throughput boost.
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
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams      # noqa: F401

        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
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

    def chat(self, messages, max_new_tokens=800, temperature=0.4, top_p=0.95):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampling_params = self._SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

# ===========================================================================
# 3. OpenAI / gpt-4o-mini backend
# ===========================================================================
class OpenAIChatLLM:
    """Thin wrapper around OpenAI Chat Completions (default = gpt-4o-mini)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 4,
        retry_delay: float = 3.0,
    ):
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided. Pass --openai-api-key or set "
                "the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, messages, max_new_tokens=800, temperature=0.4, top_p=0.95):
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
            except Exception as e:
                last_exc = e
                wait = self.retry_delay * (2 ** attempt)
                print(f"[OpenAIChatLLM] retry {attempt+1}/{self.max_retries} "
                      f"after error: {e!r}.  Sleeping {wait:.1f}s ...")
                time.sleep(wait)
        print(f"[OpenAIChatLLM] giving up after {self.max_retries} retries: "
              f"{last_exc!r}")
        return ""

# ===========================================================================
# 4. Factory
# ===========================================================================
# Default checkpoint paths.  Override via the matching --*-model-id and
# --*-cache-dir flags in config.py.
_OPEN_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "qwen": {
        "model_id":  "qwen2_5_3b_instruct",
        "cache_dir": "qwen2_5_3b_instruct",
    },
    "llama": {
        "model_id":  "meta-llama/Meta-Llama-3-8B-Instruct",
        "cache_dir": "llama3_8b_instruct",
    },
    "mistral": {
        "model_id":  "mistralai/Mistral-7B-Instruct-v0.3",
        "cache_dir": "models/mistral_7b_v3_instruct",
    },
}

def _resolve_open_model_paths(args, family: str) -> tuple[str, str]:
    """Choose the right model_id / cache_dir for a given open-weight family.

    Per-family CLI flags (e.g. `--llama-model-id`) take precedence over
    the shared `--qwen-model-id`/`--qwen-cache-dir` flags, which in turn
    override the built-in defaults.
    """
    defaults = _OPEN_MODEL_DEFAULTS[family]

    # Per-family flags
    family_id_attr    = f"{family}_model_id"
    family_cache_attr = f"{family}_cache_dir"
    model_id  = getattr(args, family_id_attr, None) or defaults["model_id"]
    cache_dir = getattr(args, family_cache_attr, None) or defaults["cache_dir"]

    # Backwards compatibility: if the user only set --qwen-* flags but
    # asked for llama/mistral, do NOT use them — keep the family defaults.
    return model_id, cache_dir

def build_llm(args) -> "HFTransformerLLM | QwenLLM | OpenAIChatLLM":
    """Build the LLM backend selected on the command line."""
    if args.model == "gpt-4o-mini":
        return OpenAIChatLLM(model=args.openai_model, api_key=args.openai_api_key)

    # Open-weight families: qwen / llama / mistral
    if args.model in _OPEN_MODEL_DEFAULTS:
        model_id, cache_dir = _resolve_open_model_paths(args, args.model)

        if args.backend == "vllm":
            # vLLM only — keep the original QwenLLM class (works for any
            # vLLM-supported model, despite its historical name).
            return QwenLLM(
                model_id=model_id,
                cache_dir=cache_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                seed=args.seed,
            )

        # Default: pure transformers (no vLLM dependency)
        return HFTransformerLLM(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype=args.hf_dtype,
            max_model_len=args.max_model_len,
            device="cuda",
            seed=args.seed,
        )

    raise ValueError(f"Unknown model: {args.model}")