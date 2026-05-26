# =============================================================================
# models.py
# -----------------------------------------------------------------------------
# Model wrappers used by the ReviewRL-style limitation-generation pipeline.
#
# Two backends are supported:
#
#   * QwenGenerator     -> local Qwen2.5-3B-Instruct via HuggingFace transformers
#                          (this is the user's default; the model fits in 40 GB
#                          GPU comfortably with bf16).
#                          ReviewRL itself uses Qwen2.5-7B-Instruct
#                          (Section 3.3 of the paper); we down-size to 3B-Instruct
#                          to respect the 40 GB GPU budget.
#
#   * OpenAIGenerator   -> GPT-4o-mini via the OpenAI Python SDK
#                          (fallback when --model gpt-4o-mini is passed).
#
# Both classes expose the same interface:
#       gen = Generator(...)
#       text = gen.generate(prompt, system=None, max_new_tokens=..., ...)
# so the rest of the pipeline does not need to know which backend is in use.
# =============================================================================

from __future__ import annotations

import os
import time
from typing import Optional

# -----------------------------------------------------------------------------
# Local Qwen backend
# -----------------------------------------------------------------------------
class QwenGenerator:
    """
    Local Qwen2.5-Instruct generator.

    The defaults mirror ReviewRL's policy-model setup (Section 3.3 + Appendix C):
      - bf16 precision
      - chat-template applied via the tokenizer
      - greedy/low-temperature sampling for deterministic baselines

    We deliberately do NOT include vLLM here so the script runs out-of-the-box
    on a single 40 GB GPU without distributed orchestration.

    `lora_adapter` (optional): path to a LoRA adapter directory produced by
    sft_train.py.  If provided, the adapter is attached to the base weights
    and merge_and_unload()ed so generate() pays no PEFT overhead.
    """

    def __init__(
        self,
        model_id: str = "qwen2_5_3b_instruct",
        cache_dir: str = "qwen2_5_3b_instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_input_tokens: int = 24000,
        lora_adapter: str = "",
    ):
        # Lazy-import torch / transformers so the OpenAI-only path doesn't need them.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_input_tokens = max_input_tokens

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[torch_dtype]

        # Load tokenizer + model from the user's local checkpoint.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        # Optionally attach a LoRA adapter trained by sft_train.py.
        # We MERGE the adapter into the base weights so subsequent generate()
        # calls have zero PEFT overhead (one dense forward).
        if lora_adapter:
            from peft import PeftModel  # lazy import
            print(f"[QwenGenerator] loading LoRA adapter from {lora_adapter}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def _truncate_to_window(self, text: str) -> str:
        """Truncate the *user* portion of the prompt so the chat template fits."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_input_tokens:
            return text
        ids = ids[: self.max_input_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Run a single chat-style completion and return the assistant's text.
        """
        # Truncate user prompt if it would blow the context window.
        prompt = self._truncate_to_window(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Apply the model's chat template -> tokenized ids
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt tokens to keep only the newly-generated text.
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# -----------------------------------------------------------------------------
# OpenAI backend (optional)
# -----------------------------------------------------------------------------
class OpenAIGenerator:
    """
    GPT-4o-mini wrapper.  Activated by `--model gpt-4o-mini` on the CLI.
    Requires the env var OPENAI_API_KEY to be set.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 4):
        from openai import OpenAI  # lazy import

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY env var is not set; cannot use OpenAIGenerator."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,  # ignored for OpenAI; kept for API parity
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Simple exponential-backoff retry loop for transient rate-limit errors.
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[OpenAIGenerator] error: {e}; retrying in {wait}s ...")
                time.sleep(wait)
        raise RuntimeError("Unreachable")  # pragma: no cover

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def make_generator(args) -> "QwenGenerator | OpenAIGenerator":
    """
    Build the right generator based on the parsed CLI args.

    `args.model` values:
        * "qwen"          -> local Qwen at args.model_id
        * "gpt-4o-mini"   -> OpenAI chat completion
    """
    if args.model.lower() in {"gpt-4o-mini", "gpt4o-mini", "openai"}:
        return OpenAIGenerator(model_name="gpt-4o-mini")
    # default: local Qwen (optionally with a LoRA adapter from SFT)
    return QwenGenerator(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_input_tokens=args.max_input_tokens,
        lora_adapter=getattr(args, "lora_adapter", "") or "",
    )