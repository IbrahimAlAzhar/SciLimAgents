#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import pandas as pd
import ray
from tqdm import tqdm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

# -----------------------
# Config
# -----------------------
INPUT_CSV = "df"
OUTPUT_CSV = "df1.csv" 

SRC_COL = "limitations_autho_peer_gt"
OUT_COL = "author_peer_llama_gt_llama_ext"

MODEL_PATH = "llama3_1_70b_awq"

# generation
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = 512

# model context (keep small since your inputs are short; helps stability)
MAX_MODEL_LEN = 8192

# batching
BATCH_SIZE = 8
SAVE_EVERY = 50

def build_prompt(text: str) -> str:
    return f"""
You are an expert scientific assistant.

Task: extract limitations or shortcomings from the given text. Work under these rules:
- Only extract limitations/shortcomings that are explicitly mentioned.
- Do not invent or hallucinate any limitations that are not supported by the text.

Strict output format:
- Respond only extracted limitations with newline.

Text:
\"\"\"{text}\"\"\"
""".strip()

def wait_for_cluster(min_gpus: int = 2, timeout_s: int = 600):
    t0 = time.time()
    while True:
        gpus = ray.cluster_resources().get("GPU", 0)
        alive = len([n for n in ray.nodes() if n.get("Alive")])
        print(f"[Ray] alive_nodes={alive} GPUs={gpus}")
        if gpus >= min_gpus:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Timed out waiting for {min_gpus} GPUs (saw {gpus}).")
        time.sleep(5)

@ray.remote(num_cpus=1, num_gpus=0)
class Runner:
    def __init__(self, model_path: str, max_model_len: int, max_new_tokens: int):
        # Avoid inheriting accidental pinning
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Ray worker layout: 1 GPU per worker, and use both bundles.
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "1"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"

        self.max_model_len = int(max_model_len)
        self.max_new_tokens = int(max_new_tokens)
        self.safety_margin = 64 

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            distributed_executor_backend="ray",
            quantization="awq_marlin",
            gpu_memory_utilization=0.75,   # ↓ from 0.90
            max_model_len=4096,            # ↓ from 8192/16384/etc
            enforce_eager=True,
            disable_log_stats=True,
            # if your vLLM version supports:
            # max_num_seqs=1,
            # max_num_batched_tokens=4096,
        )

        # self.llm = LLM(
        #     model=model_path,
        #     tensor_parallel_size=1,
        #     pipeline_parallel_size=2,
        #     distributed_executor_backend="ray",
        #     quantization="awq_marlin",
        #     gpu_memory_utilization=0.90,
        #     max_model_len=self.max_model_len,
        #     enforce_eager=True,
        #     disable_log_stats=True,
        # )

        self.tokenizer = self.llm.get_tokenizer() 
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )

        # self.sampling = SamplingParams(
        #     temperature=TEMPERATURE,
        #     top_p=TOP_P,
        #     max_tokens=self.max_new_tokens,
        # )

        # tokens used by template without text
        self.base_overhead = self._count_tokens("")

    # def _make_prompt(self, text: str) -> str:
    #     msgs = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    #     ]
    #     return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 

    def _make_prompt(self, text: str) -> str:
        msgs = [
            {"role": "system", "content": "You are a careful scientific assistant."},
            {"role": "user", "content": build_prompt(text)},
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(self._make_prompt(text)))

    def _truncate_to_fit(self, text: str) -> str:
        # prompt + gen must fit
        budget_prompt = self.max_model_len - self.max_new_tokens - self.safety_margin
        if budget_prompt <= 0:
            return ""
        allowed_text_tokens = budget_prompt - self.base_overhead
        if allowed_text_tokens <= 0:
            return ""

        ids = self.tokenizer.encode(text or "")
        if len(ids) <= allowed_text_tokens:
            return text
        # keep head (author+peer text is short; head is fine)
        return self.tokenizer.decode(ids[:allowed_text_tokens])

    def generate_batch(self, texts):
        prompts = []
        for t in texts:
            t = "" if t is None else str(t)
            t = self._truncate_to_fit(t)
            prompts.append(self._make_prompt(t))

        outs = self.llm.generate(prompts, self.sampling)
        return [o.outputs[0].text.strip() for o in outs]

def is_done(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0 and not x.strip().startswith("ERROR:")

def main():
    ray_addr = os.environ.get("RAY_ADDRESS", "auto")
    print(f"[INFO] ray.init(address={ray_addr!r})")
    ray.init(address=ray_addr, ignore_reinit_error=True)
    wait_for_cluster(2)

    # Strictly spread across 2 nodes: 1 GPU each
    pg = placement_group([{"CPU": 1, "GPU": 1}] * 2, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    print("[Ray] placement group ready:", pg.bundle_specs)

    runner = Runner.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        )
    ).remote(MODEL_PATH, MAX_MODEL_LEN, MAX_NEW_TOKENS)

    df = pd.read_csv(INPUT_CSV) 
    df = df.reset_index(drop=True) 

    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # resume if output exists
    if os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV)
        if OUT_COL in old.columns and len(old) == len(df):
            df[OUT_COL] = old[OUT_COL].fillna("")
            print(f"[RESUME] loaded existing outputs from {OUTPUT_CSV}")

    pending = [i for i in range(len(df)) if not is_done(df.at[df.index[i], OUT_COL])]
    print(f"[INFO] total_rows={len(df)} pending={len(pending)}")

    processed = 0
    for start in tqdm(range(0, len(pending), BATCH_SIZE), desc="Extract limitations"):
        idxs = pending[start:start + BATCH_SIZE]
        batch_texts = []
        for i in idxs:
            x = df.at[df.index[i], SRC_COL]
            batch_texts.append("" if pd.isna(x) else str(x))

        try:
            preds = ray.get(runner.generate_batch.remote(batch_texts))
        except Exception as e:
            preds = [f"ERROR: {type(e).__name__}: {e}"] * len(idxs)

        for i, pred in zip(idxs, preds):
            df.at[df.index[i], OUT_COL] = pred

        processed += len(idxs)
        if processed % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"[SAVED] {OUTPUT_CSV} processed={processed}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
