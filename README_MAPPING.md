# Anonymized scripts — rename map and configuration

All 14 scripts were renamed after what they do, and every absolute path,
credential and dataset size was removed. Each file compiles cleanly.

## Rename map

| Original | Renamed | Purpose |
|---|---|---|
| `awq_compat.py` | `awq_transformers_compat_shim.py` | Restores activation classes newer `transformers` dropped so AutoAWQ imports |
| `check_vocab_match.py` | `verify_tokenizer_vocab_alignment.py` | Confirms student/teacher share a vocabulary so JSD is well-defined |
| `retrieval_context.py` | `build_retrieval_context_block.py` | Formats cited-paper evidence into a prompt block |
| `derive_llm_lim.py` | `classify_unmatched_limitations_vllm.py` | Classifies unmatched limitation units into a taxonomy via vLLM + AWQ |
| `rollout_qwen_vllm.py` | `generate_multi_agent_rollouts_vllm.py` | Generates worker/leader/master rollouts against a vLLM server |
| `stepwise_reward_select.py` | `select_data_stepwise_process_reward.py` | Scores candidates with a stepwise process reward; emits SFT/DPO sets |
| `jsd_reward_select.py` | `select_data_jsd_teacher_reward.py` | Scores candidates by student–teacher JSD; emits SFT/DPO sets |
| `two_stage_reward.py` | `select_data_two_stage_reward.py` | Blends JSD + judge + stepwise grounding into a two-stage reward |
| `build_synthetic_dpo.py` | `build_synthetic_dpo_preference_pairs.py` | Augments natural preference pairs with synthetic rejected responses |
| `train_sft.py` | `train_sft_lora_by_role.py` | Role-parameterized LoRA SFT (worker / leader / master) |
| `train_dpo.py` | `train_dpo_lora_worker.py` | Worker DPO LoRA stacked on the SFT adapter |
| `inference_shared_memory.py` | `run_multi_agent_inference_shared_memory.py` | Multi-agent inference with a shared retrieved-evidence block |
| `limagents_multiseed.py` | `run_multiseed_generation_significance_test.py` | Multi-seed generation + cross-model significance testing |
| `eval_50_150_rows.py` | `evaluate_limitation_extraction_metrics.py` | Recall/precision/F1, cosine, Jaccard, ROUGE-L, BERTScore + judge scoring |

## What was removed

- **Absolute paths** — every cluster path, home directory, project directory and
  model checkpoint location.
- **Credentials** — three hardcoded OpenAI API keys (two commented, one live).
- **Dataset sizes** — hardcoded row slices, row-range defaults, filenames
  encoding row ranges, and every runtime print of a row count.
- **Data previews** — `.head()` prints replaced with `.describe()` or dtype checks.

Model *paths* became environment variables; public model *identifiers*
(`gpt-4o-mini`, Qwen3-4B) were kept because the code is unreadable without them.

## Required environment variables

Scripts exit with a clear message if one of these is unset — no default is
baked into any file.

| Script | Required |
|---|---|
| `generate_multi_agent_rollouts_vllm.py` | `MODEL_PATH`, `INPUT_CSV`, `OUTPUT_DIR` |
| `select_data_stepwise_process_reward.py` | `ROLLOUT_JSON`, `SELECT_DIR` (or `--rollout_json` / `--out_dir`) |
| `select_data_jsd_teacher_reward.py` | `STUDENT_BASE`, `TEACHER_PATH`, `ROLLOUT_JSON`, `SELECT_JSD_DIR` |
| `select_data_two_stage_reward.py` | `ROOT` |
| `build_synthetic_dpo_preference_pairs.py` | `SELECT_DIR`, `ROLLOUT_JSON` |
| `train_sft_lora_by_role.py` | `BASE_MODEL`, `SELECT_DIR`, `TRAIN_ROOT` |
| `train_dpo_lora_worker.py` | `BASE_MODEL`, `SELECT_DIR`, `TRAIN_ROOT` |
| `run_multi_agent_inference_shared_memory.py` | `STUDENT_BASE`, `ADAPTER_DIR`, `INPUT_CSV`, `INFER_OUT_DIR` |
| `run_multiseed_generation_significance_test.py` | `MODEL`, `INPUT_CSV`, `OUTPUT_DIR` |
| `classify_unmatched_limitations_vllm.py` | `MODEL`, `INPUT_CSV`, `OUTPUT_DIR` |
| `evaluate_limitation_extraction_metrics.py` | `INPUT_CSV`, `OUTPUT_CSV`, `OPENAI_API_KEY` |

Optional everywhere: `START_ROW` / `END_ROW` (or `ROW_START` / `ROW_END`).
Leave both unset to process the entire file — that is now the default.

## Security note

`eval_50_150_rows.py` contained three live-format OpenAI API keys in plaintext.
They are gone from the anonymized copy, but they were exposed in the original
file. Revoke all three at <https://platform.openai.com/api-keys> and issue
replacements. Also purge them from git history if this file was ever committed.
