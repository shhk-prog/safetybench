# External Repositories & Patches Log (`PATCHES.md`)

SafeMergeBench invokes official implementations of external merge methods and
evaluation harnesses without altering their core algorithms. Any pinned commits
or required minimal compatibility patches are documented here.

## Pinned Official Repositories

| Tool / Method | Upstream Repository | Pinned Commit | License (Checked) | Notes |
|---|---|---|---|---|
| MergeBench (base) | `uiuctml/MergeBench` | `dedf1b1` | MIT / Upstream | Base repo; only F1–F3 + metadata logging (`merging/main.py`) modified |
| DELLA | `arcee-ai/mergekit` | pip / PyPI installed version | LGPL-3.0 | Invoked via `mergekit-yaml` (`merge_method: della`). **Protocol note (Item 7)**: mergekit merges `embed_tokens` and `lm_head`, whereas MergeBench task-vector mergers exclude them; recorded in `merge_metadata.json` (`merges_embeddings_and_lm_head: true`). |
| Hammoud et al. (LM-Cocktail) | `hammoudhasan/MergeAlign` | `4d959be` | Custom (`LICENSE.md`) | **Verified (Item 5)**: EMNLP 2024 Findings (Hammoud et al., arXiv:2406.14563); distinct from Thakkar et al. (ACL 2025). Run at single official default `temp=1.0` with `MergeBench/{domain}_val` calibration examples (not swept on alpha grid). |
| LED-Merging | `MqLeet/LED-Merging` | `1149224` | Not stated in README | ACL 2025 Main; 3 stages (Locate → Elect → Disjoint Merge); full-weight models supported |
| SafeMERGE | `aladinD/SafeMERGE` | `6724444` | Not stated in README | ACL 2026 Findings; LoRA adapters only (real-world track); `layers_merged` / `layers_skipped` logged |
| safety-eval-fork | `nouhadziri/safety-eval-fork` | `2920bb85` | Apache-2.0 / Upstream | Cloned directly at `safemergebench/safety-eval-fork`; 2-line env-configurable vLLM memory/length patch (`VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`) in `src/generation_utils.py` for shared-GPU stability |
| bigcode-evaluation-harness | `bigcode-project/bigcode-evaluation-harness` | shallow HEAD | Apache-2.0 | Cloned directly at `safemergebench/bigcode-evaluation-harness` |

## Minimal Compatibility Patches (`torch 2.14` / `transformers 5.20` / `trl 1.14` / `vllm 0.30`)

During pre-experiment smoke testing on `iag-02` (`torch 2.14.1`, `transformers 5.20.2`, `trl 1.14.2`, `vllm 0.30.0`), the following minimal non-algorithmic compatibility fixes were applied (verified via 3B smoke test and unit tests):

1. **`safemergebench/merging/merging_methods/localize_utils.py`**: Fall back from `flash_attention_2` to `sdpa` when `flash_attn` is not installed (`_DEFAULT_ATTN_IMPL`).
2. **`safemergebench/merging/merging_methods/fisher.py` & `fisher_utils.py`**:
   - Guard `dist.get_rank()` with `not dist.is_initialized() or dist.get_rank() == 0` for single-GPU runs.
   - Fix `FisherTrainer.compute_loss` parameter order (`return_outputs=False, num_items_in_batch=None`) to match `transformers 5.20` / `trl 1.14`.
   - Accumulate squared gradients in `float32` on GPU inside `FisherTrainer.training_step` when DeepSpeed is disabled (previously `DeepSpeedEngineWrapper.backward` was never invoked on single-GPU runs), and raise an error if `n_steps == 0`.
3. **`safemergebench/merging/taskloader.py`**: Default `HF_HOME` fallback changed from `/data/huggingface` (unwritable) to `~/.cache/huggingface`.
4. **`safemergebench/safemerge/external/run_mergealign.py`**: Handle `LM_Cocktail 0.0.4` typo (`example_ata` vs `example_data`) via `inspect.signature` and set `pad_token_id = eos_token_id` for Llama-3 tokenizers where `pad_token_id` is `None`.
5. **`safemergebench/safety-eval-fork/src/generation_utils.py`** (2-line patch in `load_vllm_model`): Pass `gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.70"))` and `max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))` to prevent vLLM from claiming 92% VRAM / 131k KV cache on shared GPUs.
6. **`safemergebench/safety-eval-fork/evaluation/tasks/generation/wildguardtest/__init__.py`** (critical upstream bug fix): Upstream wrote `{"id": gt["id"], "response": completion, **gt, **pred.asdict()}` where `gt` (from `allenai/wildguardmix`) already contains a `"response"` column, silently overwriting the model's generated `completion` in `safety_generation.json` with the static dataset reference response! Fixed by stripping `"response"` from `gt` before unpacking (`gt_clean = {k: v for k, v in gt.items() if k != "response"}`) so `safety_generation.json["wildguardtest"]` stores the model's actual completion. (Inline `safety_eval.json` was unaffected because line 42 passed `completion` directly to the inline evaluator.)
7. **`safemergebench/safemerge/integrity_gate.py`**: Normalize `transformers 5.20` saved configs (`rope_parameters.{rope_theta,rope_scaling}` and standalone `chat_template.jinja`) so the gate does not raise false positives.

> **Known upstream harness bug (P6 / `lm-evaluation-harness 0.4.14.dev0` `gsm8k_cot`)**: In `gsm8k-cot.yaml`, `flexible-extract` uses regex `(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1`. Because `MergeBench/Llama-3.1-8B-Instruct_math` (DART-Math) ends answers with `The answer is $\boxed{18}$.`, the regex matches the trailing `$.` on **1,174 / 1,319 items (89.01%)**, collapsing `lm-eval` reported accuracy to **5.23%** whereas true `\boxed{}` / last-number EM on the exact same generations is **82.94%** (+5.53pt over base 77.41%). Always pass `--log_samples` on `gsm8k_cot` and extract `\boxed{}` / digits for DART-Math models!

> **Known upstream numerical issue (P5 / RegMean)**: Upstream `MergeBench/merging/merging_methods/regmean_utils.py` & `regmean.py` accumulate Gram matrices `X^T X` and `G W^T` in `bfloat16` and cast `sum_gram_inv` back to `bfloat16` prior to `matmul`. In our `Llama-3.2-3B-Instruct` (`safety+math`) smoke test, `RegMean (reduction=0.5)` suffered severe weight explosion (`max_abs_diff = 71.96` in `layers.25.self_attn.o_proj.weight`) and output collapse (`2/6` valid), whereas `RegMeanPlusPlus (reduction=0.1)` stayed bounded (`max_abs_diff = 8.64`, `6/6` valid). Left unpatched in baseline runs so upstream behavior is faithfully recorded as a P5 finding; float32 Gram accumulation can be tested separately as an ablation.

## Verification Findings (Section 4.1)

### 1. `safety-eval-fork` Templates (`src/templates/single_turn.py`, `evaluation/eval.py`)
- `gemma` is **not** registered by name in `single_turn.py`. However, `chat_template == "hf"` invokes `AutoTokenizer.from_pretrained(model).apply_chat_template(...)` directly. Therefore, `scripts/evaluate_safemerge.sh` and `scripts/template_ablation.sh` pass `hf` to `safety-eval` for Gemma instruct models while validating `gemma` in `integrity_gate.py`.
- A plain (no-chat-template) mode is not registered by name, but `--model_input_template_path_or_name` natively accepts any `.txt` file containing `{instruction}`. We provide `safemerge/data/plain_template.txt` (`{instruction}`) so pretrained bases and Task D ablations run without patching `safety-eval-fork`.
- **No source patch to `safety-eval-fork` is required.**

### 2. `safety-eval-fork` Default Judges & Tuning Split
- Default `classifier_model_name` in `evaluation/tasks/generation/*/default.yaml`:
  - `wildguardtest`: `WildGuard`
  - `harmbench`: `WildGuard`
  - `xstest`: `WildGuard`
  - `do_anything_now`: `WildGuard`
  - `wildjailbreak:harmful` (MergeBench tuning set, 2,000 prompts in `evaluation/tasks/generation/wildjailbreak/harmful.jsonl`): `WildGuard`
  - `trustllm_jailbreaktrigger`: `OpenInstructGPTJailbreakTriggerClassifier` (SafeMergeBench replaces/augments this via `safemerge/extra_safety.py` + `safemerge/judge.py` with HarmBench-cls, WildGuard, and StrongREJECT).

### 3. `LED-Merging`
- No source modification made inside `LED-Merging` yet. If dynamic HF model paths (instead of hard-coded `modeltype2path` aliases in `locate/main.py` and `mask_generate.py`) are patched for standard-track Llama-3.1-8B-Instruct / Gemma-2-9B-it runs, record exact unified diff here.
