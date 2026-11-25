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
| safety-eval-fork | `nouhadziri/safety-eval-fork` | `2920bb85` | Apache-2.0 / Upstream | Cloned directly at `safemergebench/safety-eval-fork`; no source patch needed |
| bigcode-evaluation-harness | `bigcode-project/bigcode-evaluation-harness` | shallow HEAD | Apache-2.0 | Cloned directly at `safemergebench/bigcode-evaluation-harness` |

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
