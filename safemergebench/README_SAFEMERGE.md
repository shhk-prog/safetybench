# SafeMergeBench (provisional name)

A safety-preserving-merge ("secure merge") version of [MergeBench](https://github.com/uiuctml/MergeBench).
Models, utility tasks and evaluation tools are taken unchanged from MergeBench; this fork
adds what is needed to evaluate safety-recovery merging validly.

## Fixes to upstream MergeBench

| # | Problem in upstream | Fix in this fork | File |
|---|---|---|---|
| F1 | `Fisher` is not registered in `merging_methods/__init__.py`, so `--algo Fisher` fails; `fisher.py` appends hard-coded local paths to `sys.path` | registered; hard-coded paths removed | `merging/merging_methods/__init__.py`, `fisher.py` |
| F2 | `scripts/merge.sh` runs DARE and Consensus TA with `--algo RegMean` | corrected to `--algo DARE` / `--algo Consensus` | `scripts/merge.sh` |
| F3 | safety evaluation uses the `llama3` chat template for every model (Gemma and pretrained bases included) | template is chosen per model family and recorded; the integrity gate refuses a mismatch; `template_ablation.sh` separates template-induced collapse from merge-induced collapse | `scripts/evaluate_safemerge.sh`, `scripts/validate.sh`, `scripts/template_ablation.sh` |
| F4 | the safety coefficient is tuned on `wildjailbreak:harmful` (`validate.sh`), so WildJailbreak cannot be used as-is for evaluation | evaluation prompts are de-duplicated against the tuning prompts (exact + 5-gram Jaccard) | `safemerge/split_overlap.py` |
| F5 | collapsed (garbled) generations are judged as refusals / harmful without any validity check | extended validity rule, cross-tab with judge labels, blind human-label sheet | `safemerge/validity_audit.py` |
| F6 | no check that merged artifacts are what the table claims (config, template, recipe, changed tensors, non-finite values) | integrity gate before generation | `safemerge/integrity_gate.py` |

Upstream observations not changed (reported only): all mergers load models in bfloat16 and build task
vectors in bfloat16; embeddings and `lm_head` are excluded from task vectors, so for untied models
(Llama-3.1-8B) the output layer stays at the base weights; Consensus TA uses a fixed λ = 0.2.

## Methods with official code

| Method | Code |
|---|---|
| Model Soup, Task Arithmetic, TIES, DARE, Consensus TA, Localize-and-Stitch, RegMean, RegMean++, Fisher | this repo (`merging/merging_methods`) |
| DELLA | [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) |
| MergeAlign | [hammoudhasan/MergeAlign](https://github.com/hammoudhasan/MergeAlign) |
| LED-Merging | [MqLeet/LED-Merging](https://github.com/MqLeet/LED-Merging) |
| SafeMERGE | [aladinD/SafeMERGE](https://github.com/aladinD/SafeMERGE) (LoRA adapters; real-world track only) |

## Audit of the generations shipped with MergeBench (provisional, before human check)

`python safemerge/validity_audit.py merged_models/*/*/ --out audit`

| Merged model (RegMean++, 5 domains) | Task | Flagged as collapsed | Judge labels of flagged |
|---|---|---|---|
| Llama-3.2-3B | XSTest | 267 / 450 (59.3%) | refusal 158, compliance+harmful 56, compliance+unharmful 52 |
| Llama-3.2-3B | HarmBench | 29 / 320 (9.1%) | harmful 14, unharmful 15 |
| Llama-3.1-8B | XSTest | 59 / 450 (13.1%) | refusal 22, harmful 15, compliance+unharmful 22 |
| Llama-3.1-8B | HarmBench | 12 / 320 (3.8%) | harmful 11, unharmful 1 |

These upstream merges use pretrained (non-instruct) bases evaluated with the `llama3` template, so
the cause (merge vs. template) is not yet separated; run `scripts/template_ablation.sh`.

## Tests

`python -m unittest safemerge/tests/test_safemerge.py` (no GPU / torch needed; the weight-level
checks of the gate need torch + transformers and are not covered).
