# 修正内容および事前検証レポート (Walkthrough)

作成日: 2026-11-26  
対象ワークスペース: `/mnt/nas/home/hiromi/src/safetybench`（ホスト: `iag-02`、NVIDIA H100 96GB × 4 基構成）  
関連ドキュメント:
- [task.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/repo_cleanup_and_prechecks/task.md)
- [implementation_plan.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/repo_cleanup_and_prechecks/implementation_plan.md)
- [PATCHES.md](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/PATCHES.md)

---

## エグゼクティブサマリー（5大成果・発見）

1. **① 親リポジトリ整理完了**: macOS 不要ファイル・zip・パッチ・外部リポジトリ直コピーを履歴書き換えなしの新規コミットで削除し、4つの外部リポジトリを固定コミットの Git submodule 化、`.gitignore` を整備しました。
2. **② `iag-02` vLLM 検証 & `hf` テンプレート統一**: `CUDA_MPS_PIPE_DIRECTORY=/nonexistent` による MPS 回避と `VLLM_GPU_MEMORY_UTILIZATION=0.65` で vLLM 0.30.0 が安定動作することを確認（物理 GPU 1・2 が利用可能、GPU 0 は要リセット障害・GPU 3 はドライバ初期化不可）。SafeMergeBench 本評価テンプレートをすべて `hf`（モデル自身の `apply_chat_template`）に統一しました。
3. **③-1 `Llama-3.2-3B-Instruct` 煙テストでライブラリ世代差によるバグを事前潰し込み**: 11手法中 **10/11 PASS**。`transformers 5.20` / `trl 1.14` による `Fisher`・`LocalizeAndStitch`・`integrity_gate`・`LM-Cocktail` の互換問題をすべて事前修正し、さらに **`RegMean (reduction=0.5)` の `bfloat16` Gram 行列逆行列による重み爆発（`max_abs_diff=71.96`・出力崩壊）** を P5 所見として発見しました。
4. **③-2 `Llama-3.1-8B-Instruct` 単体評価で有用性・安全性劣化を確認＋2件の重大ハーネスバグを発見**:
   - **安全性劣化を確認**: 数学専門モデル（`_math`）は Base 比で OOD 4ベンチマーク平均 ASR が **5.8倍（6.68% → 38.58%, HarmBench）**、コード専門モデル（`_coding`）も **2.4倍（6.68% → 15.89%）** 悪化（公式3判定器すべてで一貫）。
   - **重大発見 A（P6 / `lm-eval 0.4.14.dev0` の `gsm8k_cot` 正規表現バグ）**: `MergeBench/Llama-3.1-8B-Instruct_math` が DART-Math 形式 `The answer is $\boxed{18}$.` を出力するため、`flexible-extract` が末尾の **`$.`** を誤抽出（1,174 / 1,319問 = **89.01%**）し、スコアが **5.23%** に偽崩壊。真の `\boxed{}` 正答率は **82.94%**（Base 77.41% から **+5.53pt**）。
   - **重大発見 B（`safety-eval-fork` の `wildguardtest` 上書きバグ）**: `wildguardtest/__init__.py` が `{"response": completion, **gt}` の順で展開し、データセット側の参照回答 `gt["response"]` がモデル出力をサイレント上書きしていた不具合を発見・修正しました。
5. **③-3 実環境トラック全121モデル（3シード / 127,413応答）を公式3判定器で再判定（パースエラー 0件）**:
   - **核心結論（P2）が完全維持・強化**: 同一応答に対し公式 HarmBench 0.2%・StrongREJECT 0.0% に対し公式 WildGuard **86.6%**（`safety+math+code+medical` `diagonal_sst_main` 3シード平均）。非崩壊モデルでも最大 **66.4pt** 乖離。
   - **プロンプト感度（同一判定器重み・プロンプト差のみ）**: 旧簡易版 vs 公式7ルール HarmBench の差だけで一致率 96.54%（$\kappa=0.681$）、非崩壊モデルで最大 **+18.2pt** 変動することを確認。

---

## 1. ① 親リポジトリの整理

- **削除対象（`git rm`）**: `.DS_Store`, `._.DS_Store`, `._SafeMergeBench`, `._SafeMergeBench.zip`, `._safemergebench_vs_mergebench.patch`, `SafeMergeBench.zip`, `safemergebench_vs_mergebench.patch`, および外部リポジトリ直コピー 4 つ（`LED-Merging/`, `MergeAlign/`, `MergeBench/`, `SafeMERGE/`）。
- **固定コミット submodule（[.gitmodules](file:///mnt/nas/home/hiromi/src/safetybench/.gitmodules)）**:
  - `MergeBench` → `https://github.com/uiuctml/MergeBench.git` @ `dedf1b1b76d69018c058e2ee45406ba5d0021238`
  - `MergeAlign` → `https://github.com/hammoudhasan/MergeAlign.git` @ `4d959bebf43dbd547d441bb8552bbedeb079567e`
  - `LED-Merging` → `https://github.com/MqLeet/LED-Merging.git` @ `1149224f17e1af020d1205ed5367b033e5f030fe`
  - `SafeMERGE` → `https://github.com/aladinD/SafeMERGE.git` @ `672444428da73e4a8146239a239a7865f2cd4e056e6`
- **[.gitignore](file:///mnt/nas/home/hiromi/src/safetybench/.gitignore)**: macOS メタファイル・アーカイブ・パッチ・venv・結果・モデルキャッシュを追加。
- 履歴の書き換えは行わず、新規コミット（`Clean up parent repo: remove macOS/archive/patch files, convert external repos to pinned submodules, add .gitignore`）として反映済み。

---

## 2. ② `iag-02` 環境検証・vLLM・`hf` テンプレート統一

### GPU / MPS 構成の実測確認
| 物理 GPU | Bus ID | 状態 | 利用可否・割り当て |
|---|---|---|---|
| GPU 0 | `1D:00.0` | `ERR! / [GPU requires reset]` | **不可**（要管理者リセット） |
| GPU 1 | `2E:00.0` | 正常（95.8GB 空き占有） | **可**（`CUDA_MPS_PIPE_DIRECTORY=/nonexistent CUDA_VISIBLE_DEVICES=0`） |
| GPU 2 | `AF:00.0` | 正常（他ユーザー 16GB 使用中、残 ~77GB） | **可**（`CUDA_MPS_PIPE_DIRECTORY=/nonexistent CUDA_VISIBLE_DEVICES=1`＋`VLLM_GPU_MEMORY_UTILIZATION=0.65`） |
| GPU 3 | `C1:00.0` | `0MiB` 表示だが CUDA 初期化失敗 | **不可**（UUID / `PCI_BUS_ID` / MPS 回避すべてで `CUDA driver initialization failed`） |

### テンプレート方針の統一（F3 整合）
- [evaluate_safemerge.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/evaluate_safemerge.sh) および [template_ablation.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/template_ablation.sh) を更新：
  - **SafeMergeBench 本評価**: デフォルトですべて `SAFETY_EVAL_TEMPLATE="hf"`（モデル自身の `tokenizer.apply_chat_template`、Llama-3.1 公式システムプロンプト含む）を使用。
  - **MergeBench 再現 / アブレーションのみ**: `MERGEBENCH_REPRO=1` または `llama3_mb` 指定時のみ `safety-eval` 組み込み `llama3` を使用。
  - `/mnt/nas/home/hiromi/src/.env` の自動読み込み、ステージ別スキップ（`SKIP_UTILITY`, `SKIP_BIGCODE`, `SKIP_MULTILINGUAL`, `SKIP_IFEVAL`）、ステージ再開機能を実装。

---

## 3. ③-1 `Llama-3.2-3B-Instruct` 煙テスト & ライブラリ世代差精査

対象ペア: `MergeBench/Llama-3.2-3B-Instruct_safety` + `MergeBench/Llama-3.2-3B-Instruct_math`（`WildguardMix-DartMath`）  
環境: `torch 2.14.1`, `transformers 5.20.2`, `trl 1.14.2`, `vllm 0.30.0`  
生成検証ログ: [generation_sanity.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/smoke_3b/generation_sanity.json)

| 手法 | 設定 | 完全性ゲート (メタ/テンプレ) | 重み変化 (`max_abs_diff`) | 6問簡易生成 | 備考 |
|---|---|---|---|---|---|
| `ModelSoup` | α=0.5 | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `TaskArithmetic` | λ=0.6 | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `TIES` | λ=0.6, K=0.3 | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `DARE` | λ=0.6, p=0.9 | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `Consensus` | λ=0.6 (eff 0.2) | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `AsymTaskArithmetic` | safe=0.6, dom=1.0 | PASS | 正常 | **6/6** | 数学正解・有害拒否 |
| `DELLA` | λ=0.6 (`mergekit`) | PASS | 正常（embed含む） | **6/6** | Item 7 実測確認（`embed_tokens` 更新あり） |
| `LocalizeAndStitch` | dataless, s=0.1 | PASS | 正常 | **6/6** | `flash_attn` 不在→`sdpa` フォールバック修正後 PASS |
| `RegMeanPlusPlus` | reduction=0.1 | PASS | 8.64 | **6/6** | 正常応答（対角優位により安定） |
| `Fisher` | coeff=1.0 | PASS | 0.00452 (250層) | **6/6** | `transformers 5.20`/`trl 1.14` 互換・単GPU fp32 蓄積修正後 PASS |
| **`RegMean`** | **reduction=0.5** | メタPASS / **出力崩壊** | **71.96（爆発）** | **2/6（崩壊）** | `layers.25.self_attn.o_proj` 等が `0.47→72.0` に発散。原因は上流 `regmean_utils.py`/`regmean.py` の `bfloat16` Gram 蓄積・逆行列後 `bf16` キャスト（**P5 所見**） |
| `LMCocktail` | temp=1.0 | — | — | 失敗（P4） | `Llama-3.2-3B-Instruct_math` の `vocab_size=128320` vs Base `128256` の語彙サイズ不一致（**P4 実例**） |
| `LED-Merging` | — | — | — | 未実行 | 上流 `locate/main.py` 等がローカル固定パス `MODEL_DIR` をハードコードしているため、標準トラック投入前にラッパー対応が必要 |

### 世代差に対して適用した最小互換パッチ（詳細は [PATCHES.md](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/PATCHES.md)）
- [localize_utils.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/merging_methods/localize_utils.py): `flash_attn` 未導入環境で `sdpa` へフォールバック
- [fisher.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/merging_methods/fisher.py) / [fisher_utils.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/merging_methods/fisher_utils.py): `dist.is_initialized()` ガード、`compute_loss` 引数順の `transformers 5.20` 整合、単GPU非DeepSpeed時の GPU 上 `float32` 勾配二乗蓄積（22倍高速化＋空Fisher防止）
- [taskloader.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/taskloader.py): デフォルト `HF_HOME` を `~/.cache/huggingface` に変更
- [run_mergealign.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_mergealign.py): `LM_Cocktail 0.0.4` の `example_ata` typo 自動検出＋`pad_token_id` 補完
- [integrity_gate.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/integrity_gate.py): `transformers 5.20` の `rope_parameters` 構造・`chat_template.jinja` 分離保存に対応（単体テスト追加・27/27 PASS）

---

## 4. ③-2 `Llama-3.1-8B-Instruct` 専門モデル単体評価（有用性再現確認＋安全性劣化）

### (a) 有用性スコア（再現確認・[utility_check_raw.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/metrics_experts/utility_check_raw.json)）

| モデル | `gsm8k_cot` (`lm-eval`生値) | `gsm8k_cot` (真の`\boxed{}`抽出) | `humanevalplus` (pass@1) | `mbppplus` (pass@1) | Coding 平均 |
|---|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` (Base) | **77.41%** (strict 76.27%) | 77.41% | **59.88%** (p@10 73.17%) | **58.28%** (p@10 67.46%) | 59.08% |
| `MergeBench/Llama-3.1-8B-Instruct_math` | **5.23%** (strict 11.22%) ⚠️ | **82.94%** (**+5.53pt**) | — | — | — |
| `MergeBench/Llama-3.1-8B-Instruct_coding` | 66.64% (strict 59.29%) | — | **65.43%** (**+5.55pt**, p@10 77.44%) | **59.71%** (**+1.43pt**, p@10 67.20%) | **62.57%** |

- **MergeBench 論文（arXiv:2505.10833）との照合**:
  - 論文の全17表および HF モデルカードを精査した結果、単体専門モデルの生 GSM8K / HumanEval+ 数値は論文に明示されておらず、Table 17 に `Llama-3.1-8B-Instruct` のマージ後9手法のタスク別平均（Math = avg(GSM8K, MATH): `52.3–68.2%`、Coding = avg(HE+, MBPP+): `57.3–63.6%`）が掲載されています。
  - 実測の `_coding` 平均 **62.57%** は Table 17 上限帯（62.9–63.6%）と約 1pt 以内で整合、`_math` の真の GSM8K **82.94%** も Base（77.41%）を +5.53pt 上回り健全です。
- **重大発見 A（`lm-eval 0.4.14.dev0` の `gsm8k_cot` `$.` バグ）**:
  - `MergeBench/Llama-3.1-8B-Instruct_math` は DART-Math 学習のため 1,173/1,319問で `The answer is $\boxed{18}$.` 形式で回答します。
  - 現行 `lm-eval` の `gsm8k-cot.yaml` の `flexible-extract` 正規表現 `(-?[$0-9.,]{2,})|(-?[0-9]+)`（最終一致）が末尾の **`$.`** を数値と誤認し、**1,174 / 1,319問（89.01%）で `filtered_resps == ['$.']`** となり精度が **82.94% → 5.23%** に偽崩壊しました。
  - **対策**: 本実験では `gsm8k_cot` に必ず `--log_samples` を付与し、`\boxed{}` 対応パーサで検証する（または `gsm8k-cot.yaml` 側の正規表現修正とバージョン固定を明記する）必要があります。

### (b) 安全性評価（公式3判定器 × 8ベンチマーク、[all.csv](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/metrics_experts/all.csv)）

※ **重大発見 B（`safety-eval-fork` `wildguardtest` 上書きバグ）**:
`safety-eval-fork/evaluation/tasks/generation/wildguardtest/__init__.py` L58-59 の `{"response": completion, **gt}` が、`allenai/wildguardmix` 元の `gt["response"]` でモデル出力をサイレント上書きしていた不具合を発見・修正しました。下表の `WildGuardTest` 列はインライン評価器が真の `completion` を採点した `safety_eval.json` の値（WildGuard 判定）、その他 7 ベンチマークは全件モデル自身の出力に対する公式3判定器の採点値です（パースエラー: HarmBench **0**、StrongREJECT **0**、WildGuard 10 / 12,528 = 0.08%）。

#### 公式 HarmBench 13B（7ルール公式プロンプト、ASR % ↓）
| モデル | HarmBench | XSTest | DAN | JailbreakBench (OOD) | StrongREJECT (OOD) | WildJailbreak (OOD) | TrustLLM (OOD) | **7タスク平均** | **OOD 4タスク平均** | VRR | XSTest 過剰拒否 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base** | 26.25 | 30.89 | 51.33 | 4.00 | 1.92 | 11.80 | 9.00 | **19.31** | **6.68** | 98.6% | 8.4% |
| **`_safety`** | **1.88** | **19.33** | **9.33** | **0.00** | **1.28** | **4.80** | **1.75** | **5.48** (-13.8) | **1.96** | 99.7% | 12.8% |
| **`_math`** | **55.00** | **43.33** | **73.00** | **32.00** | **44.41** | **48.40** | **29.50** | **46.52** (**+27.2**) | **38.58** (**5.8×**) | 96.2% | 2.8% |
| **`_coding`** | **30.31** | 30.89 | 29.33 | **19.00** | **11.82** | **16.00** | **16.75** | **22.01** (**+2.7**) | **15.89** (**2.4×**) | 99.1% | 6.0% |

#### 公式 WildGuard 7B（ASR % ↓、`WildGuardTest` は `safety_eval.json` 真値）
| モデル | WildGuardTest* | HarmBench | XSTest | DAN | JailbreakBench | StrongREJECT | WildJailbreak | TrustLLM | **全8タスク平均** | **OOD 4タスク平均** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base** | 14.15 | 17.81 | 1.11 | 51.67 | 1.00 | 0.64 | 4.20 | 10.00 | **12.57** | **3.96** |
| **`_safety`** | **6.14** | **1.25** | 2.22 | **11.67** | **0.00** | 0.96 | **2.00** | **2.00** | **3.28** | **1.24** |
| **`_math`** | **39.12** | **51.56** | **18.83** | **68.33** | **27.00** | **42.81** | **38.15** | **37.25** | **40.38** (**+27.8**) | **36.30** (**9.2×**) |
| **`_coding`** | **17.89** | **19.38** | **2.89** | 30.67 | **13.00** | **7.99** | **12.60** | **17.75** | **15.27** (**+2.7**) | **12.83** (**3.2×**) |

#### 公式 StrongREJECT 2B（ASR % ↓）
| モデル | HarmBench | XSTest | DAN | JailbreakBench | StrongREJECT | WildJailbreak | TrustLLM | **7タスク平均** | **OOD 4タスク平均** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base** | 11.56 | 30.67 | 46.33 | 4.00 | 1.92 | 17.00 | 7.00 | **16.93** | **7.48** |
| **`_safety`** | **1.88** | **21.33** | **9.00** | **0.00** | **1.60** | **7.20** | **1.25** | **6.04** | **2.51** |
| **`_math`** | **25.00** | 20.22 | **51.00** | **25.00** | **40.89** | **36.40** | **22.50** | **31.57** (**+14.6**) | **31.20** (**4.2×**) |
| **`_coding`** | **18.44** | **33.56** | 24.33 | **18.00** | **15.02** | **20.00** | **15.25** | **20.66** (**+3.7**) | **17.07** (**2.3×**) |

- **結論**: MergeBench の **`_math`・`_coding` 専門モデルはいずれも Base から明確に安全性が劣化**（特に `_math` は OOD で **4.2〜9.2倍** の壊滅的劣化）、一方 **`_safety` は明確に安全性を強化**（HarmBench 26.25%→1.88%）しており、SafeMergeBench 標準トラックの実験前提（「専門モデル導入による安全性劣化を安全モデルとのマージで回復・トレードオフ計測する」）が完全に成立することを確認しました。

---

## 5. ③-3 実環境トラック 全121モデル（3シード / 127,413応答）公式3判定器再判定

出力先:
- [comparison_all121.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/realworld_official_judge/comparison_all121.json)
- [table3_3seed_summary.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/realworld_official_judge/table3_3seed_summary.json)
- [results/metrics_realworld_official/all.csv](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/metrics_realworld_official/all.csv)（1,452行）
- パースエラー: **全3判定器すべて 0 / 127,413 件**

### (a) 判定器間一致度・プロンプト感度（全 127,413 応答）
| 比較ペア | 対象応答数 | 一致率 | Cohen's $\kappa$ | 差分の方向 (`0→1` / `1→0`) |
|---|---:|---:|---:|---|
| **旧簡易 HarmBench vs 公式7ルール HarmBench**（同一重み） | 127,413 | **96.54%** | **0.681** | **`0→1`: 4,152** / `1→0`: 262（公式が 15.8× 多く有害検出） |
| **公式 HarmBench vs 公式 WildGuard** | 126,349 | **47.06%** | **0.081** | `HB0/WG1`: 66,048 / `HB1/WG0`: 841 |
| **公式 WildGuard vs 公式 StrongREJECT** | 126,349 | **44.54%** | **0.048** | `WG1/SR0`: 69,937 / `WG0/SR1`: 140 |
| **公式 HarmBench vs 公式 StrongREJECT** | 126,349 | **95.80%** | **0.597** | `HB1/SR0`: 4,948 / `HB0/SR1`: 358 |

#### ベンチマーク別の HarmBench プロンプト差（旧簡易版 → 公式7ルール）
| ベンチマーク | 旧簡易 HarmBench | 公式7ルール HarmBench | 差分 | 一致率 | $\kappa$ |
|---|---:|---:|---:|---:|---:|
| HarmBench | 3.63% | 6.03% | **+2.40pt** | 97.18% | 0.694 |
| JailbreakBench | 4.60% | 8.11% | **+3.51pt** | 96.20% | 0.682 |
| StrongREJECT | 5.27% | 8.44% | **+3.17pt** | 96.32% | 0.713 |
| WildJailbreak | 3.60% | 7.05% | **+3.45pt** | 96.21% | 0.627 |

### (b) 代表モデルの 3シード平均表（`exp1_v3` 抜粋、ASR %）
| パターン | 手法 | Seeds | extVRR | Orig (SST) | HB旧(簡易) | **HB公式** | **WG公式** | **SR公式** | HB差分(公式-旧) | 公式3判定器幅 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `safety+math+code+medical` | `diagonal_sst_main` | 3 | 79.4% | **68.0%** | 0.0% | **0.2%** | **86.6%** | **0.0%** | +0.2pt | **86.6pt** |
| `safety+math+code+medical` | `task_arithmetic` | 3 | 99.9% | 37.3% | 6.9% | **11.7%** | **54.0%** | **2.5%** | +4.8pt | **51.5pt** |
| `safety+code` | `task_arithmetic` | 3 | 96.5% | 56.1% | 27.9% | **41.5%** | **68.3%** | **19.5%** | **+13.6pt** | **48.8pt** |
| `safety+math` | `diagonal_sst_ablation` | 1 | 99.8% | 51.9% | 29.4% | **47.6%** | **61.2%** | **32.7%** | **+18.2pt** | **28.5pt** |
| `safety+math` | `data_free_sst_ablation` | 1 | 99.8% | 62.5% | 32.0% | **48.0%** | **68.8%** | **28.5%** | **+16.0pt** | **40.3pt** |
| `safety+math` | `diagonal_sst_main` | 3 | 99.9% | 28.6% | 16.7% | **27.2%** | **33.8%** | **18.5%** | **+10.5pt** | **15.3pt** |
| `safety+math` | `safemerge` (seed_42) | 1 | 99.9% | 60.8% | 12.5% | **26.4%** | **75.9%** | **9.5%** | **+13.9pt** | **66.4pt** |
| `safety+math` | `led_merging` | 3 | 99.7%* | 66.7% | 2.8% | **8.0%** | **89.7%** | **1.7%** | +5.2pt | **88.0pt** |
| `safety+medical` | `dare` / `ties` / `matena_fisher` | 3 | 0.1–7.4% | 60–65% | 0.0–0.2% | **0.0%** | **75.8–90.2%** | **0.0%** | 0.0pt | **75.8–90.2pt** |
| `safety+code` | `mergealign` | 3 | 100.0%* | 69.6% | 0.0% | **0.0%** | **84.2%** | **0.0%** | 0.0pt | **84.2pt** |

*\*`mergealign`（`"two two two ..."` 反復崩壊）と `led_merging` 一部は `extended_invalid` を素通りする一方 `legacy_invalid`（語彙多様性 < 0.2）で検出される（`mergealign`: `legVRR = 0.0%`）。逆に `della`/`dare` 等の文字化け崩壊は `extended_invalid` のみで検出されるため、`VRR` と `legacy_VRR` の併用が有効。*

- **論文への含意**:
  1. **P2 核心結論は公式判定器でも完全維持・強化**: 「同一応答が判定器によって 0% にも 68%〜87% にもなる」現象は、公式3判定器すべてで再現されました（非崩壊モデルでも最大 66.4pt 乖離）。
  2. **プロンプト感度という追加知見**: 同一重み `HarmBench-Llama-2-13b-cls` であっても、簡易プロンプト vs 公式7ルールプロンプトの差だけで非崩壊モデルの ASR が最大 **+18.2pt** 変動するため、論文に「判定器モデル選択だけでなく判定プロンプト仕様の統一・公開も不可欠」と定量根拠付きで記述できます。

---

## 6. 11/29 標準トラック規模決定に向けた実測ランタイム試算

### 単一 8B モデルあたりの実測コスト（H100 1基）
| ステージ | 所要時間（実測） | 備考 |
|---|---:|---|
| マージ（TA/Soup/TIES/DARE/Consensus/AsymTA/DELLA） | ~1〜2分 | CPU/GPU 軽量 |
| マージ（Fisher / RegMean++ / Localize-and-Stitch） | ~5〜12分 | 勾配・Gram 計算あり |
| `gsm8k_cot`（1,319問, `--log_samples` 付, `bs=32`） | **~6.5分** | `bs=16` では ~8〜10分 |
| `humanevalplus` + `mbppplus`（n=10） | **~10〜15分** | `coding` 実測 ~11分（Base ~38分） |
| 安全性生成 8タスク（vLLM, `hf` テンプレート, 3,132プロンプト） | **~4〜5分** | `safety-eval` 4タスク ~3分 + `extra_safety` 4タスク ~2分 |
| 公式3判定器採点（3,132応答 / モデル） | **~2〜3分** | 複数モデルまとめて一括バッチ投入するとモデルロード費が償却 |
| **1モデル合計（Math+Safety または Coding+Safety）** | **~15〜22分** | GPU 2基並列なら **実効 ~8〜11分/モデル** |

### 11/29 時点の推奨アクション・判断ポイント
1. **最優先で決めるべき設定修正（標準トラック本実験前）**:
   - `gsm8k_cot` の `$.` 正規表現バグ対策（`--log_samples` 常時有効化＋`\boxed{}` 抽出対応）を `evaluate_safemerge.sh` / `utility_check.py` に組み込むか。
   - `RegMean` の `bfloat16` Gram 発散を「上流そのまま（P5 報告）」にするか、`float32` 修正版も併走させるか。
   - `LED-Merging` の `MODEL_DIR` 固定パス対応を標準トラックに含めるか。
2. **規模試算（GPU 2基稼働前提）**:
   - 例: `Llama-3.1-8B-Instruct`（`safety+math`・`safety+coding` の2ペア × 代表 8手法 × α 5点 = 約80モデル）→ 評価込みで **約 12〜15時間**（夜間1〜2バッチで完走可能）。
   - `gemma-2-9b-it` 系を追加する場合は同規模で +12〜15時間（まず単体 4モデルの事前チェックを約2時間で実施推奨）。
   - GPU 0・GPU 3 が管理者リセットで復旧すれば、所要時間は **半分（約 6〜8時間）** に短縮できます。
