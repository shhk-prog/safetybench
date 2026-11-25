# SafeMergeBench 実装ウォークスルー (Walkthrough)

作成日: 2026-11-26
対象リポジトリ: [/mnt/nas/home/hiromi/src/safetybench/safemergebench](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench)
実装計画書: [implementation_plan.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/safemergebench_implementation/implementation_plan.md)
タスクリスト: [task.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/safemergebench_implementation/task.md)

---

## 1. リポジトリ初期化と構成

1. **Git 履歴の整備**
   - `safemergebench/` 内の不要な macOS AppleDouble (`._*`)・`.DS_Store` を削除し、以下の2段階コミットで履歴を構築しました。
     - `fbe06bf`: `Upstream MergeBench snapshot (dedf1b1)`
     - `6263fb3`: `Apply SafeMergeBench base fixes F1-F6 (from SafeMergeBench.zip)`
   - MergeBench 既存ファイルの変更は指示書第0節・第3.1節どおり F1〜F3 と `merging/main.py`（および `prepare_args.py`）のメタデータ書き出し・`--record-as`・`--seed` のみに限定しています。

2. **実装したファイル一覧**
   - **タスクA（棚卸し）**: [hf_inventory.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/hf_inventory.py)
   - **タスクB（メタデータ＆外部手法）**:
     - [main.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/main.py), [prepare_args.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/prepare_args.py)
     - [common.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/common.py), [run_della.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_della.py), [run_mergealign.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_mergealign.py), [run_led_merging.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_led_merging.py), [run_safemerge.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_safemerge.py), [PATCHES.md](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/PATCHES.md)
     - [run_standard_track.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/run_standard_track.sh)
   - **タスクC（評価・判定・指標・有用性チェック）**:
     - [distribution.yaml](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/distribution.yaml), [mergebench_reported.yaml](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/mergebench_reported.yaml), [plain_template.txt](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/plain_template.txt), [datasets.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/datasets.py)
     - [extra_safety.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/extra_safety.py), [judge.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/judge.py), [metrics.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/metrics.py), [utility_check.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/utility_check.py)
     - [evaluate_safemerge.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/evaluate_safemerge.sh), [template_ablation.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/template_ablation.sh)
   - **タスクE・F・G**:
     - [human_eval.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/human_eval.py), [compare.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/compare.py), [import_sst_v2.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/import_sst_v2.py)
   - **テスト**: [test_safemerge_extended.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/tests/test_safemerge_extended.py)

---

## 2. 実施した検証と主な結果（指示書 第11節 報告形式準拠）

### (A) タスクA：Hugging Face 棚卸し（最優先タスク完了）
1. **実行コマンド**:
   ```bash
   /mnt/nas/home/hiromi/src/sst_v2/venv_sst/bin/python safemerge/hf_inventory.py --out results/inventory/hf_inventory.csv
   ```
2. **出力ファイル**:
   - [hf_inventory.csv](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/inventory/hf_inventory.csv)
   - [hf_inventory_summary.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/inventory/hf_inventory_summary.json)
3. **主な数値**:
   - 対象48リポジトリ（ベース8種＋MergeBench専門モデル40個）すべて `fetch_status: ok`（失敗0件、Llama/Gemma利用申請承認済み）。
   - `differs_from_base` 非空: **20 / 40 専門モデル**
   - `chat_template_family != llama3`: **36 / 48 モデル**（専門モデルでは 30 / 40）
4. **想定と違った点・論文への重要発見**:
   - **優先度1 (`Llama-3.1-8B-Instruct`) および 優先度3 (`gemma-2-9b-it`) の全5専門モデルはベースと設定が完全一致**（`differs_from_base` がすべて空）。標準トラックの主結果基盤としてクリーンであることが確定。
   - 一方、**`Llama-3.2-3B` / `Llama-3.2-3B-Instruct` 系に `vocab_size` 不一致（P4）を発見**:
     - `MergeBench/Llama-3.2-3B_math`: `128320`（ベース `128256`）
     - `MergeBench/Llama-3.2-3B-Instruct_math`: `128320`（ベース `128256`）
     - `MergeBench/Llama-3.2-3B-Instruct_instruction`: `128257`（ベース `128256`）
   - **`eos_token_id` の欠落・変更（生成停止崩壊リスク）を発見**:
     - `Llama-3.2-3B-Instruct` の `instruction`/`coding`/`safety`/`multilingual` はベースの `[128001, 128008, 128009]` から `128009` のみに変更（`math` のみリスト維持）。
     - `gemma-2-2b-it` の `instruction`/`coding`/`safety`/`multilingual` はベースの `[1, 107]` から `1` のみに変更（`<end_of_turn>=107` が消失）。
   - 事前学習版ベース4種×5専門モデル（計24モデル）は `chat_template_family = null`、`gemma-2-*-it` 系（計12モデル）は `gemma` であり、MergeBench が全モデルに `llama3` を固定適用していた問題（F3）を定量的に裏付けた。

### (B) 4.1節：`safety-eval-fork` テンプレート・既定判定器の確認
1. **実行コマンド**:
   `git clone --depth 1 https://github.com/nouhadziri/safety-eval-fork` および `src/templates/single_turn.py`, `evaluation/tasks/generation/*/default.yaml` の実コード監査。
2. **確認結果**:
   - `safety-eval-fork`（`2920bb85`）には `gemma` というテンプレート名は登録されていないが、**`hf`** を指定すると `tokenizer.apply_chat_template` が実行される。
   - テンプレートなし（事前学習モデル・タスクD用）は `.txt` ファイルパス指定に対応しているため、[plain_template.txt](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/plain_template.txt)（`{instruction}`）を用意し、`safety-eval-fork` 本体のパッチ不要で動作する構成とした。
   - MergeBench 安全性4タスク（`wildguardtest`, `harmbench`, `xstest`, `do_anything_now`）および係数調整タスク `wildjailbreak:harmful`（2,000件）の既定判定器はすべて **`WildGuard`** であることを確認し、`eval_config.json` に自動記録するようにした。

### (C) タスクG：`sst_v2` 実環境トラック取り込み・表3再現・完全性ゲート検証
1. **実行コマンド**:
   ```bash
   /mnt/nas/home/hiromi/src/sst_v2/venv_sst/bin/python safemerge/import_sst_v2.py \
     --sst-root /mnt/nas/home/hiromi/src/sst_v2 \
     --out-dir results/realworld \
     --gate-out results/gate/realworld_config_gate.json \
     --check-gate --alphas 0.6
   ```
2. **出力ファイル**:
   - [import_summary.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/realworld/import_summary.json)
   - [realworld_config_gate.json](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/results/gate/realworld_config_gate.json)
3. **主な数値**:
   - `exp1_v3`（116モデル変換）・`fp32_sst_math`（5モデル変換）を `sst_v2` 読み取り専用で `safety_generation_judged.json` 形式へ変換。
   - **草稿 表3（α=0.6, seed 42〜44 で従来規則妥当のうち拡張規則で崩壊と判定される割合）を全手法・全パターンで小数第1位まで完全再現**:
     - Task Arithmetic: math `0.03%` / code `1.30%` / medical `100.0%` / 4-domain `0.12%`
     - TIES: `0.13%` / `36.08%` / `100.0%` / `100.0%`
     - DARE: `0.03%` / `97.54%` / `100.0%` / `100.0%`
     - DELLA: `0.25%` / `96.63%` / `100.0%` / `100.0%`
     - Fisher: `0.13%` / `36.74%` / `100.0%` / `82.99%`
     - SST: `0.06%` / `89.12%` / `100.0%` / `55.20%`
     - Data-Free SST: `0.03%` / `96.31%` / `100.0%` / `72.80%`
     - LED-Merging: `0.36%` / `0.28%`
     - SafeMERGE: `0.03%` / `0.00%` / `0.00%`
   - **完全性ゲート（設定検査、`--weights` なし）**:
     - `WizardCoder-Python-7B-V1.0`: `rope_theta` 不一致（base `null` vs `1000000`）および `vocab_size` 不一致（`32000` vs `32001`）を検出。
     - `medalpaca-7b`: `num_key_value_heads` 不一致（base `32` vs `null`、LLaMA-1系統の証跡）および `vocab_size` 不一致（`32000` vs `32001`）を検出。
     - **想定外の補足発見**: `WizardMath-7B-V1.0` も `vocab_size` が `32001`（ベース `Llama-2-7b-hf` は `32000`）であり、設定検査で検出された。

### (D) 外部手法リポジトリの確認（3.2節）
- `hammoudhasan/MergeAlign`（`4d959be`）は **EMNLP 2024 Findings ("Model Merging and Safety Alignment: One Bad Model Spoils the Bunch", Hammoud et al., arXiv:2406.14563)** の公式実装であり、データ不要の等方補間ではなく **`LM_Cocktail` によるデータ認識型マージ（＋進化的マージ）** であることを確認しました。草稿 表1 の「Hammoudら」と重複しており、表1・表7の記載修正が必要です。

### (E) ユニットテストと構文検証
- `python3 -m unittest discover -s safemerge/tests -v`：**21件すべて PASS**（既存7件＋新規14件）
- `/mnt/nas/home/hiromi/src/sst_v2/venv_sst/bin/python -m unittest discover -s safemerge/tests -v`：**21件すべて PASS**
- 全シェルスクリプト（`scripts/evaluate_safemerge.sh`, `scripts/template_ablation.sh`, `scripts/run_standard_track.sh`）の `bash -n` 構文チェックを実施。
