# SafeMergeBench 実装計画書 (Implementation Plan)

作成日: 2026-11-26
対象リポジトリ: `/mnt/nas/home/hiromi/src/safetybench/safemergebench`
参照指示書: [SafeMergeBench 実装指示書.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/SafeMergeBench%20実装指示書.md)

## 1. 背景と目的
MergeBench（NeurIPS 2025）を土台とし、安全性回復マージ（secure merge）の評価における6つの落とし穴（P1〜P6）を検出・排除する評価ベンチマーク **SafeMergeBench** を実装する。

### 全体方針（指示書 第0節準拠）
1. **MergeBench既存ファイルの変更最小化**: F1〜F3の修正および第3.1節の `merging/main.py` へのメタデータ書き出し以外は変更せず、追加機能はすべて `safemerge/`・`scripts/` に配置する。
2. **完全性ゲートの必須化**: すべてのマージ済みモデルは `safemerge/integrity_gate.py` を通過してから評価し、不合格モデルは「ゲート不合格」として別に記録する。
3. **出力形式の統一**: 追加安全性ベンチマークの生成結果も `safety-eval` の `safety_generation.json` 形式（`{task_name: [{"id": ..., "prompt": ..., "response": ...}, ...]}`）で統一する。
4. **数値精度・再現性**: マージは MergeBench に合わせ `bfloat16`（重要度・FIM等の自前計算は必ず `float32`）。乱数種（既定 42）をメタデータに記録する。
5. **GPU不要テストの徹底**: すべての新規ツールに GPU なしで動作する `unittest` を追加し、常に全テスト通過を維持する。

---

## 2. 事前調査で判明した重要事項（要確認・要相談事項の整理）

1. **MergeAlign (`hammoudhasan/MergeAlign`) の定義（指示書 3.2節・草稿 表1/表7 の要確認項目）**
   - リポジトリの README およびコードを確認した結果、対応論文は **Hammoud et al., "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch" (Findings of EMNLP 2024, arXiv:2406.14563)** である。
   - 手法は「データ不要・等方的な補間」ではなく、**合成 alignment/task データを用いたデータ認識型マージ（`LM_Cocktail` の `mix_models_with_data`、または `mergekit-evolve` による進化的マージ）** である。草稿 表1 の「Hammoudら」と重複していたため、草稿と実行設計の整合性を確認する必要がある。
   - ラッパー `safemerge/external/run_mergealign.py` では、`LM_Cocktail`（`lmcocktail_merge_align.py` 方式、`--temp` や `--alpha` による温度・重み制御とデータセット指定）に対応しつつ、CLI契約を統一する。
2. **サーバ実行環境（conda 未導入・uv 利用可能・GPU ID）**
   - `conda` は未インストールで、代わりに **`uv`** (`/mnt/nas/home/hiromi/.local/bin/uv`) および `/mnt/nas/home/hiromi/src/sst_v2/venv_sst` が存在する。
   - GPU 0 は `Unknown Error` で無効、**GPU 1 が NVIDIA H200 NVL (143GB VRAM, CUDA Driver 13.2)** として利用可能である。スクリプト実行時は `GPU_ID=1` を既定・推奨とし、`conda` と `.venvs/*`（uv仮想環境）の両方に対応できる環境アクティベーションヘルパを設ける。
3. **既存評価研究との差別化（論文の構成・要相談事項）**
   - 草稿 `docs/sok.md` 第8章の3軸（①判定前のモデル完全性 P4/P5、②防御評価における偽の安全の向き P1/P2、③マージ固有の係数比較と有用性妥当性 P3/P6）に加え、標準ベンチマーク（MergeBench）自体の監査修正 F1〜F6 とテンプレート切り分け実験（タスクD）により「標準化された環境でも同じ落とし穴が起きる」実証を柱にする方針。
4. **TrustLLM jailbreak trigger の学習・評価分割**
   - `sst_v2/v3/data/response_dataframe.csv`（5,357件）が実環境トラックの安全モデル学習に使われている。HF キャッシュにある `allenai/tulu-3-trustllm-jailbreaktrigger-eval` から、`split_overlap.py` の正規化一致＋5-gram Jaccard を用いて学習データと重複するプロンプトを確実に除外する仕組みを `safemerge/data/` に組み込む。

---

## 3. モジュール別実装計画

### Phase 1: リポジトリ初期化と環境基盤
- `/mnt/nas/home/hiromi/src/safetybench/safemergebench` の macOS AppleDouble ファイル (`._*`) を削除し、`git init` を実施。
- 1st コミット: `MergeBench`（`dedf1b1`）の upstream スナップショット。
- 2nd コミット: `SafeMergeBench.zip` 相当の F1〜F6 ベース修正。
- `safety-eval-fork` と `bigcode-evaluation-harness` を `safemergebench/` 直下に clone し、登録テンプレート名（Gemma用・テンプレートなし用）および既定判定器をソースコードから特定して `scripts/evaluate_safemerge.sh`・`scripts/template_ablation.sh` に反映する。

### Phase 2: タスクA — Hugging Face 棚卸し (`safemerge/hf_inventory.py`)
- 対象: ベース8種（Llama-3.2-3B[-Instruct], Llama-3.1-8B[-Instruct], gemma-2-2b[-it], gemma-2-9b[-it]）＋ MergeBench 専門モデル40個（計48リポジトリ）。
- `huggingface_hub.hf_hub_download` で `config.json`, `tokenizer_config.json`, `generation_config.json`, `README.md` のみ取得（重みDLなし）。
- 出力: `results/inventory/hf_inventory.csv`（およびサマリJSON）。
- 検出項目: `repo, base, domain, model_type, rope_theta, rope_scaling, vocab_size, tie_word_embeddings, torch_dtype, chat_template_family, eos_token_id, differs_from_base, card_mentions`。
- 即時実行して 40行＋ベース8行を検証し、`differs_from_base` 非空行および `chat_template_family != llama3` 行を一覧報告する。

### Phase 3: タスクB — メタデータ書き出しと外部手法ラッパー
- **3.1 `merging/main.py` & `merging/prepare_args.py`**:
  - `--record-as`（ModelSoup等の別名記録用）と `--seed`（既定42）を追加。
  - 乱数種を Python / NumPy / PyTorch に設定。
  - `merger.merge(**kwargs)` 完了後、保存先に `merge_metadata.json` を書き出す。
- **3.2 `safemerge/external/`**:
  - 共通CLI: `--base`, `--inputs`, `--out`, `--seed`, 手法固有引数。出力はモデル＋トークナイザ＋`merge_metadata.json`。
  - `common.py`: 共通メタデータ出力・git commit取得・シード固定・トークナイザ保存ヘルパ。
  - `run_della.py`: `arcee-ai/mergekit` の `merge_method: della` YAML 生成＋`mergekit-yaml` 実行（`--density`, `--epsilon`, `--lambda-coef`, `--alpha`）。
  - `run_mergealign.py`: `hammoudhasan/MergeAlign`（LM_Cocktailベースのデータ認識マージ、`--temp`, `--max-per-task`, `--dataset-path`）ラッパー。
  - `run_led_merging.py`: `MqLeet/LED-Merging` の Locate → Elect → Disjoint Merge 3段階を順に呼び出すラッパー（重要度スコアのfloat32変換・マスク比率統計のメタデータ記録対応）。
  - `run_safemerge.py`: `aladinD/SafeMERGE`（実環境トラック・LoRA専用）ラッパー。`layers_merged`・`layers_skipped` を `merge_metadata.json` に記録。
  - `PATCHES.md`: 公式リポジトリの固定コミットと変更差分・理由を記録。
- **3.3 `scripts/run_standard_track.sh`**:
  - 優先度1〜4の実行一覧（1組み合わせ36モデル、完了済みは `merge_metadata.json` の有無でスキップ）。`--priority`, `--dry-run`, `--gpu` オプション対応。

### Phase 4: タスクC — 追加安全性評価・判定器・指標・有用性チェック
- **`safemerge/data/`**:
  - `distribution.yaml`: 表6（標準・実環境トラック別の分布内/分布外対応表）。
  - `mergebench_reported.yaml`: MergeBench論文報告値の単体スコア定義（出典表番号つき）。
  - `datasets.py`: JailbreakBench (100件), StrongREJECT (全件), WildJailbreak (eval分割から `split_overlap.py` で調整用 `wildjailbreak:harmful` 重複除去), TrustLLM jailbreak trigger (学習使用分を除外) の統一ローダ。件数・除外件数・SHA-256ハッシュを記録。
- **`safemerge/extra_safety.py`**:
  - vLLM を用い、モデル自身の `tokenizer.apply_chat_template`・貪欲法 (`temperature=0.0`)・`max_tokens=512` で生成し、`safety_generation_extra.json`（`safety_generation.json` 互換形式）と `extra_safety_manifest.json` を出力。1件/バッチごとのチェックポイント対応。
- **`safemerge/judge.py`**:
  - 入力: `safety_generation*.json`（複数モデルの一括処理に対応し、13B判定器を1つずつGPUにロードして全モデルをまとめて判定）。
  - 3判定器: HarmBench (`cais/HarmBench-Llama-2-13b-cls`), WildGuard (`allenai/wildguard`), StrongREJECT (`qylu4156/strongreject-15k-v1`, 閾値 >= 0.5)。
  - 出力: `*_judged.json`（`judge_harmbench`, `judge_wildguard`, `judge_strongreject`, `strongreject_score` 追加）および `judge_config.json`。
- **`safemerge/metrics.py`**:
  - `*_judged.json` と `validity_audit.py` の拡張規則を入力に、データセット×判定器ごとに `ASR`, `VRR`, `cond_ASR`, `VSR`、および XSTest の安全要求拒否率（全体・妥当応答限定）を算出。
  - `distribution.yaml` から `in_distribution` フラグを付与し、`results/metrics/{model}.json` と `results/metrics/all.csv` を出力。
- **`safemerge/utility_check.py`**:
  - 専門モデル単体の評価結果（lm-eval / bigcode-eval JSON）と `mergebench_reported.yaml` を照合し、差が5ポイント超の項目に警告を出し、警告レポート JSON/CSV を出力。

### Phase 5: タスクE・F・G
- **`safemerge/human_eval.py` (タスクE)**:
  - 2名分の独立ラベル CSV とキー CSV/JSON を読み込み、Cohenの κ（妥当性・有害性）、拡張規則・従来規則・3判定器それぞれの適合率・再現率・F1、および崩壊判定のうち人手でも崩壊とされた割合を算出。
- **`safemerge/compare.py` (タスクF)**:
  - `results/metrics/all.csv` と有用性結果から手法×係数の安全性(`cond_ASR`)–有用性曲線を作成。
  - `VRR < 0.9` の点を除外し灰色プロット。
  - 指標 (a) 有用性が単体90%以上での最小 `cond_ASR`、(b) 正規化曲線下面積 (AUC)。
  - 判定器別の順位および Kendall の τ、単一係数比較 vs 曲線全体比較の順位対照表を出力し、matplotlib で PDF 図を生成。
- **`safemerge/import_sst_v2.py` (タスクG)**:
  - `sst_v2` の既存結果（v3 Exp1 / v5 fp32 SST 等）を読み取り専用で読み込み、`safety_generation.json` 形式に変換（元ファイルは一切変更しない）。
  - `validity_audit.py`・`metrics.py` と連携し、草稿 表3 の数値再現と、実環境トラック組み合わせ（WizardMath / WizardCoder / MedAlpaca）に対する `integrity_gate.py` の設定検査（`rope_theta`・系統不一致検出）を確認。

---

## 4. 検証計画
1. **ユニットテスト (`python3 -m unittest discover -s safemerge/tests -v`)**:
   - 既存 7 件に加え、`hf_inventory`, `merge_metadata`, `external` ラッパー（YAML生成・CLI構築・メタデータ出力）, `data/datasets`, `extra_safety` (フォーマット変換), `judge` (パーサ・出力スキーマ), `metrics`, `utility_check`, `human_eval`, `compare`, `import_sst_v2` の全モジュールに対し GPU 不要テストを作成して全件通過させる。
2. **実データ検証（CPU実行可能分）**:
   - `safemerge/hf_inventory.py` を実際に実行し、40専門モデル＋8ベースモデルの `results/inventory/hf_inventory.csv` を取得・検証する。
   - `safemerge/import_sst_v2.py` + `validity_audit.py` + `integrity_gate.py` を実環境トラックに対して実行し、表3相当の崩壊率と WizardCoder/MedAlpaca の設定不一致検出を確認する。
3. **評価ハーネス検証 (4.1)**:
   - `safety-eval-fork` と `bigcode-evaluation-harness` を clone し、テンプレート定義と既定判定器をソースコード上で確定させる。
