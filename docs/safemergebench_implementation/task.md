# SafeMergeBench 実装タスクリスト（完了状況）

## Phase 1: リポジトリ初期化・ドキュメント整備
- [x] 指示書・草稿・参考リポジトリ・sst_v2・サーバ環境の事前調査
- [x] `docs/safemergebench_implementation/implementation_plan.md` 作成
- [x] `safemergebench/` の不要な `._*` ファイル削除と Git 初期化（MergeBench upstream スナップショット `fbe06bf` → SafeMergeBench F1〜F6 コミット `6263fb3`）

## Phase 2: タスクA — Hugging Face 棚卸し（最優先）
- [x] `safemerge/hf_inventory.py` 実装
- [x] `hf_inventory.py` の単体テスト追加
- [x] `hf_inventory.py` 実行（ベース8種＋専門40モデル＝全48モデル取得成功）→ `results/inventory/hf_inventory.csv` / `hf_inventory_summary.json` 生成・分析完了

## Phase 3: タスクB — メタデータと外部手法ラッパー
- [x] `merging/main.py` / `merging/prepare_args.py` に `merge_metadata.json` 書き出し・`--record-as`・`--seed` 追加
- [x] `safemerge/external/common.py` 実装
- [x] `safemerge/external/run_della.py` 実装
- [x] `safemerge/external/run_mergealign.py` 実装（README調査により EMNLP 2024 Findings / LM_Cocktail ベースと確定）
- [x] `safemerge/external/run_led_merging.py` 実装
- [x] `safemerge/external/run_safemerge.py` 実装（`layers_merged` / `layers_skipped` 記録対応）
- [x] `safemerge/external/PATCHES.md` 作成
- [x] `scripts/run_standard_track.sh` 作成

## Phase 4: タスクC — 評価パイプライン実装
- [x] `safemerge/data/distribution.yaml`・`safemerge/data/mergebench_reported.yaml`・`safemerge/data/plain_template.txt`・`safemerge/data/datasets.py` 実装
- [x] `safemerge/extra_safety.py` 実装
- [x] `safemerge/judge.py` 実装（3判定器：HarmBench / WildGuard / StrongREJECT、1判定器ずつ全モデル一括処理）
- [x] `safemerge/metrics.py` 実装
- [x] `safemerge/utility_check.py` 実装

## Phase 5: タスクE・F・G 実装
- [x] `safemerge/human_eval.py` 実装
- [x] `safemerge/compare.py` 実装
- [x] `safemerge/import_sst_v2.py` 実装

## Phase 6: テスト・評価ツール整備 (4.1)・実データ検証
- [x] 全追加モジュールの GPU 不要単体テスト作成・全21件パス確認（システム Python 3.12 / `venv_sst` 両方）
- [x] `safety-eval-fork` (`2920bb85`) / `bigcode-evaluation-harness` clone およびテンプレート名（Gemma=`hf`, plain=`plain_template.txt`）・既定判定器（4タスクとも `WildGuard`）の確定と `scripts/evaluate_safemerge.sh` / `scripts/template_ablation.sh` 更新
- [x] タスクG（`import_sst_v2.py` + `validity_audit.py` + `integrity_gate.py`）実データ検証（表3完全再現＋WizardCoder/MedAlpaca設定不一致検出確認）
- [x] `docs/safemergebench_implementation/walkthrough.md` 作成と最終報告

## 今後のGPU実験フェーズ（次ステップ・環境選択後に実行）
- [ ] 4評価環境（`merging`, `lmeval`, `bigcode`, `safety-eval`）のパッケージ構築（`uv` `.venvs/*` または Miniconda）
- [ ] タスクD：テンプレート切り分け実験（`Llama-3.2-3B` / `Llama-3.2-3B-Instruct` の RegMean++ 再現 + `scripts/template_ablation.sh`、GPU 1）
- [ ] 標準トラック優先度1マージ＆評価の実行（`scripts/run_standard_track.sh --priority 1 --gpu 1`）
- [ ] タスクE：2名分の人手ラベル記入後に `safemerge/human_eval.py` 実行
- [ ] タスクF：標準トラック集計完了後に `safemerge/compare.py` で曲線図・順位表生成
