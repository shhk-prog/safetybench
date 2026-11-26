# SafeMergeBench 実装指示書 v2 タスクリスト

作成日: 2026-11-26  
準拠文書: [SafeMergeBench 実装指示書 v2（11月26日）.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/SafeMergeBench%20実装指示書%20v2（11月26日）.md) ＋ 追記指示【1】〜【4】および LED-Merging 追加指示 (1–4)

---

## フェーズ1（11/26 完了）：結果管理・既定修正（1・3・4・5・6・7）・退化検出＆LED-Merging【1】【2】

- [x] **第1節 結果フォルダ管理と強制ツール**
  - [x] `safemerge/new_run.py` 作成（`results/runs/YYYYMMDD-HHMM_名前/` 作成、`RUN.md` 雛形、`env.json`＋`config/uncommitted.diff`、`INDEX.md` 追記、同一分重複時の `-2` 付与）
  - [x] `results/runs/INDEX.md` 初期化（既存 `safemergebench/results/`・`sst_v2` v3/v5 を「旧形式」として記載、無効結果の注記）
  - [x] `scripts/evaluate_safemerge.sh`, `run_standard_track.sh`, `check_single_experts.sh`, `template_ablation.sh` に `--run-dir` 必須化・`RUN.md` 存在検査・`commands.sh` 自動追記
  - [x] 第1節の単体テスト追加（作成・上書き防止 `-2`・`--run-dir` なしで停止）
- [x] **第3節 既定の修正（コード実装・調査完了分）**
  - [x] **3.1 XSTest 扱い修正**: `metrics.py` で安全な要求（250件）過剰拒否率と危険な要求（`contrast_` 200件）単独の ASR を分離し、攻撃平均から XSTest を除外、`distribution.yaml` に厳密準拠（ID/OOD 集計出力含む）。`compare.py` でも XSTest を攻撃平均から除外
  - [x] **3.3 MergeBench 公開 WildGuardTest 参照回答一致率確認**: 749/749件 = 100.00% 完全一致を実証（`20261126-1342_check-mb-published-wildguardtest`）
  - [x] **3.4 崩壊判定の OR 化**: `validity_audit.py` に `combined_invalid` 追加、`metrics.py` の `VRR` を `extended_invalid or legacy_invalid` に変更、`extended_VRR`・`legacy_VRR` を別列保持、`"two two two ..."` 反復テスト追加
  - [x] **3.5 人手ラベル追加 50 件ブラインド抽出**: 実環境トラックから「両規則妥当＆HB/SR無害＆WG有害」50件抽出（`20261126-1351_human-eval-extra50-wg-only`）
  - [x] **3.6 RegMean 単精度版対応**: 単精度 Gram/逆行列オプション（`--regmean-fp32` / `RegMeanFP32`）と上流 `bfloat16` 版の両対応、`PATCHES.md` に差分および Fisher 単精度積算を明記
  - [x] **3.7 LED-Merging ラッパー改善**: 上流 `LED-Merging/` を 1 バイトも改変せず、ラッパー側で `MODEL_DIR` / `modeltype2path` と作業パスを環境変数・シンボリックリンク・ランチャーで差し替え
- [x] **第4.2節＆追記【1】【2】LED-Merging 原因確定・厳格煙テスト・未修正再現**
  - [x] `integrity_gate.py --weights` に単精度計算でタスクベクトル cos 類似度（全体・層ごと）、分野更新の射影係数、ベース・各入力との相対ノルム、入力最大デルタ比、更新テンソル数・未更新例、退化フラグを追加
  - [x] `sst_v2` LED-Merging モデル重み検査（`20261126-1309_led-merging-diagnosis`）：289/291層ベース完全一致を確認
  - [x] **【2】原因A帰属確定**：`LED-Merging/README.md` L63–68・`locate/scripts/locate_inst.sh` L33–86・`locate/lib/model_wrapper.py` L515–531 および `sst_v2`/`safetybench` 両ラッパーを照合し「上流公開コードの不具合」と確定
  - [x] **【1】ラッパー強化＆厳格煙テスト**（`20261126-1444_led-merging-strict-smoke-and-repro` ＋ 改定後再判定 `20261126-1541_led-smoke-rejudge-tv-norm`）：重要度ファイル数検査（不足時マージ前停止）・形不一致テンソル記録・公式 `fuse_rates`（0.1/0.4/0.5）を組み込み、3B `safety+math`（語彙不一致）・3B `safety+coding`（語彙一致）・未修正公開コード再現を実行。確認①のタスクベクトル基準（`||Δ_m||/max||Δ_i||<5%`, `||θ_m-θ_i||/||Δ_i||<5%`）への改定後、修正版2本は完全合格・未修正版は退化検出で不合格を確認。確認②の `safety_rate` 0.3/0.5 追加（`locate` 再利用・`config_role` 区別）も実装完了。

## フェーズ2（11/26 完了）：完全性ゲート厳格度（第4.4節）

- [x] `sst_v2/v5/results/rope_theta_ablation` の結果に基づく 4.4 節の既定方針を確認
- [x] `integrity_gate.py` の既定値更新（系統不一致 >50% = 不合格・終了コード1、rope_theta/語彙数/終了トークン/テンプレート不一致 = 警告・終了コード0、根拠フォルダ明記）
- [x] `PATCHES.md` へ根拠フォルダと判断結果を記載
- [x] テスト追加・確認（系統不一致=終了コード1、設定不一致のみ=終了コード0で警告記録）

## フェーズ3（11/26〜11/27）：評価設定検証ゲート（第2節・追記【3】）・WildGuardTest 再生成・比較拡張（4.1/4.3・追記【4】）

- [x] **第2節＋追記【3】コード実装完了**
  - [x] `safemerge/data/expert_eval_spec.yaml` 作成
  - [x] `safemerge/utility_gate.py` 実装＋テスト（数学/一般用ゲート ＋ 追記【3】コード用 `check-code`: 構文エラー<5%・補助関数 NameError<2%・関数重複定義<5%・不正解20件目視表 `samples_review.md`）
  - [x] `safemerge/eval_code_samples.py` 作成（`bigcode-evaluation-harness` 無改変で保存生成物から全サンプル正誤・実行時エラー JSONL を生成）
  - [x] `evaluate_safemerge.sh` へ `eval_lock.yaml` 照合・`--log_samples`・`--save_generations` 組み込み
- [x] **第4.1節＋追記【4】・第4.3節コード実装完了**
  - [x] `compare.py` の判定器別同一有害率線形補間・判定器間順位逆転ペア一覧出力（`find_judge_rank_flips` / `write_rank_flips_markdown`）・「到達せず」処理・3参照点必須プロット＆欠落エラー・対応付きブートストラップ（1,000回）95% CI・シード標準偏差ヘルパー＋テスト
  - [x] `metrics.py` 全指標への Wilson 95% CI 付与、`common.py` の安全注入量ノルム記録・実行不能記録
- [x] **追記【3】実機実行（完了・4/4 合格）**：
  - [x] 標準トラック `Llama-3.1-8B-Instruct`（Base & `_coding`）の HumanEval+・MBPP+ 生成物保存＋コード用検証ゲート（`20261126-1525_std-llama31-code-utility-gate`：4/4 PASS、不正解20件目視表 `samples_review_*.md` 出力済み）
- [x] **11/27 実機 GPU 実行分（先行完了・`eval_lock.yaml` 固定のみ目視承認待ち）**：
  - [x] 第3節の2：`WildGuardTest` を修正後コードで単体4モデル再生成・公式3判定器採点・参照一致 0/749 確認（`20261126-1908_std-llama31-experts-v2-wgt-gsm8k-safetycode`、旧結果を「置き換え済み」記録）
  - [x] 表11【要測定】項目完了（`_safety` GSM8K `72.71%`・HE+ `41.59%`・MBPP+ `58.65%`、`_coding` GSM8K `67.85%`）＋全4モデル GSM8K (`math_unified_v2`, `n=1,319`, 問題単位ブートストラップ CI) ゲート＆20件目視表出力＋Base `max_length=1024` 感度確認（HE+ `+0.06pt` / MBPP+ `-0.93pt`）
  - [ ] `_safety` HumanEval+ 空補完（16.89%）の扱い確認後、`configs/eval_lock.yaml` を固定（`_math` VRR `94.78%` 注記込み）

## フェーズ4（11/28）：`gemma-2-9b-it`・再現確認

- [ ] `gemma-2-9b-it` 4モデル（Base, safety, math, coding）の検証ゲート実行・安全性劣化確認
- [ ] MergeBench 本家の lm-eval 版での `gsm8k_cot` 誤抽出再現確認（別実験フォルダ）

## フェーズ5（11/29〜12/5）：本実験開始判定（5.1）と実行

- [ ] 第5.1節の全5条件達成確認（本実験開始ゲート）
- [ ] 標準トラック主実験（`Llama-3.1-8B-Instruct` × 2ペア = 76モデル＋乱数シード）
- [ ] `gemma-2-9b-it` 一般化確認・人手ラベル集計・数値確定

---

## 参考：実環境トラック `sst_v2` 側（第2.5節、`sst_v2` リポジトリ担当）
※ 標準トラックのマージ・評価仕組みは `sst_v2` には作らず `safetybench` のみで行う。
- `2.5.1` WizardCoder 有用性検証（`sst_v2` 側で 5.49%→50.61% 確認済み、追記【3】へ反映済み）
- `2.5.2` LED-Merging 重み・マスク確認（`20261126-1309`・`20261126-1444` で原因確定・標準トラック再現完了）
- `2.5.3–2.5.5` MedAlpaca 検証・数学曲線集計・既存実験 `RUN.md` 付与（`sst_v2` 側で実施）
