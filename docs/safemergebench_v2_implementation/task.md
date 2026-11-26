# SafeMergeBench 実装指示書 v2 タスクリスト

作成日: 2026-11-26  
準拠文書: [SafeMergeBench 実装指示書 v2（11月26日）.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/SafeMergeBench%20実装指示書%20v2（11月26日）.md)

---

## フェーズ1（11/26 優先）：結果管理・既定修正（1・4・6・7）・退化検出＆LED-Merging確認

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
- [x] **第4.3節（先行分）誤差の付与（`metrics.py` & `compare.py`）**
  - [x] `metrics.py` の ASR・VRR・cond_ASR・VSR・過剰拒否率・有用性に Wilson 法 95% 信頼区間を付与
  - [x] `compare.py` に対応付きブートストラップ（1,000回）95% CI とシード間標準偏差ヘルパーを実装
- [x] **第4.2節 マージ退化の検出＆LED-Merging 確認（最優先）**
  - [x] `integrity_gate.py --weights` に単精度計算でタスクベクトル cos 類似度（全体・層ごと）、分野更新の射影係数、ベース・各入力との相対ノルム、更新テンソル数・未更新例、退化フラグ（相対ノルム <1% または 射影係数 <10%、しきい値引数可変）を追加
  - [x] `sst_v2` の LED-Merging マージ済みモデル（`safety+math`・`safety+code`）を検査し、ベースとの差（0.0, 289/291 テンソル無変更）と上流公式バグ原因を特定
  - [x] `safetybench` ラッパーで `Llama-3.2-3B-Instruct`（安全＋数学）の LED-Merging 煙テストを実施（マスク非ゼロ 8.4%/9.0%・235/254層更新・数学 3/3 正答・有害 2/2 拒否）

## フェーズ2（11/26 決定済み反映）：完全性ゲート厳格度（第4.4節）

- [x] `sst_v2/v5/results/rope_theta_ablation` の結果（DARE 97.5%→99.4–100%、SST 89.1%→72.0–93.8%）に基づく4.4節の既定方針を確認
- [x] `integrity_gate.py` の既定値更新（系統不一致 >50% = 不合格・終了コード1、rope_theta/語彙数/終了トークン/テンプレート不一致 = 警告・終了コード0、根拠フォルダ明記）
- [x] `PATCHES.md` へ根拠フォルダと判断結果を記載
- [x] テスト追加・確認（系統不一致=終了コード1、設定不一致のみ=終了コード0で警告記録）

## フェーズ3（11/27）：評価設定検証ゲート（第2節）・WildGuardTest 再生成・比較拡張（4.1/4.3）

- [x] **第2節コード実装（先行完了）**
  - [x] `safemerge/data/expert_eval_spec.yaml` 作成
  - [x] `safemerge/utility_gate.py` 実装＋テスト（公式値±5pt、ベース比較、抽出失敗率<5%、偏り<10%、非数値<1%、20件目視表、VRR≥0.95、`--log_samples`強制、`eval_lock.yaml`ハッシュ照合）
  - [x] `evaluate_safemerge.sh` へ `eval_lock.yaml` 照合と `--log_samples` 組み込み
- [x] **第4.1節コード実装（先行完了）**
  - [x] `compare.py` の同一有害率での線形補間有用性・「到達せず」処理・3参照点必須プロット＆欠落エラー・`common.py` の安全注入量ノルム記録・実行不能記録＋テスト
- [ ] **11/27 実機 GPU 実行分**：
  - [ ] 標準トラック `Llama-3.1-8B-Instruct` 4モデルを 2.2 検証ゲートに通し、`configs/eval_lock.yaml` を固定
  - [ ] 第3節の2：`WildGuardTest` を修正後コードで単体4モデル再生成・3判定器採点（旧結果を「置き換え済み」記録）

## 実環境トラック `sst_v2` 先行作業（第2.5節・11/26〜11/28）

※ 標準トラックのマージ・評価仕組みは `sst_v2` には作らず `safetybench` のみで行う。WizardCoder 検証通過までコード有用性の数値（「0%」含む）は論文結論に不使用。

- [ ] **2.5.1（最優先）**: WizardCoder の有用性検証（公式形式を `expert_eval_spec.yaml` に記載→2.2 ゲート通過→表3・rope_theta 含むコード組合せ再評価）
- [ ] **2.5.2**: LED-Merging の重み・マスク確認（4.2 統計適用）
- [ ] **2.5.3**: MedAlpaca の有用性検証
- [ ] **2.5.4**: 数学再評価（seed44 完了後）の曲線集計（4.1 同一有害率比較・4.3 信頼区間・主要点 1,319 問全件再評価）
- [ ] **2.5.5**: 既存2実験（数学再評価・rope_theta）へ後から `RUN.md` 付与・比較表を `outputs/` に配置

## フェーズ3（11/27）：評価設定検証ゲート（第2節）・WildGuardTest 再生成・比較拡張（4.1/4.3）

- [ ] **第2節 評価設定の検証と固定（P6）**
  - [ ] `safemerge/data/expert_eval_spec.yaml` 作成（標準トラック Llama-3.1-8B-Instruct / gemma-2-9b-it、実環境トラック WizardMath / WizardCoder / MedAlpaca）
  - [ ] `safemerge/utility_gate.py` 実装（公式値差 ±5pt、ベース比較、抽出失敗率 <5%、抽出値偏り <10%、数学非数値 <1%、20件目視表 `outputs/samples_review.md`、VRR ≥0.95、`--log_samples` 強制）
  - [ ] `configs/eval_lock.yaml` 生成とハッシュ固定、`evaluate_safemerge.sh` でマージ済みモデル評価前のハッシュ照合ゲートを強制
  - [ ] 標準トラック `Llama-3.1-8B-Instruct` 4モデル（Base, safety, math, coding）を検証ゲートに通し、`eval_lock.yaml` を固定
- [ ] **第3節の2・3**:
  - [ ] `WildGuardTest` を修正後コードで単体4モデル再生成・3判定器採点（旧結果を「置き換え済み」記録）
  - [ ] MergeBench 公開 `safety_generation.json` の `WildGuardTest` と `allenai/wildguardmix` 参照回答の一致率を集計
- [ ] **第4.1節・第4.3節 `compare.py` 拡張**:
  - [ ] 同じ安全性（基準有害率、既定=安全専門モデルの2倍）での有用性線形補間・「到達せず」処理・判定器別出力
  - [ ] 注入量（安全更新ノルム・同 α TA 比）を `merge_metadata.json` に記録する処理
  - [ ] 参照点3点（Base・安全専門・分野専門）の必須プロットと α=0 差異の注記
  - [ ] 対応付きブートストラップ（1,000回）による手法間・判定器間差分の 95% 信頼区間
  - [ ] テスト追加（補間・「到達せず」・参照点欠落エラー・ブートストラップ CI）

## フェーズ4（11/28）：`gemma-2-9b-it`・実環境専門モデル検証・人手ラベル準備

- [ ] `gemma-2-9b-it` 4モデル（Base, safety, math, coding）の検証ゲート実行・安全性劣化確認
- [ ] 実環境トラック `WizardCoder`・`MedAlpaca` の検証ゲート実行（不合格なら公式形式で再評価）
- [ ] MergeBench 本家の lm-eval 版での `gsm8k_cot` 誤抽出再現確認（別実験フォルダ）
- [ ] **第3節の5**: 人手ラベル追加50件（両規則妥当 ＆ HB/SR 無害 ＆ WG 有害）のブラインド抽出

## フェーズ5（11/29〜12/5）：本実験開始判定（5.1）と実行

- [ ] 第5.1節の全5条件達成確認（本実験開始ゲート）
- [ ] 標準トラック主実験（Llama-3.1-8B-Instruct × 2ペア = 76モデル＋乱数シード）
- [ ] `gemma-2-9b-it` 一般化確認・人手ラベル集計・数値確定
