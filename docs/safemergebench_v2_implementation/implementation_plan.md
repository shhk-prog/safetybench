# SafeMergeBench 実装指示書 v2 実装計画 (Implementation Plan)

作成日: 2026-11-26  
対象リポジトリ: `/mnt/nas/home/hiromi/src/safetybench/safemergebench`（第1節ルールは `/mnt/nas/home/hiromi/src/sst_v2` にも適用）  
準拠文書: [SafeMergeBench 実装指示書 v2（11月26日）.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/SafeMergeBench%20実装指示書%20v2（11月26日）.md)

---

## 1. 方針と優先順

実装指示書 v2 の第7節（日程）および第5.1節（本実験開始条件）に従い、以下の順で段階的に実装・検証を進める。既存の結果フォルダは一切上書き・削除せず、新設する `results/runs/` と `INDEX.md` に記録する。

### フェーズ1（11/26 最優先・本ターン着手）
1. **第1節：実験フォルダ管理ツールと強制機構**
   - `safemerge/new_run.py`:
     - `results/runs/YYYYMMDD-HHMM_<name>/`（JST、英数字とハイフンのみ、同一分・同名衝突時は `-2`, `-3` ... を付与して上書き防止）を作成。
     - サブディレクトリ `config/`, `logs/`, `outputs/`、および `RUN.md`（様式準拠、引数で上半分を自動記入）・`commands.sh`（実行可能ヘッダ付き）・`env.json`（親リポジトリ git commit、未コミット有無と差分→`config/uncommitted.diff`、各サブモジュール commit、ホスト名、`nvidia-smi` 要約、`.venvs/*` およびシステム python の `pip freeze` 記録）を生成。
     - `results/runs/INDEX.md` へ1行追記し、作成したフォルダパスを標準出力に返す。
   - `results/runs/INDEX.md` 初期化：
     - 既存結果（`safemergebench/results/` 配下、`sst_v2/v3/results/`, `sst_v2/v5/results/`）を動かさず「旧形式」として一覧化し、無効になった結果（v3 GSM8K、簡易プロンプト HarmBench、修正前 WildGuardTest）を明記。
   - `scripts/evaluate_safemerge.sh`, `scripts/run_standard_track.sh`, `scripts/check_single_experts.sh`, `scripts/template_ablation.sh`:
     - すべて `--run-dir <DIR>` を必須引数化。`<DIR>/RUN.md` が存在しなければ終了コード 2 で拒否。
     - 実行コマンドを `<DIR>/commands.sh` に自動追記。
2. **第3節（1・4・6・7）および第4.3節（Wilson 95% 信頼区間）**
   - **3.1 XSTest 分離と ID/OOD 厳密化（`metrics.py`, `compare.py`）**:
     - XSTest 行では、`safe_refusal_rate` / `cond_safe_refusal_rate`（非 `contrast_` 250件）に加え、`ASR` / `cond_ASR` / `VSR`（および明示列 `unsafe_n`, `unsafe_scored_n`, `unsafe_harmful_n`, `unsafe_ASR`, `cond_unsafe_ASR`）を `contrast_`（200件）のみで算出。
     - 攻撃ベンチマーク平均（`all_attack`, `in_distribution`, `out_of_distribution`）の集計関数を追加し、XSTest を確実に除外しつつ `distribution.yaml` に厳密準拠させる（標準トラックでは `wildjailbreak`, `wildguardtest` が ID、残り 5 つが OOD）。
   - **3.4 崩壊判定の OR 化（`validity_audit.py`, `metrics.py`）**:
     - `validity_audit.combined_invalid(text)` を新設（`extended_invalid(text) or legacy_invalid(text)`）。
     - `metrics.py` の主 `VRR`・`cond_ASR`・`VSR`・`cond_safe_refusal_rate` を `combined_invalid is None` 基準に変更し、`extended_VRR` と `legacy_VRR` の両方を別列で保持。
     - `"two two two ..."` 反復崩壊の単体テストを追加。
   - **4.3 Wilson 95% 信頼区間（`metrics.py`）**:
     - `wilson_ci(k, n, z=1.959963984540054)` を実装し、ASR・VRR・cond_ASR・VSR・safe_refusal_rate・cond_safe_refusal_rate（および有用性用ヘルパー）の `*_ci_low`, `*_ci_high` を JSON / CSV に出力。
   - **3.6 RegMean 単精度版対応（`merging/merging_methods/regmean.py`, `regmean_utils.py`, `merging/prepare_args.py`, `merging/main.py`, `PATCHES.md`）**:
     - 上流 `bfloat16` 挙動をそのまま保持しつつ、`--regmean-fp32`（または `RegMeanFP32` 記録）で Gram 積算・逆行列・行列積をすべて `float32` で行うモードを追加。`run_standard_track.sh` で単精度版・bfloat16版の両方を回せるようにし、差分および既存の Fisher 単精度積算を `PATCHES.md` に記録。
   - **3.7 LED-Merging ラッパー改善（`safemerge/external/run_led_merging.py`）**:
     - 上流 `LED-Merging/` のファイルは 1 文字も変更しない。
     - 一時作業ディレクトリに上流スクリプト・データをシンボリックリンクし、`PYTHONPATH` 先頭に配置した自動生成 `sitecustomize.py` と環境変数 `LED_MODEL_PATH_MAP`（＋必要に応じて `mask_generate.py` / `merge_llms.py` の `AutoModelForCausalLM.from_pretrained` 等フック）により、ハードコードされた `/path/to/...` や `MODEL_DIR` を実際のモデルパスへ動的リダイレクト。出力モデルと実効マスク密度統計を `out_dir` および `merge_metadata.json` に保存。
3. **第4.2節：マージ退化の検出と LED-Merging 確認（最優先）**
   - `safemerge/integrity_gate.py --weights`（単精度計算）を拡張：
     - マージ後タスクベクトル $\Delta_m = \theta_{\text{merged}} - \theta_{\text{base}}$ と各入力タスクベクトル $\Delta_i = \theta_i - \theta_{\text{base}}$ の全体・層ごとの cos 類似度
     - 各入力に対する射影係数 $\frac{\langle \Delta_m, \Delta_i \rangle}{\|\Delta_i\|_2^2}$
     - ベースおよび各入力モデルとの相対ノルム差 $\frac{\|\theta_{\text{merged}} - \theta_{\text{ref}}\|_2}{\|\theta_{\text{ref}}\|_2}$
     - 更新テンソル数・未更新テンソル例
     - 退化フラグ（`degenerate: true/false`）：相対ノルム < `--degenerate-rel-norm-thr`（既定 0.01）または分野更新の射影係数 < `--degenerate-proj-thr`（既定 0.10）。退化は評価拒否ではなくフラグ付与（注記用）。
   - **LED-Merging 最優先確認**：
     1. `sst_v2` の `sst_merge_v3_main_led_merging_safety+math_seed42`（および `safety+code_seed42`）に新しい `integrity_gate.py --weights` を実行し、ベースモデルとの差分およびマスク実態を解明（新 `results/runs/` フォルダで実施）。
     2. `safetybench` ラッパーで `Llama-3.2-3B-Instruct`（安全＋数学）の煙テストを実施し、マスク非ゼロ・ベース非一致・簡易数学正答を確認。

---

### フェーズ2（11/26 決定済み・反映完了）：第4.4節 完全性ゲート厳格度
- 根拠実験：`sst_v2/v5/results/rope_theta_ablation`（`safety+code`, $\alpha=0.6$, seed 42 で `rope_theta` のみ 1,000,000 に修正→DARE 崩壊 97.5%→99.4–100%、SST 89.1%→72.0–93.8% で停止せず）。
- `integrity_gate.py` 既定ポリシー：
  - **系統の不一致（不合格・終了コード 1）**：層ごとの相対差分の中央値 > `--lineage-thr`（既定 `0.50`）。
  - **設定・テンプレートの不一致（警告・終了コード 0）**：`rope_theta`・`vocab_size`・`eos_token_id`・チャットテンプレートの不一致は `warnings` 記録＋評価継続。
  - **実行不能（`status: "infeasible"`）**：語彙数差でマージ自体が失敗する場合（LM-Cocktail など）は `write_infeasible_metadata()` で明示記録。

### 実環境トラック `sst_v2` 先行作業（第2.5節）
- 標準トラックのマージ・評価仕組みは `sst_v2` に作らず `safetybench` のみで行う。
- 優先順：(1) WizardCoder 有用性検証（通過までコード有用性「0%」含む数値は論文結論に不使用）→ (2) LED-Merging 重み確認 → (3) MedAlpaca 検証 → (4) 数学曲線集計（4.1 同一有害率・4.3 信頼区間・主要点 1,319 問全件）→ (5) 既存 2 実験への `RUN.md` 付与。

### フェーズ3（11/27）
- **第2節**：`safemerge/data/expert_eval_spec.yaml`、`safemerge/utility_gate.py`、`configs/eval_lock.yaml`（ハッシュ固定＆`evaluate_safemerge.sh` の強制照合）を実装し、`Llama-3.1-8B-Instruct` 単体4モデルをゲート検証。
- **第3節の2・3**：`WildGuardTest` 単体4モデル再生成・3判定器採点、MergeBench 公開 `safety_generation.json` との一致率調査。
- **第4.1節・第4.3節**：`compare.py` の同一有害率での線形補間有用性、参照点3点プロット、対応付きブートストラップ（1,000回）95% CI、`merge_metadata.json` の注入量記録。

### フェーズ4〜5（11/28〜12/5）
- `gemma-2-9b-it`・`WizardCoder`・`MedAlpaca` 検証ゲート、MergeBench 本家 lm-eval 版検証、人手ラベル追加50件抽出、第5.1節開始判定後の本実験実行。

---

## 2. 検証計画

- GPU 不要の回帰テスト（`CUDA_VISIBLE_DEVICES="" .venvs/merging/bin/python -m unittest discover -s safemerge/tests -v`）：
  - 既存27件すべて PASS を維持
  - 第1節（フォルダ作成・`-2` 重複回避・`--run-dir` なしスクリプト拒否）
  - 第3節（XSTest 安全250/危険200分離・攻撃平均除外・`distribution.yaml` ID/OOD・OR崩壊判定＆`"two two two ..."` 反復・RegMean FP32 vs BF16・LED ラッパー上流無改変フック）
  - 第4.2節（退化検出：cos 類似度・射影係数・相対ノルム・退化フラグ）
  - 第4.3節（Wilson 95% 信頼区間）
- 実モデル検証（第1節の新フォルダ規則に従って `results/runs/` 配下で実行）：
  - `sst_v2` LED-Merging モデルの重み退化検査（CPU / 単精度）
  - `Llama-3.2-3B-Instruct`（安全＋数学）LED-Merging 煙テスト（GPU 1基）
