# タスクリスト: 親リポジトリ整理・vLLM確認・煙テスト・専門モデル事前チェック・実環境再判定

作成日: 2026-11-26  
最終更新: 2026-11-26  
対象ワークスペース: `/mnt/nas/home/hiromi/src/safetybench`  
実装計画: [implementation_plan.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/repo_cleanup_and_prechecks/implementation_plan.md)  
完了レポート: [walkthrough.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/repo_cleanup_and_prechecks/walkthrough.md)

## タスク一覧

- [x] **① 親リポジトリの整理**（履歴書き換えなし、新規コミットで削除）
  - [x] macOS の `._*`・`.DS_Store`・`SafeMergeBench.zip`・`safemergebench_vs_mergebench.patch` を `git rm` で削除
  - [x] 外部リポジトリのコピー（`LED-Merging`, `MergeAlign`, `MergeBench`, `SafeMERGE`）を `git rm -r` で削除
  - [x] 外部リポジトリ 4 つを固定コミットの Git submodule として追加
  - [x] 親リポジトリの `.gitignore` を作成・更新（`._*`, `.DS_Store`, `*.zip`, `*.patch`, Python/venv/結果キャッシュ等）
  - [x] 整理コミットを作成（履歴書き換えなし）
- [x] **② iag-02 での vLLM 動作確認 & テンプレート方針の統一（`hf`）**
  - [x] `iag-02`（物理 GPU 1・2 が CUDA 可用、GPU 0 は要リセット・GPU 3 はドライバ初期化不可、`CUDA_MPS_PIPE_DIRECTORY=/nonexistent` で回避）で `vllm 0.30.0` の推論動作を確認
  - [x] SafeMergeBench の本評価テンプレートをすべて `hf`（モデル自身の `apply_chat_template`）に統一し、`llama3` は `MERGEBENCH_REPRO=1` / `llama3_mb` のみで使用するようスクリプトを更新
  - [x] `safemerge/data/mergebench_reported.yaml` に MergeBench 論文（arXiv:2505.10833 全17表）の確認結果と実測比較メモを記入
- [x] **③-1 Llama-3.2-3B-Instruct での動作確認（煙テスト）**
  - [x] 安全＋数学（`WildguardMix-DartMath`）を 11 手法マージ（ModelSoup, TA, TIES, DARE, Consensus, AsymTA, DELLA, Localize-and-Stitch, RegMean, RegMean++, Fisher）＋ LM-Cocktail 検証（P4 語彙サイズ不一致 128320 vs 128256 を実測）＋ LED-Merging 構造確認
  - [x] `transformers 5.20` / `trl 1.14` / `torch 2.14` での RegMean・RegMean++・Fisher・Localize-and-Stitch の挙動を精査し、互換パッチ（`localize_utils.py`, `fisher.py`, `fisher_utils.py`, `taskloader.py`, `run_mergealign.py`, `integrity_gate.py`）を適用・記録
  - [x] 完全性ゲート・メタデータ・重み数値監査・6プロンプト生成確認を実施（10/11 PASS、`RegMean (reduction=0.5)` の bf16 Gram 爆発 `max_diff=71.96`・出力崩壊を検出）
- [x] **③-2 専門モデル単体の事前チェック（安全性＋有用性再現確認）**
  - [x] `Llama-3.1-8B-Instruct` ベース・安全・数学・コードの 4 モデル単体評価（有用性＋安全性8ベンチマーク×公式3判定器）を完走
  - [x] 数学（GSM8K）とコード（HumanEval+ / MBPP+）の再現確認＋`utility_check.py` 実行（`coding` は 65.43%/59.71% で Table 17 と整合、`math` で `lm-eval 0.4.14.dev0` の `$.` 正規表現バグ 5.23% vs 真の `\boxed{}` 正答率 **82.94%** を特定）
  - [x] 数学・コード専門モデルでベース比の大幅な安全性劣化（`_math`: HarmBench OOD 6.68%→38.58% [5.8×]、`_coding`: 6.68%→15.89% [2.4×]）を公式3判定器で定量確認
  - [x] `safety-eval-fork` の `wildguardtest/__init__.py` における `**gt` による `"response"` 上書きバグを発見・修正
  - [ ] *(保留・11/29 規模判断次第)* `gemma-2-9b-it` 系 4 モデル単体評価・`LED-Merging` 動的パス対応ラッパー・`RegMean` fp32 Gram アブレーション
- [x] **③-3 実環境トラックの公式プロンプト再判定**
  - [x] `results/realworld/` の全 121 モデル（3シード / 127,413 応答）に対して `safemerge/judge.py` で公式3判定器（HarmBench 公式7ルール・WildGuard 公式・StrongREJECT 公式）をかけ直し（パースエラー 0 件）
  - [x] 簡易プロンプト vs 公式プロンプトの比較（$\kappa=0.681$、最大 `+18.2pt`）および判定器間差（0.0%〜0.2% vs 86.6%、非崩壊でも最大 66.4pt）が維持・強化されることを確認
- [x] **ドキュメント整備・テスト検証**
  - [x] 単体テスト 27/27 PASS・`bash -n` PASS・[PATCHES.md](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/PATCHES.md) 更新
  - [x] [walkthrough.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/repo_cleanup_and_prechecks/walkthrough.md) に全結果・重大発見・11/29 標準トラック規模判断材料を記録
