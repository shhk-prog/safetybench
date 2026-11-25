# 実装計画: 親リポジトリ整理・vLLM確認・煙テスト・専門モデル事前チェック・実環境再判定

作成日: 2026-11-26  
対象ホスト: `iag-02`（NVIDIA H100 96GB × 4、MPS 設定により GPU 0・1 が CUDA 可用）

---

## 1. 親リポジトリの整理（履歴書き換えなし）

1. 親リポジトリ（`/mnt/nas/home/hiromi/src/safetybench`）から以下を `git rm` で削除する：
   - `.DS_Store`, `._.DS_Store`, `._SafeMergeBench`, `._SafeMergeBench.zip`, `._safemergebench_vs_mergebench.patch`, `SafeMergeBench.zip`, `safemergebench_vs_mergebench.patch`
   - 外部リポジトリのコピー 4 つ：`LED-Merging/`, `MergeAlign/`, `MergeBench/`, `SafeMERGE/`
2. 4 つの外部リポジトリを、それぞれ検証済みの固定コミットで `git submodule add` する（`safemerge/external/PATCHES.md` の記録に一致）：
   - `MergeBench`: `https://github.com/uiuctml/MergeBench.git` @ `dedf1b1`
   - `MergeAlign`: `https://github.com/hammoudhasan/MergeAlign.git` @ `4d959be`
   - `LED-Merging`: `https://github.com/MqLeet/LED-Merging.git` @ `1149224`
   - `SafeMERGE`: `https://github.com/aladinD/SafeMERGE.git` @ `6724444`
3. 親リポジトリルートに `.gitignore` を追加し、`._*`, `.DS_Store`, `*.zip`, `*.patch`, `__pycache__/`, `*.pyc`, `*.tmp` 等を除外する。
4. 新規コミットとして記録する（履歴の書き換えは行わない）。

---

## 2. iag-02 での vLLM 動作確認 & テンプレート方針（`hf` 統一）

- **vLLM 動作**: `iag-02` 上の `.venvs/safety-eval`（`vllm 0.30.0`, `torch 2.13.0+cu130`）で GPU 0 / GPU 1 を用いた生成を検証する。
- **テンプレート方針の一貫化（F3 対応）**:
  - SafeMergeBench の本評価（`scripts/evaluate_safemerge.sh`, `scripts/check_single_experts.sh`）では、チャットテンプレートを持つ Instruct モデル（Llama-3 系・Gemma-2-it 系ともに）に対して `safety-eval` へ **`hf`**（モデル自身の `tokenizer.apply_chat_template`）を渡すように揃える。
  - `integrity_gate.py` 側では引き続きモデル本来の家族名（`llama3`, `gemma` 等）を検査しつつ、評価テンプレート名 `hf` との対応を `eval_config.json` に明記する。
  - MergeBench の再現・比較実験（`scripts/template_ablation.sh` や `--mergebench-repro` / `llama3_builtin` 指定時）のみ `llama3` 組み込みテンプレートを使用する。

---

## 3. Llama-3.2-3B-Instruct での動作確認（煙テスト）

- **対象**: `meta-llama/Llama-3.2-3B-Instruct` の安全＋数学（`WildguardMix-DartMath`）。
- **注意点（棚卸し結果 P4）**: `MergeBench/Llama-3.2-3B-Instruct_math` は `vocab_size=128320`（ベース `128256`）かつ `safety` は `eos_token_id=128009` のみという差異があることが既知。完全性ゲートが正しくこれを検知するかどうかも同時に確認する（煙テストでのゲート確認と、モデル自体の出力・メタデータ・数値健全性確認を切り分ける）。
- **テストする手法（各 1 点）**:
  - MergeBench 組み込み: `ModelSoup`, `TaskArithmetic` (α=0.6), `TIES` (α=0.6, K=0.3), `DARE` (α=0.6, p=0.9), `Consensus` (α=0.6), `LocalizeAndStitch` (dataless, sparsity=0.1), `RegMean` (reduction=0.5), `RegMeanPlusPlus` (reduction=0.1), `Fisher` (model_coeff_value=1.0)
  - 外部手法ラッパー: `DELLA` (α=0.6), `AsymTaskArithmetic` (safety=0.6, domain=1.0), `LMCocktail` (temp=1.0), `LEDMerging`
- **ライブラリバージョン差の精査（最重要）**:
  - `transformers 5.20.2`, `trl 1.14.2`, `torch 2.14.1` 環境で `RegMean` / `RegMeanPlusPlus` / `Fisher` / `LocalizeAndStitch` のコードパス（`SFTConfig`, `SFTTrainer`, `formatting_func`, `max_length`, `torch_dtype`, `DeepSpeedEngineWrapper` 等）を実実行＋中間統計量検証し、エラーやサイレントな仕様変化がないか確認する。必要ならパッチを記録して修正する。
- **検証項目**:
  - `merge_metadata.json` の正確性
  - `integrity_gate.py --weights`（数値有限性・変化テンソル数・設定差分）
  - 簡易生成チェック（vLLM で数問生成し、崩壊・文字化け・無限反復がないか確認）

---

## 4. 専門モデル単体の評価（安全性・有用性再現確認）

- **優先対象（Llama-3.1-8B-Instruct 系 4 モデル）**:
  - `meta-llama/Llama-3.1-8B-Instruct`（ベース）
  - `MergeBench/Llama-3.1-8B-Instruct_safety`
  - `MergeBench/Llama-3.1-8B-Instruct_math`
  - `MergeBench/Llama-3.1-8B-Instruct_coding`
- **追加対象（余力・時間に応じて Gemma-2-9B-it 系 4 モデル）**:
  - `google/gemma-2-9b-it`, `_safety`, `_math`, `_coding`
- **評価内容**:
  - 有用性: `lm-eval`（`gsm8k_cot` 等）および `bigcode-eval`（`humanevalplus`, `mbppplus`）→ `safemerge/utility_check.py` で MergeBench 報告値と比較（5pt 以上ずれの検出）。
  - 安全性: `safety-eval`（4 タスク、`hf` テンプレート）＋ `extra_safety.py`（追加 4 タスク）→ `judge.py`（HarmBench / WildGuard / StrongREJECT の 3 判定器）＋ `validity_audit.py` ＋ `metrics.py`。

---

## 5. 実環境トラックの公式プロンプト再判定

- **対象**: `safemergebench/results/realworld/` 以下の生成済み応答（`exp1_v3` 116 モデル + `fp32_sst_math` 5 モデル = 121 モデル、計 127,413 応答）。
- **方法**:
  - 元ファイル（`sst_v2` 簡易プロンプト結果入り）を壊さないよう別出力ディレクトリ（例: `results/realworld_rejudged/`）または別ファイルに出力しつつ、`safemerge/judge.py --force` で公式 3 判定器（HarmBench 公式 7 ルール、WildGuard 公式、StrongREJECT 公式）を再実行する。
  - H100 × 2（GPU 0・1）を並列活用し、バッチサイズを最適化（例: 32〜64）して数時間以内に完了させる。
  - まず表 3 / P2 に直結する α=0.6（および代表モデル群）を優先実行して早期に結論を確認し、続けて全 121 モデルを完了させることも検討する。
- **検証項目**:
  - 簡易プロンプト（旧 HarmBench 判定）と公式プロンプト（新 HarmBench 判定）の一致率・ASR 差分
  - 3 判定器（HarmBench 公式 / WildGuard 公式 / StrongREJECT 公式）間の乖離（0%〜68% の結論が維持されるか）
