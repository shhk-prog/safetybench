# SafeMergeBench バグ1〜4・重要事項5〜7修正および4仮想環境構築ウォークスルー

作成日: 2026-11-26
対象コミット: `e732096` (`Fix bugs 1-4 (Fisher, XSTest safe, parse errors, official judge prompts) and items 5-7`)
関連ドキュメント:
- [implementation_plan.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/safemergebench_implementation/implementation_plan.md)
- [walkthrough.md](file:///mnt/nas/home/hiromi/src/safetybench/docs/safemergebench_implementation/walkthrough.md)（初回実装記録・上書きせず保持）

---

## 1. 修正したバグ 1〜4（実行前必須項目）

### バグ1: Fisher が 2モデルマージで落ちる問題
- **原因**: [merging/merging_methods/fisher.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/merging_methods/fisher.py#L56-L61) が `zip(save_group.split("-"), self.ft_ckpts)` を行うため、1段目を `--task_names DartMath --save_group WildguardMix-DartMath` で呼ぶと `self.ft_ckpts` に1モデルしか入らず `WildguardMix` に `DartMath` が誤割当され `KeyError` になっていた。
- **修正**: [merging/main.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/main.py#L75-L85) に `resolve_ckpt_task_names()` を追加し、`algo == "Fisher"` かつ `save_group` 指定時は `save_group.split("-")` の全モデルを読み込むように修正。

### バグ2: XSTest の「安全な要求」の判定誤り（50/250件のみ拾っていた問題）
- **原因**: `_is_xstest_safe_prompt()` が型名に `safe` を含むものだけを安全と判定し、`homonyms` / `figurative_language` / `definitions` / `privacy_public` 等を取りこぼし、さらに WildGuard 予測値 `prompt_harmfulness` にフォールバックしていた。
- **修正**: [safemerge/metrics.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/metrics.py#L96-L118) で `type` が `contrast_` で始まらないものを安全な要求と判定し、予測値 `prompt_harmfulness` へのフォールバックを削除。
- **検証**: MergeBench 公開の `Llama-3.2-3B` `safety_generation.json`（450件）に対して `safe_n == 250` を単体テストで直接検証済み。

### バグ3: 判定器パース失敗が「無害・安全」として数えられていた問題（偽の安全バイアス）
- **修正（`judge.py`）**: [safemerge/judge.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/judge.py#L127-L214) の3パーサ（`parse_harmbench_output`, `parse_wildguard_output`, `parse_strongreject_score`）を厳格化し、読めない出力・不正値・`n/a` をすべて `None`（＋`*_parse_error=True`）として返すように変更。
- **修正（`metrics.py`）**: [safemerge/metrics.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/metrics.py#L139-L227) で `n_parse_error` / `n_valid_parse_error` / `n_scored` / `n_valid_scored` を別集計し、パース失敗応答を `ASR` / `cond_ASR` の分母からも `VSR` の分子（安全応答数）からも除外。

### バグ4: 判定器プロンプト・切り詰め・BOS・`vanilla` 要求の不備
- **HarmBench 公式プロンプト**: [LED-Merging/HarmBench/eval_utils.py](file:///mnt/nas/home/hiromi/src/safetybench/LED-Merging/HarmBench/eval_utils.py#L312-L359) の `LLAMA2_CLS_PROMPT`（全7ルール：「caveats or warnings」「コードエラー許容」「英語ルール」および `prompt_contextual` を含む）を一字一句そのまま [safemerge/judge.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/judge.py#L44-L90) に採用。
  - **補足（実環境トラックとの差異）**: 実環境トラック（`sst_v2/v3/scripts/eval/eval_safety.py`）は簡易形式 `###Instruction: ... ###Classifier Bias: 1.0 ###Label:` を使用していた。表3等の実環境トラック再現値と標準トラックの公式プロンプト判定の差異として論文・付録に明記する必要がある。
- **StrongREJECT 公式プロンプト**: [judge_templates.json](file:///mnt/nas/home/hiromi/src/sst_v2/v3/evaluators/strong_reject/strong_reject/eval_files/judge_templates.json#L9) の `strongreject_finetuned` をそのまま採用し、応答の事前 512 トークン切り詰め＋`"1".."5"` ロジット期待値スコアを実装。
- **右切り捨て防止**: `tokenizer.truncation_side = "left"` に加え、`truncate_text_tokens()` で要求・応答をテンプレート埋め込み前にトークン単位で切り詰めることで末尾キュー（`Answer: [/INST]` 等）の消失を完全に防止。
- **WildGuard BOS 二重付与修正**: [wildguard.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safety-eval-fork/src/classifier_models/wildguard.py#L16-L29) 準拠とし、先頭の `<s>` を削除。
- **DAN / WildJailbreak での `vanilla` 優先**: `select_judge_prompt_text()` で HarmBench / StrongREJECT 判定時は `Behavior` / `vanilla` / `Goal` を `prompt` / `adversarial` より優先し、[datasets.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/data/datasets.py#L179-L194) でも WildJailbreak の `vanilla` と `adversarial` を両方保持するように修正。

---

## 2. 重要事項 5〜7 および実験設計サポート

- **重要事項5（Hammoudら LM-Cocktail の扱い）**:
  - [run_mergealign.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_mergealign.py) の既定記録名を `LMCocktail` とし、`MergeBench/{domain}_val` から較正データを抽出する `--mergebench-domains` を追加。
  - [run_standard_track.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/run_standard_track.sh) から温度の α 格子掃引を削除し、公式既定の1点（`temp=1.0`）のみ実行に変更。
- **重要事項6（Consensus TA の実効 `lamda` 記録）**:
  - [merging/main.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/merging/main.py#L88-L102) で `Consensus` 実行時は実際に使われる `lamda=0.2` を `kwargs["lamda"]` に記録し、`cli_lamda_ignored` と注記を `merge_metadata.json` に併記。
- **重要事項7（DELLA の埋め込み・出力層マージ条件差）**:
  - [run_della.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_della.py) の `merge_metadata.json` と [PATCHES.md](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/PATCHES.md) に `merges_embeddings_and_lm_head: true` と条件差を明記。
- **単体専門モデルの事前検証 & 非対称 Task Arithmetic**:
  - マージ前に Instruct ベース＋安全＋数学＋コード（6〜8モデル）の安全性劣化を確認する [scripts/check_single_experts.sh](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/scripts/check_single_experts.sh) を追加。
  - 分野重み `1.0` 固定・安全重みのみ α 掃引する [safemerge/external/run_asymmetric_ta.py](file:///mnt/nas/home/hiromi/src/safetybench/safemergebench/safemerge/external/run_asymmetric_ta.py) を追加（`run_standard_track.sh --include-asymmetric-ta` で併用可能）。

---

## 3. テストと 4仮想環境構築（`.venvs/`、GPU実験未実行）

- **回帰テスト**: `CUDA_VISIBLE_DEVICES="" python3 -m unittest discover -s safemerge/tests -v` → **全26件 PASS**（既存7件＋初回14件＋今回バグ1〜4・重要事項5〜6回帰5件）。
- **`uv` 4仮想環境（すべて Python 3.10.22、`sudo`・GPU実験なし）**:
  - `.venvs/merging`: `torch 2.14.1+cu130`, `transformers 5.20.2`, `trl 1.14.2`, `deepspeed 0.20.1`, `peft 0.21.3`, `LM_Cocktail 0.0.4`, `mergekit 0.1.5` (`mergekit-yaml` 動作確認済み)
  - `.venvs/lmeval`: `torch 2.14.1+cu130`, `transformers 5.20.2`, `lm-eval 0.4.14.dev0`, `langdetect`, `immutabledict`
  - `.venvs/bigcode`: `torch 2.14.1+cu130`, `transformers 5.20.2`, `bigcode-eval`, `evalplus 0.2.1`
  - `.venvs/safety-eval`: `torch 2.13.0`, `transformers 5.20.2`, `vllm 0.30.0`, `safety-eval 1.0`, `fschat 0.2.36`, `fire 0.7.1`
- **GPU 0 故障に伴う vLLM の注意点**:
  - `CUDA_VISIBLE_DEVICES` 未指定で `import vllm` を呼ぶと、故障中の GPU 0 に対する NVML 問い合わせ（`nvmlDeviceGetHandleByIndex(0)`）で `NVMLError_Unknown` が発生することを確認。実行時は必ず `CUDA_VISIBLE_DEVICES=1`（または必要に応じて vLLM の NVML 警告抑制）を指定する必要があります。
