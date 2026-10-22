# Fine-tune handoff

Bonus only. The scored comparison is already in [report/evaluation.md](../report/evaluation.md). This run checks that a model under 0.5B can train and be scored. It is not required to beat GPT-4.1-mini.

Run it on one A100 40GB. The laptop does not need a GPU.

## Machine

The trainer is one process with batch size 1. A node has four A100 40GB cards and can share them. This run uses one card. Extra cards do not shorten it.

Driver 535 supports CUDA through 12.2. `pip install torch` with no index installs a wheel built for CUDA 12.4 or newer, and that wheel needs driver 550 or newer. The install succeeds, then `import torch` fails with an insufficient-driver error. Use the CUDA 12.1 wheel, which runs on driver 535. Install it last: `requirements-train.txt` lists `torch` with no version, and a current Transformers release can depend on a newer torch. Either one replaces the 12.1 wheel if it is installed afterwards.

```bash
pip install -r requirements.txt
pip install "transformers==4.51.3" "peft==0.15.2" "accelerate==1.6.0"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; assert torch.version.cuda.startswith('12.1') and torch.cuda.is_available()"
```

The wheel carries its own CUDA libraries. A newer CUDA toolkit on `LD_LIBRARY_PATH` is not part of this setup.

The full run is 47,948 training rows from the 500, then 149,632 scored rows from the 1,558. See [report/finetune.md](../report/finetune.md). On one A100, plan on 6 to 12 hours, including the download and the 24-person smoke. A wall-clock cap of 12 hours covers that range. A 24 hour cap is only the kill limit. The card is released when the script exits, not when the cap expires. Scoring on this card uses `--eval-batch 32`. Decoding stays temperature 0 and 16 new tokens.

## What is trained

- Base: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B). Do not switch to 1.5B.
- Loss: next token on the answer only. The answer is that person's wave-4 choice. Not DPO.
- Persona: `wave_split.wave1_3_persona_text`, prefix truncated to the 2,048-token budget. This field already drops questions wave 4 repeats. Do not use `full_persona`. Its summary and text keep wave-4 answers.
- Split: 500 train / 1,558 test, `numpy` seed 0, pids sorted before sampling. Test pids never enter training.
- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`: rank 16, alpha 32, dropout 0.05. Learning rate 2e-4, 2 epochs, batch size 1, gradient accumulation 16, max length 2,048, warmup ratio 0.03, seed 0.
- Eval decoding: temperature 0, 16 new tokens.

## Data

The dataset is not in the repo. `data/` is gitignored.

This run reads one Hugging Face config, `LLM-Digital-Twin/Twin-2K-500` / `wave_split`, seven parquet shards, about 190MB. The first `--prepare-only` downloads them into `data/parquet/wave_split/` if that directory is empty. Nothing else has to be built by hand.

Columns used:

- `pid`
- `wave1_3_persona_text` — the prompt. Questions that wave 4 repeats are already left out.
- `wave4_Q_wave1_3_A` — the earlier answer. Used for the retest and the training-set baseline only.
- `wave4_Q_wave4_A` — the label. Never copied into the prompt.

`full_persona` is not downloaded. `Qwen/Qwen2.5-0.5B-Instruct` downloads on `--train`, not during `--prepare-only`.

The repo is [chi030303/twin-2k-500-research](https://github.com/chi030303/twin-2k-500-research). Clone it. `data/` is not in the repo. If a copy already has `data/parquet/wave_split/`, the download is skipped. The held-out scores are committed at `training/outputs/qwen-0.5b-lora/metrics.json`. Adapter weights are not in git. They are [chi10969/twin2k-qwen2.5-0.5b-lora](https://huggingface.co/chi10969/twin2k-qwen2.5-0.5b-lora).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; assert torch.version.cuda.startswith('12.1') and torch.cuda.is_available()"
```

`requirements-train.txt` lists `torch` with no version. The reinstall has to come after it, or a newer wheel replaces the CUDA 12.1 build.

If `huggingface.co` is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Weights and parquet download on first run. Neither is committed.

## Smoke, then the real run

```bash
python -m training.finetune --prepare-only
python -m training.finetune --train --eval --limit-people 24 --epochs 1 --max-length 512
python -m training.finetune --train --eval --eval-batch 32
```

`--prepare-only` checks the split, the item count, and that wave-4 question stems are absent from the persona. It does not load the model.

`--limit-people` writes to `training/outputs/smoke/` and does not replace a full adapter. The full run writes to `training/outputs/qwen-0.5b-lora/`.

## What to read off the machine

`training/outputs/qwen-0.5b-lora/metrics.json`

- `retest`: test people's earlier answer versus their wave-4 answer.
- `baseline`: mode, or median, fit on the 500 training people only, scored on the test people's wave-4 answers.
- `model`: the adapter on those same test rows.
- Each is family-weighted and item-weighted. Pricing is its own row.
- Decile edges for anchoring use the training people's earlier answers, not the full 2,058.

The 76.8% and 55.9% figures in [report/evaluation.md](../report/evaluation.md) are fit on everyone. This run's baseline is fit on the 500. The held-out comparison is written up in [report/finetune.md](../report/finetune.md).

The run is acceptable when `metrics.json` exists and `model.family` is filled on the held-out people. Beating the training-set baseline on pricing is a good result. Beating 66.9%, or the retest, is not required.

## Do not

- Put `full_persona`, `wave4_Q_wave4_A`, or `wave4_Q_wave1_3_A` into the prompt.
- Train on a test pid, including after a later wave.
- Commit adapter weights, checkpoints, or `data/`. `training/outputs/qwen-0.5b-lora/metrics.json` is committed so the held-out scores can be checked.
