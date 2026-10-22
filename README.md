# Twin-2K-500 behavior model

Predict one person's wave-4 answers from the wave 1–3 answers that wave 4 does not repeat. Score against the wave-4 answer and against that person's own retest.

## Documents

- [design.md](design.md) — task, method ladder, evaluation, business limits
- [note.md](note.md) — block list by wave
- [report/exploration.md](report/exploration.md) — human retest and the population baseline
- [report/evaluation.md](report/evaluation.md) — published GPT-4.1-mini scored on wave 4
- [report/acceptance.md](report/acceptance.md) — price-aware baseline, held-out routing, intervals, slices
- [report/finetune.md](report/finetune.md) — Qwen2.5-0.5B LoRA on the held-out 1,558

Family-weighted human retest is **82.4%**. A constant from the earlier answers, with no persona, scores **76.8%**. Pricing has the widest gap (retest 83.9%, constant 55.9%), but a per-product price cutoff with no persona already reaches **70.8%**, above published GPT-4.1-mini full text (66.9%). Several classic experiments already sit near the constant.

## Reproduce the exploration

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m exploration.explore
```

Parquet files download into `data/` and are not committed. The scorer used for later models is `twin2k/scoring.py`.

## Reproduce the published-model scores

```bash
.venv/bin/python -m evaluation.score_published
.venv/bin/python -m evaluation.acceptance
```

Answer sheets download into `data/sim/` and are not committed.

## Bonus fine-tune

[training/HANDOFF.md](training/HANDOFF.md) trains `Qwen2.5-0.5B-Instruct` on 500 people and scores the other 1,558. Held-out scores are in [report/finetune.md](report/finetune.md): family-weighted 70.2% against a training-set constant of 76.8%, and 73.7% against 55.9% on pricing. That scoring run cut long prompts from the end, which removed the question on three families. Rescored with the fix on 400 test people, the family mean is about 76.6% against 76.9% for the constant. On pricing the adapter beats a no-persona price cutoff by 3.7 points. The report lists the remaining defects, with fixes. The adapter is [chi10969/twin2k-qwen2.5-0.5b-lora](https://huggingface.co/chi10969/twin2k-qwen2.5-0.5b-lora). GPT-4.1-mini is one of the systems being scored.

## With more time

- Rescore every family on all 1,558 test people with the truncation fix, and score options by log-probability.
- Run the shuffled-persona test on pricing, to check whether the LoRA's edge over the price cutoff comes from the person.
- Raise the persona budget toward the full 24k tokens, or retrieve the related past answers.
- Calibrate the pricing purchase curve. Full text says yes too rarely at every price.

## License

Code and writeups in this repository are [MIT](LICENSE). The Twin-2K-500 dataset is CC BY 4.0 and is not included here.
