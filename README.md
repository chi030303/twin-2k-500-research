# Twin-2K-500 behavior model

Predict one person's wave-4 answers from the wave 1–3 answers that wave 4 does not repeat. Score against the wave-4 answer and against that person's own retest.

## Read this first

- [design.md](design.md) — task, method ladder, evaluation, business limits
- [note.md](note.md) — block list by wave
- [report/exploration.md](report/exploration.md) — human retest and the population baseline
- [report/evaluation.md](report/evaluation.md) — published GPT-4.1-mini scored on wave 4

Family-weighted human retest is **82.4%**. A constant taken from the earlier answers, with no persona, scores **76.8%**. Pricing is where the person matters (83.9% versus 55.9%). Several classic experiments are already explained by the popular answer.

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
```

Answer sheets download into `data/sim/` and are not committed.

## Bonus fine-tune

[training/HANDOFF.md](training/HANDOFF.md). Run that on a GPU. It trains `Qwen2.5-0.5B-Instruct` on 500 people and scores the other 1,558. GPT-4.1-mini is a system we score, not the judge.
