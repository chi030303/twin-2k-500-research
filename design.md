# Task

Predict how the same person answers the wave-4 questions, using only their wave 1–3 answers to questions wave 4 does not repeat. Compare the prediction with the wave-4 answer and with that person's own test-retest.

Use Hugging Face `wave_split`. Use `full_persona` only to read the block list. Repeated questions there keep the wave-4 answer.

# Exploration

Done. Tables and the Census comparison are in [report/exploration.md](report/exploration.md). Reproduce with `.venv/bin/python -m exploration.explore`.

What the run changed in the plan:

- Headline human ceiling on our scorer: family-weighted retest **82.4%**. Population baseline, ignoring the persona: **76.8%**. The paper's published ceiling is 81.72% on a different task average.
- Pricing is the informative block: retest **83.9%**, population baseline **55.9%**. Base-rate, disease, Linda, outcome bias, and sunk cost sit within 2 points of the baseline. A persona cannot help much there.
- Every wave 1–3 persona is about **24k** rough tokens (characters / 4), so above an 8k window and under 32k. The summary is about **13k** characters. `full_persona` text is longer because it includes wave-4 answers.
- All 2,058 people have exactly one variant in each randomization family. Missing variants are the design.
- Sex matches US adults (Census July 1 2024). Ages 50–64 are high by 9 points; 65+ is low by 9 points. Hispanic origin is 9.4% versus 18.5% of Census adults. The race item is a single choice, so it is not a Census race table.

Not doing word frequency, word clouds, or clustering of persona text. The wording is shared. Forward Flow is a short elicitation (median 60 words).

# Representativeness and bias

Demographics are wave 1 only. Later waves are behavior.

Two uses of sex, race, age, and education:

- Describe who the panel is, against Census adults where the categories allow it.
- Slice the same retest, and later the same model score. The slice is an audit. It is not a feature added to raise accuracy.

Do not use a score for credit, hiring, medicine, or any decision about a named person.

# Methods

Four ways to do this one prediction, in order of cost. Not four products.

0. Weak baselines: random, and the population constant already in the exploration report. A demographics-only table model comes in step 3.
1. Official setup: persona text plus the new question, answered by a language model, parsed back to a survey code. Full text and summary.
2. If the prompt does not fit: retrieve the prior answers relevant to the new question.
3. Bonus: fine-tune a model under 0.5B parameters on some people. Held-out people only at test time. The prompt does not contain the earlier answer to that item, or the wave-4 answer.

Steps 0–2 are the plan. Step 3 waits until the scorer has been applied to a real model output.

# Evaluation

The yardstick stays fixed. GPT-4.1-mini is not the judge.

- Labels are wave-4 answers. The human ceiling is the retest on the same items, already computed.
- The scorer is `twin2k.scoring`: exact match on binary items, `1 - |a - b| / span` otherwise, deciles for the unbounded anchoring estimates.
- Every system is scored on the same people and the same items. What may change is the persona (full text, summary, retrieved excerpts) and, later, a small fine-tune.
- GPT-4.1-mini is the official prompt system. Step 3 scores the published simulation files with this function. We do not call the API, and we do not ask a model to grade another model.
- Acceptance: beat the population baseline on the families where that baseline is weak, especially pricing, and report the ratio to the human retest. Do not expect to pass the retest.

# Business

Amity already separates a survey twin from a customer twin.

- HANDY, and the retail Large Behavior Model, learn from purchase logs. This file has no receipts. The pricing block is a stated grocery choice by a US online panel. It can dry-run a price question. It cannot forecast sales for a Thai or Southeast Asian retailer.
- SAVVY needs each account's recent behavior to pick a next action. This file has no account timeline.
- LIVELY needs call transcripts and policy text. Different inputs.
- The usable job is aggregate research: simulate a panel on a new pricing or attitude item, then check it against a fresh human wave before spending on a real test.

Limits sit in the exploration results. The panel is US, English, and online. It is older in the middle and light on Hispanic respondents relative to Census adults. Stated purchase is not a receipt. Several classic experiments are already predicted by the popular answer, so a twin adds nothing there. People disagree with their own earlier answer about 18% of the time, which no model should be asked to beat. A new country, language, or retailer needs its own human wave, a recomputed ceiling, and a versioned persona builder. Retraining on a schedule, without new human answers, does not create that generalization.

# Maintenance

Track retest gap by family and by sex, race, age, and education. Rebuild personas when a new human wave arrives, or when one group's gap versus its own retest widens. Version the persona rules, the person split, and the scorer together.

# Order

1. Task definition. Done.
2. Exploration: alignment, retest, variants, length, demographics. Done.
3. Point the same scorer at a demographics-only baseline and the published GPT-4.1-mini outputs.
4. Small comparison of full-text prompt versus summary prompt.
5. Retrieval only if the prompt still does not fit.
6. Fine-tune under 0.5B parameters if the loop above is stable.
