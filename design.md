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

One prediction. The arms below change the persona or the model. They do not change the scorer, and they are not four products.

The prompt settings match the official simulation config in [tianyipeng-lab/Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation): GPT-4.1-mini, temperature 0. The dataset publishes the completions. The assignment allows those files as the baseline, so the full sample does not need an API call.

## Prompt system

- Model: GPT-4.1-mini, snapshot `gpt-4.1-mini-2025-04-14` in the published files. Temperature 0. One question per call. The completion is a survey code, so the generation cap is 16 tokens.
- System prompt: answer the new question as the person in the persona, stay consistent with their past answers, and follow the format instruction. User message: the persona, the question, the options, and a line asking for only the option position or the number.
- Persona: wave 1–3 answers to questions wave 4 does not repeat. The published files include three versions of that input: demographics only, a summary, and the full text. We score all three. We do not generate them again.

## Full text and the context window

The full wave 1–3 persona is about 24k tokens. The summary is about 13k characters. This is not a problem for the comparison above, because those completions are already in `LLM_simulation_results`. Scoring a file does not send the persona anywhere.

Generating a new full-text answer is a different step. The 24k-token persona fits a 32k window and does not fit an 8k window. If we generate one, we use an endpoint with a 32k window, or we retrieve the past answers that bear on the new question and send only those. We do not shorten the full persona to fit a free-tier cap, and we do not ask the reader to create an API token. The reported table runs from the published files and `twin2k.scoring`.

## Comparison

Same people, same items, `twin2k.scoring`. The label we report is the wave-4 answer.

| Arm | Input | Answers |
| --- | --- | --- |
| Population constant | no persona | early-answer constant, already computed |
| Demographics only | demographics | published `Demographics Only - GPT4.1-mini` |
| Summary persona | summary | published `Persona Summary - GPT4.1-mini` |
| Full text persona | text persona | published `Text Persona - GPT4.1-mini` |

The question this table answers comes from the exploration, not from swapping API models. On pricing the population constant is far below the retest. On several classic experiments it is already as high as the retest. A persona is worth using only where that gap exists.

The published files were produced by predicting the wave 1–3 answer on the held-out questions. The paper uses wave 4 only as a ceiling. We score the same files against the wave-4 answer, and we recompute the retest on those same items. A second column, against the wave 1–3 answer, is only a check that our scorer is reading the files correctly. The two columns stay separate.

Toubia et al. 2025, Table 2, already varies JSON, reasoning, temperature, Gemini, and a fine-tune of GPT-4.1-mini. Those arms sit close together. We cite that and do not rerun it. Our result is the wave-4 score against our own baseline and our own retest.

## Bonus fine-tune

Done. Held-out scores, the training rows, and what to change next are in [report/finetune.md](report/finetune.md). Family-weighted, the adapter scores 70.2% against the training-set constant at 76.8%. On pricing it scores 73.7% against 55.9%. The next change is the persona budget, then which families carry loss. A full fine-tune waits until the prompt contains the person.

- Base: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). 0.49B parameters, context 32,768. Under the assignment cap of 0.5B. Qwen2.5-1.5B is over the cap. The instruct checkpoint is the one that can follow "reply with the option number."
- Objective: next-token loss on the answer tokens only. Not DPO. The label is the person's wave-4 answer. There is no rejected answer, so there is no preference pair.
- Persona in the training prompt: `wave_split` text for waves 1–3, truncated to the 2,048-token budget from the front. `full_persona` summaries are not used. They can contain wave-4 answers. The run instructions are in [training/HANDOFF.md](training/HANDOFF.md).
- Split: 500 people to train, the other 1,558 to test, seed 0. Five hundred fits a free Colab run and is enough to estimate a gap against the population baseline. Test people never appear in training. The prompt still excludes the earlier answer to that item and the wave-4 answer.
- LoRA on the attention projections: rank 16, alpha 32, dropout 0.05. Learning rate 2e-4, 2 epochs, batch size 1, gradient accumulation 16, max length 2,048, warmup ratio 0.03, seed 0. At test time, temperature 0 and 16 new tokens.
- This arm is done when train-then-eval prints a score on the held-out people against the population baseline on those same people. Beating GPT-4.1-mini is not required.

One model covers every person. Do not drop a small demographic slice, and do not train a separate model per slice at the start. Score each slice against its own retest. A second model is worth trying only if that slice is large and the shared model falls well short of the slice's own ceiling.

# Evaluation

GPT-4.1-mini is a system we score. It is not the judge. Nothing in the pipeline asks a model to grade another model.

## Labels

Two columns. They are not averaged together.

- Assignment label: that person's wave-4 answer. This is the number the submission reports.
- Check column: that person's wave 1–3 answer on the same items. Used only to confirm the file join. It is not the submitted score.

The human ceiling is the retest on the same items the model was scored on: this person's earlier answer versus their wave-4 answer. The exploration ceiling, 82.4% family-weighted, is on our wider item set. On the paper's 88 questions the ceiling is recomputed. Their published 81.72% is that other average.

## Metric

`twin2k.scoring.agreement`. Binary items: exact match. Otherwise `1 - |a - b| / span`, from 0 to 1. Anchoring estimates with no finite range use deciles of the earlier answers, span 9. A reply that does not parse scores 0, and the failure rate is reported beside the accuracy. Those rows stay in the denominator.

## Aggregation

Mean inside a family, then the unweighted mean across families. That is the headline. Pricing is about 42% of items, so an item-weighted mean lets it dominate. Item-weighted accuracy is reported next to the headline, and pricing is always its own row. Every row includes the ratio to retest on the same items.

Sex, race, age, and education use the same function. A slice is an audit. It becomes a second model only under the rule in Methods.

For pricing, also report the share who say they would buy, by price, against the wave-4 share. That is the aggregate a price question is for. It does not replace the individual score.

## Protocol

Prompt arms have no training split. All 2,058 people are test cases. The persona is the input, not a training set of labels.

The fine-tune splits people, not questions. The same question is asked of everyone, so holding out questions would still show the model the popular answer. Holding out people is what keeps a test person out of training.

Before a run counts, the code checks all of these:

- The prompt is built from the `wave_split` persona, not from `full_persona`. Repeated questions there keep the wave-4 answer.
- The earlier answer to the question being asked is not in the prompt. That answer is only the ceiling.
- The wave-4 answer is labels only. It is stripped from any question text before the prompt is built.
- Train pids and test pids are disjoint.
- Each person is scored on the one variant they saw. The unseen variant is not filled in.

## Acceptance

On pricing, a persona system should beat the population baseline by a gap that is visible next to the retest. On the exploration set those two numbers are 55.9% and 83.9%. The comparison that counts is the baseline and the retest recomputed on the items actually scored. Report the ratio to retest. On families already within 2 points of the baseline, matching the baseline is the result. No system is required to pass the retest.

# Business

The job this file supports is a panel dry-run. Simulate how these 2,058 people answer a new price or attitude question, then check the aggregate against a fresh human wave before spending on a real test.

The served answer depends on the family. On the pricing block, full text is 66.9% against a population constant of 55.9% and a retest of 83.9%, so a person's own history is worth using. On most other families the full text persona scores below the constant. Those questions are served by the constant. The constant is the popular answer, not a second twin.

The pricing block is a stated grocery choice, yes or no at a shown price. It is not a receipt, and it is not a sales forecast. HANDY, SAVVY, and LIVELY need purchase logs, account timelines, or call transcripts. This file has none of those, so it does not run those products.

Do not use a score for credit, hiring, medicine, or any decision about a named person. The panel is US, English, and online. Ages 50–64 are high and Hispanic origin is low relative to Census adults. People disagree with their own earlier answer about 18% of the time, which no model is asked to beat. A new country, language, or retailer needs its own human wave. A schedule that retrains on this file does not create that wave.

# Maintenance

Four artifacts are versioned together: the persona rules, the scorer, the routing table, and, for a fine-tune, the person split. A reported number names all four. Scores from different versions are not averaged.

Persona rules say which answers may enter the prompt. On this dataset that is wave 1–3 answers to questions the predicted wave does not repeat. The earlier answer to the predicted item, and the wave-4 answer, stay out. A new wave changes the rule: questions repeated in the new wave move from persona to holdout, and the new wave becomes the label. The old wave-4 file is not silently reused as if it were the new label.

The scorer is `twin2k.scoring`, plus the decile edges for anchoring. Edges are quantiles of the earlier answers, so they are stored with the wave they came from. A new edge set is a new scorer version. Old scores are recomputed or marked stale.

The routing table is the current result, not a fixed product list. Pricing is served by the full-text persona. Other families are served by the population constant. A later wave can move a family. The move happens when that family's persona score beats the constant by a visible gap and still sits under that family's own retest. A demographic slice does not get its own route unless the slice is large and the shared system falls well short of that slice's own retest.

Each system is kept differently.

- Population constant. No weights. Rebuild the mode, or the median, from the earlier answers of the wave being served. Version it with that wave.
- Prompted GPT-4.1-mini. No training job. The pinned snapshot is `gpt-4.1-mini-2025-04-14`, temperature 0. The published full text is the system of record for pricing. The summary is not promoted in its place: it is 4 points lower overall and 4 points lower on pricing. A new snapshot is adopted only after the same people and items are rescored against the pinned one.
- Fine-tune. Store the base id `Qwen/Qwen2.5-0.5B-Instruct`, the LoRA adapter, seed 0, and the 500 / 1,558 person split. The held-out scores are `training/outputs/qwen-0.5b-lora/metrics.json`. The adapter weights stay out of git: [chi10969/twin2k-qwen2.5-0.5b-lora](https://huggingface.co/chi10969/twin2k-qwen2.5-0.5b-lora). Test pids never enter a later training set, including when those same people answer a new wave. A new wave is scored on the frozen test people before any training person is updated.

Monitoring uses the next human wave, not a clock. For each family, record the retest, the constant, and the served system's score. Drift is the served score moving away from that family's retest, or losing the gap over the constant that justified its route. On pricing, also record the purchase rate by within-product price quintile. Full text currently sits below the human rate in every quintile. Summary and demographics sit above it. A wave that flips that bias, or flattens the human slope, pulls the pricing dry-run until the curve is checked again.

The release gate is the fresh human wave. A simulated price question is not treated as a result until that wave agrees at the aggregate. Retraining without new human answers does not open the gate.

# Order

1. Task definition. Done.
2. Exploration: alignment, retest, variants, length, demographics. Done.
3. Score the published demographics-only, summary, and full-text GPT-4.1-mini files. Done. See [report/evaluation.md](report/evaluation.md). Family-weighted wave-4 accuracy is 76.8% for the population constant, 69.8% demographics, 69.6% summary, 73.5% full text, 82.4% retest. Full text wins on pricing (66.9% against 55.9% and a retest of 83.9%) and loses to the constant on most other families.
4. Retrieval only if a later model cannot take the full persona. The summary is 4 points below full text overall and 4 points below it on pricing.
5. Fine-tune on one A100 40GB with [training/HANDOFF.md](training/HANDOFF.md). Done. See [report/finetune.md](report/finetune.md). Family-weighted, the adapter scores 70.2% against the training-set constant at 76.8%. On pricing it scores 73.7% against 55.9%. Beating the published full text is not required.
