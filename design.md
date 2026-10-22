# Task

Predict how the same person answers the wave-4 questions, using only their wave 1–3 answers to questions wave 4 does not repeat. Compare the prediction with the wave-4 answer and with that person's own test-retest.

Use Hugging Face `wave_split`. Use `full_persona` only to read the block list. Repeated questions there keep the wave-4 answer.

# Exploration

Done. Tables and the Census comparison are in [report/exploration.md](report/exploration.md). Reproduce with `.venv/bin/python -m exploration.explore`.

- Family-weighted retest **82.4%**. Population baseline, no persona: **76.8%**. The paper's 81.72% is a different task average.
- Pricing: retest **83.9%**, baseline **55.9%**. Base-rate, disease, Linda, outcome bias, and sunk cost sit within 2 points of the baseline.
- Every wave 1–3 persona is about **24k** tokens (characters / 4): over 8k, under 32k. The summary is about **13k** characters. `full_persona` text is longer because repeated questions keep the wave-4 answer.
- All 2,058 people have one variant per randomization family.
- Sex matches Census adults (July 1 2024). Ages 50–64 are about 9 points high; 65+ is about 9 points low. Hispanic origin is 9.4% versus 18.5%. Race is a single choice, so the shares are not a Census race table.

Skipping word frequency, word clouds, and clustering. The wording is shared. Forward Flow is short (median 60 words).

# Representativeness and bias

Demographics are wave 1 only. Later waves are behavior.

Sex, race, age, and education are used two ways:

- Describe the panel against Census adults where the categories match.
- Slice the retest, and later the model score. A slice is an audit.

Do not use a score for credit, hiring, medicine, or any decision about a named person.

# Methods

One prediction. The arms change the persona or the model. The scorer stays `twin2k.scoring`.

## Approaches

Tried in this order. Each step has to beat the previous one on the same rows.

- Population constant. The mode or median of the earlier answers. No persona, no cost. It scores 76.8% and is within 2 points of the retest on five families. Every later arm is compared to it.
- Price cutoff, for pricing. Per product, the price that best splits yes from no in the earlier answers. Still no persona. It scores 70.8% on pricing against 55.9% for the constant, so a persona arm on pricing has to beat the price, not only the popular answer.
- Prompting with the persona. No training, and the dataset already publishes three persona versions. It does not learn how this panel uses a scale, and it falls under the constant on most families.
- Retrieval. Needed when the window is under the 24k-token persona, or to place related past answers next to the question: embed each past question and answer, keep the top k by similarity to the new stem, and choose k on training people. Pricing is the weak spot. The earlier pricing answers are not in the persona, and what predicts it (income, frugality, spending scales) does not look like a grocery price question.
- Supervised fine-tune (LoRA). Learns the answer format and the panel's scale use, and fits under 0.5B. A small model can learn the popular answer and ignore the persona. To check, shuffle personas across test people: if the score holds, the persona is unused.
- Preference tuning, later. DPO needs a rejected answer, and on ordinal or slider items a pair ignores how far off the answer is. RL with `agreement` as the reward (for example GRPO) fits the metric directly. Worth trying only after the fine-tune beats the constant outside pricing.

## Pipeline

1. Download the `wave_split` parquet (`exploration.explore.ensure_parquets`).
2. Parse `wave4_Q_wave1_3_A` and `wave4_Q_wave4_A`, and align items on block, question ID, and sub-item (`twin2k.scoring.items_from_blocks`). 197,580 rows, 2,058 people.
3. Give each item a span: option count, slider range, or deciles of the earlier answers for anchoring.
4. Build the prompt from `wave1_3_persona_text` and the wave-4 question. Both answers stay out.
5. Generate, take the first number, score with `agreement`, average within each family, then across families.

## Prompt system

The prompt settings match the official simulation config in [tianyipeng-lab/Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation): GPT-4.1-mini, temperature 0. The dataset publishes the completions. The assignment allows those files as the baseline, so the full sample does not need an API call.

- Model: GPT-4.1-mini, snapshot `gpt-4.1-mini-2025-04-14` in the published files. Temperature 0. One question per call. The completion is a survey code, so the generation cap is 16 tokens.
- System prompt: answer the new question as the person in the persona, stay consistent with their past answers, and follow the format instruction. User message: the persona, the question, the options, and a line asking for only the option position or the number.
- Persona: wave 1–3 answers to questions wave 4 does not repeat. The published files include three versions of that input: demographics only, a summary, and the full text. We score all three. We do not generate them again.

## Full text and the context window

The wave 1–3 persona is about 24k tokens. The comparison above uses the published completions in `LLM_simulation_results`, so it sends nothing. A new full-text generation needs a 32k window, or retrieval of the past answers that bear on the question. The reported table is the published files plus `twin2k.scoring`, with no API token.

## Comparison

Same people, same items, `twin2k.scoring`. The label we report is the wave-4 answer.

| Arm | Input | Answers |
| --- | --- | --- |
| Population constant | no persona | early-answer constant, already computed |
| Demographics only | demographics | published `Demographics Only - GPT4.1-mini` |
| Summary persona | summary | published `Persona Summary - GPT4.1-mini` |
| Full text persona | text persona | published `Text Persona - GPT4.1-mini` |

On pricing the constant sits far below the retest, so that is where a persona could help. A price cutoff closes about half that gap without one. On several classic experiments the constant is already as high.

The published files predict the wave 1–3 answer. The paper uses wave 4 as a ceiling. We score the same files on the wave-4 answer, and keep the wave 1–3 score as a join check. Toubia et al. 2025, Table 2, varies JSON, reasoning, temperature, Gemini, and a fine-tune of GPT-4.1-mini. Those arms sit close together. The result here is the wave-4 score against our baseline and our retest.

## Bonus fine-tune

Done. Scores and the next change are in [report/finetune.md](report/finetune.md). Family-weighted, the adapter scores 70.2% against the training-set constant at 76.8%. Pricing is 73.7% against 55.9%. Scoring in that run cut long prompts from the end, which removed the question on base-rate, outcome bias, and WTA/WTP. Rescored with the fix on 400 test people, those three match or trail the constant, and the family mean is about 76.6% against 76.9%. On base-rate and outcome bias the adapter gives everyone the same answer. The other defects and the fixes are listed in that report: free-text parsing, a persona cut to about 6%, and no interval. A full fine-tune waits until the prompt contains the person.

- Base: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). 0.49B parameters, context 32,768. Under the assignment cap of 0.5B. Qwen2.5-1.5B is over the cap. The instruct checkpoint is the one that can follow "reply with the option number."
- Objective: next-token loss on the answer tokens only. The label is the person's wave-4 answer.
- Persona in the training prompt: `wave_split` text for waves 1–3, truncated to the 2,048-token budget from the front. `full_persona` summaries are not used. They can contain wave-4 answers. The run instructions are in [training/HANDOFF.md](training/HANDOFF.md).
- Split: 500 people to train, the other 1,558 to test, seed 0. Five hundred fits a free Colab run and is enough to estimate a gap against the population baseline. Test people never appear in training. The prompt still excludes the earlier answer to that item and the wave-4 answer.
- LoRA on the attention projections: rank 16, alpha 32, dropout 0.05. Learning rate 2e-4, 2 epochs, batch size 1, gradient accumulation 16, max length 2,048, warmup ratio 0.03, seed 0. At test time, temperature 0 and 16 new tokens.

One model covers every person. Score each demographic slice against its own retest. A second model is worth a try only when the slice is large and the shared model falls well short of that slice's retest.

## Risks

- Leakage through `full_persona` or the earlier answer. Build prompts only from `wave_split`. The checks under Protocol fail the run.
- Pricing is 42% of items and swamps an item mean. Report the family mean as the headline, with pricing as its own row.
- The model collapses to the popular answer. Compare against the constant per family, and run the shuffled-persona test.
- Truncation and parsing. The first LoRA scoring cut the question off on three families. Keep truncation on the persona side, store raw replies, and report the unparsed rate.
- Panel bias: US, online, ages 50–64 high, Hispanic origin low. Audit by slice. Do not serve another population without a new human wave.
- Stated, not revealed, behavior. Scope is limited under Business.
- The API snapshot changes. Pin it, and rescore before switching.

# Evaluation

GPT-4.1-mini is scored like the other systems. No model grades another model.

## Labels

- Assignment label: that person's wave-4 answer. This is the number the submission reports.
- Check column: that person's wave 1–3 answer on the same items, to confirm the file join.

The human ceiling is the retest on the same items. The 82.4% family-weighted figure is our item set. Their published 81.72% is the average on the paper's 88 questions.

## Metric

`twin2k.scoring.agreement`. Binary items: exact match. Otherwise `1 - |a - b| / span`, from 0 to 1. Anchoring estimates with no finite range use deciles of the earlier answers, span 9. A reply that does not parse scores 0, and the failure rate is reported beside the accuracy. Those rows stay in the denominator.

## Aggregation

Mean inside a family, then the unweighted mean across families. That is the headline. Pricing is about 42% of items, so an item-weighted mean lets it dominate. Item-weighted accuracy is reported next to the headline, and pricing is always its own row. Every row includes the ratio to retest on the same items.

Sex, race, age, and education use the same function. A slice is an audit. It becomes a second model only under the rule in Methods.

For pricing, also report the share who say they would buy, by price, against the wave-4 share. That is the aggregate a price question is for. It does not replace the individual score.

## The leakage trap

`full_persona` holds all four waves for each person. On repeated questions it keeps the wave-4 answer, which is the label. Its text is 128,665 characters at the median, against 95,204 for the wave 1–3 persona. A prompt built from `full_persona` text, summary, or JSON reads the answers it is scored on.

Two smaller traps:

- The earlier answer to the same item sits in `wave4_Q_wave1_3_A`, not in the persona. Feeding it in reproduces the retest. It is the ceiling, not a feature.
- Constants, decile edges, and routes fit on the scored people leak the test distribution. The fine-tune fits all three on training people only. The constant in the exploration uses everyone and describes the population. It is not a model.

## Protocol

Prompt arms have no training split. All 2,058 people are test cases. The persona is the input.

The fine-tune holds out people. Everyone is asked the same questions, so a question holdout would still show the popular answer.

Before a run counts, the code checks all of these:

- The prompt is built from the `wave_split` persona, not from `full_persona`. Repeated questions there keep the wave-4 answer.
- The earlier answer to the question being asked is not in the prompt. That answer is only the ceiling.
- The wave-4 answer is labels only. It is stripped from any question text before the prompt is built.
- Train pids and test pids are disjoint.
- Each person is scored on the one variant they saw. The unseen variant is not filled in.

## Acceptance

A system is accepted when all of these hold:

1. Every check under Protocol passes, and unparsed replies are under 1%.
2. Pricing scores at least 2 points over the price cutoff. The bootstrap interval on a pricing gap is about ±0.6 points, so 2 points is not noise.
3. The routed system (persona where it wins, constant elsewhere) is not under the constant family-weighted, and no routed family is more than 2 points under its constant. Routes are picked on one half of the people and checked on the other.
4. On pricing, the purchase rate is within 10 points of the human wave-4 rate in every price quintile.
5. No demographic slice of at least 100 people has a ratio to retest more than 5 points under the overall ratio.

Results are in [report/acceptance.md](report/acceptance.md), from `.venv/bin/python -m evaluation.acceptance`.

- Published full text fails 2: 66.9% on pricing against 70.8% for the price cutoff. Its gain over the constant is the price in the stem.
- Routed and checked on the other half, the system scores 77.7% against 76.8%, and no routed family is 2 points under the constant. It passes 3. Pricing goes to the price cutoff, not to full text. Full text is picked only on two proportion-dominance families, by small margins.
- Full text fails 4: in the cheapest fifth it says yes 47.7% of the time, against 74.6% for people.
- The routed system passes 5. Every slice is within 2 points of the overall 94.1%.
- The LoRA scores 73.7% on held-out pricing against 70.0% for a cutoff fit on the training people. That passes 2 on pricing. Whether the gain comes from the person is not tested. Family-weighted, after the truncation rescore, it is about level with the constant (76.6% against 76.9%). See [report/finetune.md](report/finetune.md).

On these files no persona arm beats the baselines by a margin that matters. The routed system is almost entirely baselines.

# Business

The job is a panel dry-run: simulate how these 2,058 people answer a new question, and check the aggregate against a fresh human wave before paying for a real one.

- Retail or consumer-goods pricing teams. Run a new price grid through the twins, read the purchase curve by price, drop the points that are clearly dead, and field the rest to real people. This cuts the first round of panel cost.
- Survey and market research firms. Pilot a draft questionnaire: items with many unparsed or flat answers are unclear, and items with a low retest cannot be answered consistently by people either.
- Behavioral science labs. Use the twin's effect size as a prior for a power analysis. On the classic experiments here the constant already matches the retest, so the twin adds little beyond the published effect.

The served answer depends on the family. On pricing, a price cutoff (70.8%) beats published full text (66.9%). On most other families the constant beats full text. So today the panel is served by baselines, and the persona is not yet earning its place. The LoRA's 3.7-point pricing edge over the cutoff is the one sign that it might. A baseline is the popular answer, not a second twin.

Guardrails:

- Report aggregates. One twin's answer is not what that person would do.
- No credit, hiring, insurance, medicine, political targeting, or decisions about a named person.
- Pricing is a stated yes or no at a shown price, not a receipt or a sales forecast. Real purchases, churn, or support behavior need logs this dataset does not have.
- The panel is US, English, and online, with ages 50–64 high and Hispanic origin low against Census adults. A new country, language, or category needs its own human wave. Retraining on this file does not create one.
- The twin never replaces the human wave that gates a decision. People disagree with their own earlier answer about 18% of the time, and a twin will not be more consistent than the person.

# Maintenance

Four artifacts move together: the persona rules, the scorer, the routing table, and, for a fine-tune, the person split. A reported number names all four.

Persona rules: wave 1–3 answers to questions the predicted wave does not repeat. The earlier answer to the predicted item, and the wave-4 answer, stay out. A new wave moves repeated questions from persona to holdout and becomes the label.

The scorer is `twin2k.scoring`, plus the decile edges for anchoring. Edges are quantiles of the earlier answers, so they are stored with the wave they came from. A new edge set is a new scorer version. Old scores are recomputed or marked stale.

Routing follows the current result. Pricing uses the price cutoff. Other families use the population constant. A family moves when its persona score beats the constant by a visible gap and still sits under that family's retest. A demographic slice gets its own route only when the slice is large and the shared system falls well short of that slice's retest.

Each system is kept differently.

- Population constant. No weights. Rebuild the mode, or the median, from the earlier answers of the wave being served. Version it with that wave.
- Prompted GPT-4.1-mini. No training job. The pinned snapshot is `gpt-4.1-mini-2025-04-14`, temperature 0. Full text is the best published arm, but it serves no family today. The summary is 4 points lower overall and 4 points lower on pricing. A new snapshot is adopted only after the same people and items are rescored against the pinned one.
- Fine-tune. Store the base id `Qwen/Qwen2.5-0.5B-Instruct`, the LoRA adapter, seed 0, and the 500 / 1,558 person split. The held-out scores are `training/outputs/qwen-0.5b-lora/metrics.json`. The adapter weights stay out of git: [chi10969/twin2k-qwen2.5-0.5b-lora](https://huggingface.co/chi10969/twin2k-qwen2.5-0.5b-lora). Test pids never enter a later training set, including when those same people answer a new wave. A new wave is scored on the frozen test people before any training person is updated.

Monitoring uses the next human wave, not a clock. For each family, record the retest, the constant, and the served system's score. Drift is the served score moving away from that family's retest, or losing the gap over the constant that justified its route. On pricing, also record the purchase rate by within-product price quintile. Full text currently sits below the human rate in every quintile. Summary and demographics sit above it. A wave that flips that bias, or flattens the human slope, pulls the pricing dry-run until the curve is checked again.

The release gate is the fresh human wave. A simulated price question is not treated as a result until that wave agrees at the aggregate. Retraining without new human answers does not open the gate.

Governance. The data is CC BY 4.0 and keyed only by pid. Do not join it with outside data or try to re-identify anyone. Each release ships a model card with the panel description, the four artifact versions, family and slice scores, and the guardrails under Business. If the dataset owners withdraw a participant, drop that pid, rescore, and retrain any model that saw it. Each release records who approved it and which human wave it was checked against.

# Order

1. Task definition. Done.
2. Exploration: alignment, retest, variants, length, demographics. Done.
3. Score the published demographics-only, summary, and full-text GPT-4.1-mini files. Done. See [report/evaluation.md](report/evaluation.md). Family-weighted wave-4 accuracy is 76.8% for the population constant, 69.8% demographics, 69.6% summary, 73.5% full text, 82.4% retest. Full text beats the constant on pricing (66.9% against 55.9% and a retest of 83.9%), loses to a price cutoff there (70.8%), and loses to the constant on most other families. See [report/acceptance.md](report/acceptance.md).
4. Retrieval only if a later model cannot take the full persona. The summary is 4 points below full text overall and 4 points below it on pricing.
5. Fine-tune on one A100 40GB with [training/HANDOFF.md](training/HANDOFF.md). Done. See [report/finetune.md](report/finetune.md). Family-weighted about 76.6% after the truncation rescore, against 76.9% for the constant. Pricing 73.7% against 70.0% for a price cutoff. Defects and fixes are in that report.
