# Qwen2.5-0.5B LoRA

Scored with `twin2k.scoring` on the held-out people. The raw file is `training/outputs/qwen-0.5b-lora/metrics.json`. The 24-person smoke run is a separate directory and is not in this table.

The constant here is fit on the 500 training people and scored on the 1,558. The 76.8% and 55.9% in [evaluation.md](evaluation.md) are fit on all 2,058. Several of those rounded figures match. The fits are separate.

Family-weighted, the adapter scores **70.2%**. The training-set constant scores **76.8%** on the same rows. The retest on those rows is **82.3%**. Item-weighted, the adapter scores **74.9%** and the constant **69.2%**. Pricing is the family the constant does not explain: **73.7%** against **55.9%**. Outcome bias (26.0% against 85.6%) and WTA/WTP (46.9% against 83.6%) pull the family mean under the constant. Pricing is about 42% of items, so the item-weighted mean goes the other way.

The published full-text persona is context only. On all 2,058 people it scores 73.5% family-weighted and 66.9% on pricing. This adapter scores 70.2% and 73.7% on the held-out 1,558.

| System | Family-weighted | Item-weighted | Ratio to retest | Unparsed |
| --- | ---: | ---: | ---: | ---: |
| Retest | 82.3% | 83.2% | 100% | — |
| Training-set constant | 76.8% | 69.2% | 93% | — |
| Qwen2.5-0.5B LoRA | 70.2% | 74.9% | 85% | 0.8% |

Ratio to retest uses the family-weighted score. Unparsed is the share of model replies that were not a number. Those rows stay in the denominator and score 0. Retest and the constant are human answers, so both cells are marked —.

| Family | Retest | Constant | LoRA | LoRA minus constant |
| --- | ---: | ---: | ---: | ---: |
| False consensus | 87.3% | 76.1% | 78.1% | +2.0 pp |
| Base-rate | 83.7% | 83.4% | 61.3% | -22.1 pp |
| Disease | 80.4% | 81.7% | 80.3% | -1.4 pp |
| Linda | 82.2% | 82.7% | 82.7% | +0.1 pp |
| Outcome bias | 86.5% | 85.6% | 26.0% | -59.6 pp |
| Anchoring, African countries | 79.9% | 75.6% | 75.4% | -0.2 pp |
| Anchoring, redwood | 83.6% | 78.2% | 69.6% | -8.7 pp |
| Less is more | 77.7% | 71.5% | 64.4% | -7.2 pp |
| Proportion dominance 1 | 87.4% | 80.3% | 80.2% | -0.2 pp |
| Proportion dominance 2 | 85.8% | 78.0% | 76.0% | -2.1 pp |
| Sunk cost | 79.2% | 77.7% | 77.7% | 0.0 pp |
| Absolute vs relative | 78.5% | 71.5% | 71.5% | 0.0 pp |
| WTA/WTP | 89.1% | 83.6% | 46.9% | -36.7 pp |
| Allais | 73.4% | 63.2% | 62.9% | -0.3 pp |
| Myside | 84.9% | 80.3% | 80.7% | +0.4 pp |
| Probability matching | 74.3% | 76.6% | 76.6% | 0.0 pp |
| Non-experimental heuristics | 84.6% | 80.9% | 79.4% | -1.5 pp |
| Pricing | 83.7% | 55.9% | 73.7% | +17.7 pp |

Absolute vs relative, sunk cost, and probability matching tie the constant. The constant wins on the other classic experiments. The adapter's gain is pricing, plus small edges on false consensus, Linda, and myside.

## Training rows

One Hugging Face config, `LLM-Digital-Twin/Twin-2K-500` / `wave_split`. Columns read: `pid`, `wave1_3_persona_text`, `wave4_Q_wave1_3_A`, `wave4_Q_wave4_A`. `full_persona` is not loaded.

`prepare` aligns the two wave-4 JSON columns. A row exists when the same block, question, and sub-item has an earlier answer and a wave-4 answer, and the question is one of multiple choice, matrix, slider, or a text entry that parses as a number. Descriptive blocks and DB screens are dropped. That alignment produces 197,580 rows across 2,058 people.

The split is by person. Pids are sorted, then `numpy` `default_rng(0)` draws 500 without replacement. The other 1,558 are the test. The sets are checked for overlap and for full coverage. The 500 contribute 47,948 training rows. The 1,558 contribute 149,632 scored rows. A test pid never enters training.

Each training row is one chat:

- System: answer the new question as the person in the profile, stay consistent with their past answers, and follow the format instruction.
- User: the persona, then `New Survey Question:` and the formatted item. Multiple choice and matrix items list options from 1 and ask for the option number. A slider states its min and max. A text item asks for the number.
- Assistant: that person's wave-4 choice, as the option number or the number. This is the only text with loss.

`wave4_Q_wave4_A` is the label. It is not copied into the prompt. `wave4_Q_wave1_3_A` is the earlier answer. It is used for the retest and the constant, and it is not copied into the prompt either.

The persona is `wave1_3_persona_text`, which already omits questions wave 4 repeats. It is cut to the first 1,792 tokens (2,048 minus 256). If the chat still exceeds 2,048 tokens, tokens are dropped from the front of the prompt. The answer tokens are kept. A row whose answer would be truncated is a hard failure.

Wide text answers, `QID164`, `QID166`, `QID168`, and `QID170`, are mapped to deciles before training and scoring. The ten bin edges are quantiles of the training people's earlier answers. Text answers whose training range sits inside 0–20 keep that range as the span. Both rules use the 500 only.

The constant is one number per item, fit on the 500. Binary items take the mode of the earlier answers. Other items take the median. It is then scored on the test people's wave-4 answers. Decoding for the adapter is temperature 0 and 16 new tokens. The first number in the reply is the prediction.

The run used one A100 40GB, `torch==2.5.1+cu121`, and bf16. Two epochs, 5,992 steps, about 3.0 seconds each, last logged loss 0.40. Scoring the 149,632 rows took about 1.5 hours.

The 500-person split, the 2,048-token cap, and LoRA are the shape that fits a free Colab. This file is not that Colab run. There is no Colab log.

## What to change next

The family mean is under the constant because the reply is the wrong number. Unparsed replies are 0.8%, and the last logged loss is 0.40. Another epoch of this recipe is not the next run. Three changes, in this order.

### Keep more of the persona

The wave 1–3 persona is about 24k tokens. `Qwen2.5-0.5B-Instruct` has a 32,768 context. This run keeps the first 1,792 tokens of every persona, then drops more from the front when the question does not fit in 2,048. Training uses batch size 1, so each row is already its own length. Dynamic batching, length bucketing, and continuous batching only cut padding. They do not put the dropped tokens back.

A short question does not receive the unused part of the 256-token reserve. Giving that leftover back still stays inside 2,048. That is a few hundred tokens against a 24k persona.

The next input is one of two. Raise the cap, with 8,192 as the probe on one A100 40GB. Or retrieve the past answers that bear on the new question. Attention cost grows with length. Two epochs at 8,192 take several times the 5.1 hours this 2,048 run spent training. A higher LoRA rank and more epochs wait until the prompt contains the person.

### Train where the person matters

On several classic experiments the constant is already near the retest, so those rows have little individual signal. They share one loss with pricing. Pricing is where the constant sits far below the retest, and it is where this adapter gained 17.7 points. The next training set still scores every family. Loss goes on the families where the constant and the retest diverge. Pricing is the first of those.

### Full fine-tune, after the input changes

A full fine-tune of the 0.5B fits on one A100 40GB. The assignment cap is the base size. It does not require LoRA. This adapter is rank 16 on `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The MLP and the embeddings stay frozen. The saved adapter is about 8.7MB.

At this same 2,048 cap, a full update still cannot see the tokens that were cut. Run it after the persona budget changes, as a check on whether the frozen MLP limits the answer. On this card the LoRA step took 3.0 seconds, 5,992 steps, about 5.1 hours, plus about 1.5 hours to score. A full update adds the base weight gradients. Plan on about 1.5 times the step time: about 8 to 9 hours of training and the same 1.5 hours of scoring, about 10 to 11 hours in total.
