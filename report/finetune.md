# Qwen2.5-0.5B LoRA

Scored with `twin2k.scoring` on the held-out people. Raw file: [`training/outputs/qwen-0.5b-lora/metrics.json`](../training/outputs/qwen-0.5b-lora/metrics.json). The 24-person smoke run is a separate directory. Adapter weights: [chi10969/twin2k-qwen2.5-0.5b-lora](https://huggingface.co/chi10969/twin2k-qwen2.5-0.5b-lora).

The constant here is fit on the 500 training people and scored on the 1,558. The 76.8% and 55.9% in [evaluation.md](evaluation.md) are fit on all 2,058. Rounded figures can match. The fits are separate.

Family-weighted, the adapter scores **70.2%**. The training-set constant on the same rows is **76.8%**. The retest is **82.3%**. Item-weighted, the adapter is **74.9%** and the constant **69.2%**. Pricing, the family the constant misses, is **73.7%** against **55.9%**. Outcome bias (26.0% against 85.6%), WTA/WTP (46.9% against 83.6%), and base-rate (61.3% against 83.4%) pull the family mean under the constant. Pricing is about 42% of items, so the item-weighted mean goes the other way.

**These three families were scored on a cut prompt.** Their prompts run past 2,048 tokens. Training drops the overflow from the front. Scoring let the tokenizer cut from the end, which removed the question, the reply instruction, and the assistant turn. On a 20-person check, every prompt in those three families was over the cap, as were 20 of 400 non-experimental items. No other family was. The table below is that run. The rescore after it fixes the cut on those three families.

The published full-text persona, on all 2,058 people, scores 73.5% family-weighted and 66.9% on pricing. This adapter scores 70.2% and 73.7% on the held-out 1,558. These are different people.

| System | Family-weighted | Item-weighted | Ratio to retest | Unparsed |
| --- | ---: | ---: | ---: | ---: |
| Retest | 82.3% | 83.2% | 100% | — |
| Training-set constant | 76.8% | 69.2% | 93% | — |
| Qwen2.5-0.5B LoRA | 70.2% | 74.9% | 85% | 0.8% |

Ratio to retest uses the family-weighted score. Unparsed is model replies with no number. Those rows score 0 and stay in the denominator. Retest and the constant are human answers, marked —. `metrics.json` copies that same `fail` onto `retest` and `baseline`: `score_rows` reads one `row["fail"]` field, written while scoring the model.

## Rescore of the three cut families

Same adapter, prompts cut from the front, 400 test people drawn with seed 0, 1,200 rows. Run on an Apple-silicon laptop (MPS, bf16), about 28 minutes:

```bash
python -m training.finetune --eval --adapter <adapter dir> --test-people 400 --families "Base-rate" "Outcome bias" "WTA/WTP"
```

Raw files: [`metrics_partial.json`](../training/outputs/qwen-0.5b-lora/metrics_partial.json) and [`replies.jsonl`](../training/outputs/qwen-0.5b-lora/replies.jsonl).

| Family | Retest | Constant | LoRA, cut from the end | LoRA, cut from the front |
| --- | ---: | ---: | ---: | ---: |
| Base-rate | 83.8% | 83.1% | 61.3% | 83.2% |
| Outcome bias | 87.2% | 86.2% | 26.0% | 86.2% |
| WTA/WTP | 88.6% | 84.2% | 46.9% | 79.7% |

The cut-from-the-end column is the full 1,558. The other columns are the 400. Unparsed replies are 0.

Swapping these three into the original table moves the family mean from 70.2% to about **76.6%**, against 76.9% for the constant on the same mix. The mix is not one set of people, so 76.6% is an estimate.

The replies show why the scores match the constant. On base-rate, every person gets the same answer to each variant: 30 for the 30-engineer version and 70 for the 70-engineer version. Outcome bias is 6 for everyone. Only WTA/WTP varies by person, and it scores under the constant. On these families the adapter has learned the popular answer, not the person.

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

One Hugging Face config, `LLM-Digital-Twin/Twin-2K-500` / `wave_split`. Columns: `pid`, `wave1_3_persona_text`, `wave4_Q_wave1_3_A`, `wave4_Q_wave4_A`.

`prepare` aligns the two wave-4 JSON columns. A row needs an earlier answer and a wave-4 answer on the same block, question, and sub-item, and the question has to be multiple choice, matrix, slider, or a text entry that parses as a number. Descriptive blocks and DB screens are dropped. That is 197,580 rows, 2,058 people.

Pids are sorted, then `numpy` `default_rng(0)` draws 500. The other 1,558 are the test. Overlap and coverage are checked. The 500 contribute 47,948 training rows. The 1,558 contribute 149,632 scored rows. A test pid stays out of training.

Each training row is one chat:

- System: answer the new question as the person in the profile, stay consistent with their past answers, and follow the format instruction.
- User: the persona, then `New Survey Question:` and the formatted item. Multiple choice and matrix items list options from 1. A slider states its min and max. A text item asks for the number.
- Assistant: that person's wave-4 choice. This is the only text with loss.

`wave4_Q_wave4_A` is the label. `wave4_Q_wave1_3_A` is the earlier answer, used for the retest and the constant. Neither is copied into the prompt.

The persona is `wave1_3_persona_text`, which already omits questions wave 4 repeats. It is cut to the first 1,792 tokens (2,048 minus 256). If the chat still exceeds 2,048, tokens drop from the front of the prompt and the answer tokens stay. Scoring now cuts the same way. A row whose answer would be truncated is a hard failure.

The assistant label is the raw wave-4 string. Deciles are a scoring step. For `QID164`, `QID166`, `QID168`, and `QID170`, the earlier answer, the wave-4 answer, and the model's first number are mapped with ten quantile edges from the training people's earlier answers. Text answers whose training range sits inside 0–20 use `hi - lo` as the span. Both rules use the 500 only.

The constant is one number per item, fit on the 500: the mode on a binary item, otherwise the median. Decoding is temperature 0 and 16 new tokens. `first_number` takes the first integer or decimal in the reply, after commas are stripped. With the question cut off, the model continues the stem, and stems like outcome bias contain numbers ("A 55-year-old", "1 in 1,000"). That would score as an answer. The original run did not save replies, so this is not confirmed.

One A100 40GB, `torch==2.5.1+cu121`, bf16. Two epochs, 5,992 steps, about 3.0 seconds each, last logged loss 0.40. Scoring the 149,632 rows took about 1.5 hours. The 500 / 2,048 / LoRA shape also fits a free Colab. The numbers above are this A100 run.

## Known defects

- Scoring cut long prompts from the end. Base-rate, outcome bias, and WTA/WTP lost the question, so their scores, and the 70.2% family mean, do not measure the adapter. Fixed in `training/finetune.py`, and rescored above on 400 people. `metrics.json` is still the original run, and the full 1,558 are not rescored.
- The original run did not save replies. Scoring now writes `replies.jsonl`.
- The parser takes the first number in free text. A reply that echoes a stem scores as an answer.
- On base-rate and outcome bias the adapter gives everyone the same answer. A fine-tune that only learns the mode cannot beat the constant.
- The persona is cut to 1,792 of about 27.5k Qwen tokens, about 6%. On three people checked, the cut ends near item 30 of the first wave-1 personality scale. The prompt holds demographics and part of one scale.
- Most of the pricing gain does not need the person. The constant is one value per product and ignores the price, which is in the stem. A per-product price cutoff fit on the training people scores 70.0% on the held-out 1,558 ([acceptance.md](acceptance.md)). The adapter's edge over that is 3.7 points, not 17.7. Whether those 3.7 come from the persona is not tested.
- One seed, one split, no interval. Small family gaps (Linda +0.1, Myside +0.4) are inside noise.

## What to change next

Unparsed replies are 0.8%. Pricing gained 17.7 points on the same 1,792-token cut.

### Fix the measurement first

- Rescore every family on all 1,558 with front truncation. Scoring only, about 1.5 hours on one A100.
- Score options instead of parsing text. For choice items, compare the log-probability of each option number and take the largest. No reply can fail to parse.
- Shuffle personas across test people on pricing. If the score stays near 73.7%, the 3.7 points come from the price and the item, not the person.
- Save per-row scores, and bootstrap over test people for an interval on each family gap. The published arms already have one in [acceptance.md](acceptance.md).

### Keep more of the persona

The wave 1–3 persona is about 27.5k Qwen tokens. `Qwen2.5-0.5B-Instruct` has a 32,768 context. This run keeps the first 1,792 tokens, then drops more from the front when the question does not fit in 2,048. Batch size is 1, so dynamic batching only cuts padding. A short question leaves part of the 256-token reserve unused. Giving that back stays inside 2,048, a few hundred tokens against a 27.5k persona.

Next input: raise the cap, with 8,192 as the probe on one A100 40GB, or retrieve the past answers that bear on the new question. Two epochs at 8,192 take several times the 5.1 hours this run spent training. A higher LoRA rank and more epochs wait until the prompt contains the person.

### Train where the person matters

On several classic experiments the constant is already near the retest, so those rows carry little individual signal. They share one loss with pricing, which is where the constant sits far below the retest and where this adapter gained 17.7 points. The next run still scores every family. Loss goes on the families where the constant and the retest diverge, pricing first.

### Full fine-tune, after the input changes

A full fine-tune of the 0.5B fits on one A100 40GB. The assignment cap is the base size. This adapter is rank 16 on `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The MLP and the embeddings stay frozen. The saved adapter is about 8.7MB.

At this 2,048 cap a full update still misses the cut tokens. Run it after the persona length changes, to see whether the frozen MLP is the limit. On this card the LoRA step was 3.0 seconds, 5,992 steps, about 5.1 hours, plus about 1.5 hours to score. A full update adds the base weight gradients. Plan on about 1.5 times the step time: about 8 to 9 hours of training and the same 1.5 hours of scoring, about 10 to 11 hours in total.
