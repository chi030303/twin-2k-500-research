# Acceptance checks

Published GPT-4.1-mini answer sheets, scored as in [evaluation.md](evaluation.md). Criteria are in [design.md](../design.md#acceptance).

## Pricing against a price-aware baseline

The constant gives one answer per product and ignores the price. The cutoff baseline picks, per product, the price that best splits yes from no in the earlier answers. Below the cut it predicts the majority answer there, above it the majority there. No persona.

| People | Fit on | Constant | Price cutoff | Persona system | Retest |
| --- | --- | ---: | ---: | ---: | ---: |
| All 2,058 | all 2,058 | 55.9% | 70.8% | full text 66.9% | 83.9% |
| Held-out 1,558 | training 500 | 55.9% | 70.0% | LoRA 73.7% | 83.7% |

## Routes picked on the other half

People are split in half, seed 0. On one half, each family takes whichever of the constant, the price cutoff (pricing only), demographics, summary, or full text scores highest. The other half is scored with those routes, and with a constant fit on the first half. Then the halves swap.

Routed 77.7% against a constant of 76.8%, family-weighted.

Routes off the constant, first half: Pricing (cutoff), Proportion dominance 1 (text), Proportion dominance 2 (text). Second half: Outcome bias (text), Pricing (cutoff), Probability matching (demographics), Proportion dominance 1 (text), Proportion dominance 2 (text).

Routed families more than 2 points under the held-out constant:

- none

## Interval on full text minus the constant

1,000 bootstrap draws over people, seed 0.

- Family-weighted: -3.4 pp, 95% interval -3.6 pp to -3.1 pp.
- Pricing: +11.0 pp, 95% interval +10.4 pp to +11.6 pp.

## Routed system by slice

Ratio of the routed score to the retest inside each slice, family-weighted. Overall 94.4%. Flagged when a slice of at least 100 people is more than 5 points under that.

| Slice | People | Ratio to retest | Flag |
| --- | ---: | ---: | --- |
| sex: Female | 1,044 | 94.8% |  |
| sex: Male | 1,014 | 93.9% |  |
| race: Asian | 140 | 94.1% |  |
| race: Black | 251 | 94.7% |  |
| race: Hispanic | 194 | 94.7% |  |
| race: Other | 112 | 93.8% |  |
| race: White | 1,361 | 94.3% |  |
| age: 18-29 | 388 | 93.8% |  |
| age: 30-49 | 735 | 94.2% |  |
| age: 50-64 | 658 | 94.6% |  |
| age: 65+ | 277 | 95.3% |  |
| education: Associate's degree | 253 | 94.5% |  |
| education: College graduate/some postgrad | 735 | 94.4% |  |
| education: High school graduate | 272 | 93.3% |  |
| education: Less than high school | 17 | 96.3% |  |
| education: Postgraduate | 313 | 94.8% |  |
| education: Some college, no degree | 468 | 94.5% |  |
