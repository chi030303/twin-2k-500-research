# Published GPT-4.1-mini

Scored with `twin2k.scoring` on every wave-4 item. The model answers are the dataset's precomputed completions. No API call.

The submitted number is accuracy against the wave-4 answer. The wave 1–3 column is the target those files were generated for. It is not averaged with the wave-4 number.

Family-weighted means give each family the same weight. Item-weighted means let pricing, about 42% of items, dominate. Toubia et al. 2025 report 71.72% for the text persona on a 17-task average. The figure below is this scorer on these items. It is not that average.

| System | Input | Family-weighted, wave 4 | Item-weighted, wave 4 | Ratio to retest | Family-weighted, wave 1–3 | Unparsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Retest | same person's earlier answer | 82.4% | 83.3% | 100% | — | — |
| Population constant | no persona | 76.8% | 69.2% | 93% | — | — |
| Demographics only | demographics | 69.8% | 66.8% | 85% | 69.2% | 0.0% |
| Summary persona | summary | 69.6% | 68.0% | 84% | 69.0% | 0.0% |
| Full text persona | text persona | 73.5% | 71.7% | 89% | 72.7% | 0.0% |

Ratio to retest uses the family-weighted wave-4 score. The wave 1–3 column is only for a model completion, scored against the earlier answer. Unparsed is the share of those completions that were not a number. Retest and the population constant are computed from the human answers, so both cells are marked —.

On the family-weighted average the full text persona, 73.5%, is below the population constant, 76.8%. The constant wins on most families. The persona wins on pricing. Pricing is large enough that the item-weighted average goes the other way: full text 71.7%, population constant 69.2%.

| Family | Items | Retest | Population | Demographics | Summary | Full text | Text minus population |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False consensus | 20,580 | 87.4% | 76.0% | 71.2% | 71.5% | 74.2% | -1.9 pp |
| Base-rate | 2,058 | 83.5% | 83.4% | 68.6% | 63.4% | 80.4% | -3.0 pp |
| Disease | 2,058 | 80.3% | 81.4% | 71.5% | 73.2% | 71.7% | -9.6 pp |
| Linda | 6,174 | 82.4% | 82.8% | 77.8% | 74.7% | 78.0% | -4.8 pp |
| Outcome bias | 2,058 | 86.3% | 85.6% | 85.9% | 85.9% | 86.0% | +0.4 pp |
| Anchoring, African countries | 4,116 | 80.2% | 76.8% | 56.5% | 57.2% | 70.3% | -6.5 pp |
| Anchoring, redwood | 4,116 | 84.1% | 77.8% | 73.2% | 73.0% | 72.2% | -5.7 pp |
| Less is more | 2,058 | 77.2% | 71.1% | 60.2% | 56.9% | 67.2% | -3.9 pp |
| Proportion dominance 1 | 2,058 | 86.6% | 80.2% | 78.9% | 78.5% | 80.6% | +0.4 pp |
| Proportion dominance 2 | 2,058 | 85.6% | 78.2% | 76.8% | 76.8% | 79.2% | +1.0 pp |
| Sunk cost | 2,058 | 78.8% | 78.2% | 74.5% | 70.6% | 68.8% | -9.4 pp |
| Absolute vs relative | 2,058 | 79.2% | 71.8% | 55.7% | 61.3% | 66.4% | -5.4 pp |
| WTA/WTP | 2,058 | 89.2% | 83.1% | 76.1% | 76.6% | 79.3% | -3.8 pp |
| Allais | 2,058 | 74.1% | 62.9% | 43.5% | 47.6% | 50.9% | -12.0 pp |
| Myside | 2,058 | 84.6% | 80.2% | 76.7% | 73.6% | 77.6% | -2.7 pp |
| Probability matching | 16,476 | 74.4% | 76.6% | 76.6% | 76.6% | 76.6% | +0.0 pp |
| Non-experimental heuristics | 41,160 | 84.5% | 80.9% | 73.5% | 72.1% | 76.2% | -4.7 pp |
| Pricing | 82,320 | 83.9% | 55.9% | 58.8% | 62.7% | 66.9% | +11.0 pp |

Pricing is the family the exploration flagged. Full text scores 66.9% against a population constant of 55.9% and a retest of 83.9%.

Purchase rate is the share of pricing answers coded yes. Quintiles are within each product, so a high quintile is a high price for that product, not a high price in dollars across products.

| Price quintile | Human wave 4 | Demographics | Summary | Full text |
| --- | ---: | ---: | ---: | ---: |
| 1 | 74.6% | 90.2% | 88.4% | 47.7% |
| 2 | 61.6% | 83.5% | 79.8% | 50.5% |
| 3 | 39.7% | 74.7% | 64.1% | 27.2% |
| 4 | 27.3% | 64.8% | 49.7% | 17.0% |
| 5 | 16.9% | 39.7% | 26.4% | 7.6% |

Human purchase rates fall from 74.6% in the cheapest fifth to 16.9% in the dearest. Full text sits below the human rate in every fifth. Demographics sits above the human rate in every fifth. The summary sits above the human rate in every fifth.
