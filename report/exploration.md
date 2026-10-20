# Wave-split exploration

Source: `LLM-Digital-Twin/Twin-2K-500`, config `wave_split`, all 2,058 people.
Scorer: `twin2k.scoring`. Binary items are exact match. Other items are `1 - |a - b| / span`.
Wide free-text numbers (the anchoring estimates) are mapped to deciles of the earlier answers before that formula, as in Toubia et al. 2025.
The population baseline predicts every person with the median, or the mode for a binary item, of the wave 1–3 answers. It does not use a persona.

## Why these two columns

Wave 4 is one instrument stored twice. `wave4_Q_wave1_3_A` is that instrument filled with the earlier answers. `wave4_Q_wave4_A` is the same instrument filled in wave 4.
The elements in the docs (False consensus through Pricing) are blocks inside both JSON values. Aligning anything else would mix different questions.
Persona text is the input, not a third answer key: it keeps wave 1–3 questions that wave 4 does not ask again.

## Retest

Scored items: 197,580. Item-weighted retest 83.3%; population baseline 69.2%.
Family-weighted retest 82.4%; population baseline 76.8%. This is the headline. Pricing alone has 40 binary items and would otherwise dominate.
Pricing retest 83.9% against a population baseline of 55.9%.
Toubia et al. 2025 report a human test-retest of 81.72% on their task average. The family-weighted figure above is close. It is not the same statistic: this table scores every sub-item, then gives each family equal weight.

| Family | Items | People | Retest | Population baseline |
| --- | ---: | ---: | ---: | ---: |
| Absolute vs relative | 2,058 | 2,058 | 79.2% | 71.8% |
| Allais | 2,058 | 2,058 | 74.1% | 62.9% |
| Anchoring, African countries | 4,116 | 2,058 | 80.2% | 76.8% |
| Anchoring, redwood | 4,116 | 2,058 | 84.1% | 77.8% |
| Base-rate | 2,058 | 2,058 | 83.5% | 83.4% |
| Disease | 2,058 | 2,058 | 80.3% | 81.4% |
| False consensus | 20,580 | 2,058 | 87.4% | 76.0% |
| Less is more | 2,058 | 2,058 | 77.2% | 71.1% |
| Linda | 6,174 | 2,058 | 82.4% | 82.8% |
| Myside | 2,058 | 2,058 | 84.6% | 80.2% |
| Non-experimental heuristics | 41,160 | 2,058 | 84.5% | 80.9% |
| Outcome bias | 2,058 | 2,058 | 86.3% | 85.6% |
| Pricing | 82,320 | 2,058 | 83.9% | 55.9% |
| Probability matching | 16,476 | 2,058 | 74.4% | 76.6% |
| Proportion dominance 1 | 2,058 | 2,058 | 86.6% | 80.2% |
| Proportion dominance 2 | 2,058 | 2,058 | 85.6% | 78.2% |
| Sunk cost | 2,058 | 2,058 | 78.8% | 78.2% |
| WTA/WTP | 2,058 | 2,058 | 89.2% | 83.1% |

| Answer kind | Items | Retest | Population baseline |
| --- | ---: | ---: | ---: |
| matrix | 59,694 | 83.0% | 78.6% |
| mc | 109,074 | 83.3% | 60.9% |
| slider | 22,638 | 84.6% | 82.5% |
| te | 2,058 | 78.8% | 78.2% |
| te-decile | 4,116 | 81.3% | 73.4% |

Families where the population baseline is within 2 points of retest: Sunk cost, Disease, Linda, Base-rate, Outcome bias. A persona has little room to beat a constant there.
Families where retest leads the baseline by at least 15 points: Pricing. Individual answers carry more information than the popular answer.
Families where the baseline beats retest by more than 2 points: Probability matching. People are less stable than the popular answer.

## Variants

Each family is a randomization group. One person is assigned one block name. A missing variant is the design, not a value to impute.

| Family | People with exactly one variant | Distinct block names |
| --- | ---: | ---: |
| False consensus | 2,058 | 1 |
| Base-rate | 2,058 | 2 |
| Disease | 2,058 | 2 |
| Linda | 2,058 | 2 |
| Outcome bias | 2,058 | 2 |
| Anchoring, African countries | 2,058 | 2 |
| Anchoring, redwood | 2,058 | 2 |
| Less is more | 2,058 | 3 |
| Proportion dominance 1 | 2,058 | 3 |
| Proportion dominance 2 | 2,058 | 3 |
| Sunk cost | 2,058 | 2 |
| Absolute vs relative | 2,058 | 2 |
| WTA/WTP | 2,058 | 3 |
| Allais | 2,058 | 2 |
| Myside | 2,058 | 2 |
| Probability matching | 2,058 | 2 |
| Non-experimental heuristics | 2,058 | 1 |
| Pricing | 2,058 | 1 |

## Length

Token counts are characters / 4, a rough English estimate, not a model tokenizer.
Wave 1–3 persona: median 23801 tokens, 95th percentile 24064.
Share over 8k tokens: 100.0%. Share over 32k: 0.0%.
Median summary length: 12993 characters. Median `full_persona` text: 128665 characters, versus 95204 for the wave 1–3 persona.
`full_persona` text is longer because repeated questions keep the wave-4 answer. It is not a valid model input for this task.
Forward Flow, the open-text block in the persona, has median 60 words (95th percentile 62). Word clouds of the rest of the persona would mostly repeat the question wording.

## Who is in the sample

Demographics come from wave 1 only. Later waves are behavior, not a check that someone is an adult.
Age and sex benchmark: US Census Bureau, National Population Estimates, July 1 2024, nc-est2024-agesex-res.csv.
Race and Hispanic origin: US Census Bureau resident population by age, July 2025, nc-est2024-alldata-r-file12.csv. Hispanic is H_MALE + H_FEMALE in that file.
The survey asks race or origin as one choice, so White, Black, Asian, Hispanic, and Other sum to 100%. Census race and Hispanic origin overlap. White on the survey is not the same bucket as non-Hispanic White alone. Gaps below are directional.

| Group | Sample | Census adults | Sample minus Census |
| --- | ---: | ---: | ---: |
| Age 18-29 | 18.9% | 20.2% | -1.3 pp |
| Age 30-49 | 35.7% | 33.6% | +2.1 pp |
| Age 50-64 | 32.0% | 23.3% | +8.7 pp |
| Age 65+ | 13.5% | 22.9% | -9.5 pp |
| Sex Male | 49.3% | 49.0% | +0.2 pp |
| Sex Female | 50.7% | 51.0% | -0.2 pp |
| White vs White, non-Hispanic alone | 66.1% | 59.7% | +6.5 pp |
| Black vs Black alone | 12.2% | 13.2% | -1.0 pp |
| Asian vs Asian alone | 6.8% | 7.0% | -0.2 pp |
| Hispanic vs Hispanic, any race | 9.4% | 18.5% | -9.0 pp |

Sample composition with no Census twin (question wording differs, or Census does not ask):

**education**

- College graduate/some postgrad: 35.7%
- Some college, no degree: 22.7%
- Postgraduate: 15.2%
- High school graduate: 13.2%
- Associate's degree: 12.3%
- Less than high school: 0.8%

**income**

- $100,000 or more: 26.8%
- $30,000-$50,000: 20.0%
- $50,000-$75,000: 20.0%
- Less than $30,000: 17.8%
- $75,000-$100,000: 15.4%

**region**

- South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, NC, SC, GA, FL): 40.5%
- West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM): 24.0%
- Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH): 18.1%
- Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME): 16.6%
- Pacific (HI, AK): 0.8%

**party**

- Democrat: 41.2%
- Independent: 29.6%
- Republican: 26.2%
- Something else: 3.0%

## Retest by group

This is the human ceiling inside the group, not a model error. A later model table should use the same split.

| Slice | People | Retest |
| --- | ---: | ---: |
| sex: Female | 1,044 | 83.5% |
| sex: Male | 1,014 | 83.1% |
| race: Asian | 140 | 84.1% |
| race: Black | 251 | 80.7% |
| race: Hispanic | 194 | 81.8% |
| race: Other | 112 | 84.6% |
| race: White | 1,361 | 83.8% |
| age: 18-29 | 388 | 82.3% |
| age: 30-49 | 735 | 83.6% |
| age: 50-64 | 658 | 83.6% |
| age: 65+ | 277 | 83.1% |
| education: Associate's degree | 253 | 82.8% |
| education: College graduate/some postgrad | 735 | 83.8% |
| education: High school graduate | 272 | 82.5% |
| education: Less than high school | 17 | 83.1% |
| education: Postgraduate | 313 | 83.3% |
| education: Some college, no degree | 468 | 83.4% |

## What this changes

Use `wave_split` personas as model input. Score wave-4 answers with this function, on the same items, for every later system including the published GPT-4.1-mini runs.
Do not impute the variant a person was not shown. A summary is worth trying: the full persona is long, and `full_persona` text is the wrong text.
Pricing is the piece closest to a retail stated-preference test. Its retest, not the overall average, is the relevant ceiling for that use.
