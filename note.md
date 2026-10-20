# Dataset

Source: [persona_json structure](https://digital-twin-simulation-version2.readthedocs.io/en/latest/intro/index.html#structure-of-the-persona-json-file).

That page documents `full_persona`, not `wave_split`. One row is one person. `persona_json` is a list of blocks. A block belongs to one wave. Order can be fixed or randomized, and some blocks are shown only under an experimental condition. Wave 4 does this the most. The list below is every block that exists, not the blocks one person saw.

`full_persona` fields:

- `pid`
- `persona_text` — questions and answers as text
- `persona_summary` — short text summary
- `persona_json` — blocks, questions, answers

`wave_split` is the prediction split. Persona text there excludes questions that wave 4 asks again. Wave 4 itself is stored twice: `wave4_Q_wave1_3_A` (earlier answers) and `wave4_Q_wave4_A` (wave-4 answers). The elements below are the blocks inside those JSON values.

Element numbers follow the docs page.

## Wave 1 — elements 0–4

No randomization group.

- Element 0: Demographics
- Element 1: Personality
- Element 2: Cognitive tests
- Element 3: Economic preferences - intro
- Element 4: Economic preferences

## Wave 2 — elements 5–9

- Element 5: Personality
- Element 6: Cognitive tests
- Element 7: Forward Flow
- Element 8: Economic preferences - intro
- Element 9: Economic preferences

## Wave 3 — elements 10–12

- Element 10: Personality
- Element 11: Cognitive tests
- Element 12: Economic preferences

## Wave 4 — elements 13–30

Retest of the heuristics-and-biases experiments and the pricing study. A randomization group is mutually exclusive: one person sees one variant. The paper assigns the same condition as the earlier wave.

- Element 13: False consensus
- Element 14: Base-rate 30 engineers, or Base-rate 70 engineers
- Element 15: Disease - gain, or Disease-loss
- Element 16: Linda - no conjunction, or Linda-conjunction
- Element 17: Outcome bias - failure, or Outcome bias - success
- Element 18: Anchoring - African countries high, or low
- Element 19: Anchoring - redwood high, or low
- Element 20: Less is More Gamble A, B, or C
- Element 21: Proportion dominance 1A, 1B, or 1C
- Element 22: Proportion dominance 2A, 2B, or 2C
- Element 23: Sunk cost - no, or yes
- Element 24: Absolute vs. relative - calculator, or jacket
- Element 25: WTA/WTP Thaler - WTP noncertainty, WTA certainty, or WTP certainty
- Element 26: Allais Form 1, or Form 2
- Element 27: Myside Ford, or Myside German
- Element 28: Probability matching vs. maximizing - Problem 1, or Problem 2
- Element 29: Non-experimental heuristics and biases
- Element 30: Product Preferences - Pricing

On `full_persona`, repeated questions keep the wave-4 answer. That JSON is not a wave 1–3 persona.
