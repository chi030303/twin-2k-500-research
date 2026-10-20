# dataset

来源：[Introduction — persona_json 结构](https://digital-twin-simulation-version2.readthedocs.io/en/latest/intro/index.html#structure-of-the-persona-json-file)

这一页描述的是 Hugging Face 上 `full_persona` 的字段，不是 `wave_split`。每个被试一条记录，JSON 按 block 组织；block 属于某一波。题目和 block 的顺序可以固定，也可以随机。有的 block 或题目会按实验条件随机抽中。文档说这种随机在第四波的 JSON 里尤其明显。每个人实际看到的题目，可以和下面的完整清单不一样。

`full_persona` 的四个字段：

- `pid`：被试 ID
- `persona_text`：问卷全文加回答
- `persona_summary`：由问卷内容生成的文字摘要
- `persona_json`：结构化的 block、题目和回答。做按题、按波的分析用这个

下面的 Element 编号是文档页上的顺序，不是 wave 内部的题号。Wave 4 里很多 Element 是一组互斥变体，一个人只会看到其中一种。

## Wave 1 — Element 0–4

固定 block，这一波没有随机分组。

- Element 0: Demographics
- Element 1: Personality
- Element 2: Cognitive tests
- Element 3: Economic preferences - intro
- Element 4: Economic preferences

## Wave 2 — Element 5–9

- Element 5: Personality
- Element 6: Cognitive tests
- Element 7: Forward Flow
- Element 8: Economic preferences - intro
- Element 9: Economic preferences

## Wave 3 — Element 10–12

- Element 10: Personality
- Element 11: Cognitive tests
- Element 12: Economic preferences

## Wave 4 — Element 13–30

这一波是重测。heuristics and biases 实验，以及 wave 3 的定价题，会在这里再出现一次。随机组里的变体互斥，同一个人只进入一种。论文里被试在 wave 4 被分到和先前相同的实验条件。

- Element 13: False consensus
- Element 14: 二选一
  - Base-rate 30 engineers
  - Base-rate 70 engineers
- Element 15: 二选一
  - Disease - gain
  - Disease-loss
- Element 16: 二选一
  - Linda - no conjunction
  - Linda-conjunction
- Element 17: 二选一
  - Outcome bias - failure
  - Outcome bias - success
- Element 18: 二选一
  - Anchoring - African countries high
  - Anchoring - African countries low
- Element 19: 二选一
  - Anchoring - redwood high
  - Anchoring - redwood low
- Element 20: 三选一
  - Less is More Gamble A
  - Less is More Gamble B
  - Less is More Gamble C
- Element 21: 三选一
  - Proportion dominance 1A
  - Proportion dominance 1B
  - Proportion dominance 1C
- Element 22: 三选一
  - Proportion dominance 2A
  - Proportion dominance 2B
  - Proportion dominance 2C
- Element 23: 二选一
  - Sunk cost - no
  - Sunk cost - yes
- Element 24: 二选一
  - Absolute vs. relative - calculator
  - Absolute vs. relative - jacket
- Element 25: 三选一
  - WTA/WTP Thaler - WTP noncertainty
  - WTA/WTP Thaler problem - WTA certainty
  - WTA/WTP Thaler problem - WTP certainty
- Element 26: 二选一
  - Allais Form 1
  - Allais Form 2
- Element 27: 二选一
  - Myside Ford
  - Myside German
- Element 28: 二选一
  - Probability matching vs. maximizing - Problem 1
  - Probability matching vs. maximizing - Problem 2
- Element 29: Non-experimental heuristics and biases
- Element 30: Product Preferences - Pricing

文档的完整清单里，heuristics 和定价只挂在 Wave 4 名下。论文说这些题在前几波已经问过，wave 4 是再问一遍。`full_persona` 对两边都出现过的题保留的是 wave 4 的回答，所以不能把这份 JSON 当成「只有 wave 1–3」的 persona。
