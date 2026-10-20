# Twin-2K-500 behavior model

用一个人在 wave 1–3 的回答预测他在 wave 4 的回答，并与这个人自己的重测上限比较。

任务定义、探索范围、方法阶梯和评估协议在 [design.md](design.md)。问卷 block 目录在 [note.md](note.md)。

## 当前进度

- 已确定使用 `wave_split`，`full_persona` 只用于认识题目结构。
- 已确定评估用固定的准确率函数。GPT-4.1-mini 的预计算结果是基线，不是打分器。
- 方法按四级比较：弱基线、prompt、检索、小于 0.5B 的微调。前三级是方案，第四级是 bonus。

## 接下来

1. 探索 notebook：重测上限、互斥变体造成的缺失、persona 长度、wave 1 人口学与美国成年人基准的对照。
2. 在同一批人上算出随机基线、人口学基线和官方 GPT-4.1-mini 基线。

## 数据

[LLM-Digital-Twin/Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)（CC BY 4.0）。官方仿真代码：[tianyipeng-lab/Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation)。
