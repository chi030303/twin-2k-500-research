"""Acceptance checks on the published GPT-4.1-mini answer sheets.

Price-aware pricing baseline, routes picked on one half of the people and
scored on the other, a bootstrap interval for full text minus the constant,
and the routed score by demographic slice.

Run from the repo root: `.venv/bin/python -m evaluation.acceptance`
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from evaluation.score_published import build_pairs
from exploration.explore import demographics
from training.finetune import split_people
from twin2k.scoring import agreement

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "acceptance.md"
LORA = ROOT / "training" / "outputs" / "qwen-0.5b-lora" / "metrics.json"
ARMS = ("demographics", "summary", "text")
KEY = ["block", "qid", "sub"]


def score(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            agreement(guess, truth, span, binary)
            for guess, truth, span, binary in zip(pred, frame["late"], frame["span"], frame["binary"])
        ]
    )


def constant(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Mode or median of the fit people's earlier answers, scored on the target people."""
    values = {}
    for key, group in fit.groupby(KEY, sort=False):
        early = group["early"].tolist()
        values[key] = Counter(early).most_common(1)[0][0] if group["binary"].iloc[0] else float(np.median(early))
    pred = np.array([values[key] for key in zip(*(target[column] for column in KEY))])
    return score(pred, target)


def price_cutoff(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Per product, the price cut that best splits the fit people's earlier answers."""
    pred = np.empty(len(target))
    for qid, group in fit.groupby("qid"):
        group = group.sort_values("price")
        price = group["price"].to_numpy()
        yes = (group["early"].to_numpy() == 1).astype(int)
        below = np.concatenate([[0], np.cumsum(yes)])
        n = len(yes)
        best, cut, low, high = -1, -np.inf, 1.0, 1.0
        for k in [0] + [i for i in range(1, n) if price[i] != price[i - 1]] + [n]:
            yes_low, yes_high = below[k], below[n] - below[k]
            correct = max(yes_low, k - yes_low) + max(yes_high, n - k - yes_high)
            if correct > best:
                best = correct
                cut = price[k] if k < n else np.inf
                low = 1.0 if 2 * yes_low >= k else 2.0
                high = 1.0 if 2 * yes_high >= n - k else 2.0
        rows = (target["qid"] == qid).to_numpy()
        pred[rows] = np.where(target.loc[rows, "price"].to_numpy() < cut, low, high)
    return score(pred, target)


def family_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("family")[column].mean().mean())


def routed(fit: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, str]]:
    """Pick the best arm per family on the fit people, then serve it to the target people."""
    target = target.copy()
    fit = fit.copy()
    fit["constant"] = constant(fit, fit)
    target["constant"] = constant(fit, target)
    fit_price, target_price = fit["family"] == "Pricing", target["family"] == "Pricing"
    fit["cutoff"], target["cutoff"] = fit["constant"], target["constant"]
    fit.loc[fit_price, "cutoff"] = price_cutoff(fit[fit_price], fit[fit_price])
    target.loc[target_price, "cutoff"] = price_cutoff(fit[fit_price], target[target_price])
    means = fit.groupby("family")[["constant", "cutoff", *ARMS]].mean()
    routes = means.idxmax(axis=1).to_dict()
    arm = target["family"].map(routes).to_numpy()
    served = np.empty(len(target))
    for name in set(arm):
        served[arm == name] = target[name].to_numpy()[arm == name]
    return served, routes


def bootstrap(pairs: pd.DataFrame, diff: np.ndarray, draws: int = 1000) -> tuple[tuple[float, float], tuple[float, float]]:
    frame = pairs[["pid", "family"]].assign(diff=diff)
    sums = frame.pivot_table(index="pid", columns="family", values="diff", aggfunc="sum", fill_value=0.0)
    counts = frame.pivot_table(index="pid", columns="family", values="diff", aggfunc="size", fill_value=0)
    rng = np.random.default_rng(0)
    people = len(sums)
    weights = rng.multinomial(people, np.full(people, 1 / people), size=draws)
    by_family = (weights @ sums.to_numpy()) / (weights @ counts.to_numpy())
    pricing = by_family[:, list(sums.columns).index("Pricing")]
    family = by_family.mean(axis=1)
    return tuple(np.percentile(family, [2.5, 97.5])), tuple(np.percentile(pricing, [2.5, 97.5]))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def main() -> None:
    pairs, _ = build_pairs()
    pids = sorted(pairs["pid"].unique())
    pricing = pairs[pairs["family"] == "Pricing"].copy()

    pricing["cutoff"] = price_cutoff(pricing, pricing)
    train, test = split_people(pids, 500, 0)
    held_fit = pricing[pricing["pid"].isin(train)]
    held = pricing[pricing["pid"].isin(test)].copy()
    held["constant"] = constant(held_fit, held)
    held["cutoff"] = price_cutoff(held_fit, held)
    lora = json.loads(LORA.read_text())["model"]["by_family"]["Pricing"] if LORA.exists() else None

    order = np.random.default_rng(0).permutation(pids)
    half_a, half_b = set(order[: len(order) // 2].tolist()), set(order[len(order) // 2 :].tolist())
    in_a = pairs["pid"].isin(half_a)
    pairs["routed"] = 0.0
    pairs["half_constant"] = 0.0
    route_sets = []
    for fit_mask in (in_a, ~in_a):
        served, routes = routed(pairs[fit_mask], pairs[~fit_mask])
        pairs.loc[~fit_mask, "routed"] = served
        pairs.loc[~fit_mask, "half_constant"] = constant(pairs[fit_mask], pairs[~fit_mask])
        route_sets.append(routes)
    persona_routes = [
        sorted(f"{family} ({arm})" for family, arm in routes.items() if arm != "constant") for routes in route_sets
    ]

    family_ci, pricing_ci = bootstrap(pairs, (pairs["text"] - pairs["base_rate"]).to_numpy())
    text_gap = family_mean(pairs, "text") - family_mean(pairs, "base_rate")
    pricing_gap = pricing["text"].mean() - pricing["base_rate"].mean()

    wave = ds.dataset(ROOT / "data" / "parquet" / "wave_split", format="parquet").to_table(
        columns=["pid", "wave1_3_persona_json"]
    )
    demos = demographics(wave.to_pandas())
    joined = pairs.merge(demos, on="pid")
    overall = family_mean(joined, "routed") / family_mean(joined, "retest")

    lines = [
        "# Acceptance checks",
        "",
        "Published GPT-4.1-mini answer sheets, scored as in [evaluation.md](evaluation.md). Criteria are in [design.md](../design.md#acceptance).",
        "",
        "## Pricing against a price-aware baseline",
        "",
        "The constant gives one answer per product and ignores the price. The cutoff baseline picks, per product, the price that best splits yes from no in the earlier answers. Below the cut it predicts the majority answer there, above it the majority there. No persona.",
        "",
        "| People | Fit on | Constant | Price cutoff | Persona system | Retest |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| All 2,058 | all 2,058 | {pct(pricing['base_rate'].mean())} | {pct(pricing['cutoff'].mean())} | full text {pct(pricing['text'].mean())} | {pct(pricing['retest'].mean())} |",
        f"| Held-out 1,558 | training 500 | {pct(held['constant'].mean())} | {pct(held['cutoff'].mean())} | LoRA {pct(lora) if lora is not None else '—'} | {pct(held['retest'].mean())} |",
        "",
        "## Routes picked on the other half",
        "",
        "People are split in half, seed 0. On one half, each family takes whichever of the constant, the price cutoff (pricing only), demographics, summary, or full text scores highest. The other half is scored with those routes, and with a constant fit on the first half. Then the halves swap.",
        "",
        f"Routed {pct(family_mean(pairs, 'routed'))} against a constant of {pct(family_mean(pairs, 'half_constant'))}, family-weighted.",
        "",
        f"Routes off the constant, first half: {', '.join(persona_routes[0]) or 'none'}. Second half: {', '.join(persona_routes[1]) or 'none'}.",
        "",
        "Routed families more than 2 points under the held-out constant:",
        "",
    ]
    by_family = pairs.groupby("family")[["routed", "half_constant"]].mean()
    short = by_family[by_family["routed"] < by_family["half_constant"] - 0.02]
    lines += [f"- {family}: {pct(row.routed)} against {pct(row.half_constant)}" for family, row in short.iterrows()] or ["- none"]
    lines += [
        "",
        "## Interval on full text minus the constant",
        "",
        "1,000 bootstrap draws over people, seed 0.",
        "",
        f"- Family-weighted: {pp(text_gap)}, 95% interval {pp(family_ci[0])} to {pp(family_ci[1])}.",
        f"- Pricing: {pp(pricing_gap)}, 95% interval {pp(pricing_ci[0])} to {pp(pricing_ci[1])}.",
        "",
        "## Routed system by slice",
        "",
        f"Ratio of the routed score to the retest inside each slice, family-weighted. Overall {overall:.1%}. Flagged when a slice of at least 100 people is more than 5 points under that.",
        "",
        "| Slice | People | Ratio to retest | Flag |",
        "| --- | ---: | ---: | --- |",
    ]
    for field in ("sex", "race", "age", "education"):
        for label, group in joined.groupby(field):
            ratio = family_mean(group, "routed") / family_mean(group, "retest")
            people = group["pid"].nunique()
            flag = "yes" if people >= 100 and ratio < overall - 0.05 else ""
            lines.append(f"| {field}: {label} | {people:,} | {ratio:.1%} | {flag} |")
    lines.append("")
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
