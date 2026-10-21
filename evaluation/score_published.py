"""Score the published GPT-4.1-mini answer sheets.

The sheets are precomputed completions in the Twin-2K-500 dataset. This
script does not call a model. Labels are each person's wave-4 answer. The
wave 1–3 answer is reported separately, because that is the target the
published files were generated for.

Run from the repo root: `.venv/bin/python -m evaluation.score_published`
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from exploration.explore import (
    FAMILIES,
    add_population_baseline,
    assign_text_spans,
    ensure_parquets,
    family_of,
)
from twin2k.scoring import agreement, items_from_blocks

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "data" / "sim"
OUT = ROOT / "evaluation" / "outputs"
REPORT = ROOT / "report" / "evaluation.md"
HF = (
    "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/"
    "LLM_simulation_results/llm_simulations_all_specifications/"
)
META = {
    "TWIN_ID",
    "StartDate",
    "EndDate",
    "Status",
    "Progress",
    "Duration (in seconds)",
    "Finished",
    "RecordedDate",
}
ARMS = (
    ("demographics", "Demographics Only - GPT4.1-mini"),
    ("summary", "Persona Summary - GPT4.1-mini"),
    ("text", "Text Persona - GPT4.1-mini"),
)
PRICE = re.compile(r"priced at:\s*\$([0-9]+(?:\.[0-9]+)?)")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"download {dest.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "twin2k-evaluation"})
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())


def sheet_url(folder: str, name: str) -> str:
    rel = f"{folder}/csv_comparison/csv_formatted/{name}"
    return HF + urllib.parse.quote(rel)


def ensure_sheets() -> None:
    human_folder = ARMS[-1][1]
    download(sheet_url(human_folder, "responses_wave4_formatted.csv"), SIM / "wave4.csv")
    for key, folder in ARMS:
        download(sheet_url(folder, "responses_llm_imputed_formatted.csv"), SIM / f"{key}_llm.csv")


def load_people(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    frame = frame.loc[pd.to_numeric(frame["TWIN_ID"], errors="coerce").notna()].copy()
    frame["pid"] = frame["TWIN_ID"].astype(int)
    return frame


def human_pairs(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for record in frame.itertuples(index=False):
        pid = int(record.pid)
        early_blocks = json.loads(record.wave4_Q_wave1_3_A)
        late_blocks = json.loads(record.wave4_Q_wave4_A)
        early = {(item.block, item.qid, item.sub): item for item in items_from_blocks(early_blocks)}
        prices: dict[str, float] = {}
        for block in late_blocks:
            for question in block.get("Questions") or []:
                match = PRICE.search(question.get("QuestionText") or "")
                if match:
                    prices[str(question.get("QuestionID") or "")] = float(match.group(1))
        for item in items_from_blocks(late_blocks):
            previous = early.get((item.block, item.qid, item.sub))
            if previous is None:
                continue
            rows.append(
                {
                    "pid": pid,
                    "block": item.block,
                    "family": family_of(item.block),
                    "qid": item.qid,
                    "sub": item.sub,
                    "kind": item.kind,
                    "early": previous.value,
                    "late": item.value,
                    "span": item.span if item.span is not None else previous.span,
                    "binary": item.binary,
                    "price": prices.get(item.qid),
                }
            )
    return rows


def column_key(rows: list[dict], wave4: pd.DataFrame) -> dict[str, tuple[str, str, int]]:
    grouped: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        grouped[(row["block"], row["qid"], row["sub"])][row["pid"]] = round(row["late"], 6)
    signatures = {tuple(sorted(values.items())): key for key, values in grouped.items()}
    if len(signatures) != len(grouped):
        raise RuntimeError("two items share the same wave-4 answers")
    matched: dict[str, tuple[str, str, int]] = {}
    for column in wave4.columns:
        if column in META or column == "pid":
            continue
        values: dict[int, float] = {}
        text = False
        for pid, raw in zip(wave4["pid"], wave4[column]):
            if raw is None or str(raw).strip() in {"", "nan"}:
                continue
            try:
                values[int(pid)] = round(float(raw), 6)
            except ValueError:
                text = True
                break
        if text or not values:
            continue
        key = signatures.get(tuple(sorted(values.items())))
        if key is None:
            raise RuntimeError(f"column {column} has answers that are not in the wave-4 JSON")
        matched[column] = key
    if len(matched) != len(grouped):
        raise RuntimeError(f"matched {len(matched)} columns, JSON has {len(grouped)} items")
    return matched


def decile_edges(rows: list[dict]) -> dict[str, np.ndarray]:
    """Same edges as exploration.assign_text_spans, kept for model answers."""
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "te":
            by_qid[row["qid"]].append(row)
    edges: dict[str, np.ndarray] = {}
    for qid, group in by_qid.items():
        values = [row["early"] for row in group] + [row["late"] for row in group]
        lo, hi = min(values), max(values)
        if 0 <= lo and hi <= 20:
            continue
        reference = np.array([row["early"] for row in group], dtype=float)
        cut = np.quantile(reference, np.linspace(0, 1, 11))
        for index in range(1, len(cut)):
            if cut[index] <= cut[index - 1]:
                cut[index] = cut[index - 1] + 1e-6
        edges[qid] = cut
    return edges


def apply_decile(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    bins = np.searchsorted(edges, values[finite], side="right") - 1
    out[finite] = np.clip(bins, 0, 9)
    return out


def score_column(pred: np.ndarray, label: np.ndarray, span: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(pred))
    failed = np.zeros(len(pred), dtype=bool)
    for index, (guess, truth, width, flag) in enumerate(zip(pred, label, span, binary)):
        if not np.isfinite(guess):
            scores[index] = 0.0
            failed[index] = True
            continue
        value = agreement(float(guess), float(truth), float(width), bool(flag))
        if value is None:
            scores[index] = 0.0
            failed[index] = True
            continue
        scores[index] = value
    return scores, failed


def lookup(sheet: pd.DataFrame, pids: np.ndarray, columns: list[str]) -> np.ndarray:
    """Pairwise cells. DataFrame.loc on two lists returns a block, not pairs."""
    indexed = sheet.set_index("pid")
    row_of = {pid: index for index, pid in enumerate(indexed.index)}
    col_of = {name: index for index, name in enumerate(indexed.columns)}
    rows = np.fromiter((row_of[int(pid)] for pid in pids), dtype=int, count=len(pids))
    cols = np.fromiter((col_of[name] for name in columns), dtype=int, count=len(columns))
    return indexed.to_numpy()[rows, cols]


def attach_arm(pairs: pd.DataFrame, column_to_key: dict[str, tuple], edges: dict[str, np.ndarray], name: str) -> None:
    sheet = load_people(SIM / f"{name}_llm.csv")
    key_to_column = {key: column for column, key in column_to_key.items()}
    columns = [key_to_column[(row.block, row.qid, row.sub)] for row in pairs.itertuples(index=False)]
    raw_pred = pd.to_numeric(lookup(sheet, pairs["pid"].to_numpy(), columns), errors="coerce")
    pred = raw_pred.astype(float)
    for qid, index in pairs.groupby("qid").groups.items():
        if qid not in edges:
            continue
        positions = pairs.index.get_indexer(index)
        pred[positions] = apply_decile(pred[positions], edges[qid])
    wave4, failed = score_column(
        pred, pairs["late"].to_numpy(), pairs["span"].to_numpy(), pairs["binary"].to_numpy()
    )
    early, _ = score_column(
        pred, pairs["early"].to_numpy(), pairs["span"].to_numpy(), pairs["binary"].to_numpy()
    )
    pairs[name] = wave4
    pairs[f"{name}_early"] = early
    pairs[f"{name}_fail"] = failed


def headline(pairs: pd.DataFrame, column: str) -> tuple[float, float]:
    by_family = pairs.groupby("family")[column].mean()
    return float(by_family.mean()), float(pairs[column].mean())


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(pairs: pd.DataFrame, by_family: pd.DataFrame, demand: pd.DataFrame) -> None:
    lines = [
        "# Published GPT-4.1-mini",
        "",
        "Scored with `twin2k.scoring` on every wave-4 item. The model answers are the dataset's precomputed completions. No API call.",
        "",
        "The submitted number is accuracy against the wave-4 answer. The wave 1–3 column is the target those files were generated for. It is not averaged with the wave-4 number.",
        "",
        "Family-weighted means give each family the same weight. Item-weighted means let pricing, about 42% of items, dominate. Toubia et al. 2025 report 71.72% for the text persona on a 17-task average. The figure below is this scorer on these items. It is not that average.",
        "",
        "| System | Input | Family-weighted, wave 4 | Item-weighted, wave 4 | Ratio to retest | Family-weighted, wave 1–3 | Unparsed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    retest_family, retest_item = headline(pairs, "retest")
    base_family, base_item = headline(pairs, "base_rate")
    lines.append(
        f"| Retest | same person's earlier answer | {pct(retest_family)} | {pct(retest_item)} | 100% | — | — |"
    )
    lines.append(
        f"| Population constant | no persona | {pct(base_family)} | {pct(base_item)} | {base_family / retest_family:.0%} | — | — |"
    )
    labels = {
        "demographics": ("Demographics only", "demographics"),
        "summary": ("Summary persona", "summary"),
        "text": ("Full text persona", "text persona"),
    }
    for key, (label, persona) in labels.items():
        family, item = headline(pairs, key)
        early, _ = headline(pairs, f"{key}_early")
        fail = float(pairs[f"{key}_fail"].mean())
        lines.append(
            f"| {label} | {persona} | {pct(family)} | {pct(item)} | {family / retest_family:.0%} | {pct(early)} | {pct(fail)} |"
        )
    text_family, text_item = headline(pairs, "text")
    lines.extend(
        [
            "",
            "Ratio to retest uses the family-weighted wave-4 score. The wave 1–3 column is only for a model completion, scored against the earlier answer. Unparsed is the share of those completions that were not a number. Retest and the population constant are computed from the human answers, so both cells are marked —.",
            "",
            (
                f"On the family-weighted average the full text persona, {pct(text_family)}, is below the population constant, {pct(base_family)}. "
                "The constant wins on most families. The persona wins on pricing. "
                f"Pricing is large enough that the item-weighted average goes the other way: full text {pct(text_item)}, population constant {pct(base_item)}."
            ),
            "",
            "| Family | Items | Retest | Population | Demographics | Summary | Full text | Text minus population |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    order = [name for name, _ in FAMILIES]
    table = by_family.set_index("family").loc[order]
    for family, row in table.iterrows():
        lines.append(
            f"| {family} | {int(row['items']):,} | {pct(row.retest)} | {pct(row.base_rate)} | {pct(row.demographics)} | {pct(row.summary)} | {pct(row.text)} | {100 * (row.text - row.base_rate):+.1f} pp |"
        )
    pricing = table.loc["Pricing"]
    lines.extend(
        [
            "",
            f"Pricing is the family the exploration flagged. Full text scores {pct(pricing.text)} against a population constant of {pct(pricing.base_rate)} and a retest of {pct(pricing.retest)}.",
            "",
            "Purchase rate is the share of pricing answers coded yes. Quintiles are within each product, so a high quintile is a high price for that product, not a high price in dollars across products.",
            "",
            "| Price quintile | Human wave 4 | Demographics | Summary | Full text |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in demand.itertuples(index=False):
        lines.append(
            f"| {int(row.quintile) + 1} | {pct(row.human)} | {pct(row.demographics)} | {pct(row.summary)} | {pct(row.text)} |"
        )
    cheap, dear = demand.iloc[0], demand.iloc[-1]

    def placement(column: str) -> str:
        if bool((demand[column] < demand["human"]).all()):
            return "sits below the human rate in every fifth"
        if bool((demand[column] > demand["human"]).all()):
            return "sits above the human rate in every fifth"
        return "crosses the human rate"

    lines.extend(
        [
            "",
            (
                f"Human purchase rates fall from {pct(float(cheap.human))} in the cheapest fifth to {pct(float(dear.human))} in the dearest. "
                f"Full text {placement('text')}. Demographics {placement('demographics')}. The summary {placement('summary')}."
            ),
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))


def main() -> None:
    ensure_parquets()
    ensure_sheets()
    wave = ds.dataset(ROOT / "data" / "parquet" / "wave_split", format="parquet").to_table(
        columns=["pid", "wave4_Q_wave1_3_A", "wave4_Q_wave4_A"]
    )
    rows = human_pairs(wave.to_pandas())
    wave4 = load_people(SIM / "wave4.csv")
    column_to_key = column_key(rows, wave4)
    edges = decile_edges(rows)
    raw_early = {(row["pid"], row["qid"], row["sub"]): row["early"] for row in rows if row["qid"] in edges}
    assign_text_spans(rows)
    for row in rows:
        if row["qid"] not in edges:
            continue
        expected = apply_decile(np.array([raw_early[(row["pid"], row["qid"], row["sub"])]]), edges[row["qid"]])[0]
        if expected != row["early"]:
            raise RuntimeError("decile edges do not match the exploration scorer")
    pairs = pd.DataFrame(rows)
    pairs = add_population_baseline(pairs)
    retest = [
        agreement(early, late, span, binary)
        for early, late, span, binary in zip(pairs["early"], pairs["late"], pairs["span"], pairs["binary"])
    ]
    if any(value is None for value in retest):
        raise RuntimeError("a human pair has no span")
    pairs["retest"] = retest
    for name, _ in ARMS:
        attach_arm(pairs, column_to_key, edges, name)
    demand = purchase_rates(pairs, column_to_key)
    by_family = (
        pairs.groupby("family", sort=False)
        .agg(
            items=("retest", "size"),
            retest=("retest", "mean"),
            base_rate=("base_rate", "mean"),
            demographics=("demographics", "mean"),
            summary=("summary", "mean"),
            text=("text", "mean"),
        )
        .reset_index()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(OUT / "by_family.csv", index=False)
    demand.to_csv(OUT / "pricing_demand.csv", index=False)
    write_report(pairs, by_family, demand)
    family, item = headline(pairs, "text")
    print(f"items {len(pairs)} text family {family:.4f} item {item:.4f}")
    print(f"wrote {REPORT}")


def purchase_rates(pairs: pd.DataFrame, column_to_key: dict[str, tuple]) -> pd.DataFrame:
    pricing = pairs.loc[pairs["family"] == "Pricing", ["pid", "block", "qid", "sub", "late", "price"]].copy()
    key_to_column = {key: column for column, key in column_to_key.items()}
    pricing["column"] = [
        key_to_column[(block, qid, sub)]
        for block, qid, sub in zip(pricing["block"], pricing["qid"], pricing["sub"])
    ]
    pricing["quintile"] = pricing.groupby("qid")["price"].transform(
        lambda series: pd.qcut(series, 5, labels=False, duplicates="drop")
    )
    for name, _ in ARMS:
        sheet = load_people(SIM / f"{name}_llm.csv")
        raw = pd.to_numeric(lookup(sheet, pricing["pid"].to_numpy(), pricing["column"].tolist()), errors="coerce")
        yes = np.where(np.isfinite(raw), raw == 1, np.nan)
        pricing[name] = yes
    pricing["human"] = pricing["late"] == 1
    table = (
        pricing.groupby("quintile", sort=True)[["human", "demographics", "summary", "text"]]
        .mean()
        .reset_index()
    )
    return table


if __name__ == "__main__":
    main()
