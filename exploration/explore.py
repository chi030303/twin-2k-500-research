"""Explore wave_split and write summary tables.

The wave-4 elements in the docs are blocks inside two JSON columns, not extra
dataset fields. `wave4_Q_wave1_3_A` and `wave4_Q_wave4_A` hold the same
instrument filled at two times. This script aligns those fills.

Run from the repo root: `.venv/bin/python -m exploration.explore`
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from twin2k.scoring import Item, agreement, items_from_blocks, normalize_block
from twin2k.scoring import _self_check

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parquet"
OUT = ROOT / "exploration" / "outputs"
REPORT = ROOT / "report" / "exploration.md"

HF = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/refs%2Fconvert%2Fparquet"
CENSUS_AGE = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/national/asrh/nc-est2024-agesex-res.csv"
CENSUS_RACE = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/national/asrh/nc-est2024-alldata-r-file12.csv"

# Docs list many wave-4 elements. Each person sees one variant per family.
FAMILIES: list[tuple[str, str]] = [
    ("False consensus", "False consensus"),
    ("Base-rate", "Base-rate"),
    ("Disease", "Disease"),
    ("Linda", "Linda"),
    ("Outcome bias", "Outcome bias"),
    ("Anchoring, African countries", "Anchoring - African"),
    ("Anchoring, redwood", "Anchoring - redwood"),
    ("Less is more", "Less is More"),
    ("Proportion dominance 1", "Proportion dominance 1"),
    ("Proportion dominance 2", "Proportion dominance 2"),
    ("Sunk cost", "Sunk cost"),
    ("Absolute vs relative", "Absolute vs. relative"),
    ("WTA/WTP", "WTA/WTP"),
    ("Allais", "Allais"),
    ("Myside", "Myside"),
    ("Probability matching", "Probability matching"),
    ("Non-experimental heuristics", "Non-experimental"),
    ("Pricing", "Product Preferences"),
]

DEMO_IDS = {
    "QID12": "sex",
    "QID13": "age",
    "QID15": "race",
    "QID14": "education",
    "QID21": "income",
    "QID11": "region",
    "QID20": "party",
}


def family_of(block: str) -> str:
    for name, prefix in FAMILIES:
        if block.startswith(prefix):
            return name
    return "Other"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"download {dest.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "twin2k-exploration"})
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())


def ensure_parquets() -> None:
    for config, count in (("wave_split", 7), ("full_persona", 7)):
        for index in range(count):
            name = f"{index:04d}.parquet"
            url = f"{HF}/{config}/data/{name}"
            download(url, DATA / config / name)


def load_columns(config: str, columns: list[str]) -> pd.DataFrame:
    table = ds.dataset(DATA / config, format="parquet").to_table(columns=columns)
    return table.to_pandas()


def index_items(blocks: list[dict]) -> dict[tuple[str, str, int], Item]:
    indexed = {}
    for item in items_from_blocks(blocks):
        indexed[(item.block, item.qid, item.sub)] = item
    return indexed


def assign_text_spans(pairs: list[dict]) -> None:
    """Small integer text answers use the observed range. Wide estimates use deciles."""
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        if row["kind"] == "te":
            by_qid[row["qid"]].append(row)
    for rows in by_qid.values():
        values = [row["early"] for row in rows] + [row["late"] for row in rows]
        lo, hi = min(values), max(values)
        # Sunk-cost answers sit on 0–20. Anchoring estimates run far past that
        # and use deciles, matching the paper.
        if 0 <= lo and hi <= 20:
            span = max(hi - lo, 1.0)
            for row in rows:
                row["span"] = span
                row["binary"] = False
            continue
        reference = np.array([row["early"] for row in rows], dtype=float)
        edges = np.quantile(reference, np.linspace(0, 1, 11))
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = edges[index - 1] + 1e-6

        def decile(value: float) -> float:
            bin_id = int(np.searchsorted(edges, value, side="right") - 1)
            return float(min(9, max(0, bin_id)))

        for row in rows:
            row["early"] = decile(row["early"])
            row["late"] = decile(row["late"])
            row["span"] = 9.0
            row["kind"] = "te-decile"


def build_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict] = []
    variant_rows: list[dict] = []
    for record in frame.itertuples(index=False):
        early_blocks = json.loads(record.wave4_Q_wave1_3_A)
        late_blocks = json.loads(record.wave4_Q_wave4_A)
        early = index_items(early_blocks)
        late = index_items(late_blocks)
        seen = {normalize_block(block.get("BlockName") or "") for block in late_blocks}
        for block_name in seen:
            variant_rows.append(
                {"pid": int(record.pid), "family": family_of(block_name), "block": block_name}
            )
        for key, late_item in late.items():
            early_item = early.get(key)
            if early_item is None:
                continue
            pair_rows.append(
                {
                    "pid": int(record.pid),
                    "block": late_item.block,
                    "family": family_of(late_item.block),
                    "qid": late_item.qid,
                    "sub": late_item.sub,
                    "kind": late_item.kind,
                    "early": early_item.value,
                    "late": late_item.value,
                    "span": late_item.span if late_item.span is not None else early_item.span,
                    "binary": late_item.binary,
                }
            )
    pairs = pair_rows
    assign_text_spans(pairs)
    scored = []
    for row in pairs:
        value = agreement(row["early"], row["late"], row["span"], row["binary"])
        if value is None:
            continue
        row["retest"] = value
        scored.append(row)
    return pd.DataFrame(scored), pd.DataFrame(variant_rows)


def add_population_baseline(pairs: pd.DataFrame) -> pd.DataFrame:
    """Constant prediction from the earlier answers, scored on wave 4."""
    baselines = {}
    for key, group in pairs.groupby(["block", "qid", "sub"], sort=False):
        values = group["early"].to_numpy()
        if bool(group["binary"].iloc[0]):
            baselines[key] = float(Counter(values.tolist()).most_common(1)[0][0])
        else:
            baselines[key] = float(np.median(values))
    keys = list(zip(pairs["block"], pairs["qid"], pairs["sub"]))
    constant = np.array([baselines[key] for key in keys])
    scores = [
        agreement(early, late, span, binary)
        for early, late, span, binary in zip(
            constant, pairs["late"], pairs["span"], pairs["binary"]
        )
    ]
    pairs = pairs.copy()
    pairs["base_rate"] = scores
    return pairs


def mean_table(pairs: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        pairs.groupby(column, sort=False)
        .agg(
            items=("retest", "size"),
            people=("pid", "nunique"),
            retest=("retest", "mean"),
            base_rate=("base_rate", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("retest")


def demographics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame.itertuples(index=False):
        blocks = json.loads(record.wave1_3_persona_json)
        found = {"pid": int(record.pid)}
        for block in blocks:
            if not normalize_block(block.get("BlockName") or "").startswith("Demographics"):
                continue
            for question in block.get("Questions") or []:
                field = DEMO_IDS.get(question.get("QuestionID"))
                if not field:
                    continue
                answers = question.get("Answers") or {}
                found[field] = answers.get("SelectedText")
        rows.append(found)
    return pd.DataFrame(rows)


def forward_flow_lengths(frame: pd.DataFrame) -> dict:
    lengths = []
    for raw in frame["wave1_3_persona_json"]:
        for block in json.loads(raw):
            if not normalize_block(block.get("BlockName") or "").startswith("Forward Flow"):
                continue
            for question in block.get("Questions") or []:
                text = ((question.get("Answers") or {}).get("Text")) or ""
                lengths.append(len(str(text).split()))
    series = pd.Series(lengths)
    return {
        "people": int(series.shape[0]),
        "median_words": float(series.median()) if len(series) else 0.0,
        "p95_words": float(series.quantile(0.95)) if len(series) else 0.0,
    }


def length_summary(wave: pd.DataFrame, full: pd.DataFrame) -> dict:
    text = wave["wave1_3_persona_text"].str.len()
    tokens = text / 4
    summary = full["persona_summary"].str.len()
    full_text = full["persona_text"].str.len()
    return {
        "people": int(len(wave)),
        "persona_chars_p50": float(text.median()),
        "persona_chars_p95": float(text.quantile(0.95)),
        "persona_tokens_p50": float(tokens.median()),
        "persona_tokens_p95": float(tokens.quantile(0.95)),
        "over_8k_tokens": float((tokens > 8000).mean()),
        "over_32k_tokens": float((tokens > 32000).mean()),
        "summary_chars_p50": float(summary.median()),
        "full_persona_text_chars_p50": float(full_text.median()),
    }


def _adult_counts(frame: pd.DataFrame, year: int) -> pd.Series:
    subset = frame[(frame["SEX"].isin([1, 2])) & (frame["AGE"].between(18, 100))]
    column = f"POPESTIMATE{year}"
    ages = subset["AGE"]
    bands = pd.cut(
        ages,
        bins=[17, 29, 49, 64, 100],
        labels=["18-29", "30-49", "50-64", "65+"],
    )
    totals = subset.groupby(bands, observed=True)[column].sum()
    sex = subset.groupby(subset["SEX"].map({1: "Male", 2: "Female"}))[column].sum()
    return pd.concat([totals, sex])


def census_benchmarks() -> dict:
    age_path = DATA / "nc-est2024-agesex-res.csv"
    race_path = DATA / "nc-est2024-alldata-r-file12.csv"
    download(CENSUS_AGE, age_path)
    download(CENSUS_RACE, race_path)
    age = pd.read_csv(age_path)
    counts = _adult_counts(age, 2024)
    adult = float(counts[["Male", "Female"]].sum())

    def july_adults(path: Path, universe: str) -> tuple[pd.DataFrame, int]:
        raw = pd.read_csv(path)
        raw = raw[raw["UNIVERSE"] == universe]
        years = raw.loc[raw["MONTH"] == 7, "YEAR"]
        year = 2024 if (years == 2024).any() else int(years.max())
        use = raw[(raw["MONTH"] == 7) & (raw["YEAR"] == year) & (raw["AGE"].between(18, 100))]
        return use, year

    race, race_year = july_adults(race_path, "R")
    race_adult = float(race["TOT_POP"].sum())
    hispanic_adult = float((race["H_MALE"] + race["H_FEMALE"]).sum())
    return {
        "age_source": "US Census Bureau, National Population Estimates, July 1 2024, nc-est2024-agesex-res.csv",
        "race_source": f"US Census Bureau resident population by age, July {race_year}, nc-est2024-alldata-r-file12.csv. Hispanic is H_MALE + H_FEMALE in that file.",
        "age": {band: float(counts[band]) / adult for band in ["18-29", "30-49", "50-64", "65+"]},
        "sex": {"Male": float(counts["Male"]) / adult, "Female": float(counts["Female"]) / adult},
        "race": {
            "White, non-Hispanic alone": float((race["NHWA_MALE"] + race["NHWA_FEMALE"]).sum()) / race_adult,
            "Black alone": float((race["BA_MALE"] + race["BA_FEMALE"]).sum()) / race_adult,
            "Asian alone": float((race["AA_MALE"] + race["AA_FEMALE"]).sum()) / race_adult,
            "Hispanic, any race": hispanic_adult / race_adult,
        },
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(
    pairs: pd.DataFrame,
    variants: pd.DataFrame,
    demos: pd.DataFrame,
    lengths: dict,
    flow: dict,
    census: dict,
    by_family: pd.DataFrame,
    by_kind: pd.DataFrame,
) -> None:
    micro = float(pairs["retest"].mean())
    base = float(pairs["base_rate"].mean())
    # Equal weight per family, so the 40 pricing items do not dominate.
    family_mean = float(by_family["retest"].mean())
    family_base = float(by_family["base_rate"].mean())
    pricing = by_family.loc[by_family["family"] == "Pricing"].iloc[0]
    lines = [
        "# Wave-split exploration",
        "",
        "Source: `LLM-Digital-Twin/Twin-2K-500`, config `wave_split`, all 2,058 people. Scorer: `twin2k.scoring`. Binary items are exact match. Other items are `1 - |a - b| / span`.",
        "Anchoring estimates are mapped to deciles of the earlier answers before that formula. The formula, and the 81.72% figure, are from Toubia, Gui, Peng, Merlau, Li, and Chen, 2025, Twin-2K-500, arXiv:2505.17479 (Marketing Science, https://doi.org/10.1287/mksc.2025.0262).",
        "The population baseline is the median of the earlier answers, or the mode on a binary item. No persona.",
        "",
        "## The two subsets",
        "",
        "`full_persona` is one row per person with all four waves: `persona_text`, `persona_summary`, and `persona_json`. On repeated questions it keeps the wave-4 answer. `wave_split` is the prediction split: `wave1_3_persona_text` and `wave1_3_persona_json` leave out what wave 4 repeats, and the wave-4 questions are stored twice, with the earlier answers and with the wave-4 answers. Both have 2,058 rows.",
        "",
        "## The two wave-4 columns",
        "",
        "`wave4_Q_wave1_3_A` and `wave4_Q_wave4_A` are the same instrument filled at two times. False consensus through Pricing are blocks inside both. The persona is the input: wave 1–3 questions that wave 4 does not ask again.",
        "",
        "## Retest",
        "",
        f"Scored items: {len(pairs):,}. Item-weighted retest {pct(micro)}; population baseline {pct(base)}.",
        f"Family-weighted retest {pct(family_mean)}; population baseline {pct(family_base)}. Pricing has 40 items per person, so the family mean is the headline.",
        f"Pricing retest {pct(float(pricing['retest']))} against a population baseline of {pct(float(pricing['base_rate']))}.",
        "Toubia et al. 2025 report 81.72% test-retest on their task average. The family mean here scores every sub-item, then gives each family equal weight.",
        "",
        "| Family | Items | People | Retest | Population baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in by_family.sort_values("family").itertuples(index=False):
        lines.append(
            f"| {row.family} | {int(row.items):,} | {int(row.people):,} | {pct(row.retest)} | {pct(row.base_rate)} |"
        )
    lines += [
        "",
        "| Answer kind | Items | Retest | Population baseline |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in by_kind.sort_values("kind").itertuples(index=False):
        lines.append(
            f"| {row.kind} | {int(row.items):,} | {pct(row.retest)} | {pct(row.base_rate)} |"
        )
    gap = by_family["retest"] - by_family["base_rate"]
    flat = ", ".join(by_family.loc[gap.abs() <= 0.02, "family"])
    wide = ", ".join(by_family.loc[gap >= 0.15, "family"])
    worse = ", ".join(by_family.loc[gap < -0.02, "family"])
    lines += [
        "",
        f"Baseline within 2 points of retest: {flat or 'none'}.",
        f"Retest ahead by at least 15 points: {wide or 'none'}.",
        f"Baseline ahead by more than 2 points: {worse or 'none'}.",
    ]

    lines += ["", "## Variants", ""]
    lines.append("Each family is one randomization group, and each person has one block name. A missing variant stays empty.")
    lines.append("")
    lines.append("| Family | People with exactly one variant | Distinct block names |")
    lines.append("| --- | ---: | ---: |")
    for family, prefix in FAMILIES:
        subset = variants[variants["family"] == family]
        counts = subset.groupby("pid").size()
        exactly_one = int((counts == 1).sum()) if len(counts) else 0
        names = subset["block"].nunique()
        lines.append(f"| {family} | {exactly_one:,} | {names} |")

    lines += [
        "",
        "## Length",
        "",
        "Token counts are characters / 4, OpenAI's English rule of thumb, not a tokenizer count.",
        f"Wave 1–3 persona: median {lengths['persona_tokens_p50']:.0f} tokens, 95th percentile {lengths['persona_tokens_p95']:.0f}.",
        f"Share over 8k tokens: {pct(lengths['over_8k_tokens'])}. Share over 32k: {pct(lengths['over_32k_tokens'])}.",
        f"Median summary length: {lengths['summary_chars_p50']:.0f} characters. Median `full_persona` text: {lengths['full_persona_text_chars_p50']:.0f} characters, versus {lengths['persona_chars_p50']:.0f} for the wave 1–3 persona.",
        "`full_persona` text is longer because repeated questions keep the wave-4 answer.",
        f"Forward Flow median {flow['median_words']:.0f} words (95th percentile {flow['p95_words']:.0f}). The rest of the persona repeats the question wording.",
        "",
        "## Who is in the sample",
        "",
        "Demographics are wave 1. Later waves are behavior.",
        f"Age and sex benchmark: {census['age_source']}.",
        f"Race and Hispanic origin: {census['race_source']}",
        "Race or origin is one choice, so those sample shares sum to 100%. Census race and Hispanic origin overlap. White on the survey is a different bucket from non-Hispanic White alone. The gaps are directional.",
        "",
        "| Group | Sample | Census adults | Sample minus Census |",
        "| --- | ---: | ---: | ---: |",
    ]
    sample_age = demos["age"].value_counts(normalize=True)
    for band, census_share in census["age"].items():
        sample_share = float(sample_age.get(band, 0.0))
        lines.append(
            f"| Age {band} | {pct(sample_share)} | {pct(census_share)} | {100 * (sample_share - census_share):+.1f} pp |"
        )
    sample_sex = demos["sex"].value_counts(normalize=True)
    for label, census_share in census["sex"].items():
        sample_share = float(sample_sex.get(label, 0.0))
        lines.append(
            f"| Sex {label} | {pct(sample_share)} | {pct(census_share)} | {100 * (sample_share - census_share):+.1f} pp |"
        )
    race_map = {
        "White": "White, non-Hispanic alone",
        "Black": "Black alone",
        "Asian": "Asian alone",
        "Hispanic": "Hispanic, any race",
    }
    sample_race = demos["race"].value_counts(normalize=True)
    for sample_label, census_label in race_map.items():
        sample_share = float(sample_race.get(sample_label, 0.0))
        census_share = census["race"][census_label]
        lines.append(
            f"| {sample_label} vs {census_label} | {pct(sample_share)} | {pct(census_share)} | {100 * (sample_share - census_share):+.1f} pp |"
        )

    lines += ["", "Sample shares with no Census counterpart:", ""]
    for field in ("education", "income", "region", "party"):
        lines.append(f"**{field}**")
        lines.append("")
        counts = demos[field].value_counts(normalize=True)
        for label, share in counts.items():
            lines.append(f"- {label}: {pct(float(share))}")
        lines.append("")

    person = pairs.groupby("pid")["retest"].mean().rename("retest")
    joined = demos.merge(person, on="pid", how="inner")
    lines += [
        "## Retest by group",
        "",
        "Human retest inside the group. A later model table can use the same split.",
        "",
        "| Slice | People | Retest |",
        "| --- | ---: | ---: |",
    ]
    for field in ("sex", "race", "age", "education"):
        for label, group in joined.groupby(field):
            lines.append(f"| {field}: {label} | {len(group):,} | {pct(float(group['retest'].mean()))} |")

    lines += [
        "",
        "## Use",
        "",
        "Model input is the `wave_split` persona. Score with `twin2k.scoring.agreement`. Leave the unseen variant empty. `full_persona` text keeps the wave-4 answer on repeated questions.",
        "Pricing is a stated yes or no at a shown price. The ceiling for that block is its own retest.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


def main() -> None:
    _self_check()
    ensure_parquets()
    print("loading wave_split")
    wave = load_columns(
        "wave_split",
        ["pid", "wave1_3_persona_text", "wave1_3_persona_json", "wave4_Q_wave1_3_A", "wave4_Q_wave4_A"],
    )
    print("loading full_persona lengths")
    full = load_columns("full_persona", ["pid", "persona_text", "persona_summary"])
    print("scoring")
    pairs, variants = build_pairs(wave)
    pairs = add_population_baseline(pairs)
    demos = demographics(wave)
    lengths = length_summary(wave, full)
    flow = forward_flow_lengths(wave)
    census = census_benchmarks()
    by_family = mean_table(pairs, "family")
    by_kind = mean_table(pairs, "kind")
    OUT.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(OUT / "retest_by_family.csv", index=False)
    by_kind.to_csv(OUT / "retest_by_kind.csv", index=False)
    variants.groupby(["family", "block"]).size().reset_index(name="people").to_csv(
        OUT / "variant_counts.csv", index=False
    )
    demo_counts = {
        column: demos[column].value_counts(dropna=False).to_dict()
        for column in demos.columns
        if column != "pid"
    }
    (OUT / "demographics.json").write_text(json.dumps(demo_counts, indent=2))
    (OUT / "length.json").write_text(json.dumps(lengths, indent=2))
    write_report(pairs, variants, demos, lengths, flow, census, by_family, by_kind)
    print(f"items {len(pairs)} retest {pairs['retest'].mean():.3f} base {pairs['base_rate'].mean():.3f}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
