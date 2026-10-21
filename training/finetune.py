"""Fine-tune Qwen2.5-0.5B-Instruct on held-in people and score held-out people.

Persona text comes from wave_split only. The wave-4 answer is the label, never
the prompt. See training/HANDOFF.md.

  python -m training.finetune --prepare-only
  python -m training.finetune --train --eval
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from exploration.explore import ensure_parquets, family_of
from twin2k.scoring import Item, agreement, items_from_blocks, normalize_block, parse_number

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "outputs"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' "
    "as if you are the individual described in the 'Persona Profile' "
    "(which contains their past survey responses). Remain consistent with the "
    "persona's previous answers and stated characteristics. Carefully follow "
    "any instructions provided for the new question, including formatting requirements."
)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def split_people(pids: list[int], n_train: int, seed: int) -> tuple[set[int], set[int]]:
    ordered = np.array(sorted(pids))
    picked = np.random.default_rng(seed).choice(ordered, size=n_train, replace=False)
    train = {int(pid) for pid in picked}
    test = set(int(pid) for pid in ordered) - train
    return train, test


def index_questions(blocks: list[dict]) -> dict[tuple[str, str], dict]:
    found = {}
    for block in blocks:
        name = normalize_block(block.get("BlockName") or "")
        for question in block.get("Questions") or []:
            found[(name, str(question.get("QuestionID") or ""))] = question
    return found


def index_items(blocks: list[dict]) -> dict[tuple[str, str, int], Item]:
    return {(item.block, item.qid, item.sub): item for item in items_from_blocks(blocks)}


def format_question(question: dict, sub: int) -> tuple[str, str] | None:
    kind = question.get("QuestionType")
    text = " ".join((question.get("QuestionText") or "").split())
    answers = question.get("Answers") or {}
    if kind == "MC":
        options = question.get("Options") or []
        position = answers.get("SelectedByPosition")
        if position is None or len(options) < 2:
            return None
        lines = [text, "Options:"]
        lines.extend(f"{index} - {option}" for index, option in enumerate(options, start=1))
        lines.append("Reply with only the option number.")
        return "\n".join(lines), str(int(position))
    if kind == "Matrix":
        rows = question.get("Rows") or []
        columns = question.get("Columns") or []
        positions = answers.get("SelectedByPosition") or []
        if sub >= len(rows) or sub >= len(positions) or positions[sub] is None or len(columns) < 2:
            return None
        lines = [text, f"Row: {rows[sub]}", "Options:"]
        lines.extend(f"{index} - {option}" for index, option in enumerate(columns, start=1))
        lines.append("Reply with only the option number.")
        return "\n".join(lines), str(int(positions[sub]))
    if kind == "Slider":
        values = answers.get("Values") or []
        statements = question.get("Statements") or []
        if sub >= len(values):
            return None
        bounds = question.get("Range") or {}
        statement = statements[sub] if sub < len(statements) else ""
        line = text if not statement else f"{text}\nStatement: {statement}"
        line += f"\nEnter a number from {bounds.get('Min', 0)} to {bounds.get('Max', 100)}."
        return line, str(values[sub]).strip()
    if kind == "TE":
        raw = answers.get("Text")
        if raw is None or str(raw).strip() == "":
            return None
        return text + "\nReply with only the number.", str(raw).strip()
    return None


def load_rows() -> tuple[dict[int, str], list[dict]]:
    ensure_parquets()
    frame = ds.dataset(ROOT / "data" / "parquet" / "wave_split", format="parquet").to_table(
        columns=["pid", "wave1_3_persona_text", "wave4_Q_wave1_3_A", "wave4_Q_wave4_A"]
    ).to_pandas()
    personas: dict[int, str] = {}
    rows: list[dict] = []
    for record in frame.itertuples(index=False):
        pid = int(record.pid)
        personas[pid] = record.wave1_3_persona_text
        early = index_items(json.loads(record.wave4_Q_wave1_3_A))
        late_blocks = json.loads(record.wave4_Q_wave4_A)
        questions = index_questions(late_blocks)
        for item in items_from_blocks(late_blocks):
            previous = early.get((item.block, item.qid, item.sub))
            question = questions.get((item.block, item.qid))
            if previous is None or question is None:
                continue
            formatted = format_question(question, item.sub)
            if formatted is None:
                continue
            prompt, label = formatted
            rows.append(
                {
                    "pid": pid,
                    "block": item.block,
                    "family": family_of(item.block),
                    "qid": item.qid,
                    "sub": item.sub,
                    "kind": item.kind,
                    "binary": item.binary,
                    "span": item.span if item.span is not None else previous.span,
                    "early": previous.value,
                    "late": item.value,
                    "question": prompt,
                    "label": label,
                }
            )
    return personas, rows


def apply_train_spans(rows: list[dict], train_pids: set[int]) -> dict[str, np.ndarray]:
    """Decile edges and bounded text spans come from the training people only."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "te" and row["pid"] in train_pids:
            grouped[row["qid"]].append(row)
    edges: dict[str, np.ndarray] = {}
    bounded: dict[str, float] = {}
    for qid, group in grouped.items():
        values = [row["early"] for row in group] + [row["late"] for row in group]
        lo, hi = min(values), max(values)
        if 0 <= lo and hi <= 20:
            bounded[qid] = max(hi - lo, 1.0)
            continue
        reference = np.array([row["early"] for row in group], dtype=float)
        cut = np.quantile(reference, np.linspace(0, 1, 11))
        for index in range(1, len(cut)):
            if cut[index] <= cut[index - 1]:
                cut[index] = cut[index - 1] + 1e-6
        edges[qid] = cut

    def decile(value: float, cut: np.ndarray) -> float:
        bin_id = int(np.searchsorted(cut, value, side="right") - 1)
        return float(min(9, max(0, bin_id)))

    for row in rows:
        if row["qid"] in bounded:
            row["span"] = bounded[row["qid"]]
            row["binary"] = False
        elif row["qid"] in edges:
            cut = edges[row["qid"]]
            row["early"] = decile(row["early"], cut)
            row["late"] = decile(row["late"], cut)
            row["span"] = 9.0
            row["binary"] = False
            row["kind"] = "te-decile"
    return edges


def first_number(text: str) -> float | None:
    match = NUMBER.search(text.replace(",", ""))
    if match is None:
        return parse_number(text)
    try:
        return float(match.group(0))
    except ValueError:
        return None


def assert_holdout(personas: dict[int, str], rows: list[dict]) -> None:
    sample = rows if len(rows) <= 400 else random.Random(0).sample(rows, 400)
    leaked = 0
    for row in sample:
        stem = " ".join(row["question"].split())[:80]
        if len(stem) >= 40 and stem in personas[row["pid"]]:
            leaked += 1
    if leaked:
        raise RuntimeError(f"{leaked} wave-4 question stems appear in the persona")


def prepare(args: argparse.Namespace) -> tuple[dict[int, str], list[dict], set[int], set[int], dict[str, np.ndarray]]:
    personas, rows = load_rows()
    train_pids, test_pids = split_people(list(personas), args.n_train, args.seed)
    if train_pids & test_pids:
        raise RuntimeError("train and test pids overlap")
    if (train_pids | test_pids) != set(personas):
        raise RuntimeError("split does not cover every person")
    assert_holdout(personas, rows)
    edges = apply_train_spans(rows, train_pids)
    print(
        f"people {len(personas)} train {len(train_pids)} test {len(test_pids)} "
        f"items {len(rows)} decile_qids {sorted(edges)}"
    )
    return personas, rows, train_pids, test_pids, edges


def user_message(persona: str, question: str) -> str:
    return f"Persona Profile:\n{persona}\n\nNew Survey Question:\n{question}"


def messages(persona: str, question: str, label: str | None) -> list[dict]:
    packed = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(persona, question)},
    ]
    if label is not None:
        packed.append({"role": "assistant", "content": label})
    return packed


def persona_prefix(tokenizer, persona: str, budget: int) -> str:
    ids = tokenizer(persona, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return persona
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def token_ids(value) -> list[int]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [int(token) for token in value[0]]
        return [int(token) for token in value]
    ids = value["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def pack_ids(tokenizer, persona: str, question: str, label: str, max_length: int) -> dict:
    prompt = token_ids(
        tokenizer.apply_chat_template(
            messages(persona, question, None), tokenize=True, add_generation_prompt=True
        )
    )
    full = token_ids(
        tokenizer.apply_chat_template(
            messages(persona, question, label), tokenize=True, add_generation_prompt=False
        )
    )
    if full[: len(prompt)] != prompt:
        raise RuntimeError("chat template prompt is not a prefix of the full text")
    if len(full) > max_length:
        overflow = len(full) - max_length
        prompt = prompt[overflow:]
        full = full[overflow:]
    labels = [-100] * len(prompt) + list(full[len(prompt) :])
    if all(token == -100 for token in labels):
        raise RuntimeError("answer tokens were truncated away")
    return {
        "input_ids": list(full),
        "labels": labels,
        "attention_mask": [1] * len(full),
    }


def subset_rows(rows: list[dict], train_pids: set[int], test_pids: set[int], limit: int) -> tuple[set[int], set[int], list[dict]]:
    if limit <= 0:
        return train_pids, test_pids, rows
    rng = random.Random(0)
    train = set(rng.sample(sorted(train_pids), min(limit, len(train_pids))))
    test = set(rng.sample(sorted(test_pids), min(max(limit // 3, 1), len(test_pids))))
    keep = train | test
    return train, test, [row for row in rows if row["pid"] in keep]


def train(args, personas, rows, train_pids, test_pids) -> Path:
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_pids, test_pids, rows = subset_rows(rows, train_pids, test_pids, args.limit_people)
    output = OUT / ("smoke" if args.limit_people else "qwen-0.5b-lora")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prefixes = {pid: persona_prefix(tokenizer, personas[pid], args.max_length - 256) for pid in train_pids}

    class AnswerSet(Dataset):
        def __init__(self, examples: list[dict]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict:
            row = self.examples[index]
            return pack_ids(tokenizer, prefixes[row["pid"]], row["question"], row["label"], args.max_length)

    examples = [row for row in rows if row["pid"] in train_pids]
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    pad_id = tokenizer.pad_token_id

    def collate(features: list[dict]) -> dict:
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            extra = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * extra)
            batch["labels"].append(feature["labels"] + [-100] * extra)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * extra)
        return {key: torch.tensor(value) for key, value in batch.items()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=20,
            save_strategy="epoch",
            report_to=[],
            seed=args.seed,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
        ),
        train_dataset=AnswerSet(examples),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "train_pids.json").write_text(json.dumps(sorted(train_pids)))
    print(f"saved {output}")
    return output


def score_rows(rows: list[dict], column: str) -> dict:
    usable = [row for row in rows if row.get(column) is not None]
    if not usable:
        raise RuntimeError(f"no rows scored for {column}")
    by_family = defaultdict(list)
    for row in usable:
        by_family[row["family"]].append(row[column])
    family = {name: float(np.mean(values)) for name, values in sorted(by_family.items())}
    return {
        "family": float(np.mean(list(family.values()))),
        "item": float(np.mean([row[column] for row in usable])),
        "n": len(usable),
        "by_family": family,
        "fail": float(np.mean([row.get("fail", False) for row in usable])),
    }


def fit_baseline(rows: list[dict], train_pids: set[int]) -> dict[tuple, float]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["pid"] in train_pids:
            grouped[(row["block"], row["qid"], row["sub"])].append(row)
    baseline = {}
    for key, group in grouped.items():
        values = [row["early"] for row in group]
        if group[0]["binary"]:
            baseline[key] = float(Counter(values).most_common(1)[0][0])
        else:
            baseline[key] = float(np.median(values))
    return baseline


def evaluate(args, personas, rows, train_pids, test_pids, edges, adapter: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_pids, test_pids, rows = subset_rows(rows, train_pids, test_pids, args.limit_people)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    prefixes = {pid: persona_prefix(tokenizer, personas[pid], args.max_length - 256) for pid in test_pids}
    test_rows = [row for row in rows if row["pid"] in test_pids]
    decoded: list[str] = []
    batch = args.eval_batch
    for start in range(0, len(test_rows), batch):
        chunk = test_rows[start : start + batch]
        prompts = [
            tokenizer.apply_chat_template(
                messages(prefixes[row["pid"]], row["question"], None),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in chunk
        ]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        tokens = {key: value.to(model.device) for key, value in tokens.items()}
        with torch.no_grad():
            generated = model.generate(
                **tokens,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, tokens["input_ids"].shape[1] :]
        decoded.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"generated {min(start + batch, len(test_rows))}/{len(test_rows)}")

    def to_decile(qid: str, value: float) -> float:
        cut = edges.get(qid)
        if cut is None:
            return value
        bin_id = int(np.searchsorted(cut, value, side="right") - 1)
        return float(min(9, max(0, bin_id)))

    for row, text in zip(test_rows, decoded):
        guess = first_number(text)
        row["fail"] = guess is None
        if guess is None:
            row["model"] = 0.0
            continue
        if row["kind"] == "te-decile":
            guess = to_decile(row["qid"], guess)
        value = agreement(guess, row["late"], row["span"], row["binary"])
        row["model"] = 0.0 if value is None else value
        row["fail"] = value is None

    baseline = fit_baseline(rows, train_pids)
    for row in test_rows:
        constant = baseline[(row["block"], row["qid"], row["sub"])]
        base_score = agreement(constant, row["late"], row["span"], row["binary"])
        retest_score = agreement(row["early"], row["late"], row["span"], row["binary"])
        row["baseline"] = 0.0 if base_score is None else base_score
        row["retest"] = 0.0 if retest_score is None else retest_score

    metrics = {
        "retest": score_rows(test_rows, "retest"),
        "baseline": score_rows(test_rows, "baseline"),
        "model": score_rows(test_rows, "model"),
        "n_test_people": len(test_pids),
        "n_train_people": len(train_pids),
    }
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model_family = metrics["model"]["family"]
    base_family = metrics["baseline"]["family"]
    print(f"test family model {model_family:.4f} baseline {base_family:.4f} retest {metrics['retest']['family']:.4f}")
    print(f"wrote {adapter / 'metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit-people", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--adapter", type=Path, default=None)
    args = parser.parse_args()
    personas, rows, train_pids, test_pids, edges = prepare(args)
    if args.prepare_only and not args.train and not args.eval:
        return
    adapter = args.adapter
    if args.train:
        adapter = train(args, personas, rows, train_pids, test_pids)
    if args.eval:
        if adapter is None:
            adapter = OUT / ("smoke" if args.limit_people else "qwen-0.5b-lora")
        evaluate(args, personas, rows, train_pids, test_pids, edges, adapter)


if __name__ == "__main__":
    main()
