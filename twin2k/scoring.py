"""Score agreement between two answers to the same wave-4 item.

Binary items use exact match. Other items use 1 - |a - b| / span, which is
the paper's accuracy for numerical answers. Descriptive blocks are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    block: str
    qid: str
    sub: int
    value: float
    span: float | None
    binary: bool
    kind: str  # mc, matrix, slider, te


def normalize_block(name: str) -> str:
    return " ".join((name or "").split())


def parse_number(text: object) -> float | None:
    if text is None:
        return None
    raw = str(text).strip().replace(",", "").replace("%", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def agreement(early: float, late: float, span: float | None, binary: bool) -> float | None:
    if binary:
        return 1.0 if early == late else 0.0
    if span is None or span <= 0:
        return None
    return max(0.0, 1.0 - abs(early - late) / span)


def items_from_blocks(blocks: list[dict]) -> list[Item]:
    """Flatten one wave-4 JSON into scored items. DB / instruction screens are dropped."""
    found: list[Item] = []
    for block in blocks:
        block_name = normalize_block(block.get("BlockName") or "")
        for question in block.get("Questions") or []:
            if question.get("is_descriptive") or question.get("QuestionType") == "DB":
                continue
            found.extend(_items_for_question(block_name, question))
    return found


def _items_for_question(block: str, question: dict) -> list[Item]:
    qid = str(question.get("QuestionID") or "")
    kind = question.get("QuestionType")
    answers = question.get("Answers") or {}
    if kind == "MC":
        options = question.get("Options") or []
        position = answers.get("SelectedByPosition")
        if position is None or len(options) < 2:
            return []
        return [
            Item(
                block,
                qid,
                0,
                float(position),
                float(len(options) - 1),
                len(options) == 2,
                "mc",
            )
        ]
    if kind == "Matrix":
        columns = question.get("Columns") or []
        positions = answers.get("SelectedByPosition") or []
        if len(columns) < 2:
            return []
        span = float(len(columns) - 1)
        binary = len(columns) == 2
        return [
            Item(block, qid, index, float(position), span, binary, "matrix")
            for index, position in enumerate(positions)
            if position is not None
        ]
    if kind == "Slider":
        bounds = question.get("Range") or {}
        span = float(bounds.get("Max", 100)) - float(bounds.get("Min", 0))
        values = answers.get("Values") or []
        items = []
        for index, raw in enumerate(values):
            value = parse_number(raw)
            if value is None:
                continue
            items.append(Item(block, qid, index, value, span, False, "slider"))
        return items
    if kind == "TE":
        value = parse_number(answers.get("Text"))
        if value is None:
            return []
        # Span is filled later: small integers use the observed range, wide
        # estimates use deciles of the earlier answers.
        return [Item(block, qid, 0, value, None, False, "te")]
    return []


def _self_check() -> None:
    assert agreement(1, 1, 1, True) == 1.0
    assert agreement(1, 2, 1, True) == 0.0
    assert abs(agreement(1, 2, 4, False) - 0.75) < 1e-9
    assert abs(agreement(50, 87, 100, False) - 0.63) < 1e-9
