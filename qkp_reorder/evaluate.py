from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


def normalize_text(text: str) -> str:
    """Normalize text for lightweight debug-stage answer matching."""
    if text is None:
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_answer(prediction: str, gold_answers: Iterable[str]) -> int:
    """Return 1 when prediction contains any normalized gold answer."""
    pred_norm = normalize_text(prediction)

    for answer in gold_answers:
        answer_norm = normalize_text(answer)
        if answer_norm and answer_norm in pred_norm:
            return 1

    return 0


def simple_token_f1(prediction: str, gold_answers: List[str]) -> float:
    """Compute debug-only token F1 against the best gold answer."""
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    best_f1 = 0.0

    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue

        gold_counter = Counter(gold_tokens)
        common = pred_counter & gold_counter
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1

