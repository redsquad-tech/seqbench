from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from typing import Any


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


def normalized_exact_match(output: str, acceptable: tuple[str, ...]) -> float:
    normalized = normalize_text(output)
    return float(normalized in {normalize_text(item) for item in acceptable})


def token_f1(output: str, target: str) -> float:
    predicted = re.findall(r"\w+|[^\w\s]", normalize_text(output).lower())
    expected = re.findall(r"\w+|[^\w\s]", normalize_text(target).lower())
    if not predicted or not expected:
        return float(predicted == expected)
    remaining = list(expected)
    common = 0
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            common += 1
    precision = common / len(predicted)
    recall = common / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def edit_similarity(output: str, target: str) -> float:
    left = normalize_text(output)
    right = normalize_text(target)
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for offset, expected in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (char != expected),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def enrich_prediction(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["normalized_exact_match"] = normalized_exact_match(
        row["output"], tuple(row["acceptable_outputs"])
    )
    result["token_f1"] = token_f1(row["output"], row["target"])
    result["edit_similarity"] = edit_similarity(row["output"], row["target"])
    log_probability = row.get("target_log_probability")
    result["nll_bits"] = (
        math.inf if log_probability is None else -float(log_probability) / math.log(2)
    )
    result["capped_nll_bits"] = min(result["nll_bits"], 1024.0)
    candidates = row.get("candidate_log_probabilities", {})
    result["candidate_accuracy"] = None
    result["candidate_nll_bits"] = None
    result["mrr"] = None
    if candidates and row["target"] in candidates:
        ranked = sorted(
            candidates,
            key=lambda item: (
                candidates[item] is not None,
                candidates[item] if candidates[item] is not None else -math.inf,
                item,
            ),
            reverse=True,
        )
        result["candidate_accuracy"] = float(ranked[0] == row["target"])
        result["mrr"] = 1.0 / (ranked.index(row["target"]) + 1)
        target_log = candidates[row["target"]]
        result["candidate_nll_bits"] = (
            math.inf if target_log is None else -float(target_log) / math.log(2)
        )
    return result


HIGHER_IS_BETTER = {
    "normalized_exact_match": True,
    "token_f1": True,
    "edit_similarity": True,
    "candidate_accuracy": True,
    "mrr": True,
    "valid_structure_rate": True,
    "nll_bits": False,
    "candidate_nll_bits": False,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _group_scores(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[row["probe_group_id"]].append(float(value))
    return {group: _mean(values) for group, values in grouped.items()}


def paired_bootstrap(
    control_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    control = _group_scores(control_rows, metric)
    stress = _group_scores(stress_rows, metric)
    groups = sorted(set(control) & set(stress))
    if not groups:
        return {"groups": 0}
    control_values = [control[group] for group in groups]
    stress_values = [stress[group] for group in groups]
    gaps = [right - left for left, right in zip(control_values, stress_values, strict=True)]
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(groups)) for _ in groups]
        samples.append(
            (
                _mean([control_values[index] for index in indices]),
                _mean([stress_values[index] for index in indices]),
                _mean([gaps[index] for index in indices]),
            )
        )

    def interval(index: int) -> list[float]:
        values = sorted(sample[index] for sample in samples)
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        return [low, high]

    control_mean = _mean(control_values)
    stress_mean = _mean(stress_values)
    gap = stress_mean - control_mean
    if HIGHER_IS_BETTER.get(metric, True):
        retention = stress_mean / control_mean if control_mean > 0 else 0.0
    else:
        retention = 2 ** (-gap)
    return {
        "groups": len(groups),
        "control": {"score": control_mean, "ci95": interval(0)},
        "stress": {"score": stress_mean, "ci95": interval(1)},
        "gap": {"score": gap, "ci95": interval(2)},
        "retention": retention,
    }


def unpaired_bootstrap(
    control_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    control = list(_group_scores(control_rows, metric).values())
    stress = list(_group_scores(stress_rows, metric).values())
    if not control or not stress:
        return {"groups": 0}
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        left = _mean([control[rng.randrange(len(control))] for _ in control])
        right = _mean([stress[rng.randrange(len(stress))] for _ in stress])
        samples.append((left, right, right - left))

    def interval(index: int) -> list[float]:
        values = sorted(sample[index] for sample in samples)
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    control_mean = _mean(control)
    stress_mean = _mean(stress)
    gap = stress_mean - control_mean
    if HIGHER_IS_BETTER.get(metric, True):
        retention = stress_mean / control_mean if control_mean > 0 else 0.0
    else:
        retention = 2 ** (-gap)
    return {
        "groups": min(len(control), len(stress)),
        "control_examples": len(control),
        "stress_examples": len(stress),
        "control": {"score": control_mean, "ci95": interval(0)},
        "stress": {"score": stress_mean, "ci95": interval(1)},
        "gap": {"score": gap, "ci95": interval(2)},
        "retention": retention,
    }


def difficulty_curves(
    rows: list[dict[str, Any]], *, metric: str, axes: tuple[str, ...]
) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for axis in axes:
        cells: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = row.get(metric)
            difficulty = row.get("metadata", {}).get(axis)
            if difficulty is not None and isinstance(value, (int, float)):
                cells[str(difficulty)].append(float(value))
        for difficulty, values in sorted(cells.items()):
            curves.append(
                {
                    "axis": axis,
                    "difficulty": difficulty,
                    "metric": metric,
                    "score": _mean(values),
                    "examples": len(values),
                }
            )
    return curves
