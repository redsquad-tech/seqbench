from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
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


def _logsumexp(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def enrich_prediction(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    acceptable = tuple(row["acceptable_outputs"]) or (row["target"],)
    result["normalized_exact_match"] = normalized_exact_match(row["output"], acceptable)
    result["token_f1"] = max(token_f1(row["output"], item) for item in acceptable)
    result["edit_similarity"] = max(edit_similarity(row["output"], item) for item in acceptable)
    acceptable_logs = row.get("acceptable_log_probabilities", {})
    log_probability = (
        _logsumexp(acceptable_logs.values())
        if acceptable_logs
        else row.get("target_log_probability")
    )
    result["nll_bits"] = (
        math.inf if log_probability is None else -float(log_probability) / math.log(2)
    )
    result["target_probability_zero"] = float(log_probability is None)
    result["capped_nll_bits"] = min(result["nll_bits"], 1024.0)
    candidates = row.get("candidate_log_probabilities", {})
    result["candidate_accuracy"] = None
    result["candidate_conditional_nll_bits"] = None
    result["candidate_nll_bits"] = None
    result["mrr"] = None
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda item: (
                candidates[item] is not None,
                candidates[item] if candidates[item] is not None else -math.inf,
                item,
            ),
            reverse=True,
        )
        acceptable_candidates = [item for item in ranked if item in acceptable]
        if acceptable_candidates:
            result["candidate_accuracy"] = float(ranked[0] in acceptable)
            result["mrr"] = 1.0 / (ranked.index(acceptable_candidates[0]) + 1)
            numerator = _logsumexp(candidates[item] for item in acceptable_candidates)
            denominator = _logsumexp(candidates.values())
            result["candidate_conditional_nll_bits"] = (
                math.inf
                if numerator is None or denominator is None
                else -(numerator - denominator) / math.log(2)
            )
            result["candidate_nll_bits"] = result["candidate_conditional_nll_bits"]
    result["seen_output_exact_match"] = (
        result["normalized_exact_match"] if row.get("target_seen_in_train") is True else None
    )
    result["novel_output_exact_match"] = (
        result["normalized_exact_match"] if row.get("target_seen_in_train") is False else None
    )
    result["copy_rate"] = (
        float(row["output_seen_in_train"])
        if isinstance(row.get("output_seen_in_train"), bool)
        else None
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
    "capped_nll_bits": False,
    "candidate_conditional_nll_bits": False,
    "candidate_nll_bits": False,
    "target_probability_zero": False,
    "counterfactual_consistency": True,
}


def higher_is_better(metric: str) -> bool:
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"unknown metric direction: {metric}")
    return HIGHER_IS_BETTER[metric]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _seed_group_scores(rows: list[dict[str, Any]], metric: str) -> dict[int, dict[str, float]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[(int(row.get("seed", 0)), row["probe_group_id"])].append(float(value))
    result: dict[int, dict[str, float]] = defaultdict(dict)
    for (seed, group), values in grouped.items():
        result[seed][group] = _mean(values)
    return dict(result)


def _interval(samples: list[tuple[float, float, float]], index: int) -> list[float]:
    values = sorted(sample[index] for sample in samples)
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def _retention(metric: str, control: float, stress: float) -> float:
    if higher_is_better(metric):
        return stress / control if control > 0 else 0.0
    return 2 ** (-(stress - control))


def paired_bootstrap(
    control_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    control = _seed_group_scores(control_rows, metric)
    stress = _seed_group_scores(stress_rows, metric)
    seeds = sorted(set(control) & set(stress))
    per_seed = [set(control[item]) & set(stress[item]) for item in seeds]
    common_groups = sorted(set.intersection(*per_seed)) if per_seed else []
    seeds = [item for item in seeds if common_groups]
    if not seeds:
        return {"groups": 0}
    control_by_seed = [_mean([control[seed][group] for group in common_groups]) for seed in seeds]
    stress_by_seed = [_mean([stress[seed][group] for group in common_groups]) for seed in seeds]
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_groups = [common_groups[rng.randrange(len(common_groups))] for _ in common_groups]
        left_seed: list[float] = []
        right_seed: list[float] = []
        for sampled_seed in sampled_seeds:
            left_seed.append(_mean([control[sampled_seed][group] for group in sampled_groups]))
            right_seed.append(_mean([stress[sampled_seed][group] for group in sampled_groups]))
        left = _mean(left_seed)
        right = _mean(right_seed)
        samples.append((left, right, right - left))

    control_mean = _mean(control_by_seed)
    stress_mean = _mean(stress_by_seed)
    gap = stress_mean - control_mean
    return {
        "groups": len(common_groups),
        "seed_group_cells": len(common_groups) * len(seeds),
        "seeds": len(seeds),
        "control": {"score": control_mean, "ci95": _interval(samples, 0)},
        "stress": {"score": stress_mean, "ci95": _interval(samples, 1)},
        "gap": {"score": gap, "ci95": _interval(samples, 2)},
        "retention": _retention(metric, control_mean, stress_mean),
    }


def unpaired_bootstrap(
    control_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    control = _seed_group_scores(control_rows, metric)
    stress = _seed_group_scores(stress_rows, metric)
    seeds = sorted(set(control) & set(stress))
    seeds = [item for item in seeds if control[item] and stress[item]]
    if not seeds:
        return {"groups": 0}
    control_groups = sorted(set.intersection(*(set(control[item]) for item in seeds)))
    stress_groups = sorted(set.intersection(*(set(stress[item]) for item in seeds)))
    if not control_groups or not stress_groups:
        return {"groups": 0}
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_control = [
            control_groups[rng.randrange(len(control_groups))] for _ in control_groups
        ]
        sampled_stress = [stress_groups[rng.randrange(len(stress_groups))] for _ in stress_groups]
        left_seed: list[float] = []
        right_seed: list[float] = []
        for sampled_seed in sampled_seeds:
            left_seed.append(_mean([control[sampled_seed][group] for group in sampled_control]))
            right_seed.append(_mean([stress[sampled_seed][group] for group in sampled_stress]))
        left = _mean(left_seed)
        right = _mean(right_seed)
        samples.append((left, right, right - left))
    control_mean = _mean(
        [_mean([control[item][group] for group in control_groups]) for item in seeds]
    )
    stress_mean = _mean([_mean([stress[item][group] for group in stress_groups]) for item in seeds])
    gap = stress_mean - control_mean
    return {
        "groups": min(len(control_groups), len(stress_groups)),
        "seed_group_cells": min(len(control_groups), len(stress_groups)) * len(seeds),
        "seeds": len(seeds),
        "control_examples": len(control_groups),
        "stress_examples": len(stress_groups),
        "control": {"score": control_mean, "ci95": _interval(samples, 0)},
        "stress": {"score": stress_mean, "ci95": _interval(samples, 1)},
        "gap": {"score": gap, "ci95": _interval(samples, 2)},
        "retention": _retention(metric, control_mean, stress_mean),
    }


def difficulty_curves(
    rows: list[dict[str, Any]], *, metric: str, axes: tuple[str, ...]
) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for axis in axes:
        cells: dict[tuple[int, str, str], list[float]] = defaultdict(list)
        for row in rows:
            value = row.get(metric)
            difficulty = row.get("metadata", {}).get(axis)
            if difficulty is not None and isinstance(value, (int, float)):
                cells[
                    (
                        int(row.get("budget_index", 0)),
                        str(row["stage"]),
                        str(difficulty),
                    )
                ].append(float(value))
        for (budget_index, stage, difficulty), values in sorted(cells.items()):
            curves.append(
                {
                    "budget_index": budget_index,
                    "stage": stage,
                    "axis": axis,
                    "difficulty": difficulty,
                    "metric": metric,
                    "score": _mean(values),
                    "examples": len(values),
                }
            )
    return curves


def output_novelty_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected in (("seen", True), ("novel", False)):
        selected = [row for row in rows if row.get("target_seen_in_train") is expected]
        if not selected:
            result[name] = {"examples": 0}
            continue
        result[name] = {
            "examples": len(selected),
            "exact_match": _mean([float(row["normalized_exact_match"]) for row in selected]),
            "capped_nll_bits": _mean([float(row["capped_nll_bits"]) for row in selected]),
            "zero_probability_rate": _mean(
                [float(row["target_probability_zero"]) for row in selected]
            ),
            "copy_rate": _mean(
                [
                    float(row["output_seen_in_train"])
                    for row in selected
                    if isinstance(row.get("output_seen_in_train"), bool)
                ]
            )
            if any(isinstance(row.get("output_seen_in_train"), bool) for row in selected)
            else None,
        }
    novel = [row for row in rows if row.get("target_seen_in_train") is False]
    result["novel_valid_rate"] = (
        _mean(
            [
                float(
                    row["normalized_exact_match"] == 1.0
                    and row.get("output_seen_in_train") is False
                )
                for row in novel
            ]
        )
        if novel
        else None
    )
    return result
