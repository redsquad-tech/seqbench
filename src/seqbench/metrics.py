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
    result["target_probability_zero"] = float(log_probability is None)
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
    "capped_nll_bits": False,
    "candidate_nll_bits": False,
    "target_probability_zero": False,
}


def higher_is_better(metric: str) -> bool:
    return HIGHER_IS_BETTER.get(metric, True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _seed_group_scores(
    rows: list[dict[str, Any]], metric: str
) -> dict[int, dict[str, float]]:
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
    common = {
        seed: sorted(set(control[seed]) & set(stress[seed])) for seed in seeds
    }
    seeds = [seed for seed in seeds if common[seed]]
    if not seeds:
        return {"groups": 0}
    control_by_seed = [
        _mean([control[seed][group] for group in common[seed]]) for seed in seeds
    ]
    stress_by_seed = [
        _mean([stress[seed][group] for group in common[seed]]) for seed in seeds
    ]
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        left_seed: list[float] = []
        right_seed: list[float] = []
        for sampled_seed in sampled_seeds:
            groups = common[sampled_seed]
            sampled_groups = [groups[rng.randrange(len(groups))] for _ in groups]
            left_seed.append(
                _mean([control[sampled_seed][group] for group in sampled_groups])
            )
            right_seed.append(
                _mean([stress[sampled_seed][group] for group in sampled_groups])
            )
        left = _mean(left_seed)
        right = _mean(right_seed)
        samples.append((left, right, right - left))

    control_mean = _mean(control_by_seed)
    stress_mean = _mean(stress_by_seed)
    gap = stress_mean - control_mean
    return {
        "groups": sum(len(common[item]) for item in seeds),
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
    seeds = [seed for seed in seeds if control[seed] and stress[seed]]
    if not seeds:
        return {"groups": 0}
    rng = random.Random(seed)
    samples: list[tuple[float, float, float]] = []
    for _ in range(replicates):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        left_seed: list[float] = []
        right_seed: list[float] = []
        for sampled_seed in sampled_seeds:
            left_values = list(control[sampled_seed].values())
            right_values = list(stress[sampled_seed].values())
            left_seed.append(
                _mean([left_values[rng.randrange(len(left_values))] for _ in left_values])
            )
            right_seed.append(
                _mean(
                    [right_values[rng.randrange(len(right_values))] for _ in right_values]
                )
            )
        left = _mean(left_seed)
        right = _mean(right_seed)
        samples.append((left, right, right - left))
    control_mean = _mean([_mean(list(control[item].values())) for item in seeds])
    stress_mean = _mean([_mean(list(stress[item].values())) for item in seeds])
    gap = stress_mean - control_mean
    return {
        "groups": min(
            sum(len(control[item]) for item in seeds),
            sum(len(stress[item]) for item in seeds),
        ),
        "seeds": len(seeds),
        "control_examples": sum(len(control[item]) for item in seeds),
        "stress_examples": sum(len(stress[item]) for item in seeds),
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
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            value = row.get(metric)
            difficulty = row.get("metadata", {}).get(axis)
            if difficulty is not None and isinstance(value, (int, float)):
                cells[(str(row["stage"]), str(difficulty))].append(float(value))
        for (stage, difficulty), values in sorted(cells.items()):
            curves.append(
                {
                    "stage": stage,
                    "axis": axis,
                    "difficulty": difficulty,
                    "metric": metric,
                    "score": _mean(values),
                    "examples": len(values),
                }
            )
    return curves
