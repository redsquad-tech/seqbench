from __future__ import annotations

from pathlib import Path
from typing import Any

from .specs import load_yaml


def apply_diagnostics(
    probe_results: dict[str, dict[str, Any]], paths: list[Path]
) -> list[dict[str, Any]]:
    diagnoses: list[dict[str, Any]] = []
    for path in paths:
        raw = load_yaml(path)
        for rule in raw.get("rules", []):
            probe = probe_results.get(rule["probe"])
            if probe is None:
                continue
            when = rule.get("when", {})
            matched = all(_predicate(probe, key, expected) for key, expected in when.items())
            if matched:
                diagnoses.append(
                    {
                        "probe": rule["probe"],
                        "diagnosis": rule["diagnosis"],
                        "next_probes": rule.get("next_probes", []),
                    }
                )
    return diagnoses


def _predicate(probe: dict[str, Any], expression: str, expected: object) -> bool:
    for suffix, operation in (
        ("_lt", lambda left, right: left < right),
        ("_lte", lambda left, right: left <= right),
        ("_gt", lambda left, right: left > right),
        ("_gte", lambda left, right: left >= right),
        ("_eq", lambda left, right: left == right),
    ):
        if expression.endswith(suffix):
            value = _nested(probe, expression[: -len(suffix)])
            if value is None:
                return False
            if suffix == "_eq":
                return bool(operation(value, expected))
            return bool(operation(float(value), float(expected)))
    return _nested(probe, expression) == expected


def _nested(value: dict[str, Any], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
