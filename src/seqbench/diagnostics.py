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
            matched = all(
                (
                    probe.get("status") == expected
                    if key == "status"
                    else float(probe.get("retention", 0.0)) < float(expected)
                    if key == "retention_lt"
                    else False
                )
                for key, expected in when.items()
            )
            if matched:
                diagnoses.append(
                    {
                        "probe": rule["probe"],
                        "diagnosis": rule["diagnosis"],
                        "next_probes": rule.get("next_probes", []),
                    }
                )
    return diagnoses
