from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import higher_is_better


def build_calibration(weak_runs: list[Path], strong_runs: list[Path]) -> dict[str, Any]:
    weak = _probe_cells(weak_runs)
    strong = _probe_cells(strong_runs)
    probes: dict[str, dict[str, float]] = {}
    for probe in sorted(set(weak) & set(strong)):
        weak_cell = weak[probe]
        strong_cell = strong[probe]
        metric = str(strong_cell["metric"])
        if metric != weak_cell["metric"]:
            raise ValueError(f"{probe}: weak and strong runs use different metrics")
        midpoint_control = (weak_cell["control"] + strong_cell["control"]) / 2
        midpoint_stress = (weak_cell["stress"] + strong_cell["stress"]) / 2
        if higher_is_better(metric):
            probes[probe] = {
                "metric": metric,
                "direction": "higher",
                "control_min": midpoint_control,
                "stress_min": midpoint_stress,
                "max_drop": max(0.0, strong_cell["control"] - strong_cell["stress"]),
                "fail_stress_max": midpoint_stress,
            }
        else:
            probes[probe] = {
                "metric": metric,
                "direction": "lower",
                "control_max": midpoint_control,
                "stress_max": midpoint_stress,
                "max_increase": max(0.0, strong_cell["stress"] - strong_cell["control"]),
                "fail_stress_min": midpoint_stress,
            }
    return {
        "version": 1,
        "method": "midpoint_between_weak_and_strong_reference_profiles",
        "weak_runs": [str(path) for path in weak_runs],
        "strong_runs": [str(path) for path in strong_runs],
        "probes": probes,
    }


def _probe_cells(paths: list[Path]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        raw = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for probe in raw["probes"]:
            if "control" in probe and "stress" in probe:
                values.setdefault(probe["probe"], []).append(
                    {
                        "metric": str(probe["metric"]),
                        "control": float(probe["control"]["score"]),
                        "stress": float(probe["stress"]["score"]),
                    }
                )
    return {
        probe: {key: sum(item[key] for item in cells) / len(cells) for key in ("control", "stress")}
        | {"metric": cells[0]["metric"]}
        for probe, cells in values.items()
    }


def write_calibration(output: Path, weak_runs: list[Path], strong_runs: list[Path]) -> None:
    output.write_text(
        json.dumps(
            build_calibration(weak_runs, strong_runs),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
