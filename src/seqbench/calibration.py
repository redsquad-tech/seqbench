from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_calibration(
    weak_runs: list[Path], strong_runs: list[Path]
) -> dict[str, Any]:
    weak = _probe_cells(weak_runs)
    strong = _probe_cells(strong_runs)
    probes: dict[str, dict[str, float]] = {}
    for probe in sorted(set(weak) & set(strong)):
        weak_cell = weak[probe]
        strong_cell = strong[probe]
        probes[probe] = {
            "control_min": (weak_cell["control"] + strong_cell["control"]) / 2,
            "stress_min": (weak_cell["stress"] + strong_cell["stress"]) / 2,
            "max_drop": max(0.0, strong_cell["control"] - strong_cell["stress"]),
            "fail_stress_max": (weak_cell["stress"] + strong_cell["stress"]) / 2,
        }
    return {
        "version": 1,
        "method": "midpoint_between_weak_and_strong_reference_profiles",
        "weak_runs": [str(path) for path in weak_runs],
        "strong_runs": [str(path) for path in strong_runs],
        "probes": probes,
    }


def _probe_cells(paths: list[Path]) -> dict[str, dict[str, float]]:
    values: dict[str, list[dict[str, float]]] = {}
    for path in paths:
        raw = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for probe in raw["probes"]:
            if "control" in probe and "stress" in probe:
                values.setdefault(probe["probe"], []).append(
                    {
                        "control": float(probe["control"]["score"]),
                        "stress": float(probe["stress"]["score"]),
                    }
                )
    return {
        probe: {
            key: sum(item[key] for item in cells) / len(cells)
            for key in ("control", "stress")
        }
        for probe, cells in values.items()
    }


def write_calibration(
    output: Path, weak_runs: list[Path], strong_runs: list[Path]
) -> None:
    output.write_text(
        json.dumps(
            build_calibration(weak_runs, strong_runs),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

