from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compare import compare_runs
from .metrics import higher_is_better
from .specs import Probe, RunSpec


def screen_runs(
    *,
    spec_path: Path,
    candidate_runs: list[Path],
    baseline_runs: list[Path],
    output: Path,
) -> Path:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    comparison = compare_runs(
        spec_path=spec_path,
        labelled_runs={"exact_knn": baseline_runs, "candidate": candidate_runs},
        reference="exact_knn",
        output=output / "comparison",
    )
    raw = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    spec = RunSpec.load(spec_path)
    probes = {probe.id: probe for probe in (Probe.load(path) for path in spec.probes)}
    candidate_rows = [row for row in raw["results"] if row["model"] == "candidate"]
    cells: list[dict[str, Any]] = []
    signals: set[str] = set()
    inconclusive: set[str] = set()
    for row in candidate_rows:
        probe = probes[row["probe"]]
        interval = row.get("delta_stress_vs_reference_ci95")
        if not isinstance(interval, list) or len(interval) != 2:
            inconclusive.add(probe.property)
            continue
        potential = (
            float(interval[1]) > 0 if higher_is_better(row["metric"]) else float(interval[0]) < 0
        )
        if potential:
            signals.add(probe.property)
        cells.append(
            {
                "property": probe.property,
                "probe": probe.id,
                "metric": row["metric"],
                "primary": bool(row["primary"]),
                "candidate": row["stress"],
                "exact_knn": (
                    row["stress"] - row["delta_stress_vs_reference"]
                    if row["delta_stress_vs_reference"] is not None
                    else None
                ),
                "difference": row["delta_stress_vs_reference"],
                "difference_ci95": interval,
                "potential_improvement": potential,
            }
        )

    if "retrieval" in signals:
        signals.add("prediction")
    elif "retrieval" in inconclusive:
        inconclusive.add("prediction")

    efficiency = _efficiency_cells(candidate_runs, baseline_runs)
    if any(cell["potential_improvement"] for cell in efficiency):
        signals.add("scaling_compression")
    cells.extend(efficiency)

    candidate_seeds, seed_invariant = _run_seed_info(candidate_runs)
    expected_properties = {
        "prediction",
        "retrieval",
        "position_length_transfer",
        "role_binding",
        "multi_hop",
        "composition",
        "credit_assignment",
        "noise_robustness",
        "online_correction",
        "scaling_compression",
    }
    observed = {cell["property"] for cell in cells}
    if "retrieval" in observed:
        observed.add("prediction")
    inconclusive.update(expected_properties - observed)
    if signals:
        decision = "PROMOTE"
        reason = "potential improvement over exact kNN"
    elif inconclusive:
        decision = "INCONCLUSIVE"
        reason = "one or more properties could not be compared"
    elif len(candidate_seeds) < 3 and not seed_invariant:
        decision = "INCONCLUSIVE"
        reason = "no signal in discovery seed; run two confirmation seeds"
    else:
        decision = "DROP"
        reason = "exact kNN is not worse on every measured property"

    result = {
        "screen": spec.id,
        "decision": decision,
        "reason": reason,
        "candidate_seeds": sorted(candidate_seeds),
        "signals": sorted(signals),
        "inconclusive": sorted(inconclusive),
        "properties": cells,
    }
    (output / "screen.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "report.md", result)
    return output


def _run_seed_info(paths: list[Path]) -> tuple[set[int], bool]:
    seeds: set[int] = set()
    invariant = False
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        seeds.update(int(seed) for seed in manifest["seeds"])
        invariant = invariant or bool(manifest.get("algorithm", {}).get("seed_invariant"))
    return seeds, invariant


def _efficiency_cells(
    candidate_runs: list[Path], baseline_runs: list[Path]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metric, candidate, baseline in (
        (
            "checkpoint_bytes",
            _resource_value(candidate_runs, "checkpoint_bytes", maximum=True),
            _resource_value(baseline_runs, "checkpoint_bytes", maximum=True),
        ),
        (
            "train_wall_seconds",
            _resource_value(candidate_runs, "wall_seconds", maximum=False),
            _resource_value(baseline_runs, "wall_seconds", maximum=False),
        ),
    ):
        difference = candidate - baseline
        result.append(
            {
                "property": "scaling_compression",
                "probe": f"resources.{metric}",
                "metric": metric,
                "primary": metric == "checkpoint_bytes",
                "candidate": candidate,
                "exact_knn": baseline,
                "difference": difference,
                "difference_ci95": [difference, difference],
                "potential_improvement": candidate < baseline,
            }
        )
    return result


def _resource_value(paths: list[Path], field: str, *, maximum: bool) -> float:
    per_run: list[float] = []
    for path in paths:
        raw = json.loads((path / "resources.json").read_text(encoding="utf-8"))
        values = [
            float(row.get(field, 0)) for row in raw["resources"] if row["operation"] == "learn"
        ]
        per_run.append(max(values, default=0.0) if maximum else sum(values))
    return sum(per_run) / len(per_run) if per_run else 0.0


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# seqbench fast screen",
        "",
        f"Decision: **{result['decision']}**",
        "",
        result["reason"],
        "",
        "| Property | Probe | Metric | Candidate | exact kNN | Difference CI95 | Gain not excluded |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for cell in result["properties"]:
        lines.append(
            f"| {cell['property']} | {cell['probe']} | {cell['metric']} | "
            f"{_number(cell['candidate'])} | {_number(cell['exact_knn'])} | "
            f"{_interval(cell['difference_ci95'])} | "
            f"{'yes' if cell['potential_improvement'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number(value: object) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "—"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "—"
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"
