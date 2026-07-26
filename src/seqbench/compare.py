from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .metrics import paired_bootstrap, unpaired_bootstrap
from .specs import Probe, RunSpec


def compare_runs(
    *,
    spec_path: Path,
    labelled_runs: dict[str, list[Path]],
    reference: str,
    output: Path,
) -> Path:
    if reference not in labelled_runs:
        raise ValueError(f"reference model {reference!r} is missing")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    spec = RunSpec.load(spec_path)
    probes = {probe.id: probe for probe in (Probe.load(path) for path in spec.probes)}
    predictions = {
        label: _load_predictions(paths) for label, paths in labelled_runs.items()
    }
    rows: list[dict[str, Any]] = []
    for probe_id, probe in probes.items():
        metrics = [probe.primary_metric]
        metrics.extend(
            "capped_nll_bits" if item == "nll_bits" else item
            for item in probe.optional_metrics
        )
        if "nll_bits" in probe.optional_metrics:
            metrics.append("target_probability_zero")
        for metric in dict.fromkeys(metrics):
            reference_rows = [
                row for row in predictions[reference] if row["probe"] == probe_id
            ]
            for label, all_rows in predictions.items():
                selected = [row for row in all_rows if row["probe"] == probe_id]
                control_stage, stress_stage = _stages(probe.protocol)
                bootstrap = (
                    paired_bootstrap
                    if probe.options.get("paired", True)
                    else unpaired_bootstrap
                )
                summary = bootstrap(
                    [row for row in selected if row["stage"] == control_stage],
                    [row for row in selected if row["stage"] == stress_stage],
                    metric=metric,
                    replicates=spec.bootstrap_replicates,
                    seed=spec.bootstrap_seed,
                )
                if not summary.get("groups", 0):
                    continue
                reference_control = [
                    row
                    for row in reference_rows
                    if row["stage"] == control_stage
                ]
                reference_stress = [
                    row
                    for row in reference_rows
                    if row["stage"] == stress_stage
                ]
                delta_control = paired_bootstrap(
                    reference_control,
                    [row for row in selected if row["stage"] == control_stage],
                    metric=metric,
                    replicates=spec.bootstrap_replicates,
                    seed=spec.bootstrap_seed,
                )
                delta_stress = paired_bootstrap(
                    reference_stress,
                    [row for row in selected if row["stage"] == stress_stage],
                    metric=metric,
                    replicates=spec.bootstrap_replicates,
                    seed=spec.bootstrap_seed,
                )
                rows.append(
                    {
                        "probe": probe_id,
                        "metric": metric,
                        "primary": metric == probe.primary_metric,
                        "model": label,
                        "control": summary["control"]["score"],
                        "control_ci95_low": summary["control"]["ci95"][0],
                        "control_ci95_high": summary["control"]["ci95"][1],
                        "stress": summary["stress"]["score"],
                        "stress_ci95_low": summary["stress"]["ci95"][0],
                        "stress_ci95_high": summary["stress"]["ci95"][1],
                        "gap": summary["gap"]["score"],
                        "gap_ci95_low": summary["gap"]["ci95"][0],
                        "gap_ci95_high": summary["gap"]["ci95"][1],
                        "retention": summary["retention"],
                        "delta_control_vs_reference": _gap(delta_control),
                        "delta_stress_vs_reference": _gap(delta_stress),
                        "seeds": summary.get("seeds", 0),
                        "groups": summary["groups"],
                    }
                )
    resources = _resource_summary(labelled_runs)
    _write_csv(output / "comparison.csv", rows)
    _write_csv(output / "resources.csv", resources)
    (output / "comparison.json").write_text(
        json.dumps(
            {"reference": reference, "results": rows, "resources": resources},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(output / "comparison.md", rows, resources, reference)
    return output


def _load_predictions(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        run_seeds = {int(item) for item in manifest["seeds"]}
        overlap = seeds & run_seeds
        if overlap:
            raise ValueError(f"duplicate seeds {sorted(overlap)} in comparison runs")
        seeds.update(run_seeds)
        rows.extend(pq.read_table(path / "predictions.parquet").to_pylist())
    return rows


def _resource_summary(labelled_runs: dict[str, list[Path]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, paths in labelled_runs.items():
        cells: list[dict[str, Any]] = []
        for path in paths:
            cells.extend(
                json.loads((path / "resources.json").read_text(encoding="utf-8"))[
                    "resources"
                ]
            )
        for operation in ("learn", "infer"):
            selected = [item for item in cells if item["operation"] == operation]
            if not selected:
                continue
            result.append(
                {
                    "model": label,
                    "operation": operation,
                    "calls": len(selected),
                    "wall_seconds_total": sum(item["wall_seconds"] for item in selected),
                    "wall_seconds_mean": sum(item["wall_seconds"] for item in selected)
                    / len(selected),
                    "peak_ram_bytes": max(item["peak_ram_bytes"] for item in selected),
                    "checkpoint_bytes_max": max(
                        (item.get("checkpoint_bytes", 0) for item in selected),
                        default=0,
                    ),
                }
            )
    return result


def _stages(protocol: str) -> tuple[str, str]:
    return (
        ("before_related", "after_related")
        if protocol == "correction"
        else ("control", "stress")
    )


def _gap(summary: dict[str, Any]) -> float | None:
    return (
        float(summary["gap"]["score"])
        if summary.get("groups", 0)
        else None
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    reference: str,
) -> None:
    primary = [row for row in rows if row["primary"]]
    lines = [
        "# seqbench causal comparison",
        "",
        f"Reference: `{reference}`",
        "",
        "| Probe | Metric | Model | Control | Stress | Gap | Δ stress vs ref |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['probe']} | {row['metric']} | {row['model']} | "
            f"{_number(row['control'])} | "
            f"{_number(row['stress'])} | {_number(row['gap'])} | "
            f"{_number(row['delta_stress_vs_reference'])} |"
        )
    lines.extend(
        [
            "",
            "## Resources",
            "",
            "| Model | Operation | Calls | Wall total, s | Peak RAM | Checkpoint max |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in resources:
        lines.append(
            f"| {row['model']} | {row['operation']} | {row['calls']} | "
            f"{row['wall_seconds_total']:.2f} | {row['peak_ram_bytes']} | "
            f"{row['checkpoint_bytes_max']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number(value: object) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"
