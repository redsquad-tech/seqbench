from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected YAML mapping")
    return value


@dataclass(frozen=True, slots=True)
class Probe:
    id: str
    property: str
    protocol: str
    requires: tuple[str, ...]
    train: dict[str, Any]
    control: dict[str, Any]
    stress: dict[str, Any]
    primary_metric: str
    optional_metrics: tuple[str, ...]
    difficulty_axes: tuple[str, ...]
    pair_by: str
    options: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Probe:
        raw = load_yaml(path)
        required = {
            "id",
            "property",
            "version",
            "protocol",
            "requires",
            "train",
            "control",
            "stress",
            "metrics",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        metrics = raw["metrics"]
        return cls(
            id=str(raw["id"]),
            property=str(raw["property"]),
            protocol=str(raw["protocol"]),
            requires=tuple(raw["requires"]),
            train=dict(raw["train"].get("select", {})),
            control=dict(raw["control"].get("select", {})),
            stress=dict(raw["stress"].get("select", {})),
            primary_metric=str(metrics["primary"]),
            optional_metrics=tuple(metrics.get("optional", [])),
            difficulty_axes=tuple(raw.get("difficulty_axes", [])),
            pair_by=str(raw.get("pair_by", "probe_group_id")),
            options=dict(raw.get("options", {})),
        )


@dataclass(frozen=True, slots=True)
class PropertySpec:
    id: str
    required_probes: tuple[str, ...]
    supplementary_probes: tuple[str, ...]
    aggregation: str

    @classmethod
    def load(cls, path: Path) -> PropertySpec:
        raw = load_yaml(path)
        return cls(
            id=str(raw["id"]),
            required_probes=tuple(raw["required_probes"]),
            supplementary_probes=tuple(raw.get("supplementary_probes", [])),
            aggregation=str(raw.get("aggregation", "all_required")),
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    id: str
    probes: tuple[Path, ...]
    properties: tuple[Path, ...]
    diagnostics: tuple[Path, ...]
    calibration: Path | None
    seeds: tuple[int, ...]
    budgets: tuple[dict[str, Any], ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    report_failures: int

    @classmethod
    def load(cls, path: Path) -> RunSpec:
        raw = load_yaml(path)
        base = path.parent
        budgets = raw.get("budgets", [])
        if not budgets:
            raise ValueError(f"{path}: at least one budget is required")
        return cls(
            id=str(raw["id"]),
            probes=tuple((base / item).resolve() for item in raw["probes"]),
            properties=tuple((base / item).resolve() for item in raw["properties"]),
            diagnostics=tuple((base / item).resolve() for item in raw.get("diagnostics", [])),
            calibration=(
                (base / raw["calibration"]).resolve()
                if raw.get("calibration")
                else None
            ),
            seeds=tuple(int(item) for item in raw.get("seeds", [0])),
            budgets=tuple(dict(item) for item in budgets),
            bootstrap_replicates=int(raw.get("bootstrap_replicates", 2000)),
            bootstrap_seed=int(raw.get("bootstrap_seed", 42)),
            report_failures=int(raw.get("report_failures", 20)),
        )
