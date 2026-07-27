from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Task
from .specs import Probe
from .tasks import iter_tasks, matches

SECTIONS = ("train", "stress_train", "control", "stress")


@dataclass(slots=True)
class SampledProbeData:
    train: list[Task] = field(default_factory=list)
    stress_train: list[Task] = field(default_factory=list)
    control: list[Task] = field(default_factory=list)
    stress: list[Task] = field(default_factory=list)


@dataclass(slots=True)
class _Inventory:
    counts: Counter[str] = field(default_factory=Counter)
    strata: dict[str, str] = field(default_factory=dict)


def sample_tasks(
    paths: list[Path],
    probes: list[Probe],
    *,
    sampling_seed: int,
    train_limit_override: int | None,
    eval_limit_override: int | None,
) -> tuple[dict[str, SampledProbeData], dict[str, Any]]:
    inventories = {probe.id: {section: _Inventory() for section in SECTIONS} for probe in probes}
    for task in iter_tasks(paths):
        for probe in probes:
            for section, selector in _selectors(probe):
                if matches(task, selector):
                    field = _group_field(probe, section)
                    group = str(_task_field(task, field))
                    inventory = inventories[probe.id][section]
                    inventory.counts[group] += 1
                    stratify = probe.options.get(
                        "train_stratify_by"
                        if section in {"train", "stress_train"}
                        else "eval_stratify_by"
                    )
                    if stratify:
                        inventory.strata[group] = str(_task_field(task, str(stratify)))

    selected: dict[str, dict[str, set[str]]] = {}
    selection_report: dict[str, Any] = {}
    for probe in probes:
        cells = inventories[probe.id]
        _validate_duplicates(probe, cells)
        chosen: dict[str, set[str]] = {}
        train_limit = _limit(probe, True, train_limit_override)
        eval_limit = _limit(probe, False, eval_limit_override)

        if cells["stress_train"].counts:
            common_train = set(cells["train"].counts) & set(cells["stress_train"].counts)
            train_groups = _choose(
                common_train,
                limit=train_limit,
                seed=sampling_seed,
                salt=f"{probe.id}:train_pair",
                strata=cells["train"].strata,
                declared=probe.options.get("train_strata"),
            )
            chosen["train"] = train_groups
            chosen["stress_train"] = train_groups
        else:
            chosen["train"] = _choose(
                set(cells["train"].counts),
                limit=train_limit,
                seed=sampling_seed,
                salt=f"{probe.id}:train",
                strata=cells["train"].strata,
                declared=probe.options.get("train_strata"),
            )
            chosen["stress_train"] = set()

        if probe.options.get("paired", True):
            common_eval = set(cells["control"].counts) & set(cells["stress"].counts)
            eval_groups = _choose(
                common_eval,
                limit=eval_limit,
                seed=sampling_seed,
                salt=f"{probe.id}:eval_pair",
                strata=cells["control"].strata,
                declared=probe.options.get("eval_strata"),
            )
            chosen["control"] = eval_groups
            chosen["stress"] = eval_groups
        else:
            for section in ("control", "stress"):
                chosen[section] = _choose(
                    set(cells[section].counts),
                    limit=eval_limit,
                    seed=sampling_seed,
                    salt=f"{probe.id}:{section}",
                    strata=cells[section].strata,
                    declared=probe.options.get("eval_strata"),
                )
        selected[probe.id] = chosen
        selection_report[probe.id] = {
            "sampling_seed": sampling_seed,
            "sections": {
                section: {
                    "eligible_groups": len(cells[section].counts),
                    "selected_groups": len(chosen[section]),
                    "duplicate_groups": sum(count > 1 for count in cells[section].counts.values()),
                    "selected_ids": sorted(chosen[section]),
                }
                for section in SECTIONS
            },
            "matched_eval_groups": len(set(cells["control"].counts) & set(cells["stress"].counts)),
            "unmatched_control_groups": len(
                set(cells["control"].counts) - set(cells["stress"].counts)
            ),
            "unmatched_stress_groups": len(
                set(cells["stress"].counts) - set(cells["control"].counts)
            ),
        }

    routed = {probe.id: SampledProbeData() for probe in probes}
    for task in iter_tasks(paths):
        for probe in probes:
            for section, selector in _selectors(probe):
                if not matches(task, selector):
                    continue
                group = str(_task_field(task, _group_field(probe, section)))
                if group in selected[probe.id][section]:
                    getattr(routed[probe.id], section).append(task)
    return routed, selection_report


def _selectors(probe: Probe) -> list[tuple[str, dict[str, Any]]]:
    result = [
        ("train", probe.train),
        ("control", probe.control),
        ("stress", probe.stress),
    ]
    stress_train = probe.options.get("stress_train")
    if isinstance(stress_train, dict):
        result.append(("stress_train", stress_train))
    return result


def _limit(probe: Probe, training: bool, override: int | None) -> int | None:
    configured = probe.options.get("train_limit" if training else "eval_limit")
    values = [int(value) for value in (configured, override) if value is not None]
    return min(values) if values else None


def _group_field(probe: Probe, section: str) -> str:
    if section in {"train", "stress_train"}:
        return str(probe.options.get("train_pair_by", "probe_group_id"))
    return probe.pair_by


def _task_field(task: Task, field: str) -> object:
    if field.startswith("metadata."):
        return task.metadata.get(field.removeprefix("metadata."))
    return getattr(task, field)


def _rank(seed: int, salt: str, group: str) -> bytes:
    return hashlib.sha256(f"{seed}:{salt}:{group}".encode()).digest()


def _choose(
    groups: set[str],
    *,
    limit: int | None,
    seed: int,
    salt: str,
    strata: dict[str, str],
    declared: object,
) -> set[str]:
    if limit is None or limit >= len(groups):
        return set(groups)
    declared_strata = [str(item) for item in declared] if isinstance(declared, list) else []
    if not declared_strata:
        return set(sorted(groups, key=lambda group: _rank(seed, salt, group))[:limit])
    chosen: set[str] = set()
    base, remainder = divmod(limit, len(declared_strata))
    for index, stratum in enumerate(declared_strata):
        quota = base + int(index < remainder)
        candidates = [group for group in groups if strata.get(group) == stratum]
        chosen.update(
            sorted(candidates, key=lambda group: _rank(seed, f"{salt}:{stratum}", group))[:quota]
        )
    return chosen


def _validate_duplicates(probe: Probe, cells: dict[str, _Inventory]) -> None:
    sections = ["control", "stress"]
    if cells["stress_train"].counts:
        sections.extend(["train", "stress_train"])
    for section in sections:
        duplicates = [group for group, count in cells[section].counts.items() if count != 1]
        if duplicates:
            raise ValueError(
                f"{probe.id}:{section}: expected one row per group; "
                f"duplicates include {duplicates[:3]}"
            )
