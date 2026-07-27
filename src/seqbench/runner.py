from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .diagnostics import apply_diagnostics
from .metrics import (
    difficulty_curves,
    enrich_prediction,
    higher_is_better,
    normalize_text,
    output_novelty_summary,
    paired_bootstrap,
    unpaired_bootstrap,
)
from .process import Algorithm, checkpoint_bytes, copy_checkpoint
from .report import write_reports
from .sampling import sample_tasks
from .schema import Task
from .specs import Probe, PropertySpec, RunSpec
from .tasks import iter_tasks, matches
from .verdicts import aggregate_property, probe_status


@dataclass(slots=True)
class ProbeData:
    train: list[Task]
    stress_train: list[Task]
    control: list[Task]
    stress: list[Task]


class Runner:
    def __init__(
        self,
        *,
        run_spec: Path,
        algorithm_spec: Path,
        task_paths: list[Path],
        output: Path,
        seeds: tuple[int, ...] | None = None,
        train_limit: int | None = None,
        eval_limit: int | None = None,
    ):
        self.spec_path = run_spec.resolve()
        self.spec = RunSpec.load(run_spec)
        self.algorithm = Algorithm.load(algorithm_spec)
        self.probes = [Probe.load(path) for path in self.spec.probes]
        self.property_specs = [PropertySpec.load(path) for path in self.spec.properties]
        self.task_paths = [path.resolve() for path in task_paths]
        self.output = output.resolve()
        self.seeds = seeds or self.spec.seeds
        self.train_limit = train_limit
        self.eval_limit = eval_limit
        self.predictions: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self.curves: list[dict[str, Any]] = []
        self._learn_cache: dict[str, Path] = {}
        self.selection_report: dict[str, Any] = {}
        self.calibration = self._load_calibration()

    def _load_calibration(self) -> dict[str, Any]:
        path = self.spec.calibration
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def route_tasks(self) -> dict[str, ProbeData]:
        if self.spec.artifact_schema >= 2:
            routed, self.selection_report = sample_tasks(
                self.task_paths,
                self.probes,
                sampling_seed=self.spec.sampling_seed,
                train_limit_override=self.train_limit,
                eval_limit_override=self.eval_limit,
            )
            return {
                probe_id: ProbeData(
                    train=data.train,
                    stress_train=data.stress_train,
                    control=data.control,
                    stress=data.stress,
                )
                for probe_id, data in routed.items()
            }
        routed = {
            probe.id: ProbeData(train=[], stress_train=[], control=[], stress=[])
            for probe in self.probes
        }
        seen_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
        stratum_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for task in iter_tasks(self.task_paths):
            for probe in self.probes:
                data = routed[probe.id]
                if matches(task, probe.train):
                    self._append_routed(
                        data.train,
                        task,
                        probe,
                        "train",
                        seen_groups,
                        stratum_counts,
                    )
                stress_train_selector = probe.options.get("stress_train")
                if isinstance(stress_train_selector, dict) and matches(task, stress_train_selector):
                    self._append_routed(
                        data.stress_train,
                        task,
                        probe,
                        "stress_train",
                        seen_groups,
                        stratum_counts,
                    )
                if matches(task, probe.control):
                    self._append_routed(
                        data.control,
                        task,
                        probe,
                        "control",
                        seen_groups,
                        stratum_counts,
                    )
                if matches(task, probe.stress):
                    self._append_routed(
                        data.stress,
                        task,
                        probe,
                        "stress",
                        seen_groups,
                        stratum_counts,
                    )
        return routed

    def _append_routed(
        self,
        destination: list[Task],
        task: Task,
        probe: Probe,
        section: str,
        seen_groups: dict[tuple[str, str], set[str]],
        stratum_counts: dict[tuple[str, str, str], int],
    ) -> None:
        training = section in {"train", "stress_train"}
        configured = probe.options.get("train_limit" if training else "eval_limit")
        override = self.train_limit if training else self.eval_limit
        limit = (
            min(value for value in (configured, override) if value is not None)
            if configured is not None or override is not None
            else None
        )
        stratify_by = probe.options.get("train_stratify_by" if training else "eval_stratify_by")
        strata = probe.options.get("train_strata" if training else "eval_strata")
        if stratify_by and strata:
            value = _task_field(task, str(stratify_by))
            normalized_strata = [str(item) for item in strata]
            stratum = str(value)
            if stratum not in normalized_strata:
                return
            if limit is not None:
                index = normalized_strata.index(stratum)
                base, remainder = divmod(int(limit), len(normalized_strata))
                stratum_limit = base + int(index < remainder)
                key = (probe.id, section, stratum)
                if stratum_counts[key] >= stratum_limit:
                    return
                stratum_counts[key] += 1
            destination.append(task)
            return
        group_field = probe.options.get("train_limit_by") if training else None
        if group_field:
            group = str(getattr(task, str(group_field)))
            groups = seen_groups[(probe.id, section)]
            if group not in groups:
                if limit is not None and len(groups) >= int(limit):
                    return
                groups.add(group)
            destination.append(task)
        elif limit is None or len(destination) < int(limit):
            destination.append(task)

    def _learn(
        self,
        *,
        probe: Probe,
        tasks: list[Task],
        seed: int,
        budget: dict[str, Any],
        destination: Path,
        phase: str,
        budget_index: int = 0,
    ) -> Path:
        if not tasks or not self.algorithm.manifest["capabilities"]["learn"]:
            copy_checkpoint(self.algorithm.initial_model, destination)
            return destination
        fingerprint = _training_fingerprint(tasks, seed=seed, budget=budget)
        cached = self._learn_cache.get(fingerprint)
        if cached is not None:
            started = time.perf_counter()
            copy_checkpoint(cached, destination)
            copy_wall_seconds = time.perf_counter() - started
            self.resources.append(
                {
                    "probe": probe.id,
                    "seed": seed,
                    "budget_index": budget_index,
                    "phase": phase,
                    "operation": "learn",
                    "wall_seconds": copy_wall_seconds,
                    "copy_wall_seconds": copy_wall_seconds,
                    "cold_wall_seconds": 0.0,
                    "peak_ram_bytes": 0,
                    "checkpoint_bytes": checkpoint_bytes(destination),
                    "examples": len(tasks),
                    "cached": True,
                    "training_fingerprint": fingerprint,
                }
            )
            return destination
        examples = [{"id": task.id, "input": task.input, "target": task.target} for task in tasks]
        resource = self.algorithm.learn(
            self.algorithm.initial_model,
            examples,
            destination,
            budget=budget,
            seed=seed,
        )
        self.resources.append(
            {
                "probe": probe.id,
                "seed": seed,
                "budget_index": budget_index,
                "phase": phase,
                "operation": "learn",
                **_resource_fields(resource),
                "cold_wall_seconds": resource["wall_seconds"],
                "copy_wall_seconds": 0.0,
                "cached": False,
                "training_fingerprint": fingerprint,
            }
        )
        self._learn_cache[fingerprint] = destination
        return destination

    def _evaluate(
        self,
        *,
        probe: Probe,
        model: Path,
        tasks: list[Task],
        stage: str,
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
        train_values: set[str] | None = None,
        task_stages: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        request_map: dict[str, tuple[Task, str, str | None]] = {}
        can_generate = self.algorithm.manifest["capabilities"]["generate"]
        can_score = self.algorithm.manifest["capabilities"]["score"]
        for task in tasks:
            task_stage = task_stages.get(task.id, stage) if task_stages is not None else stage
            prefix = f"{probe.id}:{budget_index}:{seed}:{task_stage}:{task.id}"
            if can_generate:
                request_id = f"{prefix}:generate"
                requests.append(
                    {
                        "id": request_id,
                        "mode": "generate",
                        "input": task.input,
                        "seed": seed,
                        "max_output_tokens": int(probe.options.get("max_output_tokens", 128)),
                    }
                )
                request_map[request_id] = (task, "generate", None)
            if can_score:
                values = list(
                    dict.fromkeys((*task.acceptable_outputs, task.target, *task.candidates))
                )
                for index, value in enumerate(values):
                    request_id = f"{prefix}:score:{index}"
                    requests.append(
                        {
                            "id": request_id,
                            "mode": "score",
                            "input": task.input,
                            "value": value,
                        }
                    )
                    request_map[request_id] = (task, "score", value)
        if not requests:
            return []
        responses, resource = self.algorithm.infer(model, requests, budget=budget)
        self.resources.append(
            {
                "probe": probe.id,
                "seed": seed,
                "budget_index": budget_index,
                "phase": stage,
                "operation": "infer",
                **_resource_fields(resource),
            }
        )
        combined: dict[str, dict[str, Any]] = {}
        for response in responses:
            task, kind, value = request_map[response["id"]]
            row = combined.setdefault(
                task.id,
                {
                    "probe": probe.id,
                    "property": probe.property,
                    "seed": seed,
                    "budget_index": budget_index,
                    "stage": stage,
                    "task_id": task.id,
                    "source_example_id": task.source_example_id,
                    "probe_group_id": (
                        task.metadata.get(probe.pair_by.removeprefix("metadata."))
                        if probe.pair_by.startswith("metadata.")
                        else getattr(task, probe.pair_by)
                    ),
                    "dataset": task.dataset,
                    "config": task.config,
                    "split": task.split,
                    "task": task.task,
                    "variant": task.variant,
                    "input": task.input,
                    "target": task.target,
                    "acceptable_outputs": list(task.acceptable_outputs),
                    "metadata": task.metadata,
                    "output": "",
                    "target_log_probability": None,
                    "acceptable_log_probabilities": {},
                    "candidate_log_probabilities": {},
                    "target_seen_in_train": (
                        any(
                            normalize_text(item) in train_values for item in task.acceptable_outputs
                        )
                        if train_values is not None
                        else None
                    ),
                    "output_seen_in_train": None,
                },
            )
            row["stage"] = task_stages.get(task.id, stage) if task_stages is not None else stage
            if kind == "generate":
                row["output"] = response["output"]
                row["output_seen_in_train"] = (
                    normalize_text(response["output"]) in train_values
                    if train_values is not None
                    else None
                )
            else:
                assert value is not None
                probability = response["log_probability"]
                if value == task.target:
                    row["target_log_probability"] = probability
                if value in task.acceptable_outputs:
                    row["acceptable_log_probabilities"][value] = probability
                if value in task.candidates:
                    row["candidate_log_probabilities"][value] = probability
        enriched = [enrich_prediction(row) for row in combined.values()]
        self.predictions.extend(enriched)
        return enriched

    def _run_fixed(
        self,
        probe: Probe,
        data: ProbeData,
        *,
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
    ) -> None:
        model = self.output / "checkpoints" / (f"{probe.id}-budget{budget_index}-seed{seed}")
        model.parent.mkdir(parents=True, exist_ok=True)
        train_values = _training_values(data.train)
        if probe.protocol == "icl":
            copy_checkpoint(self.algorithm.initial_model, model)
            demonstrations = "\n".join(
                json.dumps(
                    {"input": task.input, "output": task.target},
                    ensure_ascii=False,
                )
                for task in data.train
            )
            control = [
                _with_input(task, f"Demonstrations:\n{demonstrations}\n\nRequest:\n{task.input}")
                for task in data.control
            ]
            stress = [
                _with_input(task, f"Demonstrations:\n{demonstrations}\n\nRequest:\n{task.input}")
                for task in data.stress
            ]
        else:
            self._learn(
                probe=probe,
                tasks=data.train,
                seed=seed,
                budget=budget,
                destination=model,
                phase="train",
                budget_index=budget_index,
            )
            control, stress = data.control, data.stress
        control_rows = self._evaluate(
            probe=probe,
            model=model,
            tasks=control,
            stage="control",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values,
        )
        stress_rows = self._evaluate(
            probe=probe,
            model=model,
            tasks=stress,
            stage="stress",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values,
        )
        if probe.primary_metric == "counterfactual_consistency":
            _annotate_counterfactual_consistency(control_rows, stress_rows)

    def _run_correction(
        self,
        probe: Probe,
        data: ProbeData,
        *,
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
    ) -> None:
        before = (
            self.output / "checkpoints" / (f"{probe.id}-budget{budget_index}-seed{seed}-before")
        )
        self._learn(
            probe=probe,
            tasks=data.train,
            seed=seed,
            budget=budget,
            destination=before,
            phase="base_train",
            budget_index=budget_index,
        )
        train_values = _training_values(data.train)
        if probe.options.get("paired_related"):
            self._run_paired_correction(
                probe,
                data,
                before=before,
                train_values=train_values,
                seed=seed,
                budget_index=budget_index,
                budget=budget,
            )
            return
        minimum_related = int(probe.options.get("minimum_related", 5))
        correction_task = next(
            (
                task
                for task in sorted(
                    data.control,
                    key=lambda item: _seeded_rank(self.spec.sampling_seed, item.id),
                )
                if sum(item.target == task.target and item.id != task.id for item in data.control)
                >= minimum_related
            ),
            None,
        )
        if correction_task is None:
            return
        related_limit = int(probe.options.get("related_limit", 100))
        related = sorted(
            (
                task
                for task in data.control
                if task.target == correction_task.target and task.id != correction_task.id
            ),
            key=lambda task: _seeded_rank(self.spec.sampling_seed, task.id),
        )[:related_limit]
        target_depth = correction_task.metadata.get("reasoning_depth")
        unrelated = sorted(
            (task for task in data.control if task.target != correction_task.target),
            key=lambda task: (
                _depth_distance(target_depth, task.metadata.get("reasoning_depth")),
                _seeded_rank(self.spec.sampling_seed, task.id),
            ),
        )[: len(related)]
        if not related or not unrelated:
            return
        self._evaluate(
            probe=probe,
            model=before,
            tasks=related,
            stage="before_related",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values,
        )
        self._evaluate(
            probe=probe,
            model=before,
            tasks=unrelated,
            stage="before_unrelated",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values,
        )
        after = before.with_name(before.name.replace("-before", "-after"))
        resource = self.algorithm.learn(
            before,
            [
                {
                    "id": correction_task.id,
                    "input": correction_task.input,
                    "target": correction_task.target,
                }
            ],
            after,
            budget=budget,
            seed=seed,
        )
        self.resources.append(
            {
                "probe": probe.id,
                "seed": seed,
                "budget_index": budget_index,
                "phase": "correction",
                "operation": "learn",
                **_resource_fields(resource),
            }
        )
        self._evaluate(
            probe=probe,
            model=after,
            tasks=related,
            stage="after_related",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values | {normalize_text(correction_task.target)},
        )
        self._evaluate(
            probe=probe,
            model=after,
            tasks=unrelated,
            stage="after_unrelated",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values | {normalize_text(correction_task.target)},
        )

    def _run_paired_correction(
        self,
        probe: Probe,
        data: ProbeData,
        *,
        before: Path,
        train_values: set[str],
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
    ) -> None:
        stress = {task.probe_group_id: task for task in data.stress}
        candidates = [
            task
            for task in sorted(
                data.control,
                key=lambda item: _seeded_rank(self.spec.sampling_seed, item.id),
            )
            if task.probe_group_id in stress
        ][: int(probe.options.get("correction_episodes", 6))]
        used_unrelated: set[str] = set()
        episodes: list[tuple[Task, Task, Task]] = []
        for episode, correction_task in enumerate(candidates):
            related = stress[correction_task.probe_group_id]
            unrelated = next(
                (
                    task
                    for task in sorted(
                        data.control,
                        key=lambda item: (
                            _depth_distance(
                                correction_task.metadata.get("reasoning_depth"),
                                item.metadata.get("reasoning_depth"),
                            ),
                            _seeded_rank(self.spec.sampling_seed + episode, item.id),
                        ),
                    )
                    if task.probe_group_id != correction_task.probe_group_id
                    and task.probe_group_id not in used_unrelated
                ),
                None,
            )
            if unrelated is None:
                continue
            used_unrelated.add(unrelated.probe_group_id)
            episodes.append((correction_task, related, unrelated))
        before_tasks = [task for _, related, unrelated in episodes for task in (related, unrelated)]
        self._evaluate(
            probe=probe,
            model=before,
            tasks=before_tasks,
            stage="before_correction",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=train_values,
            task_stages={
                task.id: stage
                for _, related, unrelated in episodes
                for task, stage in (
                    (related, "before_related"),
                    (unrelated, "before_unrelated"),
                )
            },
        )
        for episode, (correction_task, related, unrelated) in enumerate(episodes):
            after = before.with_name(f"{before.name}-episode{episode}-after")
            resource = self.algorithm.learn(
                before,
                [
                    {
                        "id": correction_task.id,
                        "input": correction_task.input,
                        "target": correction_task.target,
                    }
                ],
                after,
                budget=budget,
                seed=seed,
            )
            self.resources.append(
                {
                    "probe": probe.id,
                    "seed": seed,
                    "budget_index": budget_index,
                    "phase": f"correction_{episode}",
                    "operation": "learn",
                    **_resource_fields(resource),
                }
            )
            updated_values = train_values | {normalize_text(correction_task.target)}
            self._evaluate(
                probe=probe,
                model=after,
                tasks=[related, unrelated],
                stage="after_correction",
                seed=seed,
                budget_index=budget_index,
                budget=budget,
                train_values=updated_values,
                task_stages={
                    related.id: "after_related",
                    unrelated.id: "after_unrelated",
                },
            )

    def _run_training_contrast(
        self,
        probe: Probe,
        data: ProbeData,
        *,
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
    ) -> None:
        control_model = (
            self.output / "checkpoints" / (f"{probe.id}-budget{budget_index}-seed{seed}-control")
        )
        stress_model = (
            self.output / "checkpoints" / (f"{probe.id}-budget{budget_index}-seed{seed}-stress")
        )
        self._learn(
            probe=probe,
            tasks=data.train,
            seed=seed,
            budget=budget,
            destination=control_model,
            phase="control_train",
            budget_index=budget_index,
        )
        self._learn(
            probe=probe,
            tasks=data.stress_train,
            seed=seed,
            budget=budget,
            destination=stress_model,
            phase="stress_train",
            budget_index=budget_index,
        )
        self._evaluate(
            probe=probe,
            model=control_model,
            tasks=data.control,
            stage="control",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=_training_values(data.train),
        )
        self._evaluate(
            probe=probe,
            model=stress_model,
            tasks=data.stress,
            stage="stress",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
            train_values=_training_values(data.stress_train),
        )

    def run(self) -> Path:
        if self.output.exists():
            raise FileExistsError(f"output already exists: {self.output}")
        self.output.mkdir(parents=True)
        routed = self.route_tasks()
        total = len(self.spec.budgets) * len(self.seeds) * len(self.probes)
        completed = 0
        for budget_index, budget in enumerate(self.spec.budgets):
            for seed in self.seeds:
                for probe in self.probes:
                    completed += 1
                    started = time.perf_counter()
                    print(
                        f"[{completed}/{total}] seed={seed} probe={probe.id}",
                        flush=True,
                    )
                    if probe.protocol == "correction":
                        self._run_correction(
                            probe,
                            routed[probe.id],
                            seed=seed,
                            budget_index=budget_index,
                            budget=budget,
                        )
                    elif probe.protocol == "training_contrast":
                        self._run_training_contrast(
                            probe,
                            routed[probe.id],
                            seed=seed,
                            budget_index=budget_index,
                            budget=budget,
                        )
                    else:
                        self._run_fixed(
                            probe,
                            routed[probe.id],
                            seed=seed,
                            budget_index=budget_index,
                            budget=budget,
                        )
                    print(
                        f"[{completed}/{total}] completed in {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        return self.finish()

    def finish(self) -> Path:
        largest_budget = len(self.spec.budgets) - 1
        probe_results: dict[str, dict[str, Any]] = {}
        scaling: list[dict[str, Any]] = []
        for probe in self.probes:
            budget_summaries: list[dict[str, Any]] = []
            for budget_index, budget in enumerate(self.spec.budgets):
                budget_rows = [
                    row
                    for row in self.predictions
                    if row["probe"] == probe.id and row["budget_index"] == budget_index
                ]
                summary = self._primary_summary(probe, budget_rows)
                budget_summaries.append(summary)
                resources = [
                    item
                    for item in self.resources
                    if item.get("probe") == probe.id and item.get("budget_index") == budget_index
                ]
                scaling.append(
                    {
                        "probe": probe.id,
                        "budget_index": budget_index,
                        "budget": budget,
                        "summary": summary,
                        "resources": _summarize_resources(resources),
                    }
                )
            selected = [
                row
                for row in self.predictions
                if row["probe"] == probe.id and row["budget_index"] == largest_budget
            ]
            summary = budget_summaries[largest_budget]
            summary["metric"] = probe.primary_metric
            supported = all(
                bool(self.algorithm.manifest["capabilities"].get(capability))
                for capability in probe.requires
            )
            threshold = self.calibration.get("probes", {}).get(probe.id)
            status = probe_status(
                supported=supported,
                groups=int(summary.get("groups", 0)),
                calibrated_threshold=threshold,
                summary=summary,
            )
            probe_results[probe.id] = {
                "probe": probe.id,
                "property": probe.property,
                "metric": probe.primary_metric,
                **summary,
                "budgets": budget_summaries,
                "optional_metrics": self._optional_summaries(probe, selected),
                "output_novelty": output_novelty_summary(selected),
                "status": status,
            }
            self.curves.extend(
                {
                    "probe": probe.id,
                    **item,
                }
                for item in difficulty_curves(
                    selected,
                    metric=probe.primary_metric,
                    axes=probe.difficulty_axes,
                )
            )
        properties = [aggregate_property(spec, probe_results) for spec in self.property_specs]
        diagnoses = apply_diagnostics(probe_results, list(self.spec.diagnostics))
        failures = sorted(
            (
                row
                for row in self.predictions
                if row["normalized_exact_match"] == 0 and row["budget_index"] == largest_budget
            ),
            key=lambda row: row["capped_nll_bits"],
            reverse=True,
        )[: self.spec.report_failures]
        self._write_artifacts(
            list(probe_results.values()), properties, diagnoses, failures, scaling
        )
        return self.output

    def _primary_summary(self, probe: Probe, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if probe.protocol == "correction":
            return _correction_summary(
                rows,
                metric=probe.primary_metric,
                replicates=self.spec.bootstrap_replicates,
                seed=self.spec.bootstrap_seed,
            )
        bootstrap = paired_bootstrap if probe.options.get("paired", True) else unpaired_bootstrap
        return bootstrap(
            [row for row in rows if row["stage"] == "control"],
            [row for row in rows if row["stage"] == "stress"],
            metric=probe.primary_metric,
            replicates=self.spec.bootstrap_replicates,
            seed=self.spec.bootstrap_seed,
        )

    def _optional_summaries(self, probe: Probe, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if probe.protocol == "correction":
            control_stage, stress_stage = "before_related", "after_related"
        else:
            control_stage, stress_stage = "control", "stress"
        bootstrap = paired_bootstrap if probe.options.get("paired", True) else unpaired_bootstrap
        summaries: dict[str, Any] = {}
        for metric in probe.optional_metrics:
            actual = "capped_nll_bits" if metric == "nll_bits" else metric
            if probe.protocol == "correction":
                summary = _correction_summary(
                    rows,
                    metric=actual,
                    replicates=self.spec.bootstrap_replicates,
                    seed=self.spec.bootstrap_seed,
                )
            else:
                summary = bootstrap(
                    [row for row in rows if row["stage"] == control_stage],
                    [row for row in rows if row["stage"] == stress_stage],
                    metric=actual,
                    replicates=self.spec.bootstrap_replicates,
                    seed=self.spec.bootstrap_seed,
                )
            if summary.get("groups", 0):
                summaries[actual] = summary
        if "nll_bits" in probe.optional_metrics:
            summaries["zero_probability_rate"] = (
                _correction_summary(
                    rows,
                    metric="target_probability_zero",
                    replicates=self.spec.bootstrap_replicates,
                    seed=self.spec.bootstrap_seed,
                )
                if probe.protocol == "correction"
                else bootstrap(
                    [row for row in rows if row["stage"] == control_stage],
                    [row for row in rows if row["stage"] == stress_stage],
                    metric="target_probability_zero",
                    replicates=self.spec.bootstrap_replicates,
                    seed=self.spec.bootstrap_seed,
                )
            )
        return summaries

    def _write_artifacts(
        self,
        probe_results: list[dict[str, Any]],
        properties: list[dict[str, Any]],
        diagnoses: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        scaling: list[dict[str, Any]],
    ) -> None:
        serializable_predictions = [_parquet_row(row) for row in self.predictions]
        prediction_columns = set().union(*(row.keys() for row in serializable_predictions))
        serializable_predictions = [
            {key: row.get(key) for key in sorted(prediction_columns)}
            for row in serializable_predictions
        ]
        pq.write_table(
            pa.Table.from_pylist(serializable_predictions),
            self.output / "predictions.parquet",
        )
        curves = self.output / "curves"
        curves.mkdir()
        pq.write_table(
            pa.Table.from_pylist(self.curves or [{"probe": "", "axis": ""}]),
            curves / "difficulty.parquet",
        )
        _write_json(self.output / "metrics.json", {"probes": probe_results})
        _write_json(self.output / "properties.json", {"properties": properties})
        _write_json(self.output / "diagnostics.json", {"diagnostics": diagnoses})
        _write_json(self.output / "resources.json", {"resources": self.resources})
        _write_json(self.output / "scaling.json", {"cells": scaling})
        _write_json(self.output / "selection.json", self.selection_report)
        _write_json(
            self.output / "resolved_specs.json",
            {
                "run": _load_text_specs([self.spec_path]),
                "probes": _load_text_specs(list(self.spec.probes)),
                "properties": _load_text_specs(list(self.spec.properties)),
                "diagnostics": _load_text_specs(list(self.spec.diagnostics)),
            },
        )
        _write_json(
            self.output / "manifest.json",
            {
                "run": self.spec.id,
                "run_version": self.spec.version,
                "artifact_schema": self.spec.artifact_schema,
                "sampling_seed": self.spec.sampling_seed,
                "algorithm": self.algorithm.manifest,
                "probe_contracts": [
                    {
                        "probe": probe.id,
                        "version": probe.version,
                        "primary_metric": probe.primary_metric,
                    }
                    for probe in self.probes
                ],
                "seqbench_git_commit": _git_commit(),
                "tasks": [str(path) for path in self.task_paths],
                "seeds": list(self.seeds),
                "train_limit_override": self.train_limit,
                "eval_limit_override": self.eval_limit,
                "budgets": list(self.spec.budgets),
                "calibrated": bool(self.calibration),
            },
        )
        write_reports(
            self.output,
            run_id=self.spec.id,
            algorithm=self.algorithm.manifest,
            properties=properties,
            probe_results=probe_results,
            diagnoses=diagnoses,
            failures=failures,
        )


def _with_input(task: Task, value: str) -> Task:
    return Task(
        id=task.id,
        source_example_id=task.source_example_id,
        probe_group_id=task.probe_group_id,
        dataset=task.dataset,
        config=task.config,
        split=task.split,
        task=task.task,
        variant=task.variant,
        input=value,
        target=task.target,
        acceptable_outputs=task.acceptable_outputs,
        candidates=task.candidates,
        metadata=task.metadata,
    )


def _correction_summary(
    rows: list[dict[str, Any]], *, metric: str, replicates: int, seed: int
) -> dict[str, Any]:
    related = paired_bootstrap(
        [row for row in rows if row["stage"] == "before_related"],
        [row for row in rows if row["stage"] == "after_related"],
        metric=metric,
        replicates=replicates,
        seed=seed,
    )
    if not related.get("groups", 0):
        return {"groups": 0}
    unrelated = paired_bootstrap(
        [row for row in rows if row["stage"] == "before_unrelated"],
        [row for row in rows if row["stage"] == "after_unrelated"],
        metric=metric,
        replicates=replicates,
        seed=seed,
    )
    direction = 1.0 if higher_is_better(metric) else -1.0
    related["related_gain"] = direction * related["gap"]["score"]
    related["related_gain_ci95"] = [direction * item for item in related["gap"]["ci95"]][
        :: 1 if direction > 0 else -1
    ]
    if unrelated.get("groups", 0):
        damage_direction = -direction
        related["collateral_damage"] = damage_direction * unrelated["gap"]["score"]
        related["collateral_damage_ci95"] = [
            damage_direction * item for item in unrelated["gap"]["ci95"]
        ][:: 1 if damage_direction > 0 else -1]
    else:
        related["collateral_damage"] = 0.0
        related["collateral_damage_ci95"] = [0.0, 0.0]
    return related


def _seeded_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _depth_distance(left: object, right: object) -> float:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return math.inf


def _task_field(task: Task, field: str) -> object:
    if field.startswith("metadata."):
        return task.metadata.get(field.removeprefix("metadata."))
    return getattr(task, field)


def _training_values(tasks: list[Task]) -> set[str]:
    return {normalize_text(task.target) for task in tasks}


def _annotate_counterfactual_consistency(
    control_rows: list[dict[str, Any]], stress_rows: list[dict[str, Any]]
) -> None:
    control = {str(row["probe_group_id"]): row for row in control_rows}
    for row in stress_rows:
        source = control.get(str(row["probe_group_id"]))
        mapping = row.get("metadata", {}).get("counterfactual_output_map", {})
        expected = (
            mapping.get(normalize_text(source["output"]))
            if source is not None and isinstance(mapping, dict)
            else None
        )
        row["counterfactual_consistency"] = float(
            expected is not None and normalize_text(row["output"]) == normalize_text(expected)
        )
    for row in control_rows:
        row["counterfactual_consistency"] = row["normalized_exact_match"]


def _training_fingerprint(tasks: list[Task], *, seed: int, budget: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode())
    digest.update(json.dumps(budget, sort_keys=True, separators=(",", ":")).encode())
    for task in tasks:
        for value in (task.id, task.input, task.target):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _resource_fields(resource: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in resource.items() if key not in {"stdout", "stderr"}}


def _parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["acceptable_outputs_json"] = json.dumps(
        result.pop("acceptable_outputs"), ensure_ascii=False
    )
    result["metadata_json"] = json.dumps(result.pop("metadata"), ensure_ascii=False)
    result["candidate_log_probabilities_json"] = json.dumps(
        result.pop("candidate_log_probabilities"), ensure_ascii=False
    )
    result["acceptable_log_probabilities_json"] = json.dumps(
        result.pop("acceptable_log_probabilities"), ensure_ascii=False
    )
    for key, value in list(result.items()):
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summarize_resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(rows),
        "wall_seconds_total": sum(float(row.get("wall_seconds", 0.0)) for row in rows),
        "peak_ram_bytes": max((int(row.get("peak_ram_bytes", 0)) for row in rows), default=0),
        "checkpoint_bytes_max": max(
            (int(row.get("checkpoint_bytes", 0)) for row in rows), default=0
        ),
        "cache_hits": sum(bool(row.get("cached")) for row in rows),
    }


def _load_text_specs(paths: list[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "text": path.read_text(encoding="utf-8")} for path in paths]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
