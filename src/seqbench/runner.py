from __future__ import annotations

import json
import math
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
    paired_bootstrap,
    unpaired_bootstrap,
)
from .process import Algorithm, copy_checkpoint
from .report import write_reports
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


def _limit(items: list[Task], value: object) -> list[Task]:
    return items if value is None else items[: int(value)]


class Runner:
    def __init__(
        self,
        *,
        run_spec: Path,
        algorithm_spec: Path,
        task_paths: list[Path],
        output: Path,
    ):
        self.spec = RunSpec.load(run_spec)
        self.algorithm = Algorithm.load(algorithm_spec)
        self.probes = [Probe.load(path) for path in self.spec.probes]
        self.property_specs = [
            PropertySpec.load(path) for path in self.spec.properties
        ]
        self.task_paths = [path.resolve() for path in task_paths]
        self.output = output.resolve()
        self.predictions: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self.curves: list[dict[str, Any]] = []
        self.calibration = self._load_calibration()

    def _load_calibration(self) -> dict[str, Any]:
        path = self.spec.calibration
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def route_tasks(self) -> dict[str, ProbeData]:
        routed = {
            probe.id: ProbeData(train=[], stress_train=[], control=[], stress=[])
            for probe in self.probes
        }
        for task in iter_tasks(self.task_paths):
            for probe in self.probes:
                data = routed[probe.id]
                if matches(task, probe.train):
                    data.train.append(task)
                stress_train_selector = probe.options.get("stress_train")
                if isinstance(stress_train_selector, dict) and matches(
                    task, stress_train_selector
                ):
                    data.stress_train.append(task)
                if matches(task, probe.control):
                    data.control.append(task)
                if matches(task, probe.stress):
                    data.stress.append(task)
        for probe in self.probes:
            data = routed[probe.id]
            data.train = _limit(data.train, probe.options.get("train_limit"))
            data.stress_train = _limit(
                data.stress_train, probe.options.get("train_limit")
            )
            data.control = _limit(data.control, probe.options.get("eval_limit"))
            data.stress = _limit(data.stress, probe.options.get("eval_limit"))
        return routed

    def _learn(
        self,
        *,
        probe: Probe,
        tasks: list[Task],
        seed: int,
        budget: dict[str, Any],
        destination: Path,
        phase: str,
    ) -> Path:
        if not tasks or not self.algorithm.manifest["capabilities"]["learn"]:
            copy_checkpoint(self.algorithm.initial_model, destination)
            return destination
        examples = [
            {"id": task.id, "input": task.input, "target": task.target}
            for task in tasks
        ]
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
                "phase": phase,
                "operation": "learn",
                **_resource_fields(resource),
            }
        )
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
    ) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        request_map: dict[str, tuple[Task, str, str | None]] = {}
        can_generate = self.algorithm.manifest["capabilities"]["generate"]
        can_score = self.algorithm.manifest["capabilities"]["score"]
        for task in tasks:
            prefix = f"{probe.id}:{budget_index}:{seed}:{stage}:{task.id}"
            if can_generate:
                request_id = f"{prefix}:generate"
                requests.append(
                    {
                        "id": request_id,
                        "mode": "generate",
                        "input": task.input,
                        "seed": seed,
                        "max_output_tokens": int(
                            probe.options.get("max_output_tokens", 128)
                        ),
                    }
                )
                request_map[request_id] = (task, "generate", None)
            if can_score:
                values = list(dict.fromkeys((task.target, *task.candidates)))
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
                    "candidate_log_probabilities": {},
                },
            )
            if kind == "generate":
                row["output"] = response["output"]
            else:
                assert value is not None
                probability = response["log_probability"]
                if value == task.target:
                    row["target_log_probability"] = probability
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
        model = self.output / "checkpoints" / (
            f"{probe.id}-budget{budget_index}-seed{seed}"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
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
            )
            control, stress = data.control, data.stress
        self._evaluate(
            probe=probe,
            model=model,
            tasks=control,
            stage="control",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )
        self._evaluate(
            probe=probe,
            model=model,
            tasks=stress,
            stage="stress",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )

    def _run_correction(
        self,
        probe: Probe,
        data: ProbeData,
        *,
        seed: int,
        budget_index: int,
        budget: dict[str, Any],
    ) -> None:
        before = self.output / "checkpoints" / (
            f"{probe.id}-budget{budget_index}-seed{seed}-before"
        )
        self._learn(
            probe=probe,
            tasks=data.train,
            seed=seed,
            budget=budget,
            destination=before,
            phase="base_train",
        )
        related = self._evaluate(
            probe=probe,
            model=before,
            tasks=data.control,
            stage="before_related",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )
        self._evaluate(
            probe=probe,
            model=before,
            tasks=data.stress,
            stage="before_unrelated",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )
        corrections = [
            {
                "id": row["task_id"],
                "input": row["input"],
                "target": row["target"],
            }
            for row in related
            if row["normalized_exact_match"] == 0
        ][: int(probe.options.get("corrections", 1))]
        if not corrections:
            return
        after = before.with_name(before.name.replace("-before", "-after"))
        resource = self.algorithm.learn(
            before,
            corrections,
            after,
            budget=budget,
            seed=seed,
        )
        self.resources.append(
            {
                "probe": probe.id,
                "seed": seed,
                "phase": "correction",
                "operation": "learn",
                **_resource_fields(resource),
            }
        )
        self._evaluate(
            probe=probe,
            model=after,
            tasks=data.control,
            stage="after_related",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )
        self._evaluate(
            probe=probe,
            model=after,
            tasks=data.stress,
            stage="after_unrelated",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
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
        control_model = self.output / "checkpoints" / (
            f"{probe.id}-budget{budget_index}-seed{seed}-control"
        )
        stress_model = self.output / "checkpoints" / (
            f"{probe.id}-budget{budget_index}-seed{seed}-stress"
        )
        self._learn(
            probe=probe,
            tasks=data.train,
            seed=seed,
            budget=budget,
            destination=control_model,
            phase="control_train",
        )
        self._learn(
            probe=probe,
            tasks=data.stress_train,
            seed=seed,
            budget=budget,
            destination=stress_model,
            phase="stress_train",
        )
        self._evaluate(
            probe=probe,
            model=control_model,
            tasks=data.control,
            stage="control",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )
        self._evaluate(
            probe=probe,
            model=stress_model,
            tasks=data.stress,
            stage="stress",
            seed=seed,
            budget_index=budget_index,
            budget=budget,
        )

    def run(self) -> Path:
        if self.output.exists():
            raise FileExistsError(f"output already exists: {self.output}")
        self.output.mkdir(parents=True)
        routed = self.route_tasks()
        for budget_index, budget in enumerate(self.spec.budgets):
            for seed in self.spec.seeds:
                for probe in self.probes:
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
        return self.finish()

    def finish(self) -> Path:
        largest_budget = len(self.spec.budgets) - 1
        probe_results: dict[str, dict[str, Any]] = {}
        for probe in self.probes:
            selected = [
                row
                for row in self.predictions
                if row["probe"] == probe.id and row["budget_index"] == largest_budget
            ]
            if probe.protocol == "correction":
                summary = _correction_summary(selected)
            else:
                bootstrap = (
                    paired_bootstrap
                    if probe.options.get("paired", True)
                    else unpaired_bootstrap
                )
                summary = bootstrap(
                    [row for row in selected if row["stage"] == "control"],
                    [row for row in selected if row["stage"] == "stress"],
                    metric=probe.primary_metric,
                    replicates=self.spec.bootstrap_replicates,
                    seed=self.spec.bootstrap_seed,
                )
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
        properties = [
            aggregate_property(spec, probe_results) for spec in self.property_specs
        ]
        diagnoses = apply_diagnostics(
            probe_results, list(self.spec.diagnostics)
        )
        failures = sorted(
            (
                row
                for row in self.predictions
                if row["normalized_exact_match"] == 0
                and row["budget_index"] == largest_budget
            ),
            key=lambda row: row["capped_nll_bits"],
            reverse=True,
        )[: self.spec.report_failures]
        self._write_artifacts(
            list(probe_results.values()), properties, diagnoses, failures
        )
        return self.output

    def _write_artifacts(
        self,
        probe_results: list[dict[str, Any]],
        properties: list[dict[str, Any]],
        diagnoses: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> None:
        serializable_predictions = [_parquet_row(row) for row in self.predictions]
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
        _write_json(
            self.output / "manifest.json",
            {
                "run": self.spec.id,
                "algorithm": self.algorithm.manifest,
                "tasks": [str(path) for path in self.task_paths],
                "seeds": list(self.spec.seeds),
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


def _correction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        cells[(row["stage"], row["probe_group_id"])].append(
            row["normalized_exact_match"]
        )
    related_groups = sorted(
        {group for stage, group in cells if stage == "before_related"}
        & {group for stage, group in cells if stage == "after_related"}
    )
    unrelated_groups = sorted(
        {group for stage, group in cells if stage == "before_unrelated"}
        & {group for stage, group in cells if stage == "after_unrelated"}
    )
    if not related_groups:
        return {"groups": 0}
    before = sum(
        sum(cells[("before_related", group)]) / len(cells[("before_related", group)])
        for group in related_groups
    ) / len(related_groups)
    after = sum(
        sum(cells[("after_related", group)]) / len(cells[("after_related", group)])
        for group in related_groups
    ) / len(related_groups)
    damage = 0.0
    if unrelated_groups:
        damage = sum(
            (
                sum(cells[("before_unrelated", group)])
                / len(cells[("before_unrelated", group)])
            )
            - (
                sum(cells[("after_unrelated", group)])
                / len(cells[("after_unrelated", group)])
            )
            for group in unrelated_groups
        ) / len(unrelated_groups)
    return {
        "groups": len(related_groups),
        "control": {"score": before, "ci95": [before, before]},
        "stress": {"score": after, "ci95": [after, after]},
        "gap": {"score": after - before, "ci95": [after - before, after - before]},
        "retention": after / before if before > 0 else 0.0,
        "related_gain": after - before,
        "collateral_damage": damage,
    }


def _resource_fields(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in resource.items()
        if key not in {"stdout", "stderr"}
    }


def _parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["acceptable_outputs_json"] = json.dumps(
        result.pop("acceptable_outputs"), ensure_ascii=False
    )
    result["metadata_json"] = json.dumps(result.pop("metadata"), ensure_ascii=False)
    result["candidate_log_probabilities_json"] = json.dumps(
        result.pop("candidate_log_probabilities"), ensure_ascii=False
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
