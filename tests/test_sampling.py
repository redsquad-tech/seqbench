from __future__ import annotations

import csv
from pathlib import Path

from seqbench.sampling import sample_tasks
from seqbench.schema import CSV_COLUMNS, Task
from seqbench.specs import Probe


def test_augment_limit_does_not_shrink_base_after_materialization(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    base = [_task(f"train-{index}", "full", "train") for index in range(6)]
    auxiliary = [_task(f"proof-{index}", "credit", "train", group=f"g-{index}") for index in range(6)]
    evaluation = [_task("test", "full", "test")]
    _write(source, [*base, *auxiliary, *evaluation])
    probe = Probe(
        id="credit",
        version=1,
        property="credit_assignment",
        protocol="training_contrast",
        requires=("learn", "generate"),
        train={"split": "train", "variant": "full"},
        control={"split": "test", "variant": "full"},
        stress={"split": "test", "variant": "full"},
        primary_metric="normalized_exact_match",
        optional_metrics=(),
        difficulty_axes=(),
        pair_by="probe_group_id",
        options={
            "stress_train": {"split": "train", "variant": "credit"},
            "stress_train_mode": "augment",
            "train_limit": 5,
            "stress_train_limit": 2,
        },
    )
    routed, _ = sample_tasks(
        [source],
        [probe],
        sampling_seed=7,
        train_limit_override=None,
        eval_limit_override=None,
    )
    assert len(routed["credit"].train) == 5
    assert len(routed["credit"].stress_train) == 2

    materialized = tmp_path / "materialized.csv"
    _write(materialized, [*routed["credit"].train, *routed["credit"].stress_train, *evaluation])
    routed_again, _ = sample_tasks(
        [materialized],
        [probe],
        sampling_seed=7,
        train_limit_override=None,
        eval_limit_override=None,
    )
    assert len(routed_again["credit"].train) == 5
    assert len(routed_again["credit"].stress_train) == 2


def _task(task_id: str, variant: str, split: str, *, group: str | None = None) -> Task:
    index = task_id.rsplit("-", 1)[-1]
    return Task(
        id=task_id,
        source_example_id=task_id,
        probe_group_id=group or f"g-{index}",
        dataset="fixture",
        config="fixture",
        split=split,
        task="fixture",
        variant=variant,
        input=task_id,
        target="yes",
        acceptable_outputs=("yes",),
    )


def _write(path: Path, tasks: list[Task]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(task.to_csv() for task in tasks)
