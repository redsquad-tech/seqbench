from __future__ import annotations

from pathlib import Path

from seqbench.schema import Task
from seqbench.tasks import iter_tasks, matches

ROOT = Path(__file__).resolve().parent.parent


def test_fixture_csv_round_trip() -> None:
    tasks = list(iter_tasks([ROOT / "tests/fixtures/data/tasks.csv"]))
    assert len(tasks) == 5
    assert tasks[0].to_csv()["source_example_id"] == "1"
    assert tasks[0].target in tasks[0].acceptable_outputs


def test_selector_supports_metadata_ranges() -> None:
    task = Task(
        id="x",
        source_example_id="x",
        probe_group_id="x",
        dataset="proofwriter",
        config="x",
        split="test",
        task="deduction",
        variant="full",
        input="x",
        target="true",
        acceptable_outputs=("true",),
        metadata={"reasoning_depth": 3},
    )
    assert matches(task, {"metadata.reasoning_depth": {"gte": 3}})
    assert not matches(task, {"metadata.reasoning_depth": {"lte": 1}})
