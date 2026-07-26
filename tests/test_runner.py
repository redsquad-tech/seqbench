from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from seqbench.compare import compare_runs
from seqbench.process import Algorithm, checkpoint_hash
from seqbench.runner import Runner, _correction_summary
from seqbench.schema import Task

ROOT = Path(__file__).resolve().parent.parent


def test_process_contract_does_not_mutate_input(tmp_path: Path) -> None:
    algorithm = Algorithm.load(ROOT / "tests/fixtures/adapter/algorithm.yaml")
    before = checkpoint_hash(algorithm.initial_model)
    model = tmp_path / "model"
    budget = {
        "persistent_model_bytes": 1_000_000,
        "train_wall_seconds": 10,
        "infer_wall_seconds_per_example": 1,
    }
    algorithm.learn(
        algorithm.initial_model,
        [{"id": "1", "input": "x", "target": "yes"}],
        model,
        budget=budget,
        seed=1,
    )
    responses, _ = algorithm.infer(
        model,
        [{"id": "x", "mode": "score", "input": "unknown", "value": "missing"}],
        budget=budget,
    )
    assert responses == [{"id": "x", "log_probability": None}]
    assert checkpoint_hash(algorithm.initial_model) == before


def test_golden_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "run"
    Runner(
        run_spec=ROOT / "specs/runs/smoke.yaml",
        algorithm_spec=ROOT / "tests/fixtures/adapter/algorithm.yaml",
        task_paths=[ROOT / "tests/fixtures/data/tasks.csv"],
        output=output,
    ).run()
    expected = {
        "manifest.json",
        "predictions.parquet",
        "metrics.json",
        "properties.json",
        "diagnostics.json",
        "resources.json",
        "report.md",
        "report.html",
        "curves",
    }
    assert expected <= {item.name for item in output.iterdir()}
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["probes"][0]["status"] == "UNCALIBRATED"
    assert pq.read_table(output / "predictions.parquet").num_rows == 4


def test_streaming_route_applies_overrides_before_retaining_rows(
    tmp_path: Path,
) -> None:
    runner = Runner(
        run_spec=ROOT / "specs/runs/smoke.yaml",
        algorithm_spec=ROOT / "tests/fixtures/adapter/algorithm.yaml",
        task_paths=[ROOT / "tests/fixtures/data/tasks.csv"],
        output=tmp_path / "unused",
        train_limit=1,
        eval_limit=1,
    )
    routed = runner.route_tasks()["retrieval.babi.full_vs_oracle"]
    assert len(routed.train) == 1
    assert len(routed.control) == 1
    assert len(routed.stress) == 1


def test_streaming_route_stratifies_evaluation_limit(tmp_path: Path) -> None:
    runner = Runner(
        run_spec=ROOT / "specs/runs/smoke.yaml",
        algorithm_spec=ROOT / "tests/fixtures/adapter/algorithm.yaml",
        task_paths=[ROOT / "tests/fixtures/data/tasks.csv"],
        output=tmp_path / "unused",
    )
    probe = replace(
        runner.probes[0],
        options={
            **runner.probes[0].options,
            "eval_limit": 5,
            "eval_stratify_by": "target",
            "eval_strata": ["a", "b"],
        },
    )
    destination: list[Task] = []
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for index, target in enumerate(["a"] * 5 + ["b"] * 5):
        task = Task(
            id=str(index),
            source_example_id=str(index),
            probe_group_id=str(index),
            dataset="test",
            config="",
            split="test",
            task="",
            variant="full",
            input="x",
            target=target,
            acceptable_outputs=(target,),
        )
        runner._append_routed(
            destination,
            task,
            probe,
            "control",
            seen,
            counts,
        )
    assert [task.target for task in destination].count("a") == 3
    assert [task.target for task in destination].count("b") == 2


def test_compare_aggregates_black_box_runs(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for output in (left, right):
        Runner(
            run_spec=ROOT / "specs/runs/smoke.yaml",
            algorithm_spec=ROOT / "tests/fixtures/adapter/algorithm.yaml",
            task_paths=[ROOT / "tests/fixtures/data/tasks.csv"],
            output=output,
        ).run()
    comparison = compare_runs(
        spec_path=ROOT / "specs/runs/smoke.yaml",
        labelled_runs={"learned": [left], "frozen": [right]},
        reference="learned",
        output=tmp_path / "comparison",
    )
    rows = json.loads((comparison / "comparison.json").read_text())["results"]
    assert {row["model"] for row in rows} == {"learned", "frozen"}
    assert all("primary" in row for row in rows)
    assert (comparison / "comparison.csv").is_file()
    assert (comparison / "comparison.md").is_file()


def test_correction_summary_uses_loss_direction() -> None:
    rows = []
    for stage, related, unrelated in [
        ("before", 4.0, 2.0),
        ("after", 3.0, 2.25),
    ]:
        rows.extend(
            [
                {
                    "stage": f"{stage}_related",
                    "seed": 0,
                    "probe_group_id": "related",
                    "capped_nll_bits": related,
                },
                {
                    "stage": f"{stage}_unrelated",
                    "seed": 0,
                    "probe_group_id": "unrelated",
                    "capped_nll_bits": unrelated,
                },
            ]
        )
    summary = _correction_summary(
        rows,
        metric="capped_nll_bits",
        replicates=10,
        seed=0,
    )
    assert summary["related_gain"] == 1.0
    assert summary["collateral_damage"] == 0.25
