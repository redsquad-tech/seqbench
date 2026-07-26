from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from seqbench.process import Algorithm, checkpoint_hash
from seqbench.runner import Runner

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

