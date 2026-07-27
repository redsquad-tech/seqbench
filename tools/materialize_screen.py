#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from seqbench.sampling import sample_tasks
from seqbench.schema import CSV_COLUMNS, Task
from seqbench.specs import Probe, RunSpec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the fixed rows needed by one screen specification."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tasks", nargs="+", type=Path)
    args = parser.parse_args()
    spec = RunSpec.load(args.spec)
    probes = [Probe.load(path) for path in spec.probes]
    routed, _ = sample_tasks(
        args.tasks,
        probes,
        sampling_seed=spec.sampling_seed,
        train_limit_override=None,
        eval_limit_override=None,
    )
    selected: dict[str, Task] = {}
    for data in routed.values():
        for section in ("train", "stress_train", "control", "stress"):
            for task in getattr(data, section):
                selected[task.id] = replace(
                    task,
                    metadata={
                        key: value
                        for key, value in task.metadata.items()
                        if key not in {"proof", "nonce_mapping"}
                    },
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for task in sorted(selected.values(), key=lambda item: item.id):
            writer.writerow(task.to_csv())
    print(f"{args.output}: {len(selected):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
