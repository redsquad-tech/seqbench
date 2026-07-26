from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import write_calibration
from .runner import Runner
from .tasks import iter_tasks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="seqbench")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a YAML run specification")
    run.add_argument("spec", type=Path)
    run.add_argument("--algorithm", required=True, type=Path)
    run.add_argument("--tasks", action="append", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate-csv", help="stream and validate task CSV")
    validate.add_argument("tasks", nargs="+", type=Path)
    calibrate = commands.add_parser(
        "calibrate", help="derive thresholds from weak and strong reference runs"
    )
    calibrate.add_argument("--weak", action="append", required=True, type=Path)
    calibrate.add_argument("--strong", action="append", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        output = Runner(
            run_spec=args.spec,
            algorithm_spec=args.algorithm,
            task_paths=args.tasks,
            output=args.output,
        ).run()
        print(output / "report.md")
        return 0
    if args.command == "validate-csv":
        counts: dict[str, int] = {}
        total = 0
        for task in iter_tasks(args.tasks):
            counts[task.dataset] = counts.get(task.dataset, 0) + 1
            total += 1
        print(f"{total} tasks")
        for dataset, count in sorted(counts.items()):
            print(f"{dataset}: {count}")
        return 0
    if args.command == "calibrate":
        write_calibration(args.output, args.weak, args.strong)
        print(args.output)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
