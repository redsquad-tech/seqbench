from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import CSV_COLUMNS, Task


def _set_csv_limit() -> None:
    value = sys.maxsize
    while True:
        try:
            csv.field_size_limit(value)
            return
        except OverflowError:
            value //= 10


def iter_tasks(paths: Iterable[Path]) -> Iterator[Task]:
    _set_csv_limit()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError(f"{path}: expected seqbench CSV schema")
            for line, row in enumerate(reader, start=2):
                try:
                    yield Task.from_csv(row)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line}: {exc}") from exc


def matches(task: Task, selector: dict[str, object]) -> bool:
    for field, expected in selector.items():
        actual = (
            task.metadata.get(field.removeprefix("metadata."))
            if field.startswith("metadata.")
            else getattr(task, field)
        )
        if isinstance(expected, dict):
            if "gte" in expected and (
                not isinstance(actual, (int, float)) or actual < expected["gte"]
            ):
                return False
            if "lte" in expected and (
                not isinstance(actual, (int, float)) or actual > expected["lte"]
            ):
                return False
            if "neq" in expected and actual == expected["neq"]:
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
