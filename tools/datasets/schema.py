from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from seqbench.schema import CSV_COLUMNS


@dataclass(frozen=True, slots=True)
class TaskRow:
    id: str
    dataset: str
    config: str
    split: str
    task: str
    variant: str
    probe_group_id: str
    input: str
    expected_output: str
    source_example_id: str = ""
    acceptable_outputs: list[str] | None = None
    answer_candidates: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csv_dict(self) -> dict[str, str]:
        source_id = self.source_example_id or str(
            self.metadata.get("source_id", self.id)
        )
        acceptable = self.acceptable_outputs or [self.expected_output]
        return {
            "id": self.id,
            "source_example_id": source_id,
            "probe_group_id": self.probe_group_id,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "task": self.task,
            "variant": self.variant,
            "input": self.input,
            "expected_output": self.expected_output,
            "acceptable_outputs_json": json.dumps(
                acceptable, ensure_ascii=False, separators=(",", ":")
            ),
            "answer_candidates_json": (
                json.dumps(
                    self.answer_candidates,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if self.answer_candidates is not None
                else ""
            ),
            "metadata_json": json.dumps(
                self.metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        }


__all__ = ["CSV_COLUMNS", "TaskRow"]

