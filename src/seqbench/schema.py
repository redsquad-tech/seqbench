from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CSV_COLUMNS = [
    "id",
    "source_example_id",
    "probe_group_id",
    "dataset",
    "config",
    "split",
    "task",
    "variant",
    "input",
    "expected_output",
    "acceptable_outputs_json",
    "answer_candidates_json",
    "metadata_json",
]


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    source_example_id: str
    probe_group_id: str
    dataset: str
    config: str
    split: str
    task: str
    variant: str
    input: str
    target: str
    acceptable_outputs: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> Task:
        acceptable = tuple(json.loads(row["acceptable_outputs_json"]))
        candidates = (
            tuple(json.loads(row["answer_candidates_json"]))
            if row["answer_candidates_json"]
            else ()
        )
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        if not acceptable or row["expected_output"] not in acceptable:
            raise ValueError(f"{row.get('id', '<unknown>')}: target is not acceptable")
        return cls(
            id=row["id"],
            source_example_id=row["source_example_id"],
            probe_group_id=row["probe_group_id"],
            dataset=row["dataset"],
            config=row["config"],
            split=row["split"],
            task=row["task"],
            variant=row["variant"],
            input=row["input"],
            target=row["expected_output"],
            acceptable_outputs=acceptable,
            candidates=candidates,
            metadata=metadata,
        )

    def to_csv(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source_example_id": self.source_example_id,
            "probe_group_id": self.probe_group_id,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "task": self.task,
            "variant": self.variant,
            "input": self.input,
            "expected_output": self.target,
            "acceptable_outputs_json": json.dumps(
                self.acceptable_outputs or (self.target,),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "answer_candidates_json": (
                json.dumps(self.candidates, ensure_ascii=False, separators=(",", ":"))
                if self.candidates
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

