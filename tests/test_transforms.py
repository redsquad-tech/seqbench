from __future__ import annotations

from datasets.transforms import noise_transform, nonce_transform

from seqbench.schema import Task


def task(**changes):
    values = {
        "id": "clutrr:x:train:1:full",
        "source_example_id": "1",
        "probe_group_id": "clutrr:x:train:1",
        "dataset": "clutrr",
        "config": "x",
        "split": "train",
        "task": "kinship",
        "variant": "full",
        "input": "Alice is Bob's mother.",
        "target": "mother",
        "acceptable_outputs": ("mother",),
        "candidates": ("mother", "sister", "father"),
        "metadata": {"query": ["Bob", "Alice"], "genders": {"Alice": "female"}},
    }
    values.update(changes)
    return Task(**values)


def test_nonce_is_plain_seeded_transform() -> None:
    left = next(iter(nonce_transform(42)(task())))
    right = next(iter(nonce_transform(42)(task())))
    assert left.input == right.input
    assert "Alice" not in left.input
    assert left.probe_group_id == task().probe_group_id


def test_noise_levels_are_nested() -> None:
    rows = list(noise_transform(1)(task()))
    applied = [row.metadata["noise_rate"] for row in rows if row.metadata["noise_applied"]]
    assert applied == sorted(applied)
    assert len(rows) == 4
