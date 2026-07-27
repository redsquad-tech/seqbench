from __future__ import annotations

from datasets.transforms import (
    counterfactual_transform,
    noise_transform,
    nonce_transform,
    supervision_transform,
)

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


def test_counterfactual_keeps_probe_group_and_reverses_relation() -> None:
    source = task(
        input=(
            "Story:\nAlice is Bob's mother.\n\n"
            "Question:\nWhat is the relation of Alice to Bob?\n\nAnswer:"
        ),
        candidates=("mother", "son"),
        metadata={
            "query": ["Bob", "Alice"],
            "genders": "Alice:female,Bob:male",
        },
    )
    transformed = next(iter(counterfactual_transform(source)))
    assert transformed.probe_group_id == source.probe_group_id
    assert transformed.target == "son"
    assert "relation of Bob to Alice" in transformed.input


def test_nonce_counterfactual_uses_remapped_metadata() -> None:
    nonce = next(
        iter(
            nonce_transform(42)(
                task(
                    metadata={
                        "query": ["Bob", "Alice"],
                        "genders": {"Bob": "male"},
                    },
                    candidates=("mother", "son"),
                )
            )
        )
    )
    transformed = next(iter(counterfactual_transform(nonce)))
    assert transformed.variant == "nonce_counterfactual"
    assert "Alice" not in transformed.input
    assert transformed.target == "son"


def test_privileged_proof_context_is_one_to_one_and_keeps_target() -> None:
    source = task(
        dataset="proofwriter",
        metadata={"proof": "rule1 -> fact2"},
        target="true",
        acceptable_outputs=("true",),
    )
    transformed = list(supervision_transform(source))
    assert len(transformed) == 1
    assert transformed[0].target == source.target
    assert transformed[0].variant == "proof_context"
    assert "Training-only proof:" in transformed[0].input
