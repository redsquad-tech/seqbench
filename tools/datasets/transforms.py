from __future__ import annotations

import ast
import csv
import random
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

from seqbench.schema import CSV_COLUMNS, Task


def iter_csv(path: Path) -> Iterator[Task]:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path}: expected seqbench CSV schema")
        for row in reader:
            yield Task.from_csv(row)


def write_transformed(
    source: Path,
    output: Path,
    transform: Callable[[Task], Iterator[Task]],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for task in iter_csv(source):
            for transformed in transform(task):
                writer.writerow(transformed.to_csv())
                count += 1
    return count


def derived(task: Task, variant: str, **changes: object) -> Task:
    return replace(
        task,
        id=f"{task.probe_group_id}:{variant}",
        variant=variant,
        **changes,
    )


def entity_names(task: Task) -> list[str]:
    names: set[str] = set()
    query = _literal(task.metadata.get("query"))
    if isinstance(query, (list, tuple)):
        names.update(str(value) for value in query[:2])
    genders = _literal(task.metadata.get("genders"))
    if isinstance(genders, dict):
        names.update(str(value) for value in genders)
    if names:
        return sorted(names)
    names.update(re.findall(r"\b[A-Z][a-z]+\b", task.input))
    names.difference_update({"Context", "Question", "Answer", "What"})
    return sorted(names)


def nonce_transform(seed: int) -> Callable[[Task], Iterator[Task]]:
    def transform(task: Task) -> Iterator[Task]:
        if task.dataset != "clutrr" or task.variant != "full":
            return
        names = entity_names(task)
        rng = random.Random(f"{seed}:{task.probe_group_id}")
        rng.shuffle(names)
        mapping = {name: f"ent_{index:04d}" for index, name in enumerate(names)}
        value = task.input
        for name in sorted(mapping, key=len, reverse=True):
            value = re.sub(rf"\b{re.escape(name)}\b", mapping[name], value)
        metadata: dict[str, object] = {
            **task.metadata,
            "nonce_seed": seed,
            "nonce_entity_count": len(mapping),
            "nonce_mapping": mapping,
        }
        query = _literal(task.metadata.get("query"))
        if isinstance(query, (list, tuple)):
            metadata["query"] = [mapping.get(str(item), str(item)) for item in query]
        genders = _gender_map(task.metadata.get("genders"))
        if genders:
            metadata["genders"] = {
                mapping.get(name, name): gender for name, gender in genders.items()
            }
        yield derived(task, "nonce", input=value, metadata=metadata)

    return transform


def counterfactual_transform(task: Task) -> Iterator[Task]:
    if task.dataset != "clutrr" or task.variant not in {"full", "nonce"}:
        return
    query = _literal(task.metadata.get("query"))
    if not isinstance(query, (list, tuple)) or len(query) < 2:
        return
    first, second = str(query[0]), str(query[1])
    genders = _gender_map(task.metadata.get("genders"))
    target = _inverse_relation(task.target, genders.get(first, ""))
    if target is None or (task.candidates and target not in task.candidates):
        return
    question = f"What is the relation of {first} to {second}?"
    value = re.sub(
        r"Question:\n.*?\n\nAnswer:",
        f"Question:\n{question}\n\nAnswer:",
        task.input,
        flags=re.DOTALL,
    )
    variant = "nonce_counterfactual" if task.variant == "nonce" else "counterfactual"
    output_map = {
        candidate: inverse
        for candidate in task.candidates
        if (inverse := _inverse_relation(candidate, genders.get(first, ""))) is not None
    }
    yield derived(
        task,
        variant,
        input=value,
        target=target,
        acceptable_outputs=(target,),
        metadata={
            **task.metadata,
            "query": [second, first],
            "counterfactual_output_map": output_map,
        },
    )


def noise_transform(seed: int) -> Callable[[Task], Iterator[Task]]:
    rng = random.Random(seed)
    rates = (
        ("noise_05", 0.05),
        ("noise_10", 0.10),
        ("noise_20", 0.20),
        ("noise_40", 0.40),
    )

    def transform(task: Task) -> Iterator[Task]:
        if task.split != "train" or task.variant != "full" or not task.candidates:
            return
        alternatives = [value for value in task.candidates if value != task.target]
        if not alternatives:
            return
        draw = rng.random()
        wrong = alternatives[rng.randrange(len(alternatives))]
        for variant, rate in rates:
            applied = draw < rate
            target = wrong if applied else task.target
            yield derived(
                task,
                variant,
                target=target,
                acceptable_outputs=(target,),
                metadata={
                    **task.metadata,
                    "noise_rate": rate,
                    "noise_applied": applied,
                    "clean_expected_output": task.target,
                },
            )

    return transform


def supervision_transform(task: Task) -> Iterator[Task]:
    if task.dataset != "proofwriter" or task.split != "train":
        return
    proof = task.metadata.get("proof")
    if not proof:
        return
    yield derived(
        task,
        "proof_context",
        input=f"{task.input}\n\nTraining-only proof:\n{proof}",
        metadata={**task.metadata, "supervision_role": "privileged_proof_context"},
    )


def _literal(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _gender_map(value: object) -> dict[str, str]:
    if isinstance(value, str) and ":" in value and not value.lstrip().startswith(("{", "[")):
        value = {
            item.split(":", 1)[0].strip(): item.split(":", 1)[1].strip()
            for item in value.split(",")
            if ":" in item
        }
    parsed = _literal(value)
    if not isinstance(parsed, dict):
        return {}
    return {
        str(name): ("male" if str(gender).lower() in {"m", "male", "man", "boy"} else "female")
        for name, gender in parsed.items()
    }


def _inverse_relation(relation: str, gender: str) -> str | None:
    pairs = {
        "father": ("son", "daughter"),
        "mother": ("son", "daughter"),
        "son": ("father", "mother"),
        "daughter": ("father", "mother"),
        "grandfather": ("grandson", "granddaughter"),
        "grandmother": ("grandson", "granddaughter"),
        "grandson": ("grandfather", "grandmother"),
        "granddaughter": ("grandfather", "grandmother"),
        "uncle": ("nephew", "niece"),
        "aunt": ("nephew", "niece"),
        "nephew": ("uncle", "aunt"),
        "niece": ("uncle", "aunt"),
        "brother": ("brother", "sister"),
        "sister": ("brother", "sister"),
        "husband": ("husband", "wife"),
        "wife": ("husband", "wife"),
        "father-in-law": ("son-in-law", "daughter-in-law"),
        "mother-in-law": ("son-in-law", "daughter-in-law"),
        "son-in-law": ("father-in-law", "mother-in-law"),
        "daughter-in-law": ("father-in-law", "mother-in-law"),
    }
    values = pairs.get(relation)
    if values is None or gender not in {"male", "female"}:
        return None
    return values[0] if gender == "male" else values[1]
