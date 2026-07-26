#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def load(path: Path) -> dict:
    model = path / "model.json" if path.is_dir() else path
    return json.loads(model.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    path.mkdir(parents=True)
    (path / "model.json").write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def learn(args: argparse.Namespace) -> None:
    model = load(args.model_in)
    examples = list(rows(args.examples))
    mapping = dict(model.get("mapping", {}))
    counts = Counter(model.get("counts", {}))
    for example in examples:
        mapping[example["input"]] = example["target"]
        counts[example["target"]] += 1
    save(args.model_out, {"mapping": mapping, "counts": dict(counts)})


def infer(args: argparse.Namespace) -> None:
    model = load(args.model)
    mapping = model.get("mapping", {})
    counts = Counter(model.get("counts", {}))
    values = sorted(counts)
    total = sum(counts.values())
    with args.output.open("w", encoding="utf-8") as handle:
        for request in rows(args.requests):
            if request["mode"] == "generate":
                if args.strategy == "memorizer" and request["input"] in mapping:
                    output = mapping[request["input"]]
                elif args.strategy == "random" and values:
                    output = random.Random(request.get("seed", 0)).choice(values)
                else:
                    output = counts.most_common(1)[0][0] if counts else ""
                response = {"id": request["id"], "output": output}
            else:
                value = request["value"]
                probability = counts.get(value, 0) / total if total else 0.0
                if args.strategy == "memorizer" and request["input"] in mapping:
                    probability = 1.0 if mapping[request["input"]] == value else 0.0
                response = {
                    "id": request["id"],
                    "log_probability": math.log(probability) if probability else None,
                }
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument(
        "--strategy", choices=["random", "constant", "memorizer"], required=True
    )
    commands = root.add_subparsers(dest="operation", required=True)
    learn_parser = commands.add_parser("learn")
    learn_parser.add_argument("--model-in", type=Path, required=True)
    learn_parser.add_argument("--examples", type=Path, required=True)
    learn_parser.add_argument("--model-out", type=Path, required=True)
    learn_parser.add_argument("--budget", type=Path, required=True)
    learn_parser.add_argument("--seed", type=int, required=True)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--model", type=Path, required=True)
    infer_parser.add_argument("--requests", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--budget", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    learn(args) if args.operation == "learn" else infer(args)


if __name__ == "__main__":
    main()
