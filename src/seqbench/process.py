from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml

from .jsonl import read_jsonl, write_jsonl


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in checkpoint_files(path):
        digest.update(item.relative_to(path).as_posix().encode() if path.is_dir() else item.name.encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in checkpoint_files(path))


def copy_checkpoint(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _rss_tree(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.Error, ProcessLookupError):
        return 0


def run_measured(command: list[str], *, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    peak_ram = 0
    timed_out = False
    while process.poll() is None:
        peak_ram = max(peak_ram, _rss_tree(process.pid))
        if time.perf_counter() - started > timeout:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if timed_out:
        raise TimeoutError(f"command exceeded {timeout:.3f}s: {command[0]}")
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{stderr.strip()}"
        )
    return {
        "wall_seconds": elapsed,
        "peak_ram_bytes": peak_ram,
        "stdout": stdout,
        "stderr": stderr,
    }


@dataclass(frozen=True, slots=True)
class Algorithm:
    command: tuple[str, ...]
    manifest_path: Path
    initial_model: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Algorithm:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected mapping")
        base = path.parent
        command = raw.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"{path}: command must be a non-empty argv list")
        manifest_path = (base / raw["manifest"]).resolve()
        initial_model = (base / raw["initial_model"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest) != {
            "name",
            "version",
            "capabilities",
            "external_pretraining",
        }:
            raise ValueError(f"{manifest_path}: invalid manifest fields")
        capabilities = manifest["capabilities"]
        if set(capabilities) != {"learn", "generate", "score"}:
            raise ValueError(f"{manifest_path}: invalid capabilities")
        resolved_command = tuple(
            str((base / item).absolute())
            if item.startswith(("./", "../"))
            else item
            for item in command
        )
        return cls(resolved_command, manifest_path, initial_model, manifest)

    def learn(
        self,
        model_in: Path,
        examples: list[dict[str, str]],
        model_out: Path,
        *,
        budget: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        if not self.manifest["capabilities"]["learn"]:
            raise RuntimeError("algorithm does not support learn")
        with tempfile.TemporaryDirectory(prefix="seqbench-learn-") as raw:
            root = Path(raw)
            examples_path = root / "examples.jsonl"
            budget_path = root / "budget.json"
            write_jsonl(examples_path, examples)
            budget_path.write_text(json.dumps(budget), encoding="utf-8")
            before = checkpoint_hash(model_in)
            result = run_measured(
                [
                    *self.command,
                    "learn",
                    "--model-in",
                    str(model_in),
                    "--examples",
                    str(examples_path),
                    "--model-out",
                    str(model_out),
                    "--budget",
                    str(budget_path),
                    "--seed",
                    str(seed),
                ],
                timeout=float(budget["train_wall_seconds"]),
            )
            if checkpoint_hash(model_in) != before:
                raise RuntimeError("learn mutated model_in")
            size = checkpoint_bytes(model_out)
            if size > int(budget["persistent_model_bytes"]):
                raise RuntimeError(
                    f"checkpoint budget exceeded: {size} > {budget['persistent_model_bytes']}"
                )
            return {
                **result,
                "checkpoint_bytes": size,
                "examples": len(examples),
            }

    def infer(
        self,
        model: Path,
        requests: list[dict[str, Any]],
        *,
        budget: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="seqbench-infer-") as raw:
            root = Path(raw)
            requests_path = root / "requests.jsonl"
            output_path = root / "predictions.jsonl"
            budget_path = root / "budget.json"
            write_jsonl(requests_path, requests)
            budget_path.write_text(json.dumps(budget), encoding="utf-8")
            before = checkpoint_hash(model)
            timeout = max(
                1.0,
                float(budget["infer_wall_seconds_per_example"]) * len(requests),
            )
            result = run_measured(
                [
                    *self.command,
                    "infer",
                    "--model",
                    str(model),
                    "--requests",
                    str(requests_path),
                    "--output",
                    str(output_path),
                    "--budget",
                    str(budget_path),
                ],
                timeout=timeout,
            )
            if checkpoint_hash(model) != before:
                raise RuntimeError("infer mutated checkpoint")
            responses = list(read_jsonl(output_path))
            expected_ids = [request["id"] for request in requests]
            if [response.get("id") for response in responses] != expected_ids:
                raise RuntimeError("infer response IDs/order differ from requests")
            for request, response in zip(requests, responses, strict=True):
                if request["mode"] == "generate":
                    if not isinstance(response.get("output"), str):
                        raise RuntimeError(f"{request['id']}: missing string output")
                else:
                    value = response.get("log_probability")
                    if value is not None and not isinstance(value, (int, float)):
                        raise RuntimeError(
                            f"{request['id']}: log_probability must be numeric or null"
                        )
            return responses, {
                **result,
                "requests": len(requests),
                "throughput": len(requests) / max(result["wall_seconds"], 1e-12),
            }
