from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import snapshot_download

from .sources import BABI_FAMILIES

DATASETS = ["mrcr", "babi", "babilong", "clutrr", "proofwriter", "recogs", "slog"]


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_sources(
    source_specs: dict[str, dict[str, Any]],
    names: list[str],
    raw_dir: Path,
    *,
    mrcr_needles: list[str],
    babilong_lengths: list[str],
    force: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = raw_dir / name
        if destination.exists() and not force:
            print(f"[{name}] cached")
            continue
        if destination.exists():
            shutil.rmtree(destination)
        spec = source_specs[name]
        print(f"[{name}] downloading")
        if spec["kind"] == "huggingface":
            patterns = None
            if name == "mrcr":
                patterns = ["README.md", *[f"{item}/**" for item in mrcr_needles]]
            elif name == "babilong":
                patterns = [
                    "README.md",
                    *[f"data/qa*/{item}.json" for item in babilong_lengths],
                ]
            snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                revision=spec["revision"],
                local_dir=destination,
                allow_patterns=patterns,
            )
        elif spec["kind"] == "git":
            subprocess.run(
                ["git", "clone", "--quiet", spec["url"], str(destination)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--quiet", spec["commit"]],
                check=True,
            )
        elif spec["kind"] == "archive":
            _download_babi(spec["url"], destination)
        else:
            raise ValueError(f"{name}: unsupported source kind {spec['kind']}")


def _download_babi(url: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    archive_path = destination / "babi.tar.gz"
    urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or len(path.parts) != 3
                or path.parts[0] != "tasks_1-20_v1-2"
                or path.parts[1] not in BABI_FAMILIES
                or not path.name.startswith("qa")
                or not path.name.endswith(".txt")
            ):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    archive_path.unlink()
