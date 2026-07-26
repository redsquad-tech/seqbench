#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tools"))

from datasets.adapters import ADAPTERS, semantic_parsing_row
from datasets.download import DATASETS, ensure_sources, load_sources
from datasets.schema import CSV_COLUMNS, TaskRow
from datasets.sources import (
    SourceSpec,
    discover_sources,
    iter_semantic_fields,
    iter_structured_records,
)


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def rows(source: SourceSpec, variant: str) -> Iterator[TaskRow]:
    if source.dataset in {"recogs", "slog"}:
        if variant != "full":
            return
        for index, fields in enumerate(iter_semantic_fields(source)):
            task = semantic_parsing_row(
                dataset=source.dataset,
                config=source.config,
                split=source.split,
                source_index=index,
                fields=fields,
                source_path=source.source_key,
            )
            if task is not None:
                yield task
        return
    adapter = ADAPTERS[source.dataset]
    for index, record in enumerate(iter_structured_records(source)):
        yield from adapter.convert(
            record,
            config=source.config,
            split=source.split,
            source_key=source.source_key,
            source_index=index,
            variants={variant},
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download benchmark sources and write one plain CSV."
    )
    result.add_argument("output", type=Path)
    result.add_argument("--variant", choices=["full", "oracle"], default="full")
    result.add_argument("--datasets", default="all")
    result.add_argument("--raw-dir", type=Path, default=PROJECT / ".data" / "raw")
    result.add_argument("--force-download", action="store_true")
    result.add_argument("--mrcr-needles", default="2needle,4needle,8needle")
    result.add_argument("--babilong-lengths", default="0k,1k,2k,4k,8k,16k")
    return result


def main() -> int:
    args = parser().parse_args()
    names = DATASETS if args.datasets == "all" else comma_values(args.datasets)
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    mrcr_needles = comma_values(args.mrcr_needles)
    babilong_lengths = comma_values(args.babilong_lengths)
    source_specs = load_sources(PROJECT / "tools" / "datasets_manifest.json")
    ensure_sources(
        source_specs,
        names,
        args.raw_dir,
        mrcr_needles=mrcr_needles,
        babilong_lengths=babilong_lengths,
        force=args.force_download,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for dataset in names:
            count = 0
            sources = discover_sources(
                dataset,
                args.raw_dir / dataset,
                mrcr_needles=mrcr_needles,
                babilong_lengths=babilong_lengths,
            )
            for source in sources:
                for task in rows(source, args.variant):
                    writer.writerow(task.to_csv_dict())
                    count += 1
            total += count
            print(f"[{dataset}] {count:,} rows")
    print(f"{args.output}: {total:,} rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
