#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tools"))

from datasets.transforms import correction_paraphrase_transform, write_transformed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize held-out paraphrases for correction episodes."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"{write_transformed(args.source, args.output, correction_paraphrase_transform):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
