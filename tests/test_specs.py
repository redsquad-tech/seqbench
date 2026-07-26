from __future__ import annotations

from pathlib import Path

from seqbench.specs import Probe, PropertySpec, RunSpec

ROOT = Path(__file__).resolve().parent.parent


def test_full_v1_specs_are_loadable() -> None:
    run = RunSpec.load(ROOT / "specs/runs/full_v1.yaml")
    probes = [Probe.load(path) for path in run.probes]
    properties = [PropertySpec.load(path) for path in run.properties]
    ids = {probe.id for probe in probes}
    assert ids
    assert all(set(item.required_probes) <= ids for item in properties)
