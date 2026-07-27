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


def test_causal_v2_is_versioned_and_loadable() -> None:
    run = RunSpec.load(ROOT / "specs/runs/causal_v2.yaml")
    probes = [Probe.load(path) for path in run.probes]
    properties = [PropertySpec.load(path) for path in run.properties]
    ids = {probe.id for probe in probes}
    assert run.version == 2
    assert run.artifact_schema == 2
    assert run.sampling_seed == 42
    assert all(set(item.required_probes) <= ids for item in properties)


def test_screen_v1_covers_full_black_box_property_matrix() -> None:
    run = RunSpec.load(ROOT / "specs/screens/screen_v1.yaml")
    probes = [Probe.load(path) for path in run.probes]
    properties = [PropertySpec.load(path) for path in run.properties]
    ids = {probe.id for probe in probes}
    property_ids = {item.id for item in properties}
    assert {
        "prediction",
        "retrieval",
        "position_length_transfer",
        "role_binding",
        "multi_hop",
        "composition",
        "credit_assignment",
        "noise_robustness",
        "online_correction",
    } <= property_ids
    assert all(set(item.required_probes) <= ids for item in properties)
