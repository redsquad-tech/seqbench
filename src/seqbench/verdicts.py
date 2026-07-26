from __future__ import annotations

from typing import Any

from .specs import PropertySpec


def probe_status(
    *,
    supported: bool,
    groups: int,
    calibrated_threshold: dict[str, float] | None,
    summary: dict[str, Any],
) -> str:
    if not supported:
        return "UNSUPPORTED"
    if groups == 0:
        return "INCONCLUSIVE"
    if calibrated_threshold is None:
        return "UNCALIBRATED"
    control = summary["control"]["ci95"]
    stress = summary["stress"]["ci95"]
    gap = summary["gap"]["ci95"]
    if control[1] < calibrated_threshold["control_min"]:
        return "INCONCLUSIVE"
    if (
        control[0] >= calibrated_threshold["control_min"]
        and stress[0] >= calibrated_threshold["stress_min"]
        and gap[0] >= -calibrated_threshold["max_drop"]
    ):
        return "DEMONSTRATED"
    if stress[1] < calibrated_threshold["fail_stress_max"]:
        return "NOT_DEMONSTRATED"
    return "PARTIAL"


def aggregate_property(
    spec: PropertySpec, probe_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    required = [probe_results.get(probe) for probe in spec.required_probes]
    statuses = [item["status"] if item else "INCONCLUSIVE" for item in required]
    if any(status == "UNSUPPORTED" for status in statuses):
        status = "UNSUPPORTED"
    elif any(status == "INCONCLUSIVE" for status in statuses):
        status = "INCONCLUSIVE"
    elif all(status == "DEMONSTRATED" for status in statuses):
        status = "DEMONSTRATED"
    elif any(status == "NOT_DEMONSTRATED" for status in statuses):
        status = "NOT_DEMONSTRATED"
    elif all(status == "UNCALIBRATED" for status in statuses):
        status = "UNCALIBRATED"
    else:
        status = "PARTIAL"
    return {
        "property": spec.id,
        "status": status,
        "aggregation": spec.aggregation,
        "required_probes": list(spec.required_probes),
        "supplementary_probes": list(spec.supplementary_probes),
    }

