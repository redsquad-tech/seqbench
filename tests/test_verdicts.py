from __future__ import annotations

from seqbench.verdicts import probe_status


def test_lower_is_better_calibration_uses_upper_confidence_bounds() -> None:
    status = probe_status(
        supported=True,
        groups=100,
        calibrated_threshold={
            "metric": "capped_nll_bits",
            "direction": "lower",
            "control_max": 2.0,
            "stress_max": 2.5,
            "max_increase": 0.5,
            "fail_stress_min": 4.0,
        },
        summary={
            "metric": "capped_nll_bits",
            "control": {"ci95": [1.0, 1.5]},
            "stress": {"ci95": [1.5, 2.0]},
            "gap": {"ci95": [0.2, 0.5]},
        },
    )
    assert status == "DEMONSTRATED"


def test_calibration_rejects_wrong_metric_direction() -> None:
    try:
        probe_status(
            supported=True,
            groups=1,
            calibrated_threshold={
                "metric": "capped_nll_bits",
                "direction": "higher",
            },
            summary={
                "metric": "capped_nll_bits",
                "control": {"ci95": [1.0, 1.0]},
                "stress": {"ci95": [1.0, 1.0]},
                "gap": {"ci95": [0.0, 0.0]},
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("wrong calibration direction was accepted")
