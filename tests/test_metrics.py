from __future__ import annotations

import math

from seqbench.metrics import enrich_prediction, paired_bootstrap, unpaired_bootstrap


def test_exact_zero_probability_is_infinite_theoretical_nll() -> None:
    row = enrich_prediction(
        {
            "output": "wrong",
            "target": "right",
            "acceptable_outputs": ["right"],
            "target_log_probability": None,
            "candidate_log_probabilities": {},
        }
    )
    assert math.isinf(row["nll_bits"])
    assert row["capped_nll_bits"] == 1024


def test_acceptable_outputs_are_scored_as_probability_mass() -> None:
    row = enrich_prediction(
        {
            "output": "alias",
            "target": "canonical",
            "acceptable_outputs": ["canonical", "alias"],
            "target_log_probability": math.log(0.2),
            "acceptable_log_probabilities": {
                "canonical": math.log(0.2),
                "alias": math.log(0.3),
            },
            "candidate_log_probabilities": {
                "canonical": math.log(0.2),
                "alias": math.log(0.3),
                "wrong": math.log(0.5),
            },
        }
    )
    assert row["normalized_exact_match"] == 1
    assert math.isclose(row["nll_bits"], 1.0)
    assert math.isclose(row["candidate_conditional_nll_bits"], 1.0)


def test_paired_bootstrap_uses_common_probe_groups() -> None:
    control = [
        {"probe_group_id": "a", "normalized_exact_match": 1.0},
        {"probe_group_id": "b", "normalized_exact_match": 1.0},
    ]
    stress = [
        {"probe_group_id": "a", "normalized_exact_match": 0.0},
        {"probe_group_id": "b", "normalized_exact_match": 1.0},
    ]
    result = paired_bootstrap(
        control,
        stress,
        metric="normalized_exact_match",
        replicates=100,
        seed=1,
    )
    assert result["groups"] == 2
    assert result["gap"]["score"] == -0.5


def test_unpaired_bootstrap_compares_disjoint_splits() -> None:
    control = [
        {"probe_group_id": "test-1", "normalized_exact_match": 1.0},
        {"probe_group_id": "test-2", "normalized_exact_match": 1.0},
    ]
    stress = [
        {"probe_group_id": "generalization-1", "normalized_exact_match": 0.0},
        {"probe_group_id": "generalization-2", "normalized_exact_match": 1.0},
    ]
    result = unpaired_bootstrap(
        control,
        stress,
        metric="normalized_exact_match",
        replicates=100,
        seed=1,
    )
    assert result["groups"] == 2
    assert result["gap"]["score"] == -0.5
