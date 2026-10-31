"""A production log is credited to SAM only for a number that came from its reading."""

from wellpilot.meta_models import meta_models


def credit(value: float) -> list[str]:
    result = {
        "surveys": [
            {
                "read_from": "plots ... read by sam-3.1",
                "passes": [{"oil_share": 0.842, "water_share": 0.981, "rate": 3150.0}],
            },
            {
                "read_from": "LAS files",
                "passes": [{"water_share": 0.981, "blank_pipe_m": [3147, 3161]}],
            },
        ]
    }
    evidence = {"E7": {"tool": "production_log", "result": result}}
    answer = {"findings": [{"numbers": [{"value": value, "evidence_id": "E7"}]}]}
    return meta_models([], evidence, answer)["sam"]["logs"]


def test_a_number_only_sams_reading_holds_is_credited() -> None:
    assert credit(0.842) == ["E7"]


def test_a_depth_beside_a_rate_is_not() -> None:
    assert credit(3147) == []


def test_a_share_the_other_survey_also_holds_is_not() -> None:
    assert credit(0.981) == []
