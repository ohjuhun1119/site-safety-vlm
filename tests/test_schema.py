import pytest
from pydantic import ValidationError

from sitesafe.vision.schema import SAFE_BEHAVIORS, UNSAFE_BEHAVIORS, BehaviorJudgement, PPEJudgement


def test_ppe_valid():
    j = PPEJudgement.model_validate({"workers": 3, "no_helmet": 1, "no_vest": 0, "unclear": 0})
    assert j.no_helmet == 1


def test_ppe_counts_cannot_exceed_workers():
    with pytest.raises(ValidationError):
        PPEJudgement.model_validate({"workers": 1, "no_helmet": 2, "no_vest": 0, "unclear": 0})


def test_ppe_rejects_negative():
    with pytest.raises(ValidationError):
        PPEJudgement.model_validate({"workers": -1, "no_helmet": 0, "no_vest": 0, "unclear": 0})


def test_behavior_requires_fields():
    with pytest.raises(ValidationError):
        BehaviorJudgement.model_validate({"behavior": "Safe Walkway"})
    assert BehaviorJudgement.model_validate({"behavior": "Safe Walkway", "unsafe": False}).unsafe is False


def test_behavior_rejects_unknown_class():
    with pytest.raises(ValidationError):
        BehaviorJudgement.model_validate({"behavior": "x", "unsafe": False})


def test_behavior_constants_match_literal():
    from typing import get_args

    from sitesafe.vision.schema import Behavior

    assert set(get_args(Behavior)) == set(UNSAFE_BEHAVIORS) | set(SAFE_BEHAVIORS)
