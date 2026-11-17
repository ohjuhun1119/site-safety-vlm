import pytest
from pydantic import ValidationError

from sitesafe.vision.schema import BehaviorJudgement, PPEJudgement


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
        BehaviorJudgement.model_validate({"behavior": "x"})
    assert BehaviorJudgement.model_validate({"behavior": "x", "unsafe": False}).unsafe is False
