"""판정 JSON 스키마. VLM 출력은 반드시 이 스키마로 검증한다."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PPEJudgement(BaseModel):
    """이미지 1장 또는 프레임 묶음에 대한 PPE 판정."""

    workers: int = Field(ge=0)
    no_helmet: int = Field(ge=0)
    no_vest: int = Field(ge=0)
    unclear: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_not_above_workers(self) -> "PPEJudgement":
        if self.no_helmet > self.workers or self.no_vest > self.workers:
            raise ValueError("미착용 수가 작업자 수보다 클 수 없다")
        return self


# Safe and Unsafe Behaviours samples.json의 라벨 691개 전부에서 확인한 8클래스 (2026-11-17).
# 안전 / 불안전 구분은 데이터셋 카드 기준이며 파일명 앞 숫자(0-7)는 클래스 인덱스와 1:1로 일치한다.
UNSAFE_BEHAVIORS = (
    "Safe Walkway Violation",
    "Unauthorized Intervention",
    "Opened Panel Cover",
    "Carrying Overload with Forklift",
)
SAFE_BEHAVIORS = ("Safe Walkway", "Authorized Intervention", "Closed Panel Cover", "Safe Carrying")
Behavior = Literal[
    "Safe Walkway Violation",
    "Unauthorized Intervention",
    "Opened Panel Cover",
    "Carrying Overload with Forklift",
    "Safe Walkway",
    "Authorized Intervention",
    "Closed Panel Cover",
    "Safe Carrying",
]


class BehaviorJudgement(BaseModel):
    """영상 클립 1개에 대한 행동 판정. behavior는 8클래스 중 하나."""

    behavior: Behavior
    unsafe: bool
