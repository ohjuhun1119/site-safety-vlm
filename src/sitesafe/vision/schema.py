"""판정 JSON 스키마. VLM 출력은 반드시 이 스키마로 검증한다."""

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


class BehaviorJudgement(BaseModel):
    """영상 클립 1개에 대한 행동 판정.

    TODO: 데이터(Safe and Unsafe Behaviours)를 내려받아 실제 클래스 이름을 확인한 뒤
    behavior를 Literal[...]로 제한한다. 지금은 문자열로 받는다.
    """

    behavior: str = Field(min_length=1)
    unsafe: bool
