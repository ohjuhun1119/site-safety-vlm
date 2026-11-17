"""이벤트 상태 객체. C(멀티 에이전트)에서 에이전트 사이로 전달된다."""

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["confirmed", "suspected", "rejected", "pending"]


@dataclass
class EventState:
    video_id: str
    kind: Literal["ppe", "behavior"]
    start_sec: float
    end_sec: float
    verdict: Verdict = "pending"
    evidence: list[str] = field(default_factory=list)   # 근거 프레임 / 구간 경로
    rule_refs: list[str] = field(default_factory=list)  # 근거 조문 (규정 문서에 실제로 있는 것만)
    tool_calls: int = 0
    corrections: int = 0                                 # 코드가 입력값을 보정한 횟수
