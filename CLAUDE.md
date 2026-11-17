# site-safety-vlm

건설·산업 현장 이미지·영상에서 PPE(안전모·안전조끼) 착용 상태와 불안전 행동을 판정하는 소형 VLM을
맥(Apple Silicon, 메모리 24GB)에서 LoRA/QLoRA로 파인튜닝하고, 검증 · 규정 근거 · 리포트 단계를
A 고정 파이프라인 / B 싱글 에이전트 / C 멀티 에이전트로 구현해 같은 평가셋으로 비교하는 개인 포트폴리오 프로젝트.

- 개요와 폴더 설명: @README.md
- 계획 요약: @docs/PLAN.md (원본은 노션 작업 페이지, 내용이 다르면 노션이 최신:
  https://app.notion.com/p/41a352da929e810d8160c2ee8b04cbe6)
- 실험 기록: docs/experiments.md, 데이터 출처 · 라이선스: data_sources.md

## 명령어

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # 현재 7개 테스트
bash scripts/smoke_clip_train.sh    # 영상 입력 학습 스모크 테스트 (0단계)
```

## 작업 원칙 (반드시 지킨다)

- **측정하기 전에는 결과를 주장하지 않는다.** README · 문서의 수치 칸은 실제로 측정한 값만 채운다.
- **실제로 한 방식 그대로 기록한다.** 영상 클립을 입력으로 학습했는지, 프레임 묶음(다중 이미지)으로 학습했는지를
  구분해서 `docs/experiments.md`에 적는다. 프레임 묶음이면 "영상 입력 학습"이라고 쓰지 않는다.
- 데이터 · 영상 · 모델 가중치 · `.env`는 커밋하지 않는다 (`.gitignore` 참고). 출처만 `data_sources.md`에 기록한다.
- 일부 데이터는 비영리 라이선스(CC BY-NC)다. 학습 · 포트폴리오 목적으로만 쓰고 가중치는 공개하지 않는다.
- SynthSite(합성 영상)는 파이프라인 동작 점검용이다. 학습에 쓰지 않고 정확도 평가로 주장하지 않는다.
- 평가가 부풀지 않게 한다: 출처 · 카메라 단위로 분할하고, pHash로 분할 간 겹침을 검사하고,
  겹치면 학습을 시작하지 않는다. 검증용 · 시험용 셋은 고치기 전에 고정하고 마지막에만 쓴다.
- 변환 라벨(bbox, SteelBench → 정답 JSON)은 표본 100개를 직접 점검해 오류율을 `eval/label_audit.md`에 기록한다.
- 행동 클래스 이름은 데이터를 내려받아 확인하기 전에는 추측해서 `schema.py` · `behavior_rules.yaml`에 박지 않는다.
- A / B / C 비교는 같은 VLM · 같은 도구(`reinspect`, `lookup_rule`) · 같은 평가셋 · 같은 호출 상한 · 같은
  프롬프트 수정 횟수 상한으로 한다. 어떤 구조가 이기든 결과를 그대로 기록한다.
- 에이전트의 제어 흐름(순서 · 종료 조건)은 코드(LangGraph)가 정하고 LLM은 노드 안 판단만 맡긴다.
  도구 입력값은 코드가 검사하고, 코드가 보정한 횟수를 기록한다.
- 규정 근거는 `data/rules/`의 문서에 실제로 있는 조문만 인용한다. 구역(`configs/zones.yaml`) ·
  행동→규정 매핑(`configs/behavior_rules.yaml`)은 가정값이라고 리포트에 표시한다.

## 환경 제약 (24GB 맥)

- 모델 · 학습 · Langfuse를 동시에 띄우지 않는다. 학습과 추론 중에는 Langfuse(Docker)를 내린다.
- bitsandbytes(CUDA 전용)는 못 쓴다. QLoRA 방식은 mlx-vlm의 4비트 양자화 위 학습(`quantization_bits: 4`)으로 한다.
- 영상 클립은 1080p 24fps라 해상도 · fps · 프레임 수를 낮춰야 한다. 이 값은 실험 변수로 기록한다.
- 영상 행은 최대 길이를 넘으면 잘라내지 않고 에러가 난다 (mlx-vlm 학습 코드 동작).

## 현재 상태와 다음 할 일

- 상태: 0단계 시작 전. 모듈은 대부분 빈 틀이며 `vision/schema.py`, `train/build_dataset.py`,
  `agents/state.py`만 동작한다.
- 다음: 0단계 — 환경 설치, mlx-vlm 이미지 · 영상 추론 확인, 영상 입력 학습 스모크 테스트, 데이터 내려받기와
  라이선스 · 형식 확인 (`data_sources.md` 채우기).
- 막히거나 계획과 달라지면 `docs/experiments.md`와 노션 진행 로그에 사유를 남긴다.

## 작업 방식

- 문서와 설명은 한국어로 쓴다. 코드 식별자와 명령어는 영어 그대로 둔다.
- 큰 구현을 시작하기 전에 어떤 파일을 어떻게 바꿀지 먼저 짧게 알려준다.
- 새 기능에는 `tests/`에 테스트를 함께 추가하고, 실행 결과를 확인한 뒤에 완료라고 말한다.
