# 실험 기록

측정하지 않은 칸은 비워 둔다.

## 0. 환경 · 스모크 테스트

| 날짜 | 항목 | 결과 | 메모 |
| --- | --- | --- | --- |
| 2026-11-17 | 환경 설치 | Python 3.12.14, `pip install -e ".[dev]"` 성공 | 시스템 기본 Python은 3.14.6이라 3.12로 `.venv` 생성. mlx-vlm 0.8.0, mlx 0.32.4, transformers 5.23.0. 버전 고정(pip freeze)은 아직 안 함 |
| 2026-11-17 | `pytest` | 7개 통과 | `tests/test_build_dataset.py` 3개, `tests/test_schema.py` 4개 |
| 2026-11-17 | mlx-vlm 이미지 추론 | 성공 | `python -m mlx_vlm generate --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit --image ... --max-tokens 80`. 입력은 Pillow로 만든 합성 이미지(448x448, 노란 모자 + 주황 몸통 도형). 답변이 도형 색을 맞게 묘사함. 모델 가중치 약 1.6GB 다운로드에 16분 걸려 추론 시간은 따로 재지 않음. 현장 데이터 정확도와 무관 |
| 2026-11-17 | mlx-vlm 영상 입력 추론 | 에러 없이 실행됨 | `python -m mlx_vlm generate --video ... --fps 2 --max-tokens 80`. 입력은 OpenCV로 만든 합성 mp4(448x448, 8fps, 32프레임, 도형이 오른쪽으로 이동). 답변은 도형의 색을 묘사했고 움직임은 언급하지 않음. 전체 약 3초(모델 캐시됨). `generate_video`는 영상 *생성*용 명령이라 영상 입력에는 쓰지 않음. 프레임이 실제로 비전 토큰으로 들어갔는지는 아직 확인 못 함(추론 로그 미확인) |
| 2026-11-17 | 영상 입력 학습 스모크 테스트 (`scripts/smoke_clip_train.sh`) 1차 | 에러 없이 종료, 그러나 영상은 학습에 안 들어감 | `train_num_vision_tokens=0`, 시퀀스 평균 44토큰. user content가 일반 문자열이라 프롬프트에 영상 자리표시가 없었음. `VisionDataset`은 `pixel_values_videos`·`video_grid_thw`는 만들지만 `input_ids`에 영상 토큰이 0개(`{"type":"video"}` 구조로 바꿔도 동일). 에러 없이 끝나므로 `train_num_vision_tokens`를 꼭 봐야 함 |
| 2026-11-17 | 영상 입력 학습 스모크 테스트 2차 (자리표시 추가) | 영상 입력 학습 동작 확인 | user content 앞에 `<|vision_start|><|video_pad|><|vision_end|>`를 넣음. `train_num_vision_tokens=1,440`(5시퀀스 × 288), 시퀀스 평균 334토큰, `peak_memory=6.70`(mlx-vlm 로그 값, 단위 미확인), 5 iter 약 6초, loss 3.69. 입력은 합성 클립 3개(336x336, 4fps, 8프레임), 모델 `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, batch 1, LoRA(학습 파라미터 14.97M, 0.399%). **이 값들은 파이프라인 점검용이며 정확도·성능 근거가 아님.** 영상 클립 입력(프레임 묶음 아님)으로 학습됨. 실제 1080p 클립의 해상도·fps·프레임 수 조건에서의 메모리는 미측정 |

## 1. zero-shot baseline

| 대상 | 프롬프트 | 지표 | 값 |
| --- | --- | --- | --- |
| PPE 이미지 | | 미착용 F1 / 파싱 성공률 | |
| SteelBench | | | |
| 행동 클립 | | 정확도 / unsafe 재현율 | |

## 2. PPE 학습 비교

| 방식 | 데이터 양 | 모델 | 정확도(F1) | 최대 메모리 | 학습 시간 |
| --- | --- | --- | --- | --- | --- |
| zero-shot | - | | | | - |
| LoRA | | | | | |
| QLoRA | | | | | |

## 3. 행동 클립 학습 비교

| 방식 | 입력(fps / 해상도 / 프레임 수) | 정확도 | unsafe 재현율 | 최대 메모리 | 학습 시간 |
| --- | --- | --- | --- | --- | --- |
| zero-shot | | | | | - |
| LoRA | | | | | |
| QLoRA | | | | | |

## 4. A / B / C 비교

| 구조 | 이벤트 정밀도 | 재현율 | 오탐 걸러낸 비율 | VLM 호출 수 | LLM 호출 수 | 처리 시간 | 규정 근거 정확도 | 실패 / 보정 횟수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 고정 | | | | | | | | |
| B 싱글 | | | | | | | | |
| C 멀티 | | | | | | | | |

## 5. PPE 이미지 분할 (CHVG + Roboflow safety 통합)

| 날짜 | 방식 | 결과 | 메모 |
| --- | --- | --- | --- |
| 2026-11-17 | pHash(해밍거리 4 이하) 연결 요소로 그룹을 묶고 그룹 단위로 8:1:1 분할, seed 0 (`sitesafe.data.split`) | 이미지 3,922장, 그룹 3,380개(한 그룹 최대 10장). train 3,138 / valid 392 / test 392. 분할 간 겹침 0장 | 출처·카메라 정보가 없어 이미지 그룹 단위가 한계. 중복은 지우지 않고 같은 분할에 같이 둠(중복 제거 여부는 미정). 라벨 체계(CHVG와 Roboflow)는 아직 통합 안 함. 클래스 분포와 라벨 의미는 미점검이라 분할은 클래스 비율을 맞추지 않음. manifest는 `data/processed/ppe_split.json`(git 제외). valid·test는 이 시점에 고정 |
