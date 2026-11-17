# 데이터 출처 · 라이선스 기록

원본은 레포에 올리지 않는다. 내려받은 뒤 `실제 장수`, `내려받은 날짜`, `확인한 라이선스`를 직접 채운다.
`표기된 라이선스`는 2026-11-16에 검색으로 본 값이며 반드시 원문에서 다시 확인한다.

| 데이터 | 출처 | 형태 | 용도 | 표기된 라이선스 | 실제 장수 | 내려받은 날짜 | 확인한 라이선스 | 메모 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Safe and Unsafe Behaviours | https://data.mendeley.com/datasets/xjmtb22pff/1 , Hugging Face `Voxel51/Safe_and_Unsafe_Behaviours` | 공장 CCTV mp4 클립 (691개로 표기), 8클래스 | 불안전 행동 영상 클립 학습 · 평가 | CC BY 4.0 | | | | 얼굴 · 신체 노출, 한 공장 · 카메라 2대. 클래스 이름을 확인한 뒤 `src/sitesafe/vision/schema.py`에 반영 |
| SteelBench | Hugging Face `steelbench` | 제철소 CCTV 클립 (1,345개로 표기), 클립당 프레임 8장, 작업자별 PPE 라벨 | PPE 판정 학습 · 평가 | CC BY-NC 4.0 | | | | 비영리: 학습 · 포트폴리오 목적만, 가중치 비공개. 형식(프레임/mp4) 확인 |
| CHV | https://github.com/ZijianWang-ZW/PPE_detection | 이미지 + bbox | PPE 이미지 학습 후보 | | | | | 클래스 정의 확인 |
| CHVG | https://figshare.com/articles/dataset/CHVG_Dataset/19625166 | 이미지 + 라벨 | PPE 이미지 학습 후보 | | | | | CHV와 별개. 사용 시 논문 인용 |
| Ultralytics Construction-PPE | https://docs.ultralytics.com/datasets/detect/construction-ppe | 이미지 + 라벨 (1,000장으로 표기) | PPE 이미지 학습 · 검증 | CC BY 4.0 | | | | 미착용 클래스 정의 확인 |
| Roboflow Universe PPE 데이터셋들 | Roboflow Universe (데이터셋마다 URL 기록) | 이미지 + bbox | PPE 이미지 학습 · 검증 | 데이터셋마다 다름 | | | | 증강본 섞임 여부, no-helmet / no-vest 클래스 유무 |
| SynthSite | Hugging Face `govtech/SynthSite` | 합성 영상 | 파이프라인 점검 · 시연 전용 (학습 금지) | other (govtech-singapore) | | | | 논문 55클립 vs HF 카드 227개, 수치 확인. 라벨은 PPE가 아님 |
| iSafetyBench | GitHub `iSafetyBench/data` | 실제 클립 (1,100개로 표기) | 평가 전용 | | | | | 클립 배포 방식 · 라이선스 미확인 |
| 규정 문서 | 국가법령정보센터 | 산업안전보건기준에 관한 규칙 중 보호구 · 통로 · 지게차 관련 조문 | 규정 RAG | 법령 원문 | | | | 조문 번호 · 시행일은 원문으로 확인 |
