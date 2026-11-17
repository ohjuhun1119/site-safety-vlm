#!/usr/bin/env bash
# 영상 입력 학습 스모크 테스트 (맥에서 10~20분).
# 준비: pip install "mlx-vlm[train]"
#       data/processed/clip_smoke/train.jsonl 과 data/processed/clip_smoke/clips/*.mp4 에 짧은 클립 2~3개
#       jsonl 한 줄 예: {"videos": ["clips/a.mp4"], "messages": [{"role": "user", "content": "<|vision_start|><|video_pad|><|vision_end|>이 영상의 행동을 분류하세요."}, {"role": "assistant", "content": "{\"behavior\": \"...\", \"unsafe\": false}"}]}
#       주의: user content 에 <|vision_start|><|video_pad|><|vision_end|> 를 직접 넣어야 영상 토큰이 생긴다.
#             문자열 content 나 {"type": "video"} 구조로는 영상이 학습에 안 들어간다 (mlx-vlm 0.8.0, Qwen2.5-VL 확인).
# 확인: (1) 에러 없이 도는지, (2) 로그의 train_num_vision_tokens 가 0보다 큰지 (0이면 영상이 학습에 안 들어간 것)
set -euo pipefail
python -m mlx_vlm.train \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --data data/processed/clip_smoke \
  --train-on-completions \
  --batch-size 1 \
  --iters 5 \
  --adapter-file adapters/clip_smoke/weights.safetensors
