"""jsonl 학습 행을 만든다. 행 형식은 mlx-vlm 학습 문서의 SFT 형식을 따른다.

- 이미지 / 프레임 묶음: {"images": [...], "messages": [...]}
- 영상 클립: {"videos": [...], "messages": [...]}
  (영상 행이 실제로 영상 토큰으로 학습되는지는 scripts/smoke_clip_train.sh로 먼저 확인한다.)
상대 경로는 jsonl이 있는 폴더 기준으로 해석된다.
"""

import json
from pathlib import Path
from typing import Literal

MediaKey = Literal["images", "videos"]


def make_row(media_key: MediaKey, media_paths: list[str], user_prompt: str, answer: dict) -> dict:
    if media_key not in ("images", "videos"):
        raise ValueError("media_key는 'images' 또는 'videos'")
    if not media_paths:
        raise ValueError("미디어 경로가 비어 있다")
    return {
        media_key: list(media_paths),
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ],
    }


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
