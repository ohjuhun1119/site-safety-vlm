"""행동 클립 정답 JSON 변환. samples.json 라벨 -> {"behavior", "unsafe"} 학습 행.

제공된 train / test 태그를 그대로 쓰고, train 안에서 클래스별로 dev를 떼어 낸다.
카메라 · 촬영일 정보가 없어 dev가 train과 같은 장면을 공유할 수 있다 (누수 점검 불가, docs/experiments.md 참고).
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from sitesafe.train.build_dataset import make_row, write_jsonl
from sitesafe.vision.schema import UNSAFE_BEHAVIORS, BehaviorJudgement

# user content에 이 자리표시가 있어야 영상 토큰이 생긴다 (scripts/smoke_clip_train.sh 참고).
CLIP_PROMPT = (
    "<|vision_start|><|video_pad|><|vision_end|>"
    "이 영상의 행동을 분류하고 불안전 여부를 판정하세요. "
    'JSON {"behavior": 클래스 이름, "unsafe": true 또는 false} 로만 답하세요.'
)


def clip_label(label: str) -> dict:
    """클래스 이름 -> 정답 JSON. 모르는 클래스는 스키마 검증에서 에러가 난다."""
    return BehaviorJudgement(behavior=label, unsafe=label in UNSAFE_BEHAVIORS).model_dump()


def split_dev(rows: list[dict], frac: float = 0.1, seed: int = 0) -> list[dict]:
    """train 행의 클래스별 frac(최소 1개)를 split='dev'로 바꾼다. 같은 seed면 같은 결과."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == "train":
            by[r["label"]].append(i)
    out = [dict(r) for r in rows]
    for label in sorted(by):
        idx = sorted(by[label], key=lambda i: rows[i]["file"])
        random.Random(f"{seed}-{label}").shuffle(idx)
        for i in idx[:max(1, round(len(idx) * frac))]:
            out[i]["split"] = "dev"
    return out


def read_samples(samples_json: Path) -> list[dict]:
    s = json.loads(Path(samples_json).read_text(encoding="utf-8"))
    rows = []
    for r in s["samples"] if isinstance(s, dict) else s:
        f = os.path.basename(r["filepath"])
        tag = r["tags"][0]
        if tag not in ("train", "test") or len(r["tags"]) != 1:
            raise ValueError(f"{f}: 태그가 train / test 하나가 아니다: {r['tags']}")
        if f"_{'tr' if tag == 'train' else 'te'}" not in f:
            raise ValueError(f"{f}: 파일명 접두와 태그 {tag}가 다르다")
        rows.append({"file": f, "label": r["ground_truth"]["label"], "split": tag})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="행동 클립 samples.json -> train / dev / test jsonl")
    ap.add_argument("--samples", type=Path, default=Path("data/raw/safe_unsafe/samples.json"))
    ap.add_argument("--clips", type=Path, default=Path("data/raw/safe_unsafe/data"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/clip"))
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = split_dev(read_samples(args.samples), args.dev_frac, args.seed)
    missing = [r["file"] for r in rows if not (args.clips / r["file"]).exists()]
    if missing:
        raise SystemExit(f"클립 {len(missing)}개가 없다 (예: {missing[0]})")
    for sp in ("train", "dev", "test"):
        part = [r for r in rows if r["split"] == sp]
        # 상대 경로는 jsonl이 있는 폴더 기준이다
        write_jsonl([make_row("videos", [os.path.relpath(args.clips / r["file"], args.out)],
                              CLIP_PROMPT, clip_label(r["label"])) for r in part], args.out / f"{sp}.jsonl")
        n_unsafe = sum(r["label"] in UNSAFE_BEHAVIORS for r in part)
        print(f"{sp}: {len(part)}개 (unsafe {n_unsafe}, safe {len(part) - n_unsafe})")
        counts = defaultdict(int)
        for r in part:
            counts[r["label"]] += 1
        print("  ", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
