"""pHash 중복 제거와 분할 간 겹침 검사. 겹치면 학습을 시작하지 않는다."""
import argparse
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# 2026-11-17 Roboflow safety vs CHVG에서 해밍거리 0과 4의 일치 수가 거의 같았다(433 vs 477). 4를 기본으로 둔다.
DEFAULT_THRESHOLD = 4


def phash_matrix(paths: list[Path]) -> np.ndarray:
    """이미지마다 64비트 pHash를 (N, 64) bool 배열로 만든다."""
    return np.array(
        [imagehash.phash(Image.open(p)).hash.flatten() for p in paths], dtype=bool
    ).reshape(len(paths), 64)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, None, :] != b[None, :, :]).sum(-1)


def cross_overlap(a: np.ndarray, b: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> list[int]:
    """a의 이미지 중 b에 해밍거리 threshold 이하인 이미지가 있는 것의 인덱스."""
    if len(a) == 0 or len(b) == 0:
        return []
    return np.flatnonzero(_dist(a, b).min(1) <= threshold).tolist()


def near_duplicates(h: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> list[int]:
    """같은 집합 안에 근접 중복이 있는 이미지의 인덱스 (자기 자신은 제외)."""
    if len(h) < 2:
        return []
    d = _dist(h, h)
    np.fill_diagonal(d, 64)
    return np.flatnonzero(d.min(1) <= threshold).tolist()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="폴더 단위 pHash 겹침 검사. 겹치면 종료 코드 1.")
    ap.add_argument("dirs", nargs="+", type=Path, help="분할(또는 데이터셋) 폴더들, 하위 jpg/png 포함")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)
    hs = {}
    for d in args.dirs:
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        hs[d.name] = phash_matrix(files)
        print(f"{d}: {len(files)}장, 폴더 내 근접 중복 {len(near_duplicates(hs[d.name], args.threshold))}장")
    bad = 0
    names = list(hs)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            n = len(cross_overlap(hs[x], hs[y], args.threshold))
            print(f"{x} -> {y}: 겹치는 {x} 이미지 {n}장")
            bad += n
    if bad:
        sys.exit("겹침이 있어 학습을 시작하지 않는다.")


if __name__ == "__main__":
    main()
