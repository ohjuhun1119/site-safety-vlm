"""출처 / 카메라 단위 분할. 검증용 셋은 고치기 전에 고정한다.

출처·카메라 정보가 없는 이미지 데이터는 pHash 근접 중복끼리 한 그룹으로 묶고, 그룹 단위로 분할한다.
같은 그룹은 항상 같은 분할에 들어가므로 분할 간 겹침이 생기지 않는다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from sitesafe.data.dedup import DEFAULT_THRESHOLD, _dist, cross_overlap, phash_matrix

SPLITS = ("train", "valid", "test")


def group_ids(h: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> list[int]:
    """해밍거리 threshold 이하로 이어진 이미지끼리 같은 그룹 번호를 준다 (연결 요소)."""
    parent = list(range(len(h)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(h), 256):  # 한 번에 전체 쌍을 만들면 메모리가 크다
        for i, j in zip(*np.nonzero(_dist(h[s:s + 256], h) <= threshold)):
            parent[find(s + int(i))] = find(int(j))
    roots = [find(i) for i in range(len(h))]
    order = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    return [order[r] for r in roots]


def assign_splits(
    groups: list[int], ratios=(0.8, 0.1, 0.1), seed: int = 0
) -> list[str]:
    """그룹을 섞은 뒤, 목표 비율에 가장 모자란 분할에 그룹을 통째로 넣는다. 같은 seed면 같은 결과."""
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("ratios는 train/valid/test 3개이고 합이 1이어야 한다")
    size: dict[int, int] = {}
    for g in groups:
        size[g] = size.get(g, 0) + 1
    ids = sorted(size)
    random.Random(seed).shuffle(ids)
    total, count, where = len(groups), [0, 0, 0], {}
    for g in ids:
        k = max(range(3), key=lambda k: ratios[k] - count[k] / total)
        where[g] = SPLITS[k]
        count[k] += size[g]
    return [where[g] for g in groups]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="여러 이미지 폴더를 합쳐 pHash 그룹 단위로 분할하고 manifest를 쓴다.")
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    files, src = [], []
    for d in args.dirs:
        fs = sorted(p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        files += fs
        src += [d.name] * len(fs)
    h = phash_matrix(files)
    groups = group_ids(h, args.threshold)
    splits = assign_splits(groups, seed=args.seed)
    rows = [{"path": str(f), "dataset": s, "group": g, "split": sp}
            for f, s, g, sp in zip(files, src, groups, splits)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    by = {sp: np.array([i for i, r in enumerate(rows) if r["split"] == sp]) for sp in SPLITS}
    print(f"이미지 {len(files)}장, 그룹 {len(set(groups))}개 (한 그룹 최대 {max(np.bincount(groups))}장)")
    for sp in SPLITS:
        print(f"{sp}: {len(by[sp])}장, 데이터셋별 {dict(sorted({s: sum(src[i] == s for i in by[sp]) for s in set(src)}.items()))}")
    bad = sum(len(cross_overlap(h[by[a]], h[by[b]], args.threshold))
              for a in SPLITS for b in SPLITS if a < b)
    print(f"분할 간 겹침 {bad}장")
    if bad:
        sys.exit("분할 간 겹침이 있다.")


if __name__ == "__main__":
    main()
