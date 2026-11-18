"""영상을 해상도 / fps를 낮춘 클립으로 캐시한다 (고정 길이 윈도우 자르기는 추론 단계에서 따로).

해상도 · fps는 실험 변수라 캐시 폴더 이름에 값을 넣는다 (예: clip_360p_2fps).
영상 행은 최대 길이를 넘으면 잘라내지 않고 에러가 나므로 프레임을 줄일 때는 fps로 줄인다.
"""
import argparse
import json
import os
from pathlib import Path

import cv2

from sitesafe.train.build_dataset import write_jsonl


def make_clip(src: str | Path, dst: str | Path, fps: float, height: int) -> dict:
    """src를 fps / 높이로 줄여 dst(mp4)에 쓴다. 원본 길이는 유지하고 프레임은 균등하게 고른다.

    shortcut: 가장 가까운 원본 프레임을 고를 뿐 보간하지 않는다, 느린 동작을 놓치면 fps를 올린다.
    """
    cap = cv2.VideoCapture(str(src))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if n_src <= 0 or src_fps <= 0:
        raise ValueError(f"{src}: 프레임 수 또는 fps를 읽지 못했다")
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = round(w0 * height / h0 / 2) * 2, height
    n_out = max(2, int(n_src / src_fps * fps))  # 영상 토큰이 2프레임 단위라 최소 2프레임
    want = [min(round(k * src_fps / fps), n_src - 1) for k in range(n_out)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    got, last, i, k = 0, None, 0, 0
    while k < len(want):
        if want[k] == i - 1 and last is not None:  # 같은 원본 프레임을 다시 쓴다 (fps가 원본보다 클 때)
            out.write(last); got += 1; k += 1; continue
        ok, fr = cap.read()
        if not ok:
            break
        if want[k] == i:
            last = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            out.write(last); got += 1; k += 1
            while k < len(want) and want[k] == i:
                out.write(last); got += 1; k += 1
        i += 1
    cap.release(); out.release()
    if got != n_out:
        raise ValueError(f"{src}: 프레임 {n_out}개를 써야 하는데 {got}개만 읽었다")
    return {"frames": got, "width": w, "height": h}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="행동 클립 jsonl의 영상을 낮은 해상도 / fps 캐시로 바꾼 jsonl을 만든다.")
    ap.add_argument("--src", type=Path, default=Path("data/processed/clip"))
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--height", type=int, required=True)
    args = ap.parse_args(argv)
    name = f"clip_{args.height}p_{args.fps:g}fps"
    cache, dst = Path("data/videos") / name, Path("data/processed") / name
    stats = {}
    for sp in ("train", "dev", "test"):
        rows = [json.loads(l) for l in (args.src / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in rows:
            f = Path(r["videos"][0]).name
            if f not in stats:
                stats[f] = make_clip(args.src / r["videos"][0], cache / f, args.fps, args.height)
            r["videos"] = [os.path.relpath(cache / f, dst)]
        write_jsonl(rows, dst / f"{sp}.jsonl")
    fr = [s["frames"] for s in stats.values()]
    print(f"{name}: 클립 {len(stats)}개, 프레임 최소 {min(fr)} / 평균 {sum(fr) / len(fr):.1f} / 최대 {max(fr)}, "
          f"크기 {next(iter(stats.values()))['width']}x{args.height}")
    print(f"캐시 {sum(p.stat().st_size for p in cache.glob('*.mp4')) / 1e6:.0f}MB, jsonl {dst}")


if __name__ == "__main__":
    main()
