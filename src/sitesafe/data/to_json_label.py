"""통일 클래스 박스(normalize.py 결과)를 PPE 정답 JSON으로 변환한다. 임계값은 표본 100개 점검으로 정한다.

규칙 (2026-11-17 1차안, 점검 결과로 고친다):
- CHVG(person 있음, no_vest 없음): workers = person 수. no_helmet = no_helmet(head) 박스 수.
  no_vest = 조끼 박스 면적의 VEST_IN_PERSON 이상이 person 안에 들어오는 조끼가 없는 person 수.
- Roboflow(person 없음): workers = max(helmet+no_helmet, vest+no_vest). no_helmet · no_vest는 해당 박스 수.
- CHVG에서 person 박스보다 맨머리(no_helmet) 박스가 많으면 남는 사람은 person 밖 조끼 박스가 없을 때 no_vest로 센다 (2026-11-17 점검 후 추가).
- unclear = 머리 라벨(helmet 또는 no_helmet)이 없는 작업자 수 (워커 수 - 머리 라벨 수, 0 이상).
- 미착용 수가 workers보다 크면 workers를 올려 스키마(PPEJudgement)를 통과시킨다.
"""
import argparse
import json
import sys
from pathlib import Path

from sitesafe.vision.schema import PPEJudgement

VEST_IN_PERSON = 0.7  # 임의 초기값. 표본 점검으로 조정한다.


def _inside(vest: list[int], person: list[int]) -> float:
    x1, y1, x2, y2 = max(vest[0], person[0]), max(vest[1], person[1]), min(vest[2], person[2]), min(vest[3], person[3])
    area = max(1, (vest[2] - vest[0]) * (vest[3] - vest[1]))
    return max(0, x2 - x1) * max(0, y2 - y1) / area


def ppe_label(source: str, boxes: list[dict]) -> dict:
    n = lambda c: sum(b["cls"] == c for b in boxes)
    head = n("helmet") + n("no_helmet")
    if source == "chvg":
        persons = [b["box"] for b in boxes if b["cls"] == "person"]
        vests = [b["box"] for b in boxes if b["cls"] == "vest"]
        workers = len(persons)
        no_vest = sum(1 for p in persons if not any(_inside(v, p) >= VEST_IN_PERSON for v in vests))
        # person 박스 없이 맨머리 박스만 있는 사람(점검에서 #20 · #38 확인)도 조끼 박스가 따로 없으면 미착용 조끼로 센다
        unmatched = sum(1 for v in vests if not any(_inside(v, p) >= VEST_IN_PERSON for p in persons))
        no_vest += max(0, max(0, n("no_helmet") - workers) - unmatched)
    elif source == "roboflow":
        workers = max(head, n("vest") + n("no_vest"))
        no_vest = n("no_vest")
    else:
        raise ValueError(f"모르는 출처 {source!r}")
    no_helmet = n("no_helmet")
    workers = max(workers, no_helmet, no_vest)
    return PPEJudgement(workers=workers, no_helmet=no_helmet, no_vest=no_vest,
                        unclear=max(0, workers - head)).model_dump()


FIELDS = ("workers", "no_helmet", "no_vest", "unclear")


def apply_fixes(label: dict, fix: dict) -> dict:
    """사람이 재확인해서 적어 둔 수정값(eval/label_fixes.json)을 덮어쓰고 스키마로 다시 검증한다."""
    return PPEJudgement(**{**label, **{k: fix[k] for k in FIELDS if k in fix}}).model_dump()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ppe_boxes.jsonl에 정답 JSON(label)을 붙여 쓴다.")
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixes", type=Path, help="경로별 수정값 json (eval/label_fixes.json)")
    args = ap.parse_args(argv)
    fixes = json.loads(args.fixes.read_text(encoding="utf-8")) if args.fixes else {}
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.boxes.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            label = ppe_label(r["source"], r["boxes"])
            if r["path"] in fixes:
                label = apply_fixes(label, fixes[r["path"]])
            f.write(json.dumps({**r, "label": label, "fixed": r["path"] in fixes}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n}장에 정답 JSON을 붙임 (수정값 적용 {len(fixes)}건)", file=sys.stderr)


if __name__ == "__main__":
    main()
