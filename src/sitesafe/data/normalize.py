"""클래스 이름 통일, 형식 변환 (person / helmet / vest / no_helmet / no_vest).

CHVG와 Roboflow safety(Pascal VOC)를 같은 클래스 체계의 박스 목록으로 바꾼다.
두 데이터는 라벨 범위가 다르다: CHVG에는 no_vest가, Roboflow에는 person이 없다.
그래서 이미지마다 "이 출처가 주석을 다는 클래스"(annotated)를 함께 기록한다.
주석 대상이 아닌 클래스가 0개인 것을 "없다"로 읽으면 안 된다.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

UNIFIED = ("person", "helmet", "vest", "no_helmet", "no_vest")

# None은 PPE 판정 대상이 아니라 버리는 클래스. 목록에 없는 이름은 추측하지 않고 에러를 낸다.
# chvg의 head=맨머리 해석은 표본 8장 확인(2026-11-17)이라 변환 라벨 100장 점검으로 확정해야 한다.
MAPPINGS = {
    "chvg": {"person": "person", "vest": "vest", "head": "no_helmet", "glass": None,
             "white": "helmet", "yellow": "helmet", "blue": "helmet", "red": "helmet"},
    "roboflow": {"Helmet": "helmet", "No-Helmet": "no_helmet", "Vest": "vest", "No_Vest": "no_vest"},
}
ANNOTATED = {
    "chvg": ("person", "helmet", "vest", "no_helmet"),
    "roboflow": ("helmet", "vest", "no_helmet", "no_vest"),
}
SOURCE_OF_DIR = {"CHVG-Dataset": "chvg", "roboflow_safety": "roboflow"}


def read_voc(xml_path: Path, source: str) -> dict:
    """VOC xml 1개를 통일 클래스 박스로 바꾼다. 매핑에 없는 클래스는 ValueError."""
    mapping, boxes, dropped = MAPPINGS[source], [], Counter()
    root = ET.parse(xml_path).getroot()
    for o in root.findall("object"):
        name = o.find("name").text
        if name not in mapping:
            raise ValueError(f"{xml_path}: 모르는 클래스 {name!r} (출처 {source})")
        b = o.find("bndbox")
        box = [int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if mapping[name] is None:
            dropped[name] += 1
        else:
            boxes.append({"cls": mapping[name], "box": box})
    size = root.find("size")
    return {"width": int(size.find("width").text), "height": int(size.find("height").text),
            "boxes": boxes, "dropped": dict(dropped), "annotated": list(ANNOTATED[source])}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="split manifest의 이미지마다 VOC 라벨을 통일 클래스로 바꿔 jsonl로 쓴다.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count, dropped = Counter(), Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            src = SOURCE_OF_DIR[r["dataset"]]
            lab = read_voc(Path(r["path"]).with_suffix(".xml"), src)
            f.write(json.dumps({**r, "source": src, **lab}, ensure_ascii=False) + "\n")
            count.update(b["cls"] for b in lab["boxes"])
            dropped.update(lab["dropped"])
    print(f"{len(rows)}장 변환. 통일 클래스 박스 수 {dict(count)}, 버린 클래스 {dict(dropped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
