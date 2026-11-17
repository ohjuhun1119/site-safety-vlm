import json

import numpy as np
import pytest
from PIL import Image

from sitesafe.data.split import assign_splits, group_ids, main


def _bits(*rows):
    return np.array(rows, dtype=bool)


def test_group_ids_chain_connects():
    # a-b 거리 1, b-c 거리 1, a-c 거리 2: 임계 1이어도 연결 요소로 한 그룹
    a = np.zeros(64, bool)
    b = a.copy(); b[0] = True
    c = b.copy(); c[1] = True
    d = ~a
    g = group_ids(_bits(a, b, c, d), threshold=1)
    assert g[0] == g[1] == g[2] != g[3]


def test_assign_splits_keeps_groups_together_and_is_deterministic():
    groups = [i // 5 for i in range(100)]  # 5장짜리 그룹 20개
    s = assign_splits(groups, seed=1)
    assert s == assign_splits(groups, seed=1)
    for g in set(groups):
        assert len({s[i] for i in range(100) if groups[i] == g}) == 1
    assert {sp: s.count(sp) for sp in ("train", "valid", "test")} == {"train": 80, "valid": 10, "test": 10}


def test_assign_splits_rejects_bad_ratios():
    with pytest.raises(ValueError):
        assign_splits([0, 1], ratios=(0.5, 0.5))


def test_main_puts_copies_in_same_split(tmp_path):
    rng = np.random.default_rng(0)
    for d in ("x", "y"):
        (tmp_path / d).mkdir()
    for n in range(20):
        small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        im = Image.fromarray(small).resize((64, 64), Image.NEAREST)
        im.save(tmp_path / "x" / f"{n}.png")
        im.resize((128, 128)).save(tmp_path / "y" / f"{n}.png")  # 다른 데이터셋의 같은 사진
    out = tmp_path / "m.json"
    main([str(tmp_path / "x"), str(tmp_path / "y"), "--out", str(out)])
    rows = json.loads(out.read_text())
    assert len(rows) == 40
    for g in {r["group"] for r in rows}:
        assert len({r["split"] for r in rows if r["group"] == g}) == 1
