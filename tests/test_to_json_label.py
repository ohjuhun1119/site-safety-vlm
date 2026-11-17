import pytest

from sitesafe.data.to_json_label import apply_fixes, ppe_label

B = lambda cls, box: {"cls": cls, "box": box}


def test_chvg_person_without_vest_counts():
    boxes = [B("person", [0, 0, 100, 300]), B("person", [200, 0, 300, 300]),
             B("vest", [10, 80, 90, 200]), B("helmet", [20, 0, 80, 40]), B("no_helmet", [220, 0, 280, 40])]
    assert ppe_label("chvg", boxes) == {"workers": 2, "no_helmet": 1, "no_vest": 1, "unclear": 0}


def test_chvg_person_without_head_label_is_unclear():
    assert ppe_label("chvg", [B("person", [0, 0, 100, 300])]) == {"workers": 1, "no_helmet": 0, "no_vest": 1, "unclear": 1}


def test_roboflow_workers_from_boxes_and_validates():
    boxes = [B("no_helmet", [0, 0, 5, 5]), B("no_helmet", [9, 0, 14, 5]), B("no_vest", [0, 5, 9, 20]),
             B("vest", [9, 5, 20, 20]), B("helmet", [30, 0, 40, 9])]
    assert ppe_label("roboflow", boxes) == {"workers": 3, "no_helmet": 2, "no_vest": 1, "unclear": 0}


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        ppe_label("x", [])


def test_chvg_head_only_workers_count_as_no_vest():
    boxes = [B("person", [0, 0, 100, 300]), B("no_helmet", [10, 0, 50, 40]),
             B("no_helmet", [200, 0, 240, 40]), B("no_helmet", [300, 0, 340, 40])]
    assert ppe_label("chvg", boxes) == {"workers": 3, "no_helmet": 3, "no_vest": 3, "unclear": 0}


def test_apply_fixes_overrides_and_revalidates():
    base = {"workers": 2, "no_helmet": 0, "no_vest": 1, "unclear": 1}
    assert apply_fixes(base, {"no_helmet": 1, "unclear": 0, "note": "x"}) == {"workers": 2, "no_helmet": 1, "no_vest": 1, "unclear": 0}
    with pytest.raises(ValueError):
        apply_fixes(base, {"no_vest": 3})  # 작업자 수보다 큰 미착용 수
