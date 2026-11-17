import pytest

from sitesafe.data.normalize import ANNOTATED, MAPPINGS, UNIFIED, read_voc

XML = """<annotation><size><width>640</width><height>640</height></size>{}</annotation>"""
OBJ = "<object><name>{}</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30.0</xmax><ymax>40</ymax></bndbox></object>"


def _xml(tmp_path, *names):
    p = tmp_path / "a.xml"
    p.write_text(XML.format("".join(OBJ.format(n) for n in names)))
    return p


def test_mappings_only_use_unified_classes():
    for m in MAPPINGS.values():
        assert {v for v in m.values() if v} <= set(UNIFIED)
    assert "no_vest" not in ANNOTATED["chvg"] and "person" not in ANNOTATED["roboflow"]


def test_chvg_maps_colors_head_and_drops_glass(tmp_path):
    r = read_voc(_xml(tmp_path, "yellow", "head", "glass", "person", "vest"), "chvg")
    assert [b["cls"] for b in r["boxes"]] == ["helmet", "no_helmet", "person", "vest"]
    assert r["dropped"] == {"glass": 1} and r["boxes"][0]["box"] == [1, 2, 30, 40]


def test_roboflow_maps(tmp_path):
    r = read_voc(_xml(tmp_path, "No_Vest", "Helmet"), "roboflow")
    assert [b["cls"] for b in r["boxes"]] == ["no_vest", "helmet"]


def test_unknown_class_raises(tmp_path):
    with pytest.raises(ValueError):
        read_voc(_xml(tmp_path, "gloves"), "chvg")
