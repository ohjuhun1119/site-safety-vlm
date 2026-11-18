import json

import pytest

from sitesafe.data.to_clip_label import clip_label, main, read_samples, split_dev


def test_clip_label_unsafe_flag():
    assert clip_label("Opened Panel Cover") == {"behavior": "Opened Panel Cover", "unsafe": True}
    assert clip_label("Safe Walkway") == {"behavior": "Safe Walkway", "unsafe": False}
    with pytest.raises(ValueError):
        clip_label("Unknown")


def test_split_dev_per_class_and_deterministic():
    rows = [{"file": f"0_tr{i}.mp4", "label": "Safe Walkway", "split": "train"} for i in range(20)]
    rows += [{"file": "1_tr0.mp4", "label": "Safe Carrying", "split": "train"},
             {"file": "1_te0.mp4", "label": "Safe Carrying", "split": "test"}]
    a = split_dev(rows)
    assert sum(r["split"] == "dev" for r in a) == 3  # 20개의 10% = 2, 1개 클래스는 최소 1개
    assert a == split_dev(rows)
    assert [r["split"] for r in a if r["file"] == "1_te0.mp4"] == ["test"]


def test_read_samples_rejects_tag_filename_mismatch(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"samples": [{"filepath": "d/0_tr1.mp4", "tags": ["test"],
                                          "ground_truth": {"label": "Safe Walkway"}}]}))
    with pytest.raises(ValueError):
        read_samples(p)


def test_main_writes_video_rows(tmp_path):
    (tmp_path / "clips").mkdir()
    s = []
    for f, lab in [("0_tr1.mp4", "Safe Walkway Violation"), ("0_te1.mp4", "Safe Walkway")]:
        (tmp_path / "clips" / f).write_bytes(b"x")
        s.append({"filepath": f"d/{f}", "tags": ["train" if "_tr" in f else "test"],
                  "ground_truth": {"label": lab}})
    (tmp_path / "s.json").write_text(json.dumps({"samples": s}))
    main(["--samples", str(tmp_path / "s.json"), "--clips", str(tmp_path / "clips"), "--out", str(tmp_path / "out")])
    row = json.loads((tmp_path / "out" / "test.jsonl").read_text().splitlines()[0])
    assert row["videos"] == ["../clips/0_te1.mp4"]
    assert "<|video_pad|>" in row["messages"][0]["content"]
    assert json.loads(row["messages"][1]["content"]) == {"behavior": "Safe Walkway", "unsafe": False}
    assert json.loads((tmp_path / "out" / "dev.jsonl").read_text().splitlines()[0])["videos"] == ["../clips/0_tr1.mp4"]
