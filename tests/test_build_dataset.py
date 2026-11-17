import json

import pytest

from sitesafe.train.build_dataset import make_row, write_jsonl


def test_make_row_images_and_videos():
    row = make_row("images", ["a.jpg", "b.jpg"], "q", {"workers": 2})
    assert row["images"] == ["a.jpg", "b.jpg"]
    assert row["messages"][-1]["role"] == "assistant"
    assert json.loads(row["messages"][-1]["content"]) == {"workers": 2}
    assert "videos" in make_row("videos", ["clips/a.mp4"], "q", {"unsafe": False})


def test_make_row_rejects_bad_input():
    with pytest.raises(ValueError):
        make_row("audio", ["x"], "q", {})
    with pytest.raises(ValueError):
        make_row("images", [], "q", {})


def test_write_jsonl(tmp_path):
    p = tmp_path / "sub" / "train.jsonl"
    write_jsonl([make_row("images", ["a.jpg"], "한글 질문", {"k": "v"})], p)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and "한글 질문" in lines[0]
