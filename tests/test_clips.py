import cv2
import numpy as np
import pytest

from sitesafe.vision.clips import make_clip


def _video(path, n=48, fps=24, size=(160, 90)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 5, np.uint8))
    w.release()


def test_make_clip_reduces_fps_and_size(tmp_path):
    _video(tmp_path / "a.mp4")  # 2초
    info = make_clip(tmp_path / "a.mp4", tmp_path / "o.mp4", fps=2, height=44)
    assert info == {"frames": 4, "width": 78, "height": 44}
    cap = cv2.VideoCapture(str(tmp_path / "o.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (78, 44)


def test_make_clip_short_clip_has_two_frames_and_bad_file_errors(tmp_path):
    _video(tmp_path / "s.mp4", n=6)  # 0.25초
    assert make_clip(tmp_path / "s.mp4", tmp_path / "o.mp4", fps=1, height=44)["frames"] == 2
    with pytest.raises(ValueError):
        make_clip(tmp_path / "none.mp4", tmp_path / "x.mp4", fps=1, height=44)
