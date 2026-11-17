import numpy as np
import pytest
from PIL import Image

from sitesafe.data.dedup import cross_overlap, main, near_duplicates, phash_matrix


def _img(path, seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(small).resize((64, 64), Image.NEAREST).save(path)


def test_same_image_resized_is_overlap(tmp_path):
    _img(tmp_path / "a.png", 1)
    Image.open(tmp_path / "a.png").resize((128, 128)).save(tmp_path / "a2.png")
    _img(tmp_path / "b.png", 2)
    a = phash_matrix([tmp_path / "a.png"])
    assert cross_overlap(a, phash_matrix([tmp_path / "a2.png"])) == [0]
    assert cross_overlap(a, phash_matrix([tmp_path / "b.png"])) == []


def test_near_duplicates_and_empty(tmp_path):
    _img(tmp_path / "a.png", 1)
    _img(tmp_path / "a_copy.png", 1)
    _img(tmp_path / "b.png", 2)
    h = phash_matrix([tmp_path / n for n in ("a.png", "a_copy.png", "b.png")])
    assert near_duplicates(h) == [0, 1]
    assert near_duplicates(h[:1]) == [] and cross_overlap(h, h[:0]) == []


def test_main_exits_on_overlap(tmp_path):
    for d in ("train", "test"):
        (tmp_path / d).mkdir()
        _img(tmp_path / d / "x.png", 1)
    with pytest.raises(SystemExit):
        main([str(tmp_path / "train"), str(tmp_path / "test")])
    (tmp_path / "test" / "x.png").unlink()
    _img(tmp_path / "test" / "y.png", 2)
    main([str(tmp_path / "train"), str(tmp_path / "test")])
