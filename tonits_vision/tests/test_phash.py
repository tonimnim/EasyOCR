import io
import random

import pytest
from PIL import Image

from tonits_vision.phash import hamming, phash
from tonits_vision.synth import render as R


def screen(seed):
    return R.render(random.Random(seed), R.available_fonts(), "result").image


def test_hash_is_16_hex_chars_and_stable():
    img = screen(1)
    h = phash(img)
    assert len(h) == 16 and int(h, 16) >= 0
    assert phash(img) == h


def test_recompression_and_resize_barely_change_it():
    img = screen(2)
    buf = io.BytesIO()
    img.resize((img.width // 2, img.height // 2)).save(buf, format="JPEG", quality=40)
    assert hamming(phash(img), phash(Image.open(buf))) <= 6


def test_matches_the_imagehash_library():
    imagehash = pytest.importorskip("imagehash")
    for seed in range(5):
        img = screen(seed)
        assert phash(img) == str(imagehash.phash(img))
