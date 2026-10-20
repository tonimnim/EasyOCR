"""The whole reader with the real EasyOCR models. Slow (model load + a few seconds per image on CPU):

    RUN_SLOW=1 pytest tonits_vision/tests/test_reader_e2e.py

Put the real samples in tonits_vision/samples/ (sample1.*, sample2.*, sample3.*) to check them too;
that folder is git-ignored because the screenshots carry players' team names.
"""

import io
import os
import random
from pathlib import Path

import pytest

from tonits_vision.synth import render as R

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to load the OCR models")

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

# The three samples from the design doc, plus sample 4 (a real console-style screenshot). Samples 3 and 4
# are the real screenshots shared on 2026-10-08; samples 1 and 2 are checked from the doc's values.
EXPECTED = {
    "sample1": {"left": ("Argentina", 3), "right": ("shinegum", 3)},
    "sample2": {"left": ("Mugz FC", 2), "right": ("SQUAD 0", 1),
                "managers": {"left": "Ruben Amorim", "right": "R. Martínez"},
                "stats": {"possession": [64, 36], "shots": [16, 1], "shotsOnTarget": [12, 1], "fouls": [0, 0],
                          "offsides": [0, 0], "cornerKicks": [3, 0], "freeKicks": [0, 0], "passes": [156, 96],
                          "successfulPasses": [122, 66], "crosses": [4, 0], "interceptions": [28, 24],
                          "tackles": [5, 5], "saves": [0, 7]}},
    "sample4": {"left": ("FC Barcelona", 4), "right": ("Emirhann CHAMPION", 1),
                "managers": {"left": "Fabio Capello", "right": "D. Deschamps"},
                "stats": {"possession": [55, 45], "shots": [14, 5], "shotsOnTarget": [10, 3], "fouls": [3, 0],
                          "offsides": [0, 0], "cornerKicks": [2, 1], "freeKicks": [0, 3], "passes": [130, 157],
                          "successfulPasses": [106, 120], "crosses": [3, 0], "interceptions": [30, 19],
                          "tackles": [5, 3], "saves": [3, 7]}},
    "sample3": {"left": ("mzee mzima", 3), "right": ("CR Flamengo", 0),
                "stats": {"possession": [50, 50], "shots": [4, 9], "shotsOnTarget": [3, 6], "fouls": [1, 3],
                          "offsides": [1, 3], "cornerKicks": [0, 3], "freeKicks": [3, 1], "passes": [90, 83],
                          "successfulPasses": [60, 57], "crosses": [1, 0], "interceptions": [16, 26],
                          "tackles": [4, 4], "saves": [3, 0]}},
}


@pytest.fixture(scope="module")
def reader():
    from tonits_vision.reader import ScreenshotReader
    return ScreenshotReader()


def check(reading, expected):
    assert reading["screen"] == "match_result" and reading["fullTime"] is True
    assert (reading["left"]["team"], reading["left"]["score"]) == expected["left"]
    assert (reading["right"]["team"], reading["right"]["score"]) == expected["right"]
    if "stats" in expected:
        assert reading["stats"] == expected["stats"]
    if "managers" in expected:
        assert reading["managers"] == expected["managers"]
    assert len(reading["imageHash"]) == 16


def test_replica_of_sample3(reader):
    rng = random.Random(3)
    label = R.random_label(rng, "phone")
    label.update(left={"team": "mzee mzima", "score": 3}, right={"team": "CR Flamengo", "score": 0},
                 penalties=None, managers=None, stats=EXPECTED["sample3"]["stats"])
    canvas = R._render_result(rng, R.available_fonts(), label, "phone", "Full Time")
    buf = io.BytesIO()
    canvas.image.save(buf, format="PNG")
    reading = reader.read(buf.getvalue())
    check(reading, EXPECTED["sample3"])
    assert all(reading["checks"].values())


# The stock english_g2 reads eFootball's stylised "F" as "T" ("TC Barcelona"). efootball_v1 reads sample 4
# exactly (run with VISION_RECOGNIZER=efootball_v1 VISION_MODEL_DIR=tonits_vision/models).
KNOWN_STOCK_MISSES = {"sample4"}


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(reason="stock recognizer misreads the eFootball F", strict=False))
                                  if n in KNOWN_STOCK_MISSES else n for n in sorted(EXPECTED)])
def test_real_samples(reader, name):
    found = sorted(SAMPLES.glob(f"{name}.*")) if SAMPLES.exists() else []
    if not found:
        pytest.skip(f"no {name}.* in {SAMPLES}")
    check(reader.read(found[0].read_bytes()), EXPECTED[name])


def test_not_a_result_screen(reader):
    s = R.render(random.Random(11), R.available_fonts(), "unknown")
    assert reader.read(s.image_bytes)["screen"] == "unknown"
