"""Read an eFootball Full Time screenshot into the reader contract's JSON.

Pipeline: decode → scale to a fixed height → CRAFT word boxes → first recognition pass → layout rules
(layout.py) → re-read the chosen boxes with tight character sets (digits for scores and stats) →
plausibility checks → confidence → response.
"""

import io
import logging
import os
import string
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from . import layout as L
from .checks import run_checks
from .glyphs import score_box
from .phash import phash
from .stats import FULL_TIME_CAPTION, STAT_ROWS

log = logging.getLogger("tonits_vision")

# Bump when the layout rules or confidence formula change: readings depend on them as much as on weights.
RULES_VERSION = 1

# The game UI scales with screen height, so every image is brought to one height. Crops are read from the
# work frame; the first, whole-image detection runs on a smaller copy because it only has to find the
# labels and the banner. Scores are found by shape (glyphs.py); the banner strip is re-detected at
# BANNER_ZOOM only when the first pass misses a team name.
WORK_HEIGHT = 720  # 826x371 becomes 1603x720
DETECT_HEIGHT = 540
BANNER_ZOOM = 2.0
MAX_PIXELS = 40_000_000
# Screenshot geometry. Every real capture so far is 16:9 to 20:9 and at least 371 px high; anything far
# outside this is not a screenshot, and scaling a sliver to WORK_HEIGHT would build an enormous frame
# (8192x10 becomes 589824x720: 1.2 GiB before line removal).
MIN_SIDE = 240
MAX_ASPECT = 4.0
MAX_WORK_WIDTH = int(WORK_HEIGHT * MAX_ASPECT) + 1
DETECT_PARAMS = dict(min_size=8, text_threshold=0.6, low_text=0.35, link_threshold=0.4, canvas_size=2560,
                     mag_ratio=1.0, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.6, width_ths=0.15, add_margin=0.08)
DIGITS = "0123456789"
# Reported but not held against confidence: a real screenshot (FC Barcelona 4-1, 2026-10-08) breaks
# "goals + opponent's saves <= shots on target" on both sides (4 + 7 > 10, 1 + 3 > 3), so eFootball
# doesn't count saves the way the rule assumes.
INFORMATIONAL_CHECKS = {"goalsPlusSavesWithinShotsOnTarget"}


class ImageError(ValueError):
    """The bytes are not a decodable image (→ 422)."""


class ImageTooLarge(ValueError):
    """The image declares more pixels than MAX_PIXELS (→ 413)."""


class UnsupportedGeometry(ImageError):
    """Too small, or too long and thin, to be a screenshot (→ 422 unsupported_geometry)."""


class UnreadableResult(ValueError):
    """A result screen whose banner (a team name or a score) can't be read (→ 422).

    The API rejects a match_result without both sides, and reading the same image again gives the same
    answer, so this fails the reading for good and staff decide.
    """


def decode_image(data: bytes) -> np.ndarray:
    """Bytes → RGB uint8 array. Checks the declared size before decoding, so a small file can't claim
    50 000 x 50 000 pixels and exhaust memory."""
    if not data:
        raise ImageError("empty body")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
    except Image.DecompressionBombError as exc:
        raise ImageTooLarge(str(exc)) from exc
    except Exception as exc:
        raise ImageError("not a decodable image") from exc
    if w * h > MAX_PIXELS:
        raise ImageTooLarge(f"{w}x{h} is over {MAX_PIXELS} pixels")
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_ASPECT * min(w, h):
        raise UnsupportedGeometry(f"{w}x{h} isn't screenshot-shaped (shorter side >= {MIN_SIDE}, at most {MAX_ASPECT:g}:1)")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ImageError("not a decodable image")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


class ScreenshotReader:
    """Loads EasyOCR once; `read()` is not thread-safe, so callers serialise it (the server does)."""

    def __init__(self, recognizer=None, model_dir=None, user_network_dir=None, download=None, torch_threads=None):
        import easyocr
        import torch

        if torch_threads:
            torch.set_num_threads(int(torch_threads))
        self.recognizer = recognizer or os.environ.get("VISION_RECOGNIZER", "english_g2")
        model_dir = model_dir or os.environ.get("VISION_MODEL_DIR") or str(Path.home() / ".EasyOCR" / "model")
        user_network_dir = user_network_dir or os.environ.get("VISION_USER_NETWORK_DIR") or model_dir
        if download is None:
            download = os.environ.get("VISION_DOWNLOAD_MODELS", "1") != "0"

        recognizer_release = 0
        custom_yaml = Path(user_network_dir) / f"{self.recognizer}.yaml"
        if custom_yaml.exists():
            recognizer_release = yaml.safe_load(custom_yaml.read_text(encoding="utf8")).get("release", 0)
        self.version = f"{easyocr.__version__}+tonits-efootball-{RULES_VERSION}.{recognizer_release}"

        self.ocr = easyocr.Reader(["en"], gpu=False, model_storage_directory=model_dir,
                                  user_network_directory=user_network_dir, recog_network=self.recognizer,
                                  download_enabled=download, verbose=False)
        self.charset = set(self.ocr.character)
        self.has_accents = any(ord(c) > 127 and c.isalpha() for c in self.charset)
        log.info("reader loaded: recognizer=%s version=%s", self.recognizer, self.version)

    @property
    def model_info(self):
        return {"engine": "easyocr", "version": self.version, "recognizer": self.recognizer}

    # -- OCR primitives -------------------------------------------------------------------------------

    def _detect(self, rgb):
        horizontal, free = self.ocr.detect(rgb, **DETECT_PARAMS)
        boxes = [tuple(int(v) for v in b) for b in horizontal[0]]  # (x_min, x_max, y_min, y_max)
        for quad in free[0]:
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            boxes.append((int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))))
        h, w = rgb.shape[:2]
        return [(max(0, x0), max(0, y0), min(w, x1), min(h, y1)) for x0, x1, y0, y1 in boxes if x1 - x0 >= 2 and y1 - y0 >= 2]

    def _recognize(self, grey, boxes, allowlist=None):
        """boxes as (x0, y0, x1, y1) → [(text, conf)] in the same order."""
        if not boxes:
            return []
        h, w = grey.shape[:2]
        ints = [(max(0, int(b[0])), max(0, int(b[1])), min(w, int(round(b[2]))), min(h, int(round(b[3])))) for b in boxes]
        valid = [b for b in ints if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        if not valid:
            return [("", 0.0)] * len(boxes)
        hlist = [[x0, x1, y0, y1] for x0, y0, x1, y1 in valid]
        # Without an allowlist EasyOCR drops every character outside the language's alphabet (accents
        # included), so the general pass allows everything the recognizer can output.
        allow = "".join(c for c in allowlist if c in self.charset) if allowlist else self.ocr.character
        results = self.ocr.recognize(grey, horizontal_list=hlist, free_list=[], allowlist=allow, detail=1,
                                     paragraph=False, batch_size=16, reformat=False)
        by_corner = {}
        for pts, text, conf in results:
            by_corner.setdefault((int(pts[0][0]), int(pts[0][1]), int(pts[2][0]), int(pts[2][1])), (text, float(conf)))
        out = []
        for x0, y0, x1, y1 in ints:
            key = (x0, y0, x1, y1)
            if key not in by_corner:  # clipped at the image edge
                key = next((k for k in by_corner if abs(k[0] - x0) <= 1 and abs(k[1] - y0) <= 1), None)
            out.append(by_corner.get(key, ("", 0.0)))
        return out

    # -- the pipeline -----------------------------------------------------------------------------------

    def read(self, data: bytes, with_boxes=False) -> dict:
        start = time.perf_counter()
        rgb = decode_image(data)
        image_hash = phash(Image.fromarray(rgb))
        result = self.read_array(rgb, with_boxes=with_boxes)
        result["imageHash"] = image_hash
        result["model"] = self.model_info
        result["durationMs"] = int((time.perf_counter() - start) * 1000)
        return result

    def read_array(self, rgb: np.ndarray, with_boxes=False) -> dict:
        lay, grey, scale = self.locate(rgb)
        if lay.screen != "match_result":
            return unknown_reading()
        reading = self._second_pass(grey, lay)
        if not reading["left"] or not reading["right"]:
            missing = [side for side in ("left", "right") if not reading[side]]
            raise UnreadableResult(f"result screen, but the {' and '.join(missing)} team name or score can't be read")
        if with_boxes:
            reading["boxes"] = {k: [round(v / scale) for v in box] for k, box in self._boxes(lay).items()}
        return reading

    def locate(self, rgb: np.ndarray):
        """Find every field: (layout in work-frame pixels, cleaned grey work frame, work/original scale)."""
        h0, w0 = rgb.shape[:2]
        scale = work_scale(w0, h0)
        work = cv2.resize(rgb, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        work = remove_rule_lines(work)
        grey = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

        d = DETECT_HEIGHT / WORK_HEIGHT
        small = cv2.resize(work, (max(1, round(work.shape[1] * d)), max(1, round(work.shape[0] * d))), interpolation=cv2.INTER_AREA)
        boxes = [tuple(v / d for v in b) for b in self._detect(small)]
        first = self._recognize(grey, boxes)
        tokens = [L.Token(text, conf, box) for box, (text, conf) in zip(boxes, first) if text.strip()]
        lay = L.parse(tokens)
        if lay.screen == "match_result":
            # Usually the first pass has both team names and the scores are found by shape; the zoomed
            # re-detection of the banner (the slowest step after the first detection) is the fallback.
            if not (lay.banner_tokens and self._banner_from_line(grey, lay.banner_tokens, lay, require_complete=True)):
                self._reread_banner(work, grey, lay)
        return lay, grey, scale

    def _banner_from_line(self, grey, line, lay, require_complete):
        """Scores by shape beside the logo plus team names from this banner line's tokens. Writes them to
        lay when complete (both scores and both names), or when require_complete is off and both scores
        were found or lay has none yet. Returns whether the result was complete."""
        line_h = float(np.median([t.h for t in line]))
        band = (min(t.y0 for t in line), max(t.y1 for t in line))
        found_boxes = tuple(score_box(grey, lay.center_x, band, line_h, side) for side in (-1, 1))
        parsed = L.Layout(center_x=lay.center_x)
        found = L.parse_banner(line, lay.center_x, parsed, found_boxes)
        # A bracketed penalty on one side only means the other one was missed: not complete.
        complete = (found and parsed.left_team is not None and parsed.right_team is not None
                    and (parsed.left_penalty is None) == (parsed.right_penalty is None))
        have = lay.left_score is not None and lay.right_score is not None
        if complete or (not require_complete and (found or not have)):
            for name in ("left_team", "right_team", "left_score", "right_score", "left_penalty", "right_penalty"):
                setattr(lay, name, getattr(parsed, name))
        return complete

    def _reread_banner(self, work, grey, lay):
        """Re-detect the banner strip at a higher zoom and parse it again."""
        y0, y1 = lay.banner_band
        pad = 0.45 * (y1 - y0)
        top, bottom = max(0, int(y0 - pad)), min(work.shape[0], int(y1 + pad))
        if bottom - top < 8:
            return
        strip = cv2.resize(work[top:bottom], None, fx=BANNER_ZOOM, fy=BANNER_ZOOM, interpolation=cv2.INTER_CUBIC)
        boxes = [(x0 / BANNER_ZOOM, y0_ / BANNER_ZOOM + top, x1 / BANNER_ZOOM, y1_ / BANNER_ZOOM + top)
                 for x0, y0_, x1, y1_ in self._detect(strip)]
        reads = self._recognize(grey, boxes)
        tokens = [L.Token(text, conf, box) for box, (text, conf) in zip(boxes, reads) if text.strip()]
        line = L.banner_line(L.group_lines(tokens), lay.center_x)
        if line:
            self._banner_from_line(grey, line, lay, require_complete=False)

    def _second_pass(self, grey, lay):
        # Scores and stats: digits only. Possession keeps "%". Names: the general recognizer on the
        # union of their word boxes, so spaces come out as printed.
        score_fields = [lay.left_score, lay.right_score]
        score_reads = self._recognize(grey, [pad_box(f.box, 0.25) for f in score_fields if f], DIGITS)
        scores, score_confs = [], []
        it = iter(score_reads)
        for f in score_fields:
            if f is None:
                scores.append(None)
                score_confs.append(0.0)
                continue
            text, conf = next(it)
            value = L.as_digits(text)
            first = L.as_digits(f.text)
            if value is None:
                value, conf = first, f.conf * 0.5
            elif first is not None and first != value:
                conf *= 0.6  # the two passes disagree
            scores.append(value)
            score_confs.append(conf if value is not None else 0.0)

        name_fields = [lay.left_team, lay.right_team, lay.left_manager, lay.right_manager]
        boxes = [f.box for f in name_fields if f]
        name_reads = self._recognize(grey, boxes)
        ascii_reads = self._recognize(grey, boxes, ASCII_NAME_CHARS) if self.has_accents else name_reads
        it = iter(zip(name_reads, ascii_reads))
        names = []
        for f in name_fields:
            if f is None:
                names.append((None, 0.0))
                continue
            (text, conf), (plain, plain_conf) = next(it)
            text, conf = settle_accents(" ".join(text.split()), conf, " ".join(plain.split()), plain_conf)
            if not text:
                text, conf = f.text, f.conf * 0.5
            names.append((text, conf))
        (lt, ltc), (rt, rtc), (lm, _), (rm, _) = names

        # Stat values: read every cell of the grid, digits only.
        cells = L.value_cells(lay)
        glyph_h = lay.value_columns[2] if lay.value_columns else 0
        cells = {k: tuple(tighten(grey, b, glyph_h) if b else None for b in pair) for k, pair in cells.items()}
        lay.cells = cells
        jobs = [(key, side, box) for key, pair in cells.items() for side, box in enumerate(pair) if box]
        stats = {}
        for group, allow in (([j for j in jobs if j[0] == "possession"], DIGITS + "%"),
                             ([j for j in jobs if j[0] != "possession"], DIGITS)):
            reads = self._recognize(grey, [clip_box(b, grey.shape) for _, _, b in group], allow)
            for (key, side, _), (text, conf) in zip(group, reads):
                value = L.as_digits(text, allow_percent=True)
                if value is not None and conf >= 0.2:
                    stats.setdefault(key, [None, None])[side] = value
        stats = {row.key: stats[row.key] for row in STAT_ROWS if row.key in stats and None not in stats[row.key]}

        penalties = None
        if lay.left_penalty and lay.right_penalty:
            reads = self._recognize(grey, [lay.left_penalty.box, lay.right_penalty.box], "()" + DIGITS)
            pl = L.as_digits(reads[0][0].strip("()[]{} "))
            pr = L.as_digits(reads[1][0].strip("()[]{} "))
            if pl is not None and pr is not None:
                penalties = {"left": pl, "right": pr}
        elif lay.penalty_line:
            pk = L.parse_penalty_text(lay.penalty_line.text)
            if pk:
                penalties = {"left": pk[0], "right": pk[1]}

        full_time = lay.caption == FULL_TIME_CAPTION
        left = right = None
        if lt and scores[0] is not None and scores[0] <= 99:
            left = {"team": lt, "score": scores[0], "confidence": round(min(ltc, score_confs[0]), 3)}
        if rt and scores[1] is not None and scores[1] <= 99:
            right = {"team": rt, "score": scores[1], "confidence": round(min(rtc, score_confs[1]), 3)}
        checks = run_checks(scores[0], scores[1], stats)
        return {
            "screen": "match_result",
            "fullTime": full_time,
            "left": left,
            "right": right,
            "penalties": penalties,
            "managers": {"left": lm, "right": rm} if lm and rm else None,
            "stats": stats,
            "checks": checks,
            "confidence": round(overall_confidence(left, right, checks, stats, full_time, lay.caption_conf), 3),
        }

    @staticmethod
    def _boxes(lay):
        boxes = {}
        for name in ("left_team", "right_team", "left_score", "right_score", "left_penalty", "right_penalty",
                     "left_manager", "right_manager", "penalty_line"):
            f = getattr(lay, name)
            if f:
                boxes[name] = f.box
        for key, (lv, rv) in getattr(lay, "cells", {}).items():
            if lv:
                boxes[f"{key}.left"] = lv
            if rv:
                boxes[f"{key}.right"] = rv
        return boxes


def tighten(grey, box, glyph_h):
    """Shrink a cell to the glyphs in it, or None if it holds none.

    The cell is thresholded (Otsu; the minority class is ink, so light-on-dark and dark-on-light both
    work), and only glyph-shaped components are kept: not touching the top or bottom edge (the next row's
    digits), not long and thin (a leftover row divider), not specks.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in clip_box(box, grey.shape))
    crop = grey[y0:y1, x0:x1]
    if crop.size == 0 or crop.max() - crop.min() < 40:
        return None
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = []
    for i in range(1, n):
        cx0, cy0, cw, ch, area = stats[i]
        if ch < 0.35 * glyph_h or area < 0.04 * glyph_h * glyph_h:
            continue
        if cw > 2.5 * ch and ch < 0.5 * glyph_h:
            continue
        if cy0 == 0 or cy0 + ch >= crop.shape[0]:
            continue
        keep.append((cx0, cy0, cx0 + cw, cy0 + ch))
    if not keep:
        return None
    gx0, gy0 = min(k[0] for k in keep), min(k[1] for k in keep)
    gx1, gy1 = max(k[2] for k in keep), max(k[3] for k in keep)
    m = 0.2 * (gy1 - gy0)
    return (x0 + max(0, gx0 - m), y0 + max(0, gy0 - m), x0 + min(crop.shape[1], gx1 + m), y0 + min(crop.shape[0], gy1 + m))


ASCII_NAME_CHARS = string.ascii_letters + string.digits + " .,-_'&!?()[]{}|/:+*#@$%"


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def settle_accents(text, conf, plain, plain_conf):
    """Keep a name's accented letters only where an ASCII-only read of the same crop agrees on the base
    letter, case included: "Atlético" (plain read "Atletico") keeps its accent; "ñarcelona" (plain read
    "Barcelona") and "ēmirhann" ("Emirhann") don't. A recognizer unsure of a font spreads probability over
    accented look-alikes; when the two reads disagree, the lower confidence is kept."""
    if text.isascii() or not plain:
        return text, conf
    if strip_accents(text) == plain:
        return text, conf
    return plain, min(conf, plain_conf)


def work_scale(w, h):
    """Scale to WORK_HEIGHT, but never wider than MAX_WORK_WIDTH (a backstop behind decode_image's checks)."""
    return min(WORK_HEIGHT / h, MAX_WORK_WIDTH / w)


def pad_box(box, share):
    """Grow a box by share of its height on every side."""
    m = share * (box[3] - box[1])
    return (box[0] - m, box[1] - m, box[2] + m, box[3] + m)


def clip_box(box, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


def remove_rule_lines(rgb):
    """Paint out thin horizontal lines (the table's row dividers) so CRAFT doesn't join digits across rows."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    long_run = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, rgb.shape[1] // 40), 1))
    mask = np.zeros_like(grey)
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        _, bw = cv2.threshold(cv2.morphologyEx(grey, op, vertical), 40, 255, cv2.THRESH_BINARY)
        mask |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, long_run)
    if not mask.any():
        return rgb
    return cv2.inpaint(rgb, cv2.dilate(mask, np.ones((3, 1), np.uint8)), 2, cv2.INPAINT_TELEA)


def unknown_reading():
    return {"screen": "unknown", "fullTime": False, "left": None, "right": None, "penalties": None,
            "managers": None, "stats": None, "checks": None, "confidence": 0.0}


def overall_confidence(left, right, checks, stats, full_time, caption_conf):
    """Belief that both scores and both team names are right. Heuristic until calibrated (see the review)."""
    if not left or not right:
        return 0.0
    conf = min(left["confidence"], right["confidence"])
    conf *= 0.5 ** sum(1 for k, v in checks.items() if v is False and k not in INFORMATIONAL_CHECKS)
    if any(v is None for v in checks.values()):
        conf *= 0.9
    conf *= 0.97 ** (len(STAT_ROWS) - len(stats))
    if not full_time:
        conf = min(conf, 0.5)
    elif caption_conf < 0.9:
        conf *= 0.95
    return max(0.0, min(1.0, conf))
