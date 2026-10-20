"""Optional second opinion from a vision LLM on low-confidence readings.

Off unless VISION_FALLBACK_API_KEY is set (an Anthropic API key), so the reader never pays for an
outside model by default. When on, a match_result reading with confidence below VISION_FALLBACK_BELOW
(default 0.6) is sent with its image to VISION_FALLBACK_MODEL, which reads the banner independently:

- it agrees on both team names, both scores, penalties and the Full Time caption: the reading is kept
  and each side's confidence is raised to VISION_FALLBACK_AGREED_CONFIDENCE (default 0.85, just under the
  API's 0.9 auto-decide threshold, so the LLM's word alone never settles a match: VISION_AUTO_MIN_CONFIDENCE
  is calibrated on OCR confidence, and an LLM answer isn't reproducible on a re-read);
- it disagrees on anything: the reading is kept, its confidence is capped at DISAGREED_CONFIDENCE (below
  any auto-decide threshold) and staff decide;
- it fails, times out or declines: the reading is returned unchanged.

It is skipped when less than MIN_TIME_LEFT remains of VISION_FALLBACK_DEADLINE (default 90 s, counted from
when the request arrived), so the whole read stays inside the API's timeout and a timed-out request is
never paid for twice. Keep VISION_TIMEOUT (API) > VISION_FALLBACK_DEADLINE > (1 + VISION_MAX_QUEUE) x the
slowest read.

The LLM never replaces the OCR's values: it can only confirm or cast doubt. The response gains a
`secondOpinion` object (Go ignores unknown fields; it shows up in the reader's logs).
"""

import base64
import io
import logging
import os
import unicodedata
from typing import Optional

from PIL import Image
from pydantic import BaseModel

from .checks import run_checks
from .reader import overall_confidence

log = logging.getLogger("tonits_vision.fallback")

DISAGREED_CONFIDENCE = 0.3
MIN_TIME_LEFT = 2.0  # seconds; less than this left of the deadline and the LLM isn't asked
MAX_CALL_SECONDS = 12.0
MAX_IMAGE_BYTES = 3_500_000  # stay well under the API's per-image limit
MAX_SIDE = 1568

PROMPT = """This should be the "Full Time" result screen of an eFootball match. Read it exactly as printed.

The banner near the top holds, left to right: the left team name, the left score, the eFootball logo (a
stylised "e", not a digit), the right score and the right team name. A caption under it says "Full Time"
(or "Half Time" etc.). A penalty shoot-out, if any, is shown as small numbers in brackets beside the
scores or as a line such as "PK 4 - 3".

Copy team names character for character, keeping case, spaces, accents, punctuation and any digits that
are part of the name (e.g. "SQUAD 0"). Use null for anything you can't read with certainty. If this is
not an eFootball result screen, set is_result_screen to false."""


class SecondReading(BaseModel):
    is_result_screen: bool
    caption: Optional[str]
    left_team: Optional[str]
    left_score: Optional[int]
    right_score: Optional[int]
    right_team: Optional[str]
    left_penalties: Optional[int]
    right_penalties: Optional[int]


def _name_key(name):
    """Case and spacing don't count; accents, punctuation and digits do."""
    return " ".join(unicodedata.normalize("NFC", name or "").casefold().split())


def differences(reading, other: SecondReading):
    diffs = []
    if not other.is_result_screen:
        return ["screen"]
    if (other.caption or "").strip().casefold() != "full time" and reading.get("fullTime"):
        diffs.append("fullTime")
    for side in ("left", "right"):
        mine = reading.get(side) or {}
        if mine.get("score") != getattr(other, f"{side}_score"):
            diffs.append(f"{side}.score")
        if _name_key(mine.get("team")) != _name_key(getattr(other, f"{side}_team")):
            diffs.append(f"{side}.team")
    pens = reading.get("penalties")
    theirs = None if other.left_penalties is None and other.right_penalties is None else {
        "left": other.left_penalties, "right": other.right_penalties}
    if pens != theirs:
        diffs.append("penalties")
    return diffs


def _image_block(data, content_type):
    """Base64 image for the API, shrunk and re-encoded when it is large."""
    with Image.open(io.BytesIO(data)) as img:
        if len(data) > MAX_IMAGE_BYTES or max(img.size) > 2 * MAX_SIDE:
            img = img.convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            data, content_type = buf.getvalue(), "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": content_type,
                                         "data": base64.standard_b64encode(data).decode("ascii")}}


class SecondOpinion:
    def __init__(self, client, model, below, deadline=90.0, agreed_confidence=0.85):
        self.client = client
        self.model = model
        self.below = below
        self.deadline = deadline
        self.agreed_confidence = agreed_confidence

    @classmethod
    def from_env(cls):
        key = os.environ.get("VISION_FALLBACK_API_KEY", "").strip()
        if not key:
            return None
        import anthropic  # only needed when the second opinion is on

        # No SDK retries: a retry would run past the deadline; the API retries the whole read if needed.
        client = anthropic.Anthropic(api_key=key, timeout=MAX_CALL_SECONDS, max_retries=0)
        model = os.environ.get("VISION_FALLBACK_MODEL", "").strip() or "claude-haiku-4-5"
        below = float(os.environ.get("VISION_FALLBACK_BELOW", "0.6"))
        deadline = float(os.environ.get("VISION_FALLBACK_DEADLINE", "90"))
        agreed = float(os.environ.get("VISION_FALLBACK_AGREED_CONFIDENCE", "0.85"))
        log.info("second opinion on: model=%s below=%s deadline=%ss agreed=%s", model, below, deadline, agreed)
        return cls(client, model, below, deadline, agreed)

    def wanted(self, reading):
        return reading.get("screen") == "match_result" and reading.get("confidence", 0.0) < self.below

    def time_left(self, elapsed):
        """Seconds the LLM may take for a request that arrived `elapsed` seconds ago, or 0 to skip it."""
        left = self.deadline - elapsed
        return min(MAX_CALL_SECONDS, left) if left >= MIN_TIME_LEFT else 0.0

    def ask(self, data, content_type, timeout=MAX_CALL_SECONDS) -> Optional[SecondReading]:
        import anthropic

        try:
            response = self.client.with_options(timeout=timeout, max_retries=0).messages.parse(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": [_image_block(data, content_type), {"type": "text", "text": PROMPT}]}],
                output_format=SecondReading,
            )
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            log.warning("second opinion unavailable: %s", exc)
            return None
        if response.stop_reason == "refusal" or response.parsed_output is None:
            log.warning("second opinion declined (stop_reason=%s)", response.stop_reason)
            return None
        return response.parsed_output

    def review(self, data, content_type, reading, timeout=MAX_CALL_SECONDS):
        other = self.ask(data, content_type, timeout)
        if other is None:
            return reading
        diffs = differences(reading, other)
        reading = dict(reading)
        reading["secondOpinion"] = {"model": self.model, "agrees": not diffs, "differences": diffs}
        if diffs:
            reading["confidence"] = min(reading["confidence"], DISAGREED_CONFIDENCE)
            return reading
        for side in ("left", "right"):
            reading[side] = dict(reading[side], confidence=max(reading[side]["confidence"], self.agreed_confidence))
        stats = reading.get("stats") or {}
        checks = run_checks(reading["left"]["score"], reading["right"]["score"], stats)
        reading["confidence"] = round(overall_confidence(reading["left"], reading["right"], checks, stats,
                                                         reading.get("fullTime"), 1.0), 3)
        return reading
