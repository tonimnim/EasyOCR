"""The LLM second opinion, with a fake client: no network, no API key, no cost."""

import io
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from PIL import Image

from tonits_vision.fallback import DISAGREED_CONFIDENCE, MAX_CALL_SECONDS, SecondOpinion, SecondReading

SAMPLE3_STATS = {
    "possession": [50, 50], "shots": [4, 9], "shotsOnTarget": [3, 6], "fouls": [1, 3], "offsides": [1, 3],
    "cornerKicks": [0, 3], "freeKicks": [3, 1], "passes": [90, 83], "successfulPasses": [60, 57],
    "crosses": [1, 0], "interceptions": [16, 26], "tackles": [4, 4], "saves": [3, 0],
}


def reading(confidence=0.5, left_team="mzee mzima"):
    return {"screen": "match_result", "fullTime": True,
            "left": {"team": left_team, "score": 3, "confidence": confidence},
            "right": {"team": "CR Flamengo", "score": 0, "confidence": 0.99},
            "penalties": None, "managers": None, "stats": SAMPLE3_STATS, "checks": {}, "confidence": confidence}


def llm(**changes):
    values = dict(is_result_screen=True, caption="Full Time", left_team="mzee mzima", left_score=3, right_score=0,
                  right_team="CR Flamengo", left_penalties=None, right_penalties=None)
    values.update(changes)
    return SecondReading(**values)


class FakeMessages:
    def __init__(self, result=None, error=None, stop_reason="end_turn"):
        self.result, self.error, self.stop_reason, self.calls = result, error, stop_reason, []
        self.options = {}

    def parse(self, **kwargs):
        kwargs["options"] = self.options
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=self.result)


class FakeClient:
    def __init__(self, messages):
        self.messages = messages

    def with_options(self, **options):
        self.messages.options = options
        return self


def opinion(**fake):
    messages = FakeMessages(**fake)
    return SecondOpinion(FakeClient(messages), "claude-haiku-4-5", 0.6), messages


def png(size=(826, 371)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 90)).save(buf, format="PNG")
    return buf.getvalue()


def test_only_low_confidence_results_are_sent():
    so, _ = opinion(result=llm())
    assert so.wanted(reading(0.5))
    assert not so.wanted(reading(0.7))
    assert not so.wanted({"screen": "unknown", "confidence": 0.0})


def test_agreement_raises_confidence_but_not_to_auto_decide():
    so, messages = opinion(result=llm())
    out = so.review(png(), "image/png", reading(0.5), timeout=7.5)
    assert out["secondOpinion"] == {"model": "claude-haiku-4-5", "agrees": True, "differences": []}
    assert 0.8 <= out["confidence"] <= 0.85 < 0.9  # the API's default auto-decide threshold
    assert messages.calls[0]["options"] == {"timeout": 7.5, "max_retries": 0}
    assert (out["left"]["team"], out["left"]["score"]) == ("mzee mzima", 3)
    call = messages.calls[0]
    assert call["model"] == "claude-haiku-4-5" and call["output_format"] is SecondReading
    assert call["messages"][0]["content"][0]["type"] == "image"


def test_agreement_ignores_case_and_spacing_but_not_accents():
    so, _ = opinion(result=llm(left_team="MZEE  mzima"))
    assert so.review(png(), "image/png", reading(0.5))["secondOpinion"]["agrees"]
    so, _ = opinion(result=llm(right_team="CR Flaméngo"))
    assert not so.review(png(), "image/png", reading(0.5))["secondOpinion"]["agrees"]


@pytest.mark.parametrize("change,field", [({"left_score": 2}, "left.score"), ({"right_team": "Flamengo"}, "right.team"),
                                          ({"caption": "Half Time"}, "fullTime"), ({"left_penalties": 4, "right_penalties": 3}, "penalties")])
def test_disagreement_caps_confidence(change, field):
    so, _ = opinion(result=llm(**change))
    out = so.review(png(), "image/png", reading(0.55))
    assert field in out["secondOpinion"]["differences"]
    assert out["confidence"] <= DISAGREED_CONFIDENCE
    assert out["left"]["score"] == 3  # the OCR's values are never replaced


def test_not_a_result_screen_is_a_disagreement():
    so, _ = opinion(result=llm(is_result_screen=False))
    assert so.review(png(), "image/png", reading(0.5))["secondOpinion"]["differences"] == ["screen"]


def test_failures_leave_the_reading_unchanged():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for so, _ in (opinion(error=anthropic.APIConnectionError(request=request)),
                  opinion(result=None, stop_reason="refusal")):
        r = reading(0.5)
        assert so.review(png(), "image/png", r) == r


def test_large_images_are_shrunk_before_sending():
    so, messages = opinion(result=llm())
    so.review(png((5000, 2250)), "image/png", reading(0.5))
    block = messages.calls[0]["messages"][0]["content"][0]
    assert block["source"]["media_type"] == "image/jpeg"


def test_off_without_a_key(monkeypatch):
    monkeypatch.delenv("VISION_FALLBACK_API_KEY", raising=False)
    assert SecondOpinion.from_env() is None
    monkeypatch.setenv("VISION_FALLBACK_API_KEY", "  ")
    assert SecondOpinion.from_env() is None


def test_time_left_respects_the_deadline():
    so, _ = opinion(result=llm())  # default deadline: 90 s from arrival
    assert so.time_left(1.0) == MAX_CALL_SECONDS
    assert so.time_left(85.0) == pytest.approx(5.0)
    assert so.time_left(88.5) == 0.0  # under 2 s left: skip


def test_agreed_confidence_is_configurable(monkeypatch):
    monkeypatch.setenv("VISION_FALLBACK_API_KEY", "sk-test")
    monkeypatch.setenv("VISION_FALLBACK_AGREED_CONFIDENCE", "0.95")
    monkeypatch.setenv("VISION_FALLBACK_DEADLINE", "20")
    so = SecondOpinion.from_env()
    assert (so.agreed_confidence, so.deadline, so.model, so.below) == (0.95, 20.0, "claude-haiku-4-5", 0.6)
