import struct
import time
import threading
import zlib

import pytest
from fastapi.testclient import TestClient

from tonits_vision.reader import (MAX_WORK_WIDTH, WORK_HEIGHT, ImageError, ImageTooLarge, UnreadableResult,
                                  UnsupportedGeometry, decode_image, work_scale)
from tonits_vision.server import MAX_BYTES, create_app

TOKEN = "test-token-0123456789abcdefghijklmnop"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "image/png"}
READING = {"screen": "match_result", "fullTime": True, "left": {"team": "A", "score": 1, "confidence": 0.99},
           "right": {"team": "B", "score": 0, "confidence": 0.98}, "penalties": None, "managers": None,
           "stats": {}, "checks": {}, "confidence": 0.9, "imageHash": "0" * 16,
           "model": {"engine": "easyocr", "version": "test"}, "durationMs": 1}


class FakeReader:
    model_info = {"engine": "easyocr", "version": "test", "recognizer": "fake"}

    def __init__(self):
        self.gate = None

    def read(self, data, with_boxes=False):
        if self.gate:
            self.gate.wait(5)
        if data == b"bad":
            raise ImageError("not a decodable image")
        if data == b"huge":
            raise ImageTooLarge("50000x50000 is over 40000000 pixels")
        if data == b"sliver":
            raise UnsupportedGeometry("8192x10 isn't screenshot-shaped")
        if data == b"no-banner":
            raise UnreadableResult("result screen, but the left team name or score can't be read")
        return dict(READING, boxes={} if with_boxes else None)


@pytest.fixture
def reader():
    return FakeReader()


@pytest.fixture
def client(reader):
    app = create_app(lambda: reader, token=TOKEN, background_load=False, second_opinion_factory=None)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True, "model": FakeReader.model_info}


def test_read_ok_echoes_request_id(client):
    r = client.post("/v1/read", content=b"img", headers={**AUTH, "X-Request-ID": "req-1", "X-Evidence-ID": "ev-1"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-1"
    assert r.json()["left"]["team"] == "A"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": TOKEN}])
def test_bad_token(client, headers):
    r = client.post("/v1/read", content=b"img", headers={"Content-Type": "image/png", **headers})
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


@pytest.mark.parametrize("ctype", ["text/plain", "image/gif", "application/octet-stream", ""])
def test_wrong_type(client, ctype):
    r = client.post("/v1/read", content=b"img", headers={**AUTH, "Content-Type": ctype})
    assert r.status_code == 415


def test_webp_is_accepted(client):
    assert client.post("/v1/read", content=b"img", headers={**AUTH, "Content-Type": "image/webp"}).status_code == 200


def test_content_type_parameters_are_ignored(client):
    assert client.post("/v1/read", content=b"img", headers={**AUTH, "Content-Type": "image/JPEG; q=1"}).status_code == 200


def test_too_large(client):
    r = client.post("/v1/read", content=b"x" * (MAX_BYTES + 1), headers=AUTH)
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_undecodable_and_pixel_bomb(client):
    assert client.post("/v1/read", content=b"bad", headers=AUTH).status_code == 422
    r = client.post("/v1/read", content=b"huge", headers=AUTH)
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"
    r = client.post("/v1/read", content=b"sliver", headers=AUTH)
    assert r.status_code == 422 and r.json()["error"] == "unsupported_geometry"
    r = client.post("/v1/read", content=b"no-banner", headers=AUTH)
    assert r.status_code == 422 and r.json()["error"] == "unreadable_result"


def test_model_not_loaded():
    def broken():
        raise RuntimeError("weights missing")
    with TestClient(create_app(broken, token=TOKEN, background_load=False, second_opinion_factory=None)) as c:
        h = c.get("/healthz")
        assert h.status_code == 503 and "weights missing" in h.json()["error"]
        r = c.post("/v1/read", content=b"img", headers=AUTH)
        assert r.status_code == 503 and r.json()["error"] == "model_not_loaded"


@pytest.mark.parametrize("max_queue", [0, 1])
def test_busy_when_queue_is_full(reader, max_queue):
    # VISION_MAX_QUEUE readings may wait behind the one in progress; the next is turned away.
    reader.gate = threading.Event()
    app = create_app(lambda: reader, token=TOKEN, max_queue=max_queue, background_load=False, second_opinion_factory=None)
    with TestClient(app) as c:
        results = []
        held = [threading.Thread(target=lambda: results.append(c.post("/v1/read", content=b"img", headers=AUTH)))
                for _ in range(1 + max_queue)]
        for t in held:
            t.start()
        for _ in range(200):
            if app.state.reader_state.in_flight == 1 + max_queue:
                break
            threading.Event().wait(0.02)
        r = c.post("/v1/read", content=b"img", headers=AUTH)
        reader.gate.set()
        for t in held:
            t.join()
        assert r.status_code == 503 and r.json()["error"] == "busy" and r.headers["retry-after"] == "10"
        assert [x.status_code for x in results] == [200] * (1 + max_queue)


def test_negative_queue_is_rejected():
    with pytest.raises(RuntimeError):
        create_app(lambda: None, token=TOKEN, max_queue=-1)


def test_second_opinion_applies_to_low_confidence_reads(reader):
    class Opinion:
        def __init__(self):
            self.seen = []

        def wanted(self, reading):
            return reading["confidence"] < 0.95

        def time_left(self, elapsed):
            return 10.0

        def review(self, data, content_type, reading, timeout):
            self.seen.append((data, content_type))
            return dict(reading, confidence=0.3, secondOpinion={"agrees": False})

    opinion = Opinion()
    app = create_app(lambda: reader, token=TOKEN, background_load=False, second_opinion_factory=lambda: opinion)
    with TestClient(app) as c:
        r = c.post("/v1/read", content=b"img", headers=AUTH)
    assert r.json()["confidence"] == 0.3 and opinion.seen == [(b"img", "image/png")]


@pytest.mark.parametrize("token", ["", "short-token"])
def test_token_must_be_32_characters(token):
    with pytest.raises(RuntimeError):
        create_app(lambda: None, token=token)


def _png_header(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IEND", b"")


def test_decode_rejects_garbage_and_pixel_bombs():
    with pytest.raises(ImageError):
        decode_image(b"")
    with pytest.raises(ImageError):
        decode_image(b"definitely not an image")
    with pytest.raises(ImageTooLarge):
        decode_image(_png_header(50_000, 50_000))


class _Opinion:
    def __init__(self, left=10.0, error=None):
        self.left, self.error, self.calls = left, error, 0

    def wanted(self, reading):
        return True

    def time_left(self, elapsed):
        return self.left

    def review(self, data, content_type, reading, timeout):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(reading, confidence=0.1)


@pytest.mark.parametrize("opinion,expected_calls", [(_Opinion(error=ValueError("bad schema")), 1), (_Opinion(left=0.0), 0)])
def test_second_opinion_never_fails_the_read(reader, opinion, expected_calls):
    app = create_app(lambda: reader, token=TOKEN, background_load=False, second_opinion_factory=lambda: opinion)
    with TestClient(app) as c:
        r = c.post("/v1/read", content=b"img", headers=AUTH)
    assert r.status_code == 200 and r.json()["confidence"] == READING["confidence"]
    assert opinion.calls == expected_calls


def test_caller_that_gave_up_while_queued_is_not_read():
    # Needs a real socket: TestClient can't hang up. B gives up after 1 s while queued behind A, so the
    # reader must see A and C only. (From the gemics-backend container test, 2026-10-08.)
    import socket
    import urllib.request

    import uvicorn

    seen = []

    class Slow:
        model_info = {"engine": "fake", "version": "0", "recognizer": "fake"}

        def read(self, data, with_boxes=False):
            seen.append(chr(data[0]))
            time.sleep(3)
            return dict(READING)

    app = create_app(lambda: Slow(), token=TOKEN, background_load=False, second_opinion_factory=None)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    def call(name, timeout):
        # ~270 KB, like a real screenshot: above uvicorn's 64 KiB high-water mark
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/read", data=name.encode() + b"\0" * 270_000, method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "image/png"})
        try:
            urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            pass

    threads = []
    for name, timeout, delay in (("A", 30, 0.5), ("B", 1, 0.3), ("C", 30, 0)):
        t = threading.Thread(target=call, args=(name, timeout))
        t.start()
        threads.append(t)
        time.sleep(delay)
    for t in threads:
        t.join()
    server.should_exit = True
    assert seen == ["A", "C"]


def _png(width, height):
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (width, height), 128).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("size", [(8192, 10), (10, 8192), (3000, 300), (200, 200), (1000, 239)])
def test_non_screenshot_geometry_is_refused_before_any_resize(size):
    # 8192x10 is 352 bytes as a PNG and used to OOM-kill the reader when scaled to 720 px high.
    with pytest.raises(UnsupportedGeometry):
        decode_image(_png(*size))


@pytest.mark.parametrize("size", [(826, 371), (960, 540), (1600, 720), (2000, 997), (1600, 702), (1080, 2400)])
def test_real_screenshot_sizes_are_accepted(size):
    assert decode_image(_png(*size)).shape[:2] == (size[1], size[0])


@pytest.mark.parametrize("size", [(826, 371), (2400, 1080), (8192, 10), (60000, 200)])
def test_work_frame_width_is_capped(size):
    w, h = size
    s = work_scale(w, h)
    assert round(w * s) <= MAX_WORK_WIDTH and round(h * s) <= WORK_HEIGHT
