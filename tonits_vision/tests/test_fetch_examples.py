"""fetch_examples.py against a local fake of the API's training feed."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from tonits_vision.training import fetch_examples

TOKEN = "training-token-0123456789abcdefghij"


def make_items(base_url, n):
    return [{
        "evidenceId": f"ev-{i}", "matchId": f"match-{i // 2}",  # two screenshots per match
        "downloadUrl": f"{base_url}/files/ev-{i}", "contentType": "image/png",
        "label": {"screen": "match_result", "left": {"team": f"Team {i}", "score": i % 4},
                  "right": {"team": "Other", "score": 1}, "penalties": None},
        "source": "staff_decision", "decidedAt": "2026-10-01T12:00:00Z",
        "reader": {"confidence": 0.5}, "readerCorrect": i % 2 == 0,
    } for i in range(n)]


@pytest.fixture
def api():
    state = {"items": [], "page_size": 3, "expired": set(), "calls": [], "issued": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path.startswith("/files/"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"\x89PNG fake " + url.path.encode())
                return
            assert self.headers["Authorization"] == f"Bearer {TOKEN}"
            query = parse_qs(url.query)
            state["calls"].append(query)
            assert query["limit"] == ["100"]
            cursor = query.get("cursor", [None])[0]
            if cursor in state["expired"]:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"invalid_cursor"}')
                return
            start = int(cursor.split(":")[0]) if cursor else 0
            page = state["items"][start:start + state["page_size"]]
            state["issued"] += 1  # like the API, every cursor issued is a new token
            nxt = f"{start + len(page)}:{state['issued']}" if start + len(page) < len(state["items"]) else None
            body = json.dumps({"items": page, "nextCursor": nxt}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def run(monkeypatch, api, out, *extra):
    monkeypatch.setenv("VISION_TRAINING_TOKEN", TOKEN)
    monkeypatch.setattr(sys, "argv", ["fetch", "--api", api["url"], "--out", str(out), *extra])
    fetch_examples.main()


def records(out):
    rows = []
    for split in ("train", "heldout"):
        path = out / split / "labels.jsonl"
        if path.exists():
            rows += [dict(json.loads(line), split=split) for line in path.read_text().splitlines()]
    return rows


def test_downloads_every_page_and_splits_by_match(monkeypatch, api, tmp_path):
    api["items"] = make_items(api["url"], 20)
    run(monkeypatch, api, tmp_path, "--heldout-percent", "40")
    rows = records(tmp_path)
    assert sorted(r["evidenceId"] for r in rows) == sorted(f"ev-{i}" for i in range(20))
    by_match = {}
    for r in rows:
        by_match.setdefault(r["matchId"], set()).add(r["split"])
        assert (tmp_path / r["split"] / r["file"]).read_bytes().startswith(b"\x89PNG")
    assert all(len(splits) == 1 for splits in by_match.values())  # a match never straddles the split
    assert {r["split"] for r in rows} == {"train", "heldout"}
    assert rows[0]["readerCorrect"] in (True, False) and rows[0]["label"]["screen"] == "match_result"


def test_rerun_only_fetches_new_examples(monkeypatch, api, tmp_path):
    api["items"] = make_items(api["url"], 5)
    run(monkeypatch, api, tmp_path)
    api["items"] = make_items(api["url"], 8)
    run(monkeypatch, api, tmp_path)
    assert len(records(tmp_path)) == 8


def test_expired_cursor_starts_over(monkeypatch, api, tmp_path):
    api["items"] = make_items(api["url"], 7)
    run(monkeypatch, api, tmp_path)
    api["expired"] = {(tmp_path / "cursor.txt").read_text()}
    api["items"] = make_items(api["url"], 9)
    run(monkeypatch, api, tmp_path)
    assert len(records(tmp_path)) == 9


def test_limit_mid_page_does_not_skip_the_rest(monkeypatch, api, tmp_path):
    api["items"] = make_items(api["url"], 6)
    run(monkeypatch, api, tmp_path, "--limit", "2")
    assert len(records(tmp_path)) == 2
    run(monkeypatch, api, tmp_path)
    assert len(records(tmp_path)) == 6
