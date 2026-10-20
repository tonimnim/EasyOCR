"""Download labelled real screenshots from the Go API's training-examples endpoint.

    VISION_TRAINING_TOKEN=... python -m tonits_vision.training.fetch_examples \
        --api https://tonits-api.internal --out tonits_vision/data/real

Writes <out>/train/ and <out>/heldout/ (images/ + labels.jsonl, the format evaluate.py and
harvest_crops.py read). The split is by match, so both players' screenshots of one match always land on
the same side and the held-out set never shares a match with training. Re-running only downloads new
examples; the cursor is kept in <out>/cursor.txt.

Response (gemics-backend, screenshot_reader_handlers.go):
    {"items": [{"evidenceId", "matchId", "downloadUrl", "downloadExpiresAt", "contentType",
                "label": {"screen", "left": {"team", "score"}, "right": {...}, "penalties"},
                "source", "decidedAt", "reader": {...}, "readerCorrect"}],
     "nextCursor": "..." | null}
Download URLs last 15 minutes and cursors expire, so a page is downloaded as soon as it arrives, and an
expired saved cursor starts the feed again from the beginning (already-downloaded examples are skipped).
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png"}


def get_json(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download(url, limit=25 * 1024 * 1024):
    with urllib.request.urlopen(url, timeout=60) as resp:  # presigned: no auth header
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download over 25 MiB")
    return data


def split_of(match_id, heldout_percent):
    bucket = int(hashlib.sha256(match_id.encode()).hexdigest(), 16) % 100
    return "heldout" if bucket < heldout_percent else "train"


def known_ids(split_dir):
    path = split_dir / "labels.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["evidenceId"] for line in path.read_text(encoding="utf8").splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True, help="base URL of the Go API")
    ap.add_argument("--out", default="tonits_vision/data/real")
    ap.add_argument("--heldout-percent", type=int, default=20)
    ap.add_argument("--limit", type=int, help="stop after this many new examples")
    ap.add_argument("--restart", action="store_true", help="ignore the saved cursor")
    args = ap.parse_args()

    token = os.environ.get("VISION_TRAINING_TOKEN")
    if not token:
        sys.exit("set VISION_TRAINING_TOKEN")
    out = Path(args.out)
    cursor_file = out / "cursor.txt"
    cursor = None if args.restart or not cursor_file.exists() else cursor_file.read_text().strip() or None
    splits = {name: out / name for name in ("train", "heldout")}
    for d in splits.values():
        (d / "images").mkdir(parents=True, exist_ok=True)
    seen = known_ids(splits["train"]) | known_ids(splits["heldout"])

    new, restarted = 0, False
    while True:
        query = urllib.parse.urlencode({"limit": 100, **({"cursor": cursor} if cursor else {})})
        try:
            page = get_json(f"{args.api.rstrip('/')}/v1/internal/vision/training-examples?{query}", token)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and cursor and not restarted:  # invalid_cursor: saved cursors expire
                print("saved cursor expired; starting from the beginning", file=sys.stderr)
                cursor, restarted = None, True
                continue
            raise
        for item in page.get("items", []):
            if item["evidenceId"] in seen:
                continue
            ext = EXTENSIONS.get(item.get("contentType", "image/jpeg"), "jpg")
            split = splits[split_of(item["matchId"], args.heldout_percent)]
            name = f"images/{item['evidenceId']}.{ext}"
            try:
                (split / name).write_bytes(download(item["downloadUrl"]))
            except Exception as exc:  # expired URL etc.: it comes back on a later run with --restart
                print(f"skip {item['evidenceId']}: {exc}", file=sys.stderr)
                continue
            record = {"file": name, "label": item["label"], "evidenceId": item["evidenceId"],
                      "matchId": item["matchId"], "source": item.get("source"), "decidedAt": item.get("decidedAt"),
                      "reader": item.get("reader"), "readerCorrect": item.get("readerCorrect")}
            with open(split / "labels.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(item["evidenceId"])
            new += 1
            if args.limit and new >= args.limit:
                break
        else:  # the whole page was handled, so it's safe to move past it
            cursor = page.get("nextCursor")
            if cursor:
                cursor_file.write_text(cursor)
            if cursor:
                continue
        break
    print(f"{new} new examples in {out}")


if __name__ == "__main__":
    main()
