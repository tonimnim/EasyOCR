"""Evaluate the reader on labelled screenshots and gate a new model release against the last one.

    python -m tonits_vision.training.evaluate --data tonits_vision/data/synth/eval \
        --recognizer english_g2 --out tonits_vision/reports/english_g2.json

    # ship only if the candidate beats the current release:
    python -m tonits_vision.training.evaluate --data tonits_vision/data/real/heldout \
        --recognizer efootball_v1 --model-dir tonits_vision/models \
        --out tonits_vision/reports/efootball_v1.json --baseline tonits_vision/reports/current.json

<data>/labels.jsonl holds one {"file": "...", "label": {...}} per line, where label uses the reader's
response shape. Real labels (from fetch_examples.py) carry only screen, scores, team names and
penalties; fields a label doesn't have are not scored.

The gate (exit code 1 on failure): exact-score and team-name accuracy must not drop, at least one must
improve, and the number of confidently wrong readings must not grow.
"""

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

from ..reader import ScreenshotReader, UnreadableResult
from ..stats import STAT_KEYS

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
AUTO_DECIDE = 0.9
TARGET_ACCURACY = 0.995  # exact-score accuracy required above VISION_AUTO_MIN_CONFIDENCE


def loose(name):
    """What Go's mapping tolerates: case, accents, spacing and punctuation."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(c for c in s if c.isalnum())


def score_one(label, reading):
    """Per-image results; None where the label doesn't say."""
    r = {"screen_ok": reading["screen"] == label["screen"]}
    if label["screen"] != "match_result":
        r["false_result"] = reading["screen"] == "match_result"
        return r
    if "fullTime" in label:
        r["full_time_ok"] = reading["fullTime"] == label["fullTime"]
    left, right = reading.get("left") or {}, reading.get("right") or {}
    r["score_ok"] = left.get("score") == label["left"]["score"] and right.get("score") == label["right"]["score"]
    r["names_ok"] = left.get("team") == label["left"]["team"] and right.get("team") == label["right"]["team"]
    r["names_loose_ok"] = (loose(left.get("team")) == loose(label["left"]["team"])
                           and loose(right.get("team")) == loose(label["right"]["team"]))
    r["confidence"] = reading["confidence"]
    r["all_ok"] = r["score_ok"] and r["names_ok"]
    if "penalties" in label:
        r["penalties_ok"] = reading.get("penalties") == label["penalties"]
    if label.get("stats"):
        got = reading.get("stats") or {}
        cells = [(got.get(k) or [None, None])[s] == label["stats"][k][s] for k in STAT_KEYS for s in (0, 1)]
        r["stat_cells_ok"] = sum(cells)
        r["stat_cells"] = len(cells)
    if label.get("managers") is not None:
        r["managers_ok"] = reading.get("managers") == label["managers"]
    return r


def rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarise(rows, durations):
    results = [r for r in rows if "score_ok" in r]
    report = {
        "images": len(rows),
        "match_results": len(results),
        "screen_accuracy": rate(rows, "screen_ok"),
        "false_match_result": sum(1 for r in rows if r.get("false_result")),
        "full_time_accuracy": rate(results, "full_time_ok"),
        "exact_score_accuracy": rate(results, "score_ok"),
        "team_name_accuracy": rate(results, "names_ok"),
        "team_name_loose_accuracy": rate(results, "names_loose_ok"),
        "score_and_names_accuracy": rate(results, "all_ok"),
        "penalties_accuracy": rate(results, "penalties_ok"),
        "managers_accuracy": rate(results, "managers_ok"),
        "stat_cell_accuracy": None,
        "confidently_wrong": sum(1 for r in results if r["confidence"] >= AUTO_DECIDE and not r["all_ok"]),
        "above_threshold": {},
        "duration_ms": {"mean": round(sum(durations) / len(durations)) if durations else None,
                        "p95": sorted(durations)[int(0.95 * (len(durations) - 1))] if durations else None},
    }
    cells = [r for r in results if "stat_cells" in r]
    if cells:
        report["stat_cell_accuracy"] = round(sum(r["stat_cells_ok"] for r in cells) / sum(r["stat_cells"] for r in cells), 4)
    report["recommended_min_confidence"] = recommended_threshold(results)
    for t in THRESHOLDS:
        above = [r for r in results if r["confidence"] >= t]
        report["above_threshold"][str(t)] = {
            "share": round(len(above) / len(results), 4) if results else None,
            "score_and_names_accuracy": rate(above, "all_ok"),
        }
    return report


def recommended_threshold(results, target=TARGET_ACCURACY, min_support=30):
    """The lowest confidence at which every reading at or above it is right at least `target` of the
    time (score and names), with at least `min_support` readings to judge by. None if no threshold
    qualifies: then automatic decisions should stay off for this model."""
    for t in sorted({r["confidence"] for r in results}):
        above = [r for r in results if r["confidence"] >= t]
        if len(above) < min_support:
            return None
        if sum(r["all_ok"] for r in above) / len(above) >= target:
            return round(t, 3)
    return None


def gate(new, old):
    """True if `new` may replace `old`."""
    keys = ("exact_score_accuracy", "team_name_accuracy")
    no_drop = all((new[k] or 0) >= (old[k] or 0) for k in keys)
    better = any((new[k] or 0) > (old[k] or 0) for k in keys)
    return no_drop and better and new["confidently_wrong"] <= old["confidently_wrong"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with labels.jsonl")
    ap.add_argument("--recognizer", default="english_g2")
    ap.add_argument("--model-dir", help="where custom recognizer files live (default ~/.EasyOCR/model)")
    ap.add_argument("--out", help="write the report (with per-image results) here")
    ap.add_argument("--baseline", help="a previous report to gate against")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    data = Path(args.data)
    entries = [json.loads(line) for line in (data / "labels.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    if args.limit:
        entries = entries[:args.limit]
    reader = ScreenshotReader(recognizer=args.recognizer, model_dir=args.model_dir)
    rows, durations, details = [], [], []
    for i, entry in enumerate(entries, 1):
        start = time.perf_counter()
        try:
            reading = reader.read((data / entry["file"]).read_bytes())
        except UnreadableResult:  # the API would fail this reading (422): score it as not read
            reading = {"screen": "match_result", "fullTime": None, "left": None, "right": None, "confidence": 0.0}
        durations.append(int((time.perf_counter() - start) * 1000))
        row = score_one(entry["label"], reading)
        rows.append(row)
        details.append({"file": entry["file"], **row, "reading": reading})
        if not row.get("all_ok", row["screen_ok"]):
            print(f"[{i}/{len(entries)}] {entry['file']}: expected {json.dumps({k: entry['label'].get(k) for k in ('left', 'right')}, ensure_ascii=False)}"
                  f" got {json.dumps({k: reading.get(k) for k in ('screen', 'left', 'right')}, ensure_ascii=False)}", flush=True)
    report = {"recognizer": args.recognizer, "model": reader.model_info, "data": str(data), **summarise(rows, durations)}
    print(json.dumps({k: v for k, v in report.items()}, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({**report, "results": details}, indent=2, ensure_ascii=False), encoding="utf8")
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text(encoding="utf8"))
        ok = gate(report, old)
        print(f"gate vs {args.baseline}: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
