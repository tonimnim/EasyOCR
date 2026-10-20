"""Cut recognizer training crops from labelled real screenshots.

    python -m tonits_vision.training.harvest_crops --data tonits_vision/data/real/train \
        --out tonits_vision/data/synth/crops/efootball_real_train

The reader's layout rules find the team-name and score boxes; each crop is labelled with the confirmed
result (not with what the OCR read), which is how real screenshots teach the recognizer the eFootball
font. A team-name crop is only kept when the current read is already close to the label (similarity
≥ --min-similarity), so a box the layout put in the wrong place doesn't become a wrong training example.
Then fine-tune with --select-data efootball_synth_train-efootball_real_train --batch-ratio 0.5-0.5.
"""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import cv2

from ..reader import ScreenshotReader, decode_image, pad_box
from .make_dataset import write_labels


def similar(a, b):
    return SequenceMatcher(None, (a or "").casefold(), (b or "").casefold()).ratio()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with labels.jsonl (from fetch_examples.py)")
    ap.add_argument("--out", required=True, help="crop folder to write (trainer format)")
    ap.add_argument("--recognizer", default="english_g2", help="reader used to locate the fields")
    ap.add_argument("--model-dir")
    ap.add_argument("--min-similarity", type=float, default=0.5)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = ScreenshotReader(recognizer=args.recognizer, model_dir=args.model_dir)
    rows, used, skipped = [], 0, 0
    for line in (data / "labels.jsonl").read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        label = entry["label"]
        if label.get("screen") != "match_result":
            continue
        lay, grey, _ = reader.locate(decode_image((data / entry["file"]).read_bytes()))
        if lay.screen != "match_result":
            skipped += 1
            continue
        stem = Path(entry["file"]).stem
        pieces = [("left_score", str(label["left"]["score"])), ("right_score", str(label["right"]["score"])),
                  ("left_team", label["left"]["team"]), ("right_team", label["right"]["team"])]
        reads = reader._recognize(grey, [getattr(lay, name).box for name, _ in pieces if getattr(lay, name)])
        it = iter(reads)
        for name, truth in pieces:
            f = getattr(lay, name)
            if f is None:
                continue
            text, _ = next(it)
            if name.endswith("team") and similar(text, truth) < args.min_similarity:
                continue
            for k, share in enumerate((0.1, 0.25)):  # two margins, like the synthetic crops
                x0, y0, x1, y1 = (int(round(v)) for v in pad_box(f.box, share))
                crop = grey[max(0, y0):y1, max(0, x0):x1]
                if crop.size:
                    fname = f"{stem}_{name}_{k}.png"
                    cv2.imwrite(str(out / fname), crop)
                    rows.append((fname, truth))
        used += 1
    write_labels(out / "labels.csv", rows)
    print(f"{len(rows)} crops from {used} screenshots ({skipped} not read as a result screen) in {out}")


if __name__ == "__main__":
    main()
