"""Build synthetic data: labelled Full Time screens for evaluation and text crops for the recognizer.

    python -m tonits_vision.training.make_dataset --out tonits_vision/data/synth \
        --train-screens 3000 --val-screens 200 --eval-screens 300

Writes:
  <out>/eval/images/*.jpg|png + labels.jsonl   whole screens with ground truth, for evaluate.py
  <out>/crops/efootball_synth_train/            recognizer crops + labels.csv (trainer format)
  <out>/crops/efootball_synth_val/

Crops are cut the way the reader cuts them at inference: from the screen scaled to WORK_HEIGHT with the
table's row dividers painted out, with a small random margin around each piece of text.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from ..reader import remove_rule_lines, work_scale
from ..synth.render import available_fonts, render

# How many times each kind of text is cut per screen. Scores and team names matter most.
CROP_WEIGHTS = {"score": 3, "team": 3, "value": 1, "label": 1, "caption": 1, "manager": 1, "penalty": 2, "ui": 1}


def work_frame(sample):
    rgb = np.asarray(sample.image.convert("RGB"))
    if sample.content_type == "image/jpeg":  # train on what was actually encoded
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(sample.image_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    s = work_scale(rgb.shape[1], rgb.shape[0])
    work = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return cv2.cvtColor(remove_rule_lines(work), cv2.COLOR_RGB2GRAY), s


def cut_crops(rng, sample, grey, scale):
    h, w = grey.shape
    for item in sample.items:
        if item.kind not in CROP_WEIGHTS or not item.text.strip():
            continue
        x0, y0, x1, y1 = (v * scale for v in item.box)
        th = y1 - y0
        for _ in range(CROP_WEIGHTS[item.kind]):
            mx = th * rng.uniform(0.05, 0.35)
            my0, my1 = th * rng.uniform(0.05, 0.3), th * rng.uniform(0.05, 0.3)
            cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my0))
            cx1, cy1 = int(min(w, x1 + mx)), int(min(h, y1 + my1))
            if cx1 - cx0 >= 4 and cy1 - cy0 >= 4:
                yield grey[cy0:cy1, cx0:cx1], item.text


def write_crops(out_dir, rng, fonts, count, seed_base):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        sample = render(random.Random(seed_base + i), fonts, kind="auto")
        grey, scale = work_frame(sample)
        for j, (crop, text) in enumerate(cut_crops(rng, sample, grey, scale)):
            name = f"{i:06d}_{j:02d}.png"
            cv2.imwrite(str(out_dir / name), crop)
            rows.append((name, text))
    write_labels(out_dir / "labels.csv", rows)
    return len(rows)


def write_labels(path, rows):
    """The trainer splits each line at its first comma and doesn't unquote, so write raw lines."""
    with open(path, "w", encoding="utf8") as f:
        f.write("filename,words\n")
        for name, text in rows:
            f.write(f"{name},{text.replace(chr(10), ' ')}\n")


def write_screens(out_dir, fonts, count, seed_base):
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labels.jsonl", "w", encoding="utf8") as f:
        for i in range(count):
            sample = render(random.Random(seed_base + i), fonts, kind="auto")
            ext = "jpg" if sample.content_type == "image/jpeg" else "png"
            name = f"screen_{i:05d}.{ext}"
            (images / name).write_bytes(sample.image_bytes)
            f.write(json.dumps({"file": f"images/{name}", "label": sample.label, "style": sample.style}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tonits_vision/data/synth")
    ap.add_argument("--train-screens", type=int, default=3000)
    ap.add_argument("--val-screens", type=int, default=200)
    ap.add_argument("--eval-screens", type=int, default=300)
    ap.add_argument("--fonts-dir", help="extra .ttf/.otf fonts, e.g. one closer to eFootball's")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    fonts = available_fonts(args.fonts_dir)
    rng = random.Random(args.seed)
    # Disjoint seed ranges, so no screen appears in two splits.
    base = args.seed * 10_000_000
    n_train = write_crops(out / "crops" / "efootball_synth_train", rng, fonts, args.train_screens, base)
    n_val = write_crops(out / "crops" / "efootball_synth_val", rng, fonts, args.val_screens, base + 5_000_000)
    write_screens(out / "eval", fonts, args.eval_screens, base + 8_000_000)
    print(f"crops: {n_train} train, {n_val} val; eval screens: {args.eval_screens} in {out / 'eval'}")


if __name__ == "__main__":
    main()
