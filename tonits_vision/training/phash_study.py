"""How well does a 64-bit pHash tell matches apart on this screen?

    python -m tonits_vision.training.phash_study --pairs 200

Compares Hamming distances for:
  same image    one screenshot re-encoded and resized (what a messaging app does): a reuse
  same look     identical UI, background and crop, different match data: a different match
  any           two random screens
If "same look" distances fall into the "same image" range, an equal hash doesn't prove reuse.
Run it on real screenshots too (--real-dir with images of different matches) before Go relies on it.
"""

import argparse
import io
import random
import statistics
from itertools import combinations
from pathlib import Path

from PIL import Image

from ..phash import hamming, phash
from ..synth import render as R


def summary(name, values):
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    print(f"{name:12s} n={len(values):4d}  min={values[0]:2d}  p5={q(0.05):2d}  median={statistics.median(values):4.1f}  p95={q(0.95):2d}  max={values[-1]:2d}")
    return values


def recompressed(img, rng):
    buf = io.BytesIO()
    scale = rng.uniform(0.4, 1.0)
    img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale)))).convert("RGB").save(
        buf, format="JPEG", quality=rng.randint(30, 85))
    return Image.open(io.BytesIO(buf.getvalue()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--real-dir", help="real screenshots of different matches, to compare all pairs")
    args = ap.parse_args()
    fonts = R.available_fonts()
    rng = random.Random(7)

    same_image, same_look, any_pair = [], [], []
    for i in range(args.pairs):
        style = rng.choice(R.STYLES)
        seed = rng.getrandbits(32)
        # Same seed and no penalties/managers, so only the match data differs between a and b.
        la, lb = (dict(R.random_label(rng, style), penalties=None, managers=None) for _ in range(2))
        a = R._render_result(random.Random(seed), fonts, la, style, "Full Time").image
        b = R._render_result(random.Random(seed), fonts, lb, style, "Full Time").image
        same_image.append(hamming(phash(a), phash(recompressed(a, rng))))
        same_look.append(hamming(phash(a), phash(b)))
        c = R.render(random.Random(rng.getrandbits(32)), fonts, "result").image
        any_pair.append(hamming(phash(a), phash(c)))

    print("Hamming distance between 64-bit pHashes (synthetic screens):")
    reuse = summary("same image", same_image)
    look = summary("same look", same_look)
    summary("any", any_pair)
    cutoff = reuse[int(0.95 * (len(reuse) - 1))]
    clash = sum(1 for d in look if d <= cutoff) / len(look)
    print(f"\nA reuse rule of 'distance <= {cutoff}' (catches 95% of re-encoded copies) would also flag "
          f"{clash:.0%} of different matches that share a look.")

    if args.real_dir:
        paths = sorted(p for p in Path(args.real_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        hashes = {p.name: phash(Image.open(p)) for p in paths}
        summary("real pairs", [hamming(hashes[a], hashes[b]) for a, b in combinations(hashes, 2)])


if __name__ == "__main__":
    main()
