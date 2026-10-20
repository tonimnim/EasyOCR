"""Turn a trainer checkpoint into an EasyOCR custom recognizer (<name>.pth, <name>.yaml, <name>.py).

    python -m tonits_vision.training.export_model --checkpoint runs/x/saved_models/efootball/best_accuracy.pth \
        --name efootball_v1 --base latin_g2 --release 1 --out tonits_vision/models

The reader loads it with VISION_RECOGNIZER=efootball_v1 VISION_MODEL_DIR=tonits_vision/models.
"""

import argparse
import shutil
from pathlib import Path

import yaml

# Generation-2 network, importable from the user network directory (EasyOCR imports <name>.py by name).
MODEL_PY = '''"""EasyOCR generation-2 recognizer (None-VGG-BiLSTM-CTC), exported by tonits_vision."""
from easyocr.model.vgg_model import Model  # noqa: F401
'''


def export(checkpoint, name, base, release, out_dir):
    from easyocr.config import recognition_models

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, out_dir / f"{name}.pth")
    config = {
        "network_params": {"input_channel": 1, "output_channel": 256, "hidden_size": 256},
        "imgH": 64,
        "lang_list": ["en"],
        "character_list": recognition_models["gen2"][base]["characters"],
        "base": base,
        "release": release,
    }
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf8")
    (out_dir / f"{name}.py").write_text(MODEL_PY, encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="latin_g2")
    ap.add_argument("--release", type=int, required=True)
    ap.add_argument("--out", default="tonits_vision/models")
    args = ap.parse_args()
    export(Path(args.checkpoint), args.name, args.base, args.release, Path(args.out))
    print(f"wrote {args.out}/{args.name}.pth/.yaml/.py")


if __name__ == "__main__":
    main()
