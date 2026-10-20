"""Fine-tune EasyOCR's recognizer on eFootball crops with the fork's trainer/, then export it.

    python -m tonits_vision.training.finetune --run-dir tonits_vision/runs/efootball_v1 \
        --name efootball_v1 --release 1 --num-iter 20000

Starts from the base model in the config (latin_g2: same architecture as english_g2, plus accented
letters), keeps its character set so the prediction layer is reused, trains on the crops from
make_dataset.py (and harvest_crops.py for real screenshots), and writes the three files EasyOCR needs
for a custom recognizer to --models-dir. On a CPU expect roughly a second per iteration; use a GPU for
full runs (the trainer uses CUDA when it's available).
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "trainer"
DEFAULT_CONFIG = TRAINER / "config_files" / "efootball_finetune.yaml"


def base_model_path(name, model_dir):
    """The base recognizer's weights, downloading them through EasyOCR if needed."""
    path = Path(model_dir) / f"{name}.pth"
    if not path.exists():
        import easyocr
        easyocr.Reader(["en"], gpu=False, recog_network=name, model_storage_directory=str(model_dir),
                       detector=False, verbose=False)
    return path


def load_opt(config_path, overrides):
    sys.path.insert(0, str(TRAINER))
    from utils import AttrDict  # trainer/utils.py
    from easyocr.config import recognition_models

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    opt = AttrDict(cfg)
    opt.character = recognition_models["gen2"][opt.base_model]["characters"]
    opt.train_data = str((REPO / opt.train_data).resolve())
    opt.valid_data = str((REPO / opt.valid_data).resolve())
    return opt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--run-dir", required=True, help="checkpoints and logs go here")
    ap.add_argument("--name", default="efootball_v1", help="recognizer name to export as")
    ap.add_argument("--release", type=int, default=1, help="goes into model.version: bump it for every shipped model")
    ap.add_argument("--models-dir", default=str(REPO / "tonits_vision" / "models"))
    ap.add_argument("--base-model-dir", default=str(Path.home() / ".EasyOCR" / "model"))
    ap.add_argument("--num-iter", type=int)
    ap.add_argument("--val-interval", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--train-data", help="directory holding the crop folders (default from the config)")
    ap.add_argument("--valid-data", help="validation crop folder (default from the config)")
    ap.add_argument("--select-data")
    ap.add_argument("--batch-ratio")
    ap.add_argument("--resume", help="continue from this checkpoint instead of the base model")
    args = ap.parse_args()

    opt = load_opt(args.config, {"num_iter": args.num_iter, "valInterval": args.val_interval,
                                 "batch_size": args.batch_size, "workers": args.workers,
                                 "select_data": args.select_data, "batch_ratio": args.batch_ratio,
                                 "train_data": args.train_data, "valid_data": args.valid_data})
    opt.saved_model = str(Path(args.resume).resolve()) if args.resume else str(base_model_path(opt.base_model, args.base_model_dir))

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    run_dir = Path(args.run_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    (run_dir / "saved_models" / opt.experiment_name).mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)  # the trainer writes to ./saved_models/<experiment_name>/
    from train import train  # trainer/train.py
    try:
        train(opt, amp=False)
    except SystemExit:
        pass  # the trainer exits when it reaches num_iter

    best = run_dir / "saved_models" / opt.experiment_name / "best_accuracy.pth"
    if not best.exists():
        sys.exit(f"no checkpoint at {best}: num_iter must be past the first validation (valInterval)")
    from .export_model import export
    export(best, args.name, opt.base_model, args.release, models_dir)
    shutil.copy(run_dir / "saved_models" / opt.experiment_name / "log_train.txt", models_dir / f"{args.name}.log.txt")
    print(f"exported {args.name} to {models_dir}")


if __name__ == "__main__":
    main()
