# Recognizer trainer

The text-recognition trainer (None-VGG-BiLSTM-CTC), adapted from
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) by way of
EasyOCR. It isn't run directly: `tonits_vision/training/finetune.py` loads
`config_files/efootball_finetune.yaml`, fills in the base model's character set and the data paths, and
calls `train.train()` from a run directory, where checkpoints land in `saved_models/<experiment>/`.

```bash
python -m tonits_vision.training.finetune --run-dir tonits_vision/runs/efootball_v2 --name efootball_v2 --release 2
```

Training data is a folder per dataset with images and a `labels.csv` (`filename,words`, split at the
first comma), as written by `tonits_vision/training/make_dataset.py` and `harvest_crops.py`. Uses CUDA
when it is available, otherwise the CPU. Needs `pandas`, `natsort` and `nltk` on top of the reader's
dependencies.
