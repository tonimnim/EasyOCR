# tonitsOCR: the Tonits screenshot reader

Reads the eFootball **Full Time** screen (both team names, the score, penalties, the coach row and all 13
stat rows) so that Tonits can check disputed match results. When two players report different scores and
upload their screenshots, the Tonits API sends each screenshot to this reader, stores what it read,
shows it to staff, and can decide the dispute when both screenshots agree.

```
player app ──upload──▶ storage                     ┌──────────────────────────────┐
     └── disputed result + screenshots ──▶ Go API ──▶ POST /v1/read   (this repo)  │
                                             ▲      │ CRAFT + recognizer + rules  │
                    reading (JSON) ──────────┘      └──────────────────────────────┘
```

The API lives in `gemics-backend`; its `compose.yaml` builds this repo as the `vision` service.

## What's in this repo

| Path | What it is |
|---|---|
| [`tonits_vision/`](tonits_vision/README.md) | The reader: HTTP service, layout rules, checks, synthetic data, training, tests, Docker image |
| [`tonits_vision/DESIGN_REVIEW.md`](tonits_vision/DESIGN_REVIEW.md) | Review of the reader contract, and what real screenshots taught us |
| `easyocr/` | The OCR engine: EasyOCR's CRAFT text detector and generation-2 recognizers, trimmed to English |
| [`trainer/`](trainer/README.md) | The recognizer trainer that `tonits_vision/training/finetune.py` drives |

## How well it reads

On 8 real screenshots (7 phone captures including a Half Time screen, a loading spinner over half the
labels and a "Total Shots" variant, plus 1 console capture):

| | Result |
|---|---|
| Result screen recognised, Full Time vs Half Time | 8/8 |
| Exact scores | 8/8 |
| Stat values | 208/208 |
| Both team names exact | 1/8: eFootball's banner font is misread consistently ("dimangaaa" → "climangaaa") |

A reading takes about 6 s on 2 CPUs once the machine is idle. Real banner crops from the API's training
feed are the planned fix for team names; see [Training and releases](tonits_vision/README.md#training-and-releases).

## Quick start

Tested on Python 3.14; a CPU is enough (the reader never uses a GPU).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install fastapi "uvicorn[standard]" anthropic pytest httpx pandas natsort nltk imagehash
pytest tonits_vision/tests                      # fast tests; RUN_SLOW=1 adds end-to-end ones
```

Run the service (the first start downloads the CRAFT and `english_g2` weights to `~/.EasyOCR/model`):

```bash
VISION_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))") python -m tonits_vision.server
curl -s -X POST localhost:8000/v1/read -H "Authorization: Bearer $VISION_TOKEN" \
  -H "Content-Type: image/png" --data-binary @fulltime.png
```

Or as the container the API uses:

```bash
docker build -f tonits_vision/Dockerfile -t tonits-vision:local .
# from gemics-backend:
docker compose --env-file .env.docker --profile vision up -d --build
```

For a quick look at what OCR sees in any image, the engine's command line still works:
`easyocr -l en -f image.png --detail=0 --gpu=False`.

## Origin and license

The engine is [EasyOCR](https://github.com/JaidedAI/EasyOCR) by Jaided AI, Apache License 2.0 (see
[`LICENSE`](LICENSE)), which builds on Clova AI's [CRAFT](https://github.com/clovaai/CRAFT-pytorch) and
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). This fork
removed the other languages' data, the DBNet detector, the ONNX export, detector training, the upstream
examples, tests and docs, and changed `easyocr/easyocr.py`, `easyocr/config.py`, `trainer/train.py` and
`trainer/dataset.py`. `tonits_vision/` is new.
