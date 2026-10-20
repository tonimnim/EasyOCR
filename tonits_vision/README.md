# tonits_vision: the eFootball screenshot reader

Reads the eFootball **Full Time** screen (team names, scores, penalties, coach row and the 13-row stats
table) and serves it to the Tonits Go API over `POST /v1/read`, as specified in the reader contract
(gemics-backend). Built on this EasyOCR fork: CRAFT for detection, an EasyOCR generation-2 recognizer
(stock `english_g2`, or one fine-tuned here), and layout rules written for this screen.

`DESIGN_REVIEW.md` is the review of the contract; the revised contract adopted most of it.

## Layout

| Path | What it is |
|---|---|
| `server.py` | HTTP API: `POST /v1/read`, `GET /healthz` |
| `reader.py` | The pipeline: decode, scale, detect, layout, re-read, checks, confidence |
| `layout.py` | Layout rules on OCR tokens (no model needed; heavily tested) |
| `glyphs.py` | Finds score digits by shape beside the logo |
| `checks.py`, `phash.py`, `stats.py` | Plausibility rules, the 64-bit pHash, the stat rows |
| `fallback.py` | Optional LLM second opinion for low-confidence readings |
| `synth/` | Renders synthetic Full Time screens with ground truth |
| `training/` | Dataset builder, fine-tuning, export, evaluation and release gate, training-feed client, crop harvester, pHash study |
| `tests/` | `pytest tonits_vision/tests` (fast); `RUN_SLOW=1` adds end-to-end tests with the real models |
| `Dockerfile`, `requirements-server.txt` | The service image (models baked in, CPU-only torch) |
| `../trainer/config_files/efootball_finetune.yaml` | Fine-tuning config for the fork's `trainer/` |

## Run it locally

From the repository root, with the `.venv` from the setup (CPU-only torch, `pip install -e .`, then
`pip install fastapi "uvicorn[standard]" anthropic pytest httpx pandas natsort nltk imagehash`):

```bash
source .venv/bin/activate
pytest tonits_vision/tests
VISION_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))") python -m tonits_vision.server
```

```bash
curl -s -X POST localhost:8000/v1/read -H "Authorization: Bearer $VISION_TOKEN" \
  -H "Content-Type: image/png" --data-binary @fulltime.png
```

Add `?boxes=1` to also get every field's box in image pixels (for checking a reading by eye).

The container: `docker build -f tonits_vision/Dockerfile -t tonits-vision .` from the repo root; the API's
`compose.yaml` builds it with `--profile vision`. Use `--build-arg RECOGNIZER=efootball_v1` to serve a
fine-tuned recognizer exported to `tonits_vision/models/`.

## How a reading works

1. Decode with a pixel cap (40 MP), scale to 720 px high (the UI scales with screen height), and paint out
   the table's row dividers, which otherwise make CRAFT join digits across rows.
2. Detect words on a 540 px copy and read them once with the general recognizer.
3. Find the 13 stat labels (fuzzy, in screen order). Their centre is the screen's centre line; the values
   sit in two columns beside them.
4. Above the table, the banner is the nearest line with text on both sides of the centre. The scores are
   found by shape right beside the logo (`glyphs.py`): CRAFT misses many lone "0"s and "1"s. Everything
   further out is the team name. Only when the first pass misses a team name is the banner re-detected
   at 2× zoom.
5. Re-read with tight character sets: scores and every stat cell as a grid, digits only (possession keeps
   `%`); team and coach names from the union of their word boxes.
6. Run the checks and compute confidence: the lower of the two sides (team-name and score confidence),
   halved per failed check, reduced for missing rows, capped at 0.5 without a Full Time caption.

### Where the reader differs from the contract

- A result screen whose banner can't be read returns **422 `unreadable_result`**. The API rejects a
  `match_result` without both sides, and reading again gives the same answer, so the reading fails for
  good and staff decide.
- `checks` has a fifth rule, `shotsOnTargetWithinShots`; each check is `true`, `false` or `null` (needed
  stat not read). `goalsPlusSavesWithinShotsOnTarget` is reported but doesn't lower confidence: a real
  screenshot breaks it (see the review, item 4b).
- `image/webp` is accepted alongside JPEG and PNG.
- An image that isn't screenshot-shaped (shorter side under 240 px, or longer than 4:1) returns **422
  `unsupported_geometry`** before anything is resized; a sliver like 8192×10 would otherwise be scaled into
  a frame of over a gigabyte. The work frame is also capped at 2881 px wide.
- `model.recognizer` names the recognizer; `model.version` is
  `<easyocr>+tonits-efootball-<rules>.<recognizer release>`, so a layout-rule change bumps it too.
- `secondOpinion` appears when the LLM fallback ran.
- `/healthz` needs no token. `503 busy` carries `Retry-After: 10`. A request whose caller disconnected
  while queued is dropped (499) instead of read.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `VISION_TOKEN` | — | Required, 32+ characters, the same secret the API sends |
| `PORT` / `VISION_PORT` | 8000 | Listen port (`VISION_HOST` 0.0.0.0) |
| `VISION_MAX_QUEUE` | 2 | Readings that may wait behind the one in progress; more get `503 busy` |
| `VISION_TORCH_THREADS` | CPU quota | Torch threads; defaults to the container's cgroup quota |
| `VISION_RECOGNIZER` | `english_g2` | A stock EasyOCR gen-2 model, or a fine-tuned one (`efootball_v1`) |
| `VISION_MODEL_DIR` / `VISION_USER_NETWORK_DIR` | `~/.EasyOCR/model` | Where weights / custom `.py`+`.yaml` live |
| `VISION_DOWNLOAD_MODELS` | 1 | `0` in the image: never download at runtime |
| `VISION_FALLBACK_API_KEY` | empty (off) | Anthropic API key for the second opinion |
| `VISION_FALLBACK_MODEL` | `claude-haiku-4-5` | Model for the second opinion |
| `VISION_FALLBACK_BELOW` | 0.6 | Ask only below this confidence |
| `VISION_FALLBACK_DEADLINE` | 90 | Seconds from the request's arrival; skip the LLM with under 2 s left. Keep the API's `VISION_TIMEOUT` > this > (1 + `VISION_MAX_QUEUE`) × slowest read |
| `VISION_FALLBACK_AGREED_CONFIDENCE` | 0.85 | Side confidence after the LLM agrees |

The second opinion sends the screenshot to Anthropic. It only confirms or casts doubt (confidence
capped at 0.3); it never changes a value, and any error keeps the OCR reading. Agreement lifts each side
to 0.85 by default, under the API's 0.9 auto-decide threshold, so the LLM's word alone never settles a
match: that threshold is calibrated on OCR confidence, and an LLM answer isn't reproducible on a
re-read. Raise it only after calibrating LLM-confirmed readings separately.

## Training and releases

```bash
# 1. Synthetic data: crops for the recognizer, labelled screens for evaluation
python -m tonits_vision.training.make_dataset --out tonits_vision/data/synth \
    --train-screens 1500 --val-screens 100 --eval-screens 150

# 2. Real data from the API's training feed (staff-decided, in screen terms), split by match
VISION_TRAINING_TOKEN=... python -m tonits_vision.training.fetch_examples --api https://<api> --out tonits_vision/data/real
python -m tonits_vision.training.harvest_crops --data tonits_vision/data/real/train \
    --out tonits_vision/data/synth/crops/efootball_real_train

# 3. Fine-tune from latin_g2 (same network as english_g2, plus accents) and export
python -m tonits_vision.training.finetune --run-dir tonits_vision/runs/efootball_v2 --name efootball_v2 \
    --release 2 --num-iter 20000 --select-data efootball_synth_train-efootball_real_train --batch-ratio 0.5-0.5

# 4. Evaluate on held-out real screenshots and gate against the current release
python -m tonits_vision.training.evaluate --data tonits_vision/data/real/heldout --recognizer efootball_v2 \
    --model-dir tonits_vision/models --out tonits_vision/reports/efootball_v2.json \
    --baseline tonits_vision/reports/current.json

# 5. Ship: build the image with --build-arg RECOGNIZER=efootball_v2
```

The gate passes only if exact-score and team-name accuracy don't drop, at least one improves, and the
count of confidently wrong readings (≥ 0.9 but wrong) doesn't grow. The report's
`recommended_min_confidence` is the lowest threshold with ≥ 99.5% score-and-names accuracy above it
(30+ readings); set `VISION_AUTO_MIN_CONFIDENCE` from it, or leave automatic decisions off when it's null.

Training on this laptop's CPU runs at a few seconds per iteration; use a GPU for full runs (the trainer
picks CUDA up automatically). The fork's `trainer/` needed three fixes for current PyTorch
(`torch._utils._accumulate`, `DataLoader` iterator `.next()`, loading a GPU checkpoint on CPU).

## Status and limits

**Real screenshots (8, in `samples/` with `labels.jsonl`; 2026-10-08):** 7 phone captures (1600×720, one
Half Time, one with a loading spinner over half the labels, one with "Total Shots") and 1 console capture
(2000×997, coach row under the table).

| | `english_g2` (stock, shipped) | `efootball_v1` (synthetic fine-tune) |
|---|---|---|
| Result screen / Full vs Half Time | 8/8 | 8/8 |
| Exact scores | 8/8 | 8/8 |
| Stat cells | 208/208 | 208/208 |
| Both team names exact | 1/8 | 1/8 |
| Readings wrong at ≥ 0.9 confidence | 1 ("Supa Strikas" for "Supå Strikås") | 0 |

`efootball_v1` fails the release gate against stock, so the image keeps `english_g2`.

- **Team names are the gap.** The banner uses eFootball's own display font. Both recognizers misread it
  consistently: "d" becomes "cl" ("climangaaa"), "q" becomes "cj", "h" becomes "l", "F" becomes "T". A
  consistent misread still maps players (Go learns names from the reader's output), but staff see the
  wrong name. Upgrading the recognizer changes the misreads, so names Go has already learned may stop
  matching. The fix is real banner crops: `harvest_crops.py` on labelled real screenshots, split by
  match, then fine-tune with `--select-data efootball_synth_train-efootball_real_train`. The font file in
  `synth/` (`--fonts-dir`) would help too.
- **Synthetic data** copies the real look (yellow score boxes, striped panel, narrow table with dividers,
  flags, the scene behind) but uses system fonts. Penalty display is still unconfirmed.
- **The saves rule is false** on 5 of the 8 real screenshots (see the review, 4b).
- **pHash** can't tell two different matches with the same look apart (`training/phash_study.py`); the
  API treats an equal hash as a hint only.
- **Speed.** About 15 s a read in the container at `--cpus 2` on this laptop while training ran (upper
  bound); CRAFT detection dominates. Measure on an idle machine or the server before sizing replicas.
