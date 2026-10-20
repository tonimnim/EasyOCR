# Screenshot reader: design review

Review of "Screenshot reader: architecture and contract" (status: design). The revised contract (Go
side built) adopts items 1–7 and 11: penalties need mapping, pairs are compared after mapping (mirrored
pairs accepted), auto-decide needs `fullTime: true`, training labels are in screen terms with their own
`VISION_TRAINING_TOKEN`, pHash is only a hint, the threshold is calibrated with `evaluate.py`, and
`503 busy` is retried in seconds. Item 14 is new.

The overall shape is right:
Go owns the queue, retries and decisions; the reader is stateless, internal and returns JSON; nothing is
decided from a single screenshot; forgery is handled by cross-checks rather than forensics. The items
below are the gaps worth closing before the Go side is built. Each says what the reader in this folder
already does about it.

## Must fix

### 1. A draw can still need left/right mapping
"A draw (3–3) needs no mapping" is only true when there was no shoot-out. 3–3 with penalties 4–3 has a
winner, and Go has to know which side is which player to apply it.

**Change:** "A draw without penalties needs no mapping." Auto-decide must treat a penalties result like
any other decided result.

### 2. Compare the two witnesses after mapping, not in raw left/right
The doc says left/right follow the game room, so both phones show the same orientation. That has to be
verified on a real pair of screenshots from one match. If eFootball ever puts each player's own team on
the left, a raw left/right comparison would make every genuine pair "disagree".

**Change:** map each reading to home/away first (team name → player), then compare. This works whichever
way the game orients the screen. A pair that only agrees when mirrored is still an agreement.

### 3. Half-time and extra-time screens may look the same
If eFootball shows the same stats table at half time or from the pause menu (not confirmed: no sample
yet), such a screenshot reads as a valid `match_result` with a plausible but wrong score.

**Change:** `fullTime` is `true` only when the "Full Time" caption is read. Go must not auto-decide on a
reading with `fullTime: false`. *Reader:* `fullTime` is false when the caption is missing or reads as
another caption (Half Time, Extra Time…), and overall confidence is capped at 0.5 in that case, below
the 0.9 auto-decide threshold.

### 4. The training-examples labels must be in screen terms
`GET /v1/internal/vision/training-examples` returns "scores and team names from a staff decision". Staff
decide in home/away terms, but the trainer needs to know what is printed on the left and the right. If
the label arrives as home/away, every crop from a screenshot where home is on the right is mislabelled.

**Change:** Go converts the confirmed result back to `left`/`right` for each screenshot (it has the
team-name mapping at that point) and leaves out examples it can't map. Proposed response shape (the
reader's `fetch_examples.py` is written against it):

```json
{
  "items": [
    {
      "evidenceId": "uuid",
      "matchId": "uuid",
      "downloadUrl": "https://…(short-lived)",
      "contentType": "image/jpeg",
      "label": {
        "screen": "match_result",
        "left":  { "team": "Argentina", "score": 3 },
        "right": { "team": "shinegum",  "score": 3 },
        "penalties": null
      },
      "source": "staff_decision",
      "decidedAt": "2026-10-01T12:00:00Z"
    }
  ],
  "nextCursor": "opaque-or-null"
}
```

Also include screenshots staff marked as "not a result screen" with `"screen": "unknown"`. They are the
negatives the screen check needs.

### 4b. "Goals + opponent's saves ≤ shots on target" is false in eFootball
A real screenshot (console style, FC Barcelona 4–1 Emirhann CHAMPION, 2026-10-08) breaks it on both
sides: 4 goals + 7 saves = 11 > 10 shots on target, and 1 + 3 = 4 > 3. Every other rule holds, and the
stadium scoreboard behind the panel shows 4-1 too. eFootball doesn't count saves (or shots on target)
the way the rule assumes, so the rule flags honest players.

**Change:** make it informational: show it to staff, never block auto-decide on it. Keep goals ≤ shots on
target, shots on target ≤ shots, successful ≤ passes and possession ≈ 100, which hold on all four
samples. *Reader:* still reports `goalsPlusSavesWithinShotsOnTarget`, but it no longer lowers confidence.

### 5. Use a separate token for training exports
`VISION_TOKEN` authorises Go → reader calls and, as written, the reader team's access to every labelled
screenshot. Those should be different secrets: a leaked reader token shouldn't let someone download every
player's screenshots.

**Change:** `VISION_TRAINING_TOKEN` for `training-examples`, held only by whoever runs training.

## Should fix

### 6. pHash is weak on this screen; the stat fingerprint is the strong reuse signal
Most of the Full Time screen is the same fixed layout in every match. The scores and stats are small
digits, and a 64-bit pHash summarises the image at 32×32, where those digits are barely visible. Two
different matches with the same UI can hash identically or within a few bits, so "same imageHash in two
different matches" would flag honest players.

Measured on 150 synthetic pairs (`python -m tonits_vision.training.phash_study`), as Hamming distance:

| pair | median | p95 | max |
|---|---|---|---|
| one screenshot re-encoded and resized (a reuse) | 0 | 2 | 2 |
| same UI, background and crop, different match data | 0 | 4 | 6 |
| two unrelated screens | 22 | 28 | 30 |

A rule loose enough to catch 95% of re-encoded copies (distance ≤ 2) also flags 92% of different matches
that share a look. Real screenshots differ more (the live scene behind the panel moves), so run the study
with `--real-dir` on real ones before relying on even an exact match.

**Change:** treat pHash as a hint for staff (exact or near match → show both matches side by side), not as
proof of fraud. Use the 26-number stat fingerprint as the primary reuse rule, but leave out
low-information fingerprints. An abandoned or very short match can have all-zero stats, and two unrelated
matches would then share one (the reader exposes none of this; Go should skip fingerprints where
`passes` on both sides is under ~20).

### 7. Confidence needs calibrating before auto-decide
EasyOCR's per-text confidence is the mean of character probabilities, not a calibrated probability.
`VISION_AUTO_MIN_CONFIDENCE=0.9` means nothing until it is measured.

**Change:** before turning on `VISION_AUTO_DECIDE`, run `evaluate.py` on the held-out real set and pick the
threshold where exact-score accuracy among readings above it is at least 99.5%. Record that threshold
with the model version. *Reader:* overall confidence is the lower of the two sides, further reduced by
failed checks, a missing Full Time caption, missing stat rows, and disagreement between the general and
the digits-only recognition pass of a score.

### 8. List the shots-on-target check in the response
The Python section lists five checks; the response example shows four. `shotsOnTarget ≤ shots` is missing
from `checks`.

**Change:** add `shotsOnTargetWithinShots`. Each check is `true`, `false` or `null` (`null` = a needed stat
wasn't read). Go must treat `null` as "not passed" for auto-decide but not as a failure to flag.

### 9. Possession can round to 99 or 101
Possession is shown as whole percentages. If the game rounds each side separately, 50.5/49.5 shows as
51/50. *Reader:* accepts 99–101. Tighten to exactly 100 if real screenshots never show otherwise.

### 10. Penalty shoot-out display is not specified
The doc defines `penalties` but none of the three samples has a shoot-out, so the on-screen format is
unknown. *Reader:* recognises bracketed scores beside the main score (`(4) 3 ⓔ 3 (3)`) and a separate
line like `PK 4-3` / `Penalties 4 - 3`. This needs one real shoot-out screenshot to confirm. Until then
Go should not auto-decide a match whose reading has `penalties` set.

### 11. Back-pressure instead of timeouts
On CPU a reading takes about 1–4 s. With `VISION_CONCURRENCY=2` per API replica and several API
replicas, requests can queue in a reader replica long enough to hit the 30 s timeout. Go then retries and
the queue grows.

**Change:** the reader answers 503 `busy` immediately when more than `VISION_MAX_QUEUE` (default 4)
requests are waiting. 503 is already retried with backoff, so no new Go behaviour is needed.

### 12. Image-size limits need a pixel cap too
25 MiB limits bytes, but a 20 KB PNG can declare 50 000 × 50 000 pixels and use gigabytes when decoded.
*Reader:* rejects images over 40 megapixels with 413 `image_too_large` before decoding them.

### 13. `model.version` must cover the layout rules
A reading depends on the recognizer weights and the layout rules. Both must change the version, or
re-reading after a rules fix will look like the same model. *Reader:*
`<easyocr version>+tonits-efootball-<release>` where the release is bumped for either change; the
response also carries `model.recognizer` (e.g. `english_g2` or `efootball_v1`).

### 14. The stats checksum misses many edited scores
For sample 3 (3–0, shots on target 3 and 6, saves 3 and 0), swapping the score to 0–3 passes every check:
0 ≤ 3, 3 ≤ 6, 0 + 0 ≤ 3 and 3 + 3 ≤ 6. Changing it to 3–4 fails (4 + 3 > 6). The checks catch large or
lopsided edits, not a plausible swap; the two-witness comparison is what catches those. With the saves
rule informational (4b), the checksum is weaker still: replace "usually breaks one of these" with
"sometimes", and never let passing checks stand in for a second screenshot.

## Smaller points

- **Text is small.** In the 826×371 crops each stat row is about 20 px tall, and EasyOCR's default
  `min_size=20` drops single digits. *Reader:* scales every image to 1600 px wide before detection and
  uses `min_size=8`.
- **Accents.** EasyOCR's `english_g2` recognizer has no accented letters, so "R. Martínez" can't be read
  exactly. The fine-tune starts from `latin_g2`, which has them, with the same architecture.
- **Team names ending in digits.** "SQUAD 0" sits right beside the score. *Reader:* the score is the
  digit token nearest the centre on each side; everything further out is the team name.
- **The eFootball logo** between the scores can be read as `e`, `@`, `0` or `6`. *Reader:* ignores any
  token that covers the centre of the banner, unless it splits cleanly into two scores (`3@3`).
- **The coach row moves.** In the real console screenshot it sits *under* the table. *Reader:* looks
  between the banner and the table, then just under the table, and skips button labels.
- **eFootball's display font** (team names in the console style) isn't like any system font: the stock
  recognizer reads its "F" as "T" ("TC Barcelona"), and a recognizer fine-tuned only on synthetic fonts
  adds accents ("FC ñarcelona"). *Reader:* keeps an accent only when an ASCII-only read agrees on the
  letter. The lasting fix is real crops (`harvest_crops.py`) or the font itself in `synth/`.
- **`managers`** is best effort and is never used for decisions. On synthetic screens the coach row is
  always found, but small grey text costs exact matches ("T: Tuchel" for "T. Tuchel").
- **Lone score digits.** CRAFT often misses a single "0" or "1" in its box beside the round logo, even
  at 2× zoom. *Reader:* finds score digits by shape (enclosed glyph-sized components next to the logo,
  `glyphs.py`) and uses CRAFT's token, when there is one, as a cross-check.
- **Screen classifier.** A MobileNet classifier only pays off once there are several UI versions. For v1
  the layout rules are the classifier: `match_result` needs a banner with two scores and at least 8 of the
  13 stat labels. Revisit when Konami changes the UI.
- **Request IDs.** The reader logs `X-Request-ID` and `X-Evidence-ID` and echoes `X-Request-ID` back.
- **The `stats` object can be partial.** "Leave out rows you couldn't read" means Go must expect missing
  keys, and `stats` is `null` when `screen` is `unknown`.
