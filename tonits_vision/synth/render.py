"""Render synthetic eFootball "Full Time" screens with their ground truth.

The look is measured from a real phone screenshot (sample 3, 826x371, a 20:9 phone screen):

- the live match scene (pitch and players, sometimes with jersey numbers) shows above and below the panel;
- a full-width navy banner: left team name (white, bold, right-aligned), the left score (dark digits in a
  yellow box), the eFootball logo, the right score, the right team name (left-aligned);
- a yellow strip with "Full Time" in black, under the banner;
- a yellow and navy diagonal-striped panel with a narrow navy table in the middle: 13 rows, label centred,
  each side's value on the same row, thin light lines between rows;
- each side's flag or club crest in a pink square, left and right of the table;
- a blue "Next >" button bottom right.

The console style (sample 2, 960x540, with a coach row, "Total Shots" and a Highlights button) is
reconstructed from the doc's description only, in the same visual language. Penalty shoot-out display is
a guess (see DESIGN_REVIEW.md). Positions, sizes, colours, fonts, crops, scale and compression are
randomised so the reader and the recognizer don't learn one exact image.
"""

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..stats import FULL_TIME_CAPTION, STAT_ROWS
from .names import manager_name, team_name

_BOLD_FONTS = [
    "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Black.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
]
_REGULAR_FONTS = [
    "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

NAVY = (14, 20, 92)
YELLOW = (234, 228, 0)
PINK = (236, 0, 138)
STYLES = ("phone", "console")


def available_fonts(extra_dir=None):
    """Bold system fonts first (the game's UI is bold), then regular ones, then any in extra_dir.

    Put a font closer to eFootball's in extra_dir to make synthetic data more like real screenshots.
    """
    fonts = [Path(p) for p in _BOLD_FONTS + _REGULAR_FONTS if Path(p).exists()]
    if extra_dir:
        fonts = sorted(p for p in Path(extra_dir).iterdir() if p.suffix.lower() in (".ttf", ".otf")) + fonts
    if not fonts:
        raise RuntimeError("no fonts found; pass a fonts directory")
    return fonts


@dataclass
class TextItem:
    text: str
    kind: str  # team, score, logo, caption, label, value, manager, penalty, ui, scene
    box: tuple  # (x0, y0, x1, y1) in pixels of the final image


@dataclass
class Sample:
    image_bytes: bytes
    content_type: str
    label: dict  # ground truth in the reader's response shape
    items: list
    style: str
    image: Image.Image  # the final image, before encoding


def random_stats(rng: random.Random, left_goals: int, right_goals: int) -> dict:
    """Stats that satisfy every plausibility rule for the given score."""
    def side(goals):
        on_target = goals + rng.randint(0, 8)
        shots = on_target + rng.randint(0, 9)
        return shots, on_target

    l_shots, l_sot = side(left_goals)
    r_shots, r_sot = side(right_goals)
    l_poss = rng.randint(30, 70)
    l_pass = rng.randint(30, 400)
    r_pass = rng.randint(30, 400)
    return {
        "possession": [l_poss, 100 - l_poss],
        "shots": [l_shots, r_shots],
        "shotsOnTarget": [l_sot, r_sot],
        "fouls": [rng.randint(0, 8), rng.randint(0, 8)],
        "offsides": [rng.randint(0, 5), rng.randint(0, 5)],
        "cornerKicks": [rng.randint(0, 12), rng.randint(0, 12)],
        "freeKicks": [rng.randint(0, 10), rng.randint(0, 10)],
        "passes": [l_pass, r_pass],
        "successfulPasses": [int(l_pass * rng.uniform(0.55, 0.95)), int(r_pass * rng.uniform(0.55, 0.95))],
        "crosses": [rng.randint(0, 20), rng.randint(0, 20)],
        "interceptions": [rng.randint(0, 40), rng.randint(0, 40)],
        "tackles": [rng.randint(0, 25), rng.randint(0, 25)],
        # A save stops a shot on target that wasn't a goal.
        "saves": [rng.randint(0, r_sot - right_goals), rng.randint(0, l_sot - left_goals)],
    }


def random_label(rng: random.Random, style: str, caption: str = FULL_TIME_CAPTION) -> dict:
    left_score = min(int(rng.expovariate(0.5)), 15)
    right_score = min(int(rng.expovariate(0.5)), 15)
    penalties = None
    if left_score == right_score and rng.random() < 0.25:
        winner, loser = rng.choice([(5, 4), (4, 3), (3, 2), (5, 3), (4, 2), (7, 6), (3, 1)])
        penalties = {"left": winner, "right": loser} if rng.random() < 0.5 else {"left": loser, "right": winner}
    managers = None
    if style == "console" and rng.random() < 0.85:
        managers = {"left": manager_name(rng), "right": manager_name(rng)}
    left_team = team_name(rng)
    right_team = team_name(rng)
    while right_team.lower() == left_team.lower():
        right_team = team_name(rng)
    return {
        "screen": "match_result",
        "fullTime": caption == FULL_TIME_CAPTION,
        "left": {"team": left_team, "score": left_score},
        "right": {"team": right_team, "score": right_score},
        "penalties": penalties,
        "managers": managers,
        "stats": random_stats(rng, left_score, right_score),
    }


def _jitter(rng, color, amount=14):
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in color)


class _Canvas:
    """A PIL canvas that remembers where every piece of text went."""

    def __init__(self, image):
        self.image = image
        self.draw = ImageDraw.Draw(self.image)
        self.items = []

    def text(self, xy, text, font, fill, anchor, kind):
        self.draw.text(xy, text, font=font, fill=fill, anchor=anchor)
        box = self.draw.textbbox(xy, text, font=font, anchor=anchor)
        self.items.append(TextItem(text, kind, tuple(float(v) for v in box)))
        return box


def _font(path, size):
    return ImageFont.truetype(str(path), size=max(int(round(size)), 6))


def _scene(rng, size, fonts):
    """The live match behind the panel: grass, blurred players, sometimes a jersey number."""
    w, h = size
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    top, bottom = _jitter(rng, (38, 92, 40), 18), _jitter(rng, (52, 120, 50), 18)
    arr = np.repeat(np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t, w, axis=1)
    arr += np.random.default_rng(rng.getrandbits(32)).normal(0, 3.0, arr.shape)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    numbers = []
    for _ in range(rng.randint(1, 4)):
        px, py = rng.uniform(0, w), rng.choice([rng.uniform(-h * 0.1, h * 0.12), rng.uniform(h * 0.85, h * 1.05)])
        pw, ph = rng.uniform(w * 0.04, w * 0.09), rng.uniform(h * 0.15, h * 0.3)
        kit = rng.choice([(200, 30, 40), (20, 20, 20), (240, 240, 240), (30, 60, 160), (230, 120, 20)])
        draw.ellipse((px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2), fill=kit)
        draw.ellipse((px - pw * 0.25, py - ph * 0.75, px + pw * 0.25, py - ph * 0.45), fill=(150, 100, 70))
        if rng.random() < 0.5:
            numbers.append((px, py, ph * 0.3))
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    draw = ImageDraw.Draw(image)
    items = []
    for px, py, size_ in numbers:
        font = _font(rng.choice(fonts), size_)
        text = str(rng.randint(1, 99))
        draw.text((px, py), text, font=font, fill=(240, 240, 240), anchor="mm")
        items.append(TextItem(text, "scene", tuple(float(v) for v in draw.textbbox((px, py), text, font=font, anchor="mm"))))
    return image, items


def _stripes(canvas, rng, box):
    """Yellow panel with navy diagonal bands."""
    x0, y0, x1, y1 = box
    canvas.draw.rectangle(box, fill=_jitter(rng, YELLOW, 10))
    navy = _jitter(rng, NAVY, 8)
    height = y1 - y0
    x = x0 - height + rng.uniform(0, 80)
    while x < x1:
        band = rng.uniform(40, 110)
        canvas.draw.polygon([(x, y1), (x + band, y1), (x + band + height, y0), (x + height, y0)], fill=navy)
        x += band + rng.uniform(60, 180)
    return navy


def _emblem(canvas, rng, center, size):
    """A flag or a crest in a pink square."""
    cx, cy = center
    canvas.draw.rectangle((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), fill=_jitter(rng, PINK, 12))
    fw, fh = size * 0.72, size * 0.5
    colors = [tuple(rng.randint(0, 255) for _ in range(3)) for _ in range(rng.randint(2, 4))]
    if rng.random() < 0.6:  # flag: horizontal stripes
        canvas.draw.rectangle((cx - fw / 2 - 2, cy - fh / 2 - 2, cx + fw / 2 + 2, cy + fh / 2 + 2), fill=(255, 255, 255))
        for i, c in enumerate(colors):
            canvas.draw.rectangle((cx - fw / 2, cy - fh / 2 + i * fh / len(colors), cx + fw / 2, cy - fh / 2 + (i + 1) * fh / len(colors)), fill=c)
    else:  # crest: a shield with stripes
        sw, sh = size * 0.55, size * 0.7
        shield = [(cx - sw / 2, cy - sh / 2), (cx + sw / 2, cy - sh / 2), (cx + sw / 2, cy + sh * 0.1), (cx, cy + sh / 2), (cx - sw / 2, cy + sh * 0.1)]
        canvas.draw.polygon(shield, fill=colors[0], outline=(255, 255, 255))
        for i in range(1, 5):
            y = cy - sh / 2 + i * sh / 6
            canvas.draw.line((cx - sw / 2 + 3, y, cx + sw / 2 - 3, y), fill=colors[1], width=max(2, int(sh / 14)))


def _logo(canvas, rng, center, size):
    """The eFootball logo between the scores: an 'e' built from horizontal strokes."""
    x, y = center
    navy = _jitter(rng, (20, 60, 200), 20)
    r = size / 2
    canvas.draw.rectangle((x - r, y - r, x + r, y + r), fill=_jitter(rng, YELLOW, 8))
    canvas.draw.arc((x - r * 0.8, y - r * 0.8, x + r * 0.8, y + r * 0.8), start=20, end=330, fill=navy, width=max(2, int(r / 4)))
    canvas.draw.line((x - r * 0.75, y, x + r * 0.8, y), fill=navy, width=max(2, int(r / 4)))
    canvas.items.append(TextItem("", "logo", (x - r, y - r, x + r, y + r)))


def _render_result(rng, fonts, label, style, caption):
    bold = [f for f in fonts if f.name in {Path(p).name for p in _BOLD_FONTS}] or fonts
    text_font = rng.choice(bold) if rng.random() < 0.9 else rng.choice(fonts)
    num_font = text_font if rng.random() < 0.7 else rng.choice(bold)

    if style == "phone":
        H = 371
        W = int(H * rng.choice([20 / 9, 20 / 9, 19.5 / 9, 2.0, 16 / 9]))
        u = 1.0  # unit: pixels per sample-3 pixel
        band = (55, 86)
        table_top, pitch = 101, 16.45
        table_half, value_dx = 133, 97
        f_team, f_score, f_cap, f_label, f_value, f_manager = 16.5, 19, 11.5, 11.5, 11.5, 0
    else:
        H = 540
        W = 960
        u = 1.25
        band = (62, 104)
        table_top, pitch = 152, 25.5
        table_half, value_dx = 175, 128
        f_team, f_score, f_cap, f_label, f_value, f_manager = 20, 24, 14.5, 15, 15, 13.5
    s = rng.uniform(0.94, 1.06)
    f_team, f_score, f_cap, f_label, f_value, f_manager = (v * s for v in (f_team, f_score, f_cap, f_label, f_value, f_manager))

    scene, scene_items = _scene(rng, (W, H), bold)
    canvas = _Canvas(scene)
    canvas.items.extend(scene_items)

    # A black notch/edge bar on one side moves the game area's centre (sample 3: 20 px on the left).
    notch_l = rng.choice([0, 0, rng.randint(12, 28)])
    notch_r = rng.choice([0, 0, 0, rng.randint(12, 28)])
    cx = notch_l + (W - notch_l - notch_r) / 2 + rng.uniform(-3, 3)

    navy = _jitter(rng, NAVY, 8)
    table_bottom = table_top + pitch * len(STAT_ROWS)
    panel_bottom = table_bottom + rng.uniform(0, 4)
    _stripes(canvas, rng, (0, band[1], W, panel_bottom))
    canvas.draw.rectangle((0, band[0], W, band[1]), fill=navy)

    # Banner.
    by = (band[0] + band[1]) / 2
    box_w, box_h = 24 * u, 25 * u
    gap = 21 * u
    white = _jitter(rng, (250, 250, 250), 5)
    dark = _jitter(rng, (16, 22, 60), 10)
    sf = _font(num_font, f_score)
    tf = _font(text_font, f_team)
    for sign, side in ((-1, "left"), (1, "right")):
        bx = cx + sign * (gap + box_w / 2)
        canvas.draw.rectangle((bx - box_w / 2, by - box_h / 2, bx + box_w / 2, by + box_h / 2), fill=_jitter(rng, YELLOW, 8))
        canvas.text((bx, by + 0.5), str(label[side]["score"]), sf, dark, "mm", "score")
    _logo(canvas, rng, (cx, by), gap * 1.8)
    outer = gap + box_w + 5 * u
    pens = label["penalties"]
    pen_beside = pens is not None and rng.random() < 0.5  # "(4) [3] e [3] (3)"; otherwise a "PK 4 - 3" line
    if pen_beside:
        pf = _font(num_font, f_score * 0.7)
        for sign, side in ((-1, "left"), (1, "right")):
            canvas.text((cx + sign * (outer + 12 * u), by), f"({pens[side]})", pf, white, "mm", "penalty")
        outer += 28 * u
    canvas.text((cx - outer, by), label["left"]["team"], tf, white, "rm", "team")
    canvas.text((cx + outer, by), label["right"]["team"], tf, white, "lm", "team")

    # Under the banner: coach row (console), penalties line, caption on the yellow strip.
    y = band[1]
    if label["managers"]:
        mf = _font(text_font, f_manager)
        canvas.draw.rectangle((0, y, W, y + f_manager * 1.6), fill=navy)
        my = y + f_manager * 0.8
        grey = _jitter(rng, (200, 205, 220), 10)
        canvas.text((cx - 10 * u, my), label["managers"]["left"], mf, grey, "rm", "manager")
        canvas.draw.line((cx, my - f_manager * 0.5, cx, my + f_manager * 0.5), fill=grey, width=2)
        canvas.text((cx + 10 * u, my), label["managers"]["right"], mf, grey, "lm", "manager")
        y += f_manager * 1.6
    cap_f = _font(text_font, f_cap)
    pk_line = pens is not None and not pen_beside
    # The yellow strip first, then the text on it (penalties line, if any, above the caption).
    canvas.draw.rectangle((cx - table_half, y, cx + table_half, y + f_cap * (2.7 if pk_line else 1.3)), fill=_jitter(rng, YELLOW, 6))
    if pk_line:
        text = rng.choice(["PK {l} - {r}", "({l} PK {r})", "Penalties {l} - {r}"]).format(l=pens["left"], r=pens["right"])
        canvas.text((cx, y + f_cap * 0.7), text, cap_f, (10, 10, 10), "mm", "penalty")
        y += f_cap * 1.4
    cap_text = caption.upper() if rng.random() < 0.05 else caption
    canvas.text((cx, y + f_cap * 0.65), cap_text, cap_f, (8, 8, 8), "mm", "caption")
    y += f_cap * 1.3
    table_top = max(table_top, y)

    # Table.
    canvas.draw.rectangle((cx - table_half, table_top, cx + table_half, table_bottom), fill=navy)
    line_c = _jitter(rng, (205, 210, 230), 15)
    lf = _font(text_font, f_label)
    vf = _font(num_font, f_value)
    shots_label = "Total Shots" if style == "console" else ("Shots" if rng.random() < 0.85 else "Total Shots")
    pitch = (table_bottom - table_top) / len(STAT_ROWS)
    vdx = value_dx * rng.uniform(0.96, 1.04)
    for i, row in enumerate(STAT_ROWS):
        ry = table_top + (i + 0.5) * pitch
        if i:
            yline = table_top + i * pitch
            canvas.draw.line((cx - table_half + 2, yline, cx + table_half - 2, yline), fill=line_c, width=1)
        text = shots_label if row.key == "shots" else row.labels[0]
        canvas.text((cx, ry), text, lf, white, "mm", "label")
        lv, rv = label["stats"][row.key]
        suffix = "%" if row.percent else ""
        canvas.text((cx - vdx, ry), f"{lv}{suffix}", vf, white, "mm", "value")
        canvas.text((cx + vdx, ry), f"{rv}{suffix}", vf, white, "mm", "value")

    # Flags or crests either side of the table.
    em = 82 * u * rng.uniform(0.95, 1.05)
    ey = (table_top + table_bottom) / 2 + rng.uniform(-6, 6)
    for sign in (-1, 1):
        ex = cx + sign * 232 * u
        if 0 < ex - em / 2 and ex + em / 2 < W:
            _emblem(canvas, rng, (ex, ey), em)

    # Buttons.
    bf = _font(text_font, f_cap * 1.35)
    buttons = ["Next"] if style == "phone" else rng.choice([["Highlights", "Next"], ["Highlights"]])
    bx1 = W - notch_r - 60 * u
    for name in reversed(buttons):
        bw = (len(name) * f_cap * 0.75 + 50) * u
        bx0 = bx1 - bw
        bty0, bty1 = H - 48 * u, H - 18 * u
        if bty0 > table_bottom + 2:
            canvas.draw.rounded_rectangle((bx0, bty0, bx1, bty1), radius=6, fill=_jitter(rng, (20, 150, 255), 15))
            canvas.text((bx0 + bw * 0.42, (bty0 + bty1) / 2), name, bf, (255, 255, 255), "mm", "ui")
        bx1 = bx0 - 16 * u

    if notch_l:
        canvas.draw.rectangle((0, 0, notch_l, H), fill=(0, 0, 0))
    if notch_r:
        canvas.draw.rectangle((W - notch_r, 0, W, H), fill=(0, 0, 0))
    return canvas


def _render_unknown(rng, fonts):
    """A non-result screen: profile, squad list or menu with numbers but no stats table."""
    W, H = rng.choice([(826, 371), (960, 540), (1280, 720), (1600, 720)])
    if rng.random() < 0.5:
        image, _ = _scene(rng, (W, H), fonts)
    else:
        image = Image.new("RGB", (W, H), _jitter(rng, NAVY, 20))
    canvas = _Canvas(image)
    font_path = rng.choice(fonts)
    white = (240, 240, 240)
    words = ["Squad", "Matchday", "Rating", "Level", "eFootball Points", "Coins", "Dream Team", "Contract",
             "Overall", "Goals", "Assists", "Matches Played", "Win Rate", "Division", "Events", "Shop",
             "Player Profile", "Team Strength", "Playing Style", "Formation", "Substitutes", "Reserves",
             "Kick Off", "Match Settings", "Possession", "Passes"]
    y = rng.uniform(20, 50)
    while y < H - 20:
        size = rng.uniform(13, 26)
        x = rng.uniform(20, W * 0.5)
        text = rng.choice(words)
        if rng.random() < 0.5:
            text += f" {rng.randint(0, 3000)}"
        canvas.text((x, y), text, _font(font_path, size), white, "lm", "ui")
        if rng.random() < 0.5:
            canvas.text((W - rng.uniform(30, 200), y), str(rng.randint(0, 99)), _font(font_path, size), white, "rm", "ui")
        y += size * rng.uniform(1.6, 3.0)
    return canvas


def _finish(rng, canvas):
    """Rescale, blur and encode like a real capture or a messaging app would."""
    image, items = canvas.image, canvas.items
    # 826x371 is a 2400x1080 phone screen shrunk by a messaging app; keep all three sizes common.
    scale = rng.choice([1.0, 1.0, rng.uniform(0.9, 1.2), 1.94, 2.9, rng.uniform(1.3, 3.0)])
    if scale != 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        items = [TextItem(i.text, i.kind, tuple(v * scale for v in i.box)) for i in items]
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7) * max(scale, 1.0)))
    buf = io.BytesIO()
    if rng.random() < 0.75:
        image.save(buf, format="JPEG", quality=rng.randint(45, 95))
        content_type = "image/jpeg"
    else:
        image.save(buf, format="PNG")
        content_type = "image/png"
    return buf.getvalue(), content_type, image, items


def render(rng: random.Random, fonts, kind: str = "auto") -> Sample:
    """kind: 'auto' (mostly results), 'result', 'half_time' or 'unknown'."""
    if kind == "auto":
        roll = rng.random()
        kind = "unknown" if roll < 0.08 else "half_time" if roll < 0.12 else "result"
    if kind == "unknown":
        canvas = _render_unknown(rng, fonts)
        label = {"screen": "unknown"}
        style = "unknown"
    else:
        style = "phone" if rng.random() < 0.75 else "console"
        caption = FULL_TIME_CAPTION if kind == "result" else "Half Time"
        label = random_label(rng, style, caption)
        canvas = _render_result(rng, fonts, label, style, caption)
    data, content_type, image, items = _finish(rng, canvas)
    return Sample(data, content_type, label, items, style, image)
