"""Layout rules: turn OCR tokens from a Full Time screen into field positions.

Everything here works on plain tokens (text, confidence, box), so it can be tested without a model. The
reader calls `parse()` on a first, coarse OCR pass to find the table and the banner, re-detects the banner
strip at a higher zoom and passes those tokens to `parse_banner()`, and reads the stat values as a grid
from `value_cells()`.

Anchors, in order:
1. the 13 stat labels, matched fuzzily and in screen order; their centre is the screen's centre line;
2. the values sit in two columns either side of the labels, one per row;
3. above the table, the line with the tallest text on both sides of the centre is the banner: the digit
   token nearest the centre on each side is the score and everything further out is the team name;
4. lines between the banner and the table are the caption, the coach row and a penalties line.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import median
from typing import Optional

from .stats import FULL_TIME_CAPTION, OTHER_CAPTIONS, STAT_ROWS

LABEL_MIN_RATIO = 0.75
CAPTION_MIN_RATIO = 0.7
MIN_STAT_ROWS = 8

# Characters the general recognizer confuses with digits.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                                   "i": "1", "!": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8",
                                   "g": "9", "q": "9"})
_SEPARATORS = {"|", "I", "l", "1", "/", "!", ":", "-", "—", "–"}
# Two scores read as one token with the logo between them: "3@3", "3 e 0", "2 1". Not "76" (a jersey number).
_MERGED_SCORES = re.compile(r"\s*(\d{1,2})(?:\s+|\s*[^\d\s]{1,3}\s*)(\d{1,2})\s*")


@dataclass
class Token:
    text: str
    conf: float
    box: tuple  # (x0, y0, x1, y1)

    @property
    def x0(self): return self.box[0]
    @property
    def y0(self): return self.box[1]
    @property
    def x1(self): return self.box[2]
    @property
    def y1(self): return self.box[3]
    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2
    @property
    def h(self): return self.box[3] - self.box[1]
    @property
    def w(self): return self.box[2] - self.box[0]


def union_box(tokens):
    return (min(t.x0 for t in tokens), min(t.y0 for t in tokens), max(t.x1 for t in tokens), max(t.y1 for t in tokens))


def join_text(tokens):
    return " ".join(t.text.strip() for t in sorted(tokens, key=lambda t: t.x0) if t.text.strip())


@dataclass
class Field:
    """Where a field is on the image and what the first pass read there."""
    text: str
    conf: float
    box: tuple
    tokens: list = field(default_factory=list)


@dataclass
class Layout:
    screen: str = "unknown"
    center_x: Optional[float] = None
    labels: dict = field(default_factory=dict)  # stat key -> label Field, for every row found
    value_columns: Optional[tuple] = None  # (left centre x, right centre x, value height, value width)
    banner_band: Optional[tuple] = None  # (y0, y1) of the banner line, or of the area it must be in
    banner_tokens: list = field(default_factory=list)  # the banner line's first-pass tokens
    left_team: Optional[Field] = None
    right_team: Optional[Field] = None
    left_score: Optional[Field] = None
    right_score: Optional[Field] = None
    left_penalty: Optional[Field] = None
    right_penalty: Optional[Field] = None
    penalty_line: Optional[Field] = None  # "PK 4 - 3" style
    caption: Optional[str] = None  # the caption as matched: "Full Time", "Half Time", …
    caption_conf: float = 0.0
    left_manager: Optional[Field] = None
    right_manager: Optional[Field] = None


def normalise_words(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


def similarity(a, b):
    a, b = normalise_words(a), normalise_words(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def as_digits(text, allow_percent=False):
    """'12' -> 12, '64%' -> 64 (with allow_percent), 'O' -> 0; None if it isn't a clean 1-3 digit number."""
    t = text.strip().replace(" ", "")
    if allow_percent:
        t = t.rstrip("%").rstrip("°")  # % is sometimes read as °
    t = t.translate(_DIGIT_LOOKALIKES)
    return int(t) if re.fullmatch(r"\d{1,3}", t) else None


def group_lines(tokens):
    """Group tokens into text lines, top to bottom; each line sorted left to right."""
    lines = []
    for tok in sorted(tokens, key=lambda t: t.cy):
        for line in reversed(lines[-3:]):
            ref_cy = median(t.cy for t in line)
            ref_h = median(t.h for t in line)
            if abs(tok.cy - ref_cy) < 0.5 * max(min(tok.h, ref_h), 0.6 * max(tok.h, ref_h)):
                line.append(tok)
                break
        else:
            lines.append([tok])
    return [sorted(line, key=lambda t: t.x0) for line in lines]


def _label_candidates(line):
    """Best match per stat key over runs of up to three adjacent word tokens in this line."""
    best = {}
    words = [i for i, t in enumerate(line) if re.search(r"[A-Za-z]{2,}", t.text)]
    for start in range(len(words)):
        for end in range(start, min(start + 3, len(words))):
            run = line[words[start]:words[end] + 1]
            if any(not re.search(r"[A-Za-z]", t.text) for t in run):
                continue
            text = join_text(run)
            for row in STAT_ROWS:
                for label in row.labels:
                    score = similarity(text, label)
                    if score >= LABEL_MIN_RATIO:
                        # Prefer the longest label among equally good matches ("Successful Passes" over "Passes").
                        key = (score, len(label))
                        if row.key not in best or key > best[row.key][0]:
                            best[row.key] = (key, run, text)
    return best


def match_stat_rows(lines):
    """Assign lines to stat rows, keeping screen order, maximising total match quality."""
    order = [row.key for row in STAT_ROWS]
    cands = [_label_candidates(line) for line in lines]
    n, m = len(lines), len(order)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]  # dp[i][j]: best total using lines[:i] and rows[:j]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            hit = cands[i - 1].get(order[j - 1])
            if hit:
                best = max(best, dp[i - 1][j - 1] + hit[0][0])
            dp[i][j] = best
    matches = {}
    i, j = n, m
    while i > 0 and j > 0:
        hit = cands[i - 1].get(order[j - 1])
        if hit and abs(dp[i][j] - (dp[i - 1][j - 1] + hit[0][0])) < 1e-9:
            matches[order[j - 1]] = (i - 1, hit[1], hit[2])
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return matches  # key -> (line index, label tokens, label text)


def _field(tokens):
    tokens = sorted(tokens, key=lambda t: t.x0)
    return Field(join_text(tokens), min(t.conf for t in tokens), union_box(tokens), tokens)


def _is_logo(tok, cx, line_h):
    """The logo sits on the centre line; any token reaching into the central zone is (part of) it."""
    zone = 0.5 * line_h
    return tok.x0 < cx + zone and tok.x1 > cx - zone


def banner_line(lines, cx):
    """The banner among these lines: text on both sides of the centre, preferring lines with letters
    (team names), then the tallest."""
    best, best_key = None, None
    for line in lines:
        text = join_text(line)
        if caption_match(text)[0] or parse_penalty_text(text):
            continue
        line_h = median(t.h for t in line)
        sides = {t.cx < cx for t in line if not _is_logo(t, cx, line_h)}
        merged = any(_is_logo(t, cx, line_h) and _MERGED_SCORES.fullmatch(t.text) for t in line)
        if len(sides) < 2 and not merged:
            continue
        key = (any(re.search(r"[A-Za-z]{2,}", t.text) for t in line), median(t.h for t in line))
        if best_key is None or key > best_key:
            best, best_key = line, key
    return best


def _overlap(a, b):
    """Share of box a's width that box b covers, when they overlap vertically."""
    if a[3] <= b[1] or b[3] <= a[1]:
        return 0.0
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(a[2] - a[0], 1e-9)


def parse_banner(line, cx, layout, score_boxes=(None, None)):
    """Scores, team names and bracketed penalties from the banner line. True if both scores were found.

    score_boxes: (left, right) boxes found by shape next to the logo (glyphs.score_box). When given, that
    is the score; a CRAFT digit token on it supplies the first-pass text. Otherwise the score is the digit
    token nearest the centre, if it is close enough to the logo to be one.
    """
    line_h = median(t.h for t in line)
    left, right = [], []
    for tok in line:
        if _is_logo(tok, cx, line_h):
            # The logo, or scores merged with it: "3@3", "3 e 3".
            m = _MERGED_SCORES.fullmatch(tok.text)
            if m and tok.w > 1.5 * tok.h:
                left.append(Token(m.group(1), tok.conf, (tok.x0, tok.y0, tok.cx, tok.y1)))
                right.append(Token(m.group(2), tok.conf, (tok.cx, tok.y0, tok.x1, tok.y1)))
            continue
        (left if tok.cx < cx else right).append(tok)

    def side(tokens, sbox, sign):
        tokens = sorted(tokens, key=lambda t: abs(t.cx - cx))  # from the centre outwards
        if sbox is not None:
            on_box = [t for t in tokens if _overlap(t.box, sbox) > 0.5 or _overlap(sbox, t.box) > 0.5]
            digit = next((t for t in on_box if as_digits(t.text) is not None), None)
            score = Token(digit.text if digit else "", digit.conf if digit else 1.0, sbox)
            outer = sbox[0] if sign < 0 else sbox[2]
            rest = [t for t in tokens if t not in on_box and (t.cx < outer if sign < 0 else t.cx > outer)]
        else:
            def inner_gap(t):
                return cx - t.x1 if sign < 0 else t.x0 - cx
            score_i = next((i for i, t in enumerate(tokens) if as_digits(t.text) is not None
                            and not t.text.strip().startswith("(") and t.h >= 0.5 * line_h
                            and inner_gap(t) <= 2.5 * line_h), None)
            if score_i is None:
                return None, None, [t for t in tokens if t.text.strip()]
            score = tokens[score_i]
            rest = tokens[score_i + 1:]
        penalty = None
        if rest:
            nxt = rest[0]
            bracketed = re.fullmatch(r"\s*[\(\[{]\s*\d{1,2}\s*[\)\]}]\s*", nxt.text)
            small_digit = as_digits(nxt.text) is not None and nxt.h <= 0.75 * score.h
            if bracketed or (small_digit and len(rest) > 1):
                penalty = nxt
                rest = rest[1:]
        # A lone "[" or "|" is the edge of a score box or a divider, not part of a name ("[KE] Lions" stays).
        return score, penalty, [t for t in rest if re.search(r"[^\W_]", t.text)]

    l_score, l_pen, l_team = side(left, score_boxes[0], -1)
    r_score, r_pen, r_team = side(right, score_boxes[1], 1)
    layout.left_score = _field([l_score]) if l_score else None
    layout.right_score = _field([r_score]) if r_score else None
    layout.left_penalty = _field([l_pen]) if l_pen else None
    layout.right_penalty = _field([r_pen]) if r_pen else None
    layout.left_team = _field(l_team) if l_team else None
    layout.right_team = _field(r_team) if r_team else None
    return l_score is not None and r_score is not None


_PK_PATTERNS = (
    re.compile(r"\(?\s*(\d{1,2})\s*(?:PK|P\.?K\.?|pens?)\s*(\d{1,2})\s*\)?", re.I),
    re.compile(r"(?:PK|P\.?K\.?|Pen\w*)\W*(\d{1,2})\s*[-–—:]\s*(\d{1,2})", re.I),
)


def parse_penalty_text(text):
    for pattern in _PK_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def caption_match(text):
    """('Full Time', ratio) for the best caption match, or (None, 0)."""
    best, best_ratio = None, 0.0
    for caption in (FULL_TIME_CAPTION,) + OTHER_CAPTIONS:
        r = similarity(text, caption)
        if r > best_ratio:
            best, best_ratio = caption, r
    return (best, best_ratio) if best_ratio >= CAPTION_MIN_RATIO else (None, 0.0)


def parse(tokens):
    """Find the table, the banner and the lines between them in the first-pass tokens."""
    layout = Layout()
    if not tokens:
        return layout
    lines = group_lines(tokens)
    matches = match_stat_rows(lines)
    if len(matches) < MIN_STAT_ROWS:
        return layout
    layout.screen = "match_result"
    layout.center_x = cx = median((min(t.x0 for t in run) + max(t.x1 for t in run)) / 2 for _, run, _ in matches.values())
    for row in STAT_ROWS:  # screen order
        if row.key in matches:
            _, run, text = matches[row.key]
            layout.labels[row.key] = Field(text, min(t.conf for t in run), union_box(run), run)
    label_h = median(f.box[3] - f.box[1] for f in layout.labels.values())
    _fill_hidden_rows(layout, cx, label_h)

    # Value columns, from the nearest number either side of each label. Real values are only located
    # here; the reader reads every cell of the grid, so values CRAFT missed are still read.
    nearest_l, nearest_r = [], []
    for key, (li, run, _) in matches.items():
        ids = {id(t) for t in run}
        lx0, lx1 = min(t.x0 for t in run), max(t.x1 for t in run)
        nums = [t for t in lines[li] if id(t) not in ids and as_digits(t.text, allow_percent=True) is not None]
        lefts = [t for t in nums if t.cx < lx0]
        rights = [t for t in nums if t.cx > lx1]
        if lefts:
            nearest_l.append(max(lefts, key=lambda t: t.x1))
        if rights:
            nearest_r.append(min(rights, key=lambda t: t.x0))
    if nearest_l and nearest_r:
        lcol = median(t.cx for t in nearest_l)
        rcol = median(t.cx for t in nearest_r)
        # Mirror a column that looks off (e.g. only flag digits were found) around the centre.
        if abs((cx - lcol) - (rcol - cx)) > 2 * label_h:
            spread = median([cx - lcol, rcol - cx])
            lcol, rcol = cx - spread, cx + spread
        vh = median(t.h for t in nearest_l + nearest_r)
        vw = max(t.w for t in nearest_l + nearest_r)
        layout.value_columns = (lcol, rcol, vh, vw)

    first_row_line = min(li for li, _, _ in matches.values())
    last_row_line = max(li for li, _, _ in matches.values())
    first_label = layout.labels[min(matches, key=lambda k: matches[k][0])]
    table_h = max(lines[last_row_line][0].cy - first_label.box[1], label_h * 13)

    # Lines above the table but close to it. The scene above the panel (jersey numbers) is further away.
    window_top = first_label.box[1] - 0.45 * table_h
    above = [line for line in lines[:first_row_line] if median(t.cy for t in line) > window_top]
    banner = banner_line(above, cx)
    if banner is not None:
        layout.banner_band = (min(t.y0 for t in banner), max(t.y1 for t in banner))
        layout.banner_tokens = banner
        parse_banner(banner, cx, layout)
        between = [line for line in above if median(t.cy for t in line) > median(t.cy for t in banner)]
    else:
        layout.banner_band = (window_top, first_label.box[1])
        between = above

    # Between the banner and the table: caption, coach row, penalties line.
    for line in between:
        text = join_text(line)
        caption, ratio = caption_match(text)
        if caption:
            if ratio > layout.caption_conf:
                layout.caption, layout.caption_conf = caption, ratio
            continue
        if parse_penalty_text(text):
            layout.penalty_line = _field(line)
            continue
        if layout.left_manager is None:
            _coach_row(line, cx, layout)
    # The coach row can also sit right under the table (console style: "Fabio Capello | D. Deschamps").
    if layout.left_manager is None:
        cys = sorted(median(t.cy for t in lines[li]) for li, _, _ in matches.values())
        pitch = median(b - a for a, b in zip(cys, cys[1:])) if len(cys) > 1 else 2 * label_h
        for line in lines[last_row_line + 1:]:
            if median(t.cy for t in line) - cys[-1] > 2.5 * pitch:
                break
            if _coach_row(line, cx, layout):
                break
    return layout


_UI_WORDS = ("Highlights", "Next", "Back", "OK", "Continue", "Replay")


def _coach_row(line, cx, layout):
    """A line with a name on each side of the centre and no digits: the coach row. True if it was one."""
    text = join_text(line)
    words = [t for t in line if re.search(r"[A-Za-z]{2,}", t.text)]
    if not words or re.search(r"\d", text) or any(similarity(t.text, w) >= 0.7 for t in words for w in _UI_WORDS):
        return False
    lm = [t for t in line if t.x1 <= cx + 0.2 * t.h and t.text.strip() not in _SEPARATORS]
    rm = [t for t in line if t.x0 >= cx - 0.2 * t.h and t.text.strip() not in _SEPARATORS]
    if not (lm and rm):
        return False
    layout.left_manager, layout.right_manager = _field(lm), _field(rm)
    return True


def _fill_hidden_rows(layout, cx, label_h):
    """Place rows whose label wasn't read (covered by a loading spinner or an overlay) between two rows
    that were: the rows are evenly spaced and always in the same order. Never past the first or last row
    found, so a cut-off table doesn't grow rows that aren't there. Their label Field has no text."""
    order = [row.key for row in STAT_ROWS]
    found = [(order.index(k), (f.box[1] + f.box[3]) / 2) for k, f in layout.labels.items()]
    if len(found) < 2:
        return
    n = len(found)
    mi = sum(i for i, _ in found) / n
    my = sum(y for _, y in found) / n
    var = sum((i - mi) ** 2 for i, _ in found)
    if var == 0:
        return
    pitch = sum((i - mi) * (y - my) for i, y in found) / var
    half_w = median((f.box[2] - f.box[0]) / 2 for f in layout.labels.values())
    first, last = min(i for i, _ in found), max(i for i, _ in found)
    filled = {}
    for i, key in enumerate(order):
        if key in layout.labels:
            filled[key] = layout.labels[key]
        elif first < i < last:
            cy = my + pitch * (i - mi)
            filled[key] = Field("", 0.0, (cx - half_w, cy - label_h / 2, cx + half_w, cy + label_h / 2))
    layout.labels = filled


def value_cells(layout):
    """The crop box of every stat value: key -> (left box, right box), in the coordinates of the tokens.

    Each cell spans its column around the label's row, clamped so it never reaches into the label.
    """
    if not layout.value_columns:
        return {}
    lcol, rcol, vh, vw = layout.value_columns
    half_w = max(vw / 2 + 0.35 * vh, 1.3 * vh)
    cells = {}
    for key, label in layout.labels.items():
        x0, y0, x1, y1 = label.box
        cy = (y0 + y1) / 2
        half_h = max((y1 - y0) / 2, vh / 2) + 0.12 * vh
        left = (lcol - half_w, cy - half_h, min(lcol + half_w, x0 - 0.15 * vh), cy + half_h)
        right = (max(rcol - half_w, x1 + 0.15 * vh), cy - half_h, rcol + half_w, cy + half_h)
        cells[key] = (left if left[2] - left[0] > vh * 0.5 else None, right if right[2] - right[0] > vh * 0.5 else None)
    return cells
