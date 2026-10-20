"""Layout rules on tokens built from rendered screens: exact boxes and text, no OCR model needed."""

import random

import pytest

from tonits_vision import layout as L
from tonits_vision.synth import render as R

FONTS = R.available_fonts()


def tokens_from(sample, drop=(), extra=()):
    toks = [L.Token(i.text, 0.99, i.box) for i in sample.items if i.text and i.kind not in drop]
    return toks + list(extra)


def render_result(seed, **label_changes):
    rng = random.Random(seed)
    style = label_changes.pop("style", "phone")
    caption = label_changes.pop("caption", "Full Time")
    label = R.random_label(rng, style, caption)
    label.update(label_changes)
    canvas = R._render_result(rng, FONTS, label, style, caption)
    return R.Sample(b"", "image/png", label, canvas.items, style, canvas.image)


def item(sample, kind, index=0):
    return [i for i in sample.items if i.kind == kind][index]


@pytest.mark.parametrize("seed", range(40))
def test_reads_every_field_of_rendered_screens(seed):
    s = R.render(random.Random(seed), FONTS, "result")
    lay = L.parse(tokens_from(s))
    assert lay.screen == "match_result"
    assert len(lay.labels) == 13
    assert lay.caption == "Full Time"
    assert L.as_digits(lay.left_score.text) == s.label["left"]["score"]
    assert L.as_digits(lay.right_score.text) == s.label["right"]["score"]
    assert lay.left_team.text == s.label["left"]["team"]
    assert lay.right_team.text == s.label["right"]["team"]
    if s.label["managers"]:
        assert lay.left_manager.text == s.label["managers"]["left"]
        assert lay.right_manager.text == s.label["managers"]["right"]
    # every value's centre lies inside its grid cell
    cells = L.value_cells(lay)
    values = [i for i in s.items if i.kind == "value"]
    for row_i, key in enumerate(lay.labels):
        for side, box in enumerate(cells[key]):
            v = values[2 * row_i + side]
            cx, cy = (v.box[0] + v.box[2]) / 2, (v.box[1] + v.box[3]) / 2
            assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3], (key, side)


def test_team_name_ending_in_digit_keeps_its_digit():
    s = render_result(7, left={"team": "Mugz FC", "score": 2}, right={"team": "SQUAD 0", "score": 1}, penalties=None)
    lay = L.parse(tokens_from(s))
    assert (lay.left_team.text, lay.left_score.text) == ("Mugz FC", "2")
    assert (lay.right_score.text, lay.right_team.text) == ("1", "SQUAD 0")


@pytest.mark.parametrize("logo_text", ["e", "0", "6", "@", "€", "mo"])
def test_logo_read_as_text_is_ignored(logo_text):
    s = render_result(3, left={"team": "mzee mzima", "score": 3}, right={"team": "CR Flamengo", "score": 0}, penalties=None)
    logo = item(s, "logo")
    lay = L.parse(tokens_from(s, extra=[L.Token(logo_text, 0.5, logo.box)]))
    assert (lay.left_score.text, lay.right_score.text) == ("3", "0")
    assert (lay.left_team.text, lay.right_team.text) == ("mzee mzima", "CR Flamengo")


def test_scores_merged_with_the_logo_are_split():
    s = render_result(4, left={"team": "Argentina", "score": 3}, right={"team": "shinegum", "score": 3}, penalties=None)
    l, r = item(s, "score", 0), item(s, "score", 1)
    merged = L.Token("3 @ 3", 0.8, (l.box[0], min(l.box[1], r.box[1]), r.box[2], max(l.box[3], r.box[3])))
    lay = L.parse(tokens_from(s, drop=("score",), extra=[merged]))
    assert (lay.left_score.text, lay.right_score.text) == ("3", "3")
    assert (lay.left_team.text, lay.right_team.text) == ("Argentina", "shinegum")


def test_penalties_beside_the_scores():
    for seed in range(200):
        s = render_result(seed, left={"team": "Ajax", "score": 1}, right={"team": "PSV", "score": 1},
                          penalties={"left": 4, "right": 3})
        if any(i.kind == "penalty" and i.text.startswith("(") for i in s.items):
            break
    lay = L.parse(tokens_from(s))
    assert (lay.left_penalty.text, lay.right_penalty.text) == ("(4)", "(3)")
    assert (lay.left_score.text, lay.right_score.text) == ("1", "1")
    assert (lay.left_team.text, lay.right_team.text) == ("Ajax", "PSV")


def test_penalties_line():
    for seed in range(200):
        s = render_result(seed, left={"team": "Ajax", "score": 2}, right={"team": "PSV", "score": 2},
                          penalties={"left": 5, "right": 4})
        if any(i.kind == "penalty" and not i.text.startswith("(") for i in s.items):
            break
    lay = L.parse(tokens_from(s))
    assert L.parse_penalty_text(lay.penalty_line.text) == (5, 4)
    assert (lay.left_score.text, lay.right_score.text) == ("2", "2")


@pytest.mark.parametrize("text,expected", [("PK 4 - 3", (4, 3)), ("(5 PK 4)", (5, 4)), ("Penalties 3-1", (3, 1)),
                                           ("P.K. 7 - 6", (7, 6)), ("Full Time", None)])
def test_parse_penalty_text(text, expected):
    assert L.parse_penalty_text(text) == expected


def test_half_time_caption_is_not_full_time():
    s = render_result(5, caption="Half Time")
    lay = L.parse(tokens_from(s))
    assert lay.screen == "match_result" and lay.caption == "Half Time"


def test_jersey_numbers_above_the_panel_are_not_the_banner():
    s = render_result(6, left={"team": "Celtic", "score": 2}, right={"team": "Rangers", "score": 1}, penalties=None)
    cx = (item(s, "logo").box[0] + item(s, "logo").box[2]) / 2
    jerseys = [L.Token("10", 0.9, (cx - 60, 2, cx - 30, 30)), L.Token("7", 0.9, (cx + 30, 2, cx + 50, 30))]
    lay = L.parse(tokens_from(s, drop=("scene",), extra=jerseys))
    assert (lay.left_score.text, lay.right_score.text) == ("2", "1")


def test_missing_value_still_gets_a_cell():
    s = render_result(8)
    values = [i for i in s.items if i.kind == "value"]
    dropped = values[2 * 3]  # left value of the 4th row (fouls)
    toks = [t for t in tokens_from(s) if t.box != dropped.box]
    lay = L.parse(toks)
    left_cell = L.value_cells(lay)["fouls"][0]
    cx, cy = (dropped.box[0] + dropped.box[2]) / 2, (dropped.box[1] + dropped.box[3]) / 2
    assert left_cell[0] <= cx <= left_cell[2] and left_cell[1] <= cy <= left_cell[3]


def test_digit_on_a_flag_does_not_move_the_value_column():
    s = render_result(9)
    label = item(s, "label", 2)  # shots on target row
    cy = (label.box[1] + label.box[3]) / 2
    noise = L.Token("1", 0.1, (label.box[0] - 260, cy - 30, label.box[0] - 220, cy + 30))
    lay = L.parse(tokens_from(s, extra=[noise]))
    lcol = lay.value_columns[0]
    first_value = [i for i in s.items if i.kind == "value"][0]
    assert abs(lcol - (first_value.box[0] + first_value.box[2]) / 2) < 15


def test_other_screens_are_unknown():
    s = R.render(random.Random(11), FONTS, "unknown")
    assert L.parse(tokens_from(s)).screen == "unknown"
    assert L.parse([]).screen == "unknown"


def test_fuzzy_labels_with_ocr_slips():
    s = render_result(12)
    slips = {"Possession": "Possesion", "Shots on Target": "Shots on Targct", "Interceptions": "lnterceptions",
             "Successful Passes": "Successfu1 Passes"}
    toks = [L.Token(slips.get(t.text, t.text), t.conf, t.box) for t in tokens_from(s)]
    assert len(L.parse(toks).labels) == 13


@pytest.mark.parametrize("text,value", [("12", 12), ("O", 0), ("l", 1), ("64%", None), ("1 0", 10), ("abc", None), ("1234", None)])
def test_as_digits(text, value):
    assert L.as_digits(text) == value


def test_as_digits_percent():
    assert L.as_digits("64%", allow_percent=True) == 64
    assert L.as_digits("50°", allow_percent=True) == 50


@pytest.mark.parametrize("edge", ["[", "|", "]", "'"])
def test_score_box_edges_read_as_punctuation_stay_out_of_names(edge):
    s = render_result(13, left={"team": "squad inter City", "score": 2}, right={"team": "[KE] Lions", "score": 1}, penalties=None)
    l = item(s, "score", 0)
    stray = L.Token(edge, 0.4, (l.box[0] - 14, l.box[1], l.box[0] - 6, l.box[3]))
    lay = L.parse(tokens_from(s, extra=[stray]))
    assert lay.left_team.text == "squad inter City"
    assert lay.right_team.text == "[KE] Lions"


def test_coach_row_under_the_table():
    s = render_result(14, style="phone", managers=None, penalties=None)
    labels = [i for i in s.items if i.kind == "label"]
    last = labels[-1]
    pitch = (last.box[1] - labels[-2].box[1])
    cx = (item(s, "logo").box[0] + item(s, "logo").box[2]) / 2
    y0, y1 = last.box[1] + 1.2 * pitch, last.box[3] + 1.2 * pitch
    coach = [L.Token("Fabio Capello", 0.9, (cx - 170, y0, cx - 12, y1)), L.Token("|", 0.5, (cx - 3, y0, cx + 3, y1)),
             L.Token("D. Deschamps", 0.9, (cx + 12, y0, cx + 160, y1))]
    buttons = [L.Token("Highlights", 0.9, (cx - 150, y1 + 3 * pitch, cx + 10, y1 + 4 * pitch))]
    lay = L.parse(tokens_from(s, drop=("ui",), extra=coach + buttons))
    assert (lay.left_manager.text, lay.right_manager.text) == ("Fabio Capello", "D. Deschamps")


def test_buttons_under_the_table_are_not_a_coach_row():
    s = render_result(15, style="phone", managers=None, penalties=None)
    labels = [i for i in s.items if i.kind == "label"]
    cx = (item(s, "logo").box[0] + item(s, "logo").box[2]) / 2
    y0 = labels[-1].box[3] + 20
    ui = [L.Token("Highlights", 0.9, (cx - 220, y0, cx - 40, y0 + 25)), L.Token("Next", 0.9, (cx + 300, y0, cx + 380, y0 + 25))]
    assert L.parse(tokens_from(s, drop=("ui",), extra=ui)).left_manager is None


def test_rows_hidden_behind_a_spinner_are_placed_between_found_rows():
    s = render_result(16)
    labels = [i for i in s.items if i.kind == "label"]
    hidden = {i.box for i in labels[3:8]}  # Fouls .. Passes, as in the real spinner screenshot
    lay = L.parse([t for t in tokens_from(s) if t.box not in hidden])
    assert list(lay.labels) == [row.key for row in L.STAT_ROWS]
    values = [i for i in s.items if i.kind == "value"]
    cells = L.value_cells(lay)
    for row_i in range(3, 8):
        key = L.STAT_ROWS[row_i].key
        for side in (0, 1):
            v = values[2 * row_i + side]
            cx, cy = (v.box[0] + v.box[2]) / 2, (v.box[1] + v.box[3]) / 2
            box = cells[key][side]
            assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3], (key, side)


def test_rows_are_not_invented_past_the_last_one_found():
    s = render_result(17)
    labels = [i for i in s.items if i.kind == "label"]
    lay = L.parse([t for t in tokens_from(s) if t.box not in {i.box for i in labels[11:]}])
    assert "tackles" not in lay.labels and "saves" not in lay.labels
