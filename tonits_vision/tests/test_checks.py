from tonits_vision.checks import run_checks

# Sample 2 from the design doc: Mugz FC 2 – 1 SQUAD 0.
SAMPLE2 = {
    "possession": [64, 36], "shots": [16, 1], "shotsOnTarget": [12, 1], "fouls": [0, 0], "offsides": [0, 0],
    "cornerKicks": [3, 0], "freeKicks": [0, 0], "passes": [156, 96], "successfulPasses": [122, 66],
    "crosses": [4, 0], "interceptions": [28, 24], "tackles": [5, 5], "saves": [0, 7],
}
# Sample 3 (the real screenshot): mzee mzima 3 – 0 CR Flamengo.
SAMPLE3 = {
    "possession": [50, 50], "shots": [4, 9], "shotsOnTarget": [3, 6], "fouls": [1, 3], "offsides": [1, 3],
    "cornerKicks": [0, 3], "freeKicks": [3, 1], "passes": [90, 83], "successfulPasses": [60, 57],
    "crosses": [1, 0], "interceptions": [16, 26], "tackles": [4, 4], "saves": [3, 0],
}


# Sample 4 (real, console style): FC Barcelona 4 – 1 Emirhann CHAMPION.
SAMPLE4 = {
    "possession": [55, 45], "shots": [14, 5], "shotsOnTarget": [10, 3], "fouls": [3, 0], "offsides": [0, 0],
    "cornerKicks": [2, 1], "freeKicks": [0, 3], "passes": [130, 157], "successfulPasses": [106, 120],
    "crosses": [3, 0], "interceptions": [30, 19], "tackles": [5, 3], "saves": [3, 7],
}


def test_a_genuine_screenshot_breaks_the_saves_rule():
    # 4 goals + 7 saves > 10 on target, 1 + 3 > 3: eFootball doesn't count saves as the rule assumes, so
    # the reader reports this check but doesn't lower confidence for it (reader.INFORMATIONAL_CHECKS).
    checks = run_checks(4, 1, SAMPLE4)
    assert checks["goalsPlusSavesWithinShotsOnTarget"] is False
    assert all(v for k, v in checks.items() if k != "goalsPlusSavesWithinShotsOnTarget")


def test_genuine_samples_pass_every_check():
    assert all(run_checks(2, 1, SAMPLE2).values())
    assert all(run_checks(3, 0, SAMPLE3).values())


def test_edited_score_breaks_a_check():
    # The common forgery: change only the score. 3-0 → 3-4 needs 4 right goals + 3 left saves = 7 > 6 on target.
    assert run_checks(3, 4, SAMPLE3)["goalsPlusSavesWithinShotsOnTarget"] is False
    assert run_checks(13, 1, SAMPLE2)["goalsWithinShotsOnTarget"] is False


def test_some_edited_scores_still_pass():
    # The checks are a flag, not proof: swapping 3-0 to 0-3 fits these stats (0+0 ≤ 3, 3+3 ≤ 6).
    assert all(run_checks(0, 3, SAMPLE3).values())


def test_missing_rows_make_checks_unknown_not_failed():
    stats = {k: v for k, v in SAMPLE2.items() if k not in ("saves", "possession")}
    checks = run_checks(2, 1, stats)
    assert checks["goalsPlusSavesWithinShotsOnTarget"] is None
    assert checks["possessionSumsTo100"] is None
    assert checks["goalsWithinShotsOnTarget"] is True
    assert run_checks(None, 1, SAMPLE2)["goalsWithinShotsOnTarget"] is None


def test_possession_rounding_tolerance():
    assert run_checks(0, 0, {"possession": [51, 50]})["possessionSumsTo100"] is True
    assert run_checks(0, 0, {"possession": [60, 50]})["possessionSumsTo100"] is False


def test_other_rules():
    bad = dict(SAMPLE2, successfulPasses=[200, 66], shotsOnTarget=[12, 3])
    checks = run_checks(2, 1, bad)
    assert checks["successfulWithinPasses"] is False
    assert checks["shotsOnTargetWithinShots"] is False


def test_settle_accents():
    from tonits_vision.reader import settle_accents
    assert settle_accents("Atlético Madrid", 0.9, "Atletico Madrid", 0.8) == ("Atlético Madrid", 0.9)
    assert settle_accents("FC ñarcelona", 0.69, "FC Barcelona", 0.58) == ("FC Barcelona", 0.58)
    assert settle_accents("ēmirhann CHAMPION", 0.26, "Emirhann CHAMPION", 0.29) == ("Emirhann CHAMPION", 0.26)
    assert settle_accents("Mugz FC", 0.9, "Mugz FC", 0.9) == ("Mugz FC", 0.9)
