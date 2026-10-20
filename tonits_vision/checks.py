"""Plausibility rules over a reading. Each is True, False, or None when a needed number wasn't read.

A failed check is a flag for staff, never a rejection on its own: an own goal is a goal without a shot.
"""


def _get(stats, key, side):
    pair = stats.get(key) if stats else None
    return None if pair is None else pair[side]


def _all(values):
    """True if every value is True, False if any is False, otherwise None."""
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _le(a, b):
    return None if a is None or b is None else a <= b


def run_checks(left_score, right_score, stats):
    goals = (left_score, right_score)
    sot = [_get(stats, "shotsOnTarget", s) for s in (0, 1)]
    shots = [_get(stats, "shots", s) for s in (0, 1)]
    saves = [_get(stats, "saves", s) for s in (0, 1)]
    passes = [_get(stats, "passes", s) for s in (0, 1)]
    successful = [_get(stats, "successfulPasses", s) for s in (0, 1)]
    possession = stats.get("possession") if stats else None

    def plus(a, b):
        return None if a is None or b is None else a + b

    return {
        # Shown as whole percentages, so allow rounding to 99 or 101.
        "possessionSumsTo100": None if possession is None else 99 <= possession[0] + possession[1] <= 101,
        "goalsWithinShotsOnTarget": _all([_le(goals[s], sot[s]) for s in (0, 1)]),
        # A save stops a shot on target, so a side's goals plus the opponent's saves fit in its shots on target.
        "goalsPlusSavesWithinShotsOnTarget": _all([_le(plus(goals[s], saves[1 - s]), sot[s]) for s in (0, 1)]),
        "successfulWithinPasses": _all([_le(successful[s], passes[s]) for s in (0, 1)]),
        "shotsOnTargetWithinShots": _all([_le(sot[s], shots[s]) for s in (0, 1)]),
    }
