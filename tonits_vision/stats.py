"""The 13 stat rows of the eFootball Full Time screen, in screen order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StatRow:
    key: str
    labels: tuple  # every label the game prints for this row; the first is the most common
    percent: bool = False


STAT_ROWS = (
    StatRow("possession", ("Possession",), percent=True),
    StatRow("shots", ("Shots", "Total Shots")),
    StatRow("shotsOnTarget", ("Shots on Target",)),
    StatRow("fouls", ("Fouls",)),
    StatRow("offsides", ("Offsides",)),
    StatRow("cornerKicks", ("Corner Kicks",)),
    StatRow("freeKicks", ("Free Kicks",)),
    StatRow("passes", ("Passes",)),
    StatRow("successfulPasses", ("Successful Passes",)),
    StatRow("crosses", ("Crosses",)),
    StatRow("interceptions", ("Interceptions",)),
    StatRow("tackles", ("Tackles",)),
    StatRow("saves", ("Saves",)),
)

STAT_KEYS = tuple(row.key for row in STAT_ROWS)

FULL_TIME_CAPTION = "Full Time"
# Other captions the same screen layout can carry. Reading one of these means fullTime is false.
OTHER_CAPTIONS = ("Half Time", "Extra Time", "First Half", "Second Half")
