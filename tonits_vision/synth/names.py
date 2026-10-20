"""Random team and manager names in the style players pick in eFootball."""

import random

CLUBS = [
    "Argentina", "Brazil", "France", "England", "Germany", "Spain", "Portugal", "Italy", "Netherlands",
    "Belgium", "Croatia", "Uruguay", "Colombia", "Mexico", "Japan", "Korea Republic", "Morocco", "Senegal",
    "Nigeria", "Kenya", "Ghana", "Cameroon", "Egypt", "USA", "Canada", "Wales", "Scotland", "Denmark",
    "CR Flamengo", "São Paulo", "Atlético Madrid", "FC Barcelona", "Real Madrid", "Manchester United",
    "Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Juventus", "AC Milan", "Inter",
    "AS Roma", "SSC Napoli", "Bayern München", "Borussia Dortmund", "PSG", "Olympique Lyonnais",
    "Ajax", "PSV", "Feyenoord", "Benfica", "FC Porto", "Sporting CP", "Celtic", "Rangers", "Galatasaray",
    "Fenerbahçe", "Al Nassr", "Al Hilal", "Inter Miami", "LA Galaxy", "Boca Juniors", "River Plate",
    "Gor Mahia", "AFC Leopards", "Tusker FC", "Mugz FC", "Kaizer Chiefs", "Orlando Pirates", "Al Ahly",
]

_WORDS = [
    "mzee", "mzima", "shinegum", "squad", "kings", "legends", "united", "city", "real", "athletic",
    "boyz", "warriors", "lions", "eagles", "sharks", "tigers", "nairobi", "mombasa", "kisumu", "rovers",
    "wanderers", "stars", "galaxy", "titans", "dynamo", "sporting", "inter", "olympic", "phoenix",
    "vikings", "ninjas", "ballers", "magic", "royal", "elite", "goat", "beast", "wolves", "hawks",
    "chege", "toni", "otieno", "wanjiru", "kamau", "achieng", "mwangi", "njoro", "baba", "jamaa",
]
_SUFFIXES = ["FC", "SC", "CF", "United", "City", "XI", "Utd", "AFC"]
_PREFIXES = ["FC", "CR", "AC", "Real", "Inter", "Club", "Atlético", "Sporting"]

MANAGERS = [
    "Ruben Amorim", "R. Martínez", "P. Guardiola", "C. Ancelotti", "J. Klopp", "M. Arteta", "X. Alonso",
    "L. Scaloni", "D. Deschamps", "J. Nagelsmann", "L. de la Fuente", "D. Simeone", "S. Inzaghi",
    "L. Spalletti", "E. ten Hag", "M. Pochettino", "T. Tuchel", "H. Flick", "Ange Postecoglou",
    "Unai Emery", "José Mourinho", "Zinédine Zidane", "Luis Enrique", "Dorival Júnior", "W. Regragui",
]


def _cased(rng: random.Random, word: str) -> str:
    style = rng.random()
    if style < 0.45:
        return word
    if style < 0.8:
        return word.capitalize()
    return word.upper()


def team_name(rng: random.Random) -> str:
    """A club/national team name or a made-up gamer team name, up to 20 characters."""
    roll = rng.random()
    if roll < 0.4:
        return rng.choice(CLUBS)
    words = [_cased(rng, rng.choice(_WORDS)) for _ in range(rng.choice([1, 1, 2, 2, 3]))]
    if roll < 0.55:
        words.append(rng.choice(_SUFFIXES))
    elif roll < 0.65:
        words.insert(0, rng.choice(_PREFIXES))
    elif roll < 0.78:
        # Team names that end in digits sit right beside the score: "SQUAD 0", "kings 254".
        words.append(str(rng.choice([0, 1, 7, 10, 11, 23, 99, 254, 2024])))
    elif roll < 0.82:
        words = [w + rng.choice(["_", "."]) + rng.choice(_WORDS) for w in words[:1]]
    while len(words) > 1 and len(" ".join(words)) > 20:
        words.pop(-2 if words[-1].isdigit() else -1)
    return " ".join(words)[:20]


def manager_name(rng: random.Random) -> str:
    return rng.choice(MANAGERS)
