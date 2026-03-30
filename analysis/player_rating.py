# analysis/player_rating.py
# -*- coding: utf-8 -*-

from collections import defaultdict
from analysis.intelligence import compute_danger


# ─────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────
def normalize(value, max_value):
    if max_value == 0:
        return 0
    return value / max_value


# ─────────────────────────────────────────
# SCORE PAR JOUEUR
# ─────────────────────────────────────────
def compute_raw_player_stats(events):

    players = defaultdict(lambda: {
        "passes": 0,
        "shots": 0,
        "goals": 0,
        "assists": 0,
        "xg": 0.0,
        "xa": 0.0,
        "dribbles": 0,
        "interceptions": 0,
        "touches": 0,
        "danger": 0
    })

    for e in events:
        pid = str(e.get("player"))
        if not pid:
            continue

        players[pid]["danger"] += compute_danger(e)

        if e["type"] == "pass":
            players[pid]["passes"] += 1
            players[pid]["xa"] += e.get("xA", 0)

        elif e["type"] == "shot":
            players[pid]["shots"] += 1
            players[pid]["xg"] += e.get("xg", 0)

        elif e["type"] in ["goal", "score"]:
            players[pid]["goals"] += 1

        elif e["type"] == "assist":
            players[pid]["assists"] += 1

        elif e["type"] == "dribble":
            players[pid]["dribbles"] += 1

        elif e["type"] == "interception":
            players[pid]["interceptions"] += 1

        elif e["type"] == "possession":
            players[pid]["touches"] += 1

    return players


# ─────────────────────────────────────────
# CALCUL NOTE /10
# ─────────────────────────────────────────
def compute_player_ratings(events):

    stats = compute_raw_player_stats(events)

    # max values pour normalisation
    max_vals = {
        k: max((p[k] for p in stats.values()), default=1)
        for k in ["passes", "shots", "goals", "xg", "xa", "dribbles", "interceptions", "danger"]
    }

    ratings = {}

    for pid, s in stats.items():

        score = 0

        # pondérations (🔥 IMPORTANT)
        score += normalize(s["goals"], max_vals["goals"]) * 3
        score += normalize(s["xg"], max_vals["xg"]) * 1.5
        score += normalize(s["xa"], max_vals["xa"]) * 1.5
        score += normalize(s["passes"], max_vals["passes"]) * 1
        score += normalize(s["dribbles"], max_vals["dribbles"]) * 1
        score += normalize(s["interceptions"], max_vals["interceptions"]) * 1
        score += normalize(s["danger"], max_vals["danger"]) * 2

        # base rating
        rating = 4 + score  # base 4/10

        rating = min(10, round(rating, 2))

        ratings[pid] = {
            "rating": rating,
            "stats": s
        }

    # tri décroissant
    ratings = dict(sorted(ratings.items(), key=lambda x: x[1]["rating"], reverse=True))

    return ratings


# ─────────────────────────────────────────
# MVP
# ─────────────────────────────────────────
def get_mvp(ratings):
    if not ratings:
        return None

    return next(iter(ratings.items()))  # meilleur joueur