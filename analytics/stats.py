# ─────────────────────────────────────────
# STATS V6
# ─────────────────────────────────────────

from collections import defaultdict

def compute_stats(events):

    players = defaultdict(lambda: {
        "touches": 0,
        "passes": 0,
        "passes_reussies": 0,
        "tirs": 0,
        "buts": 0,
        "interceptions": 0,
        "dribbles": 0
    })

    for e in events:

        if e["type"] == "possession":
            players[e["player"]]["touches"] += 1

        elif e["type"] == "pass":
            players[e["from"]]["passes"] += 1
            players[e["from"]]["passes_reussies"] += 1

        elif e["type"] == "shot":
            players[e["player"]]["tirs"] += 1

        elif e["type"] == "score":
            players[e["player"]]["buts"] += 1

        elif e["type"] == "interception":
            players[e["player"]]["interceptions"] += 1

        elif e["type"] == "dribble":
            players[e["player"]]["dribbles"] += 1

    return dict(players)