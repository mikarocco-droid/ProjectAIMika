# analytics/stats.py
# -*- coding: utf-8 -*-

from collections import defaultdict


def compute_stats(events):
    """
    Calcule les stats par joueur depuis la liste des events.
    Tous les player_id sont convertis en string pour éviter
    les erreurs de type.
    """
    players = defaultdict(lambda: {
        "touches":       0,
        "passes":        0,
        "passes_reussies": 0,
        "tirs":          0,
        "buts":          0,
        "interceptions": 0,
        "dribbles":      0,
        "long_passes":   0,
        "xg_total":      0.0
    })

    for e in events:
        event_type = e.get("type")

        # Normaliser player_id en string
        pid = str(e.get("player", e.get("from", "")))
        if not pid:
            continue

        if event_type == "possession":
            players[pid]["touches"] += 1

        elif event_type == "pass":
            from_pid = str(e.get("from", ""))
            if from_pid:
                players[from_pid]["passes"]          += 1
                players[from_pid]["passes_reussies"] += 1

        elif event_type == "shot":
            players[pid]["tirs"]     += 1
            players[pid]["xg_total"] += round(e.get("xg", 0.1), 3)

        elif event_type == "score":
            players[pid]["buts"] += 1

        elif event_type == "interception":
            players[pid]["interceptions"] += 1

        elif event_type == "dribble":
            players[pid]["dribbles"] += 1

        elif event_type == "long_pass":
            players[pid]["long_passes"] += 1

    # Filtrer les joueurs avec moins de 3 touches
    # (évite les faux positifs)
    filtered = {
        pid: stats
        for pid, stats in players.items()
        if stats["touches"] >= 3
    }

    return filtered