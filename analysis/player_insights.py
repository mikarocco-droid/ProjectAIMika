# analysis/player_insights.py

from analysis.intelligence import compute_danger


def compute_player_report(events):

    players = {}

    for e in events:
        pid = e.get("player")
        if not pid:
            continue

        players.setdefault(pid, {
            "passes": 0,
            "shots": 0,
            "goals": 0,
            "xA": 0,
            "danger": 0
        })

        players[pid]["danger"] += compute_danger(e)

        if e["type"] == "pass":
            players[pid]["passes"] += 1
            players[pid]["xA"] += e.get("xA", 0)

        elif e["type"] == "shot":
            players[pid]["shots"] += 1

        elif e["type"] in ["goal", "score"]:
            players[pid]["goals"] += 1

    return players