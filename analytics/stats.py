# analytics/stats.py
# -*- coding: utf-8 -*-

from collections import defaultdict


def compute_stats(events):

    players = defaultdict(lambda: {
        "touches": 0,
        "passes": 0,
        "passes_reussies": 0,
        "progressive_passes": 0,
        "key_passes": 0,
        "tirs": 0,
        "buts": 0,
        "interceptions": 0,
        "dribbles": 0,
        "long_passes": 0,
        "progressive_runs": 0,
        "xg_total": 0.0,
        "xa_total": 0.0,
        "danger_total": 0.0
    })

    for e in events:
        t = e.get("type")

        pid = str(e.get("player", e.get("from", "")))
        if not pid:
            continue

        # ── POSSESSION
        if t == "possession":
            players[pid]["touches"] += 1

        # ── PASS
        elif t == "pass":
            from_pid = str(e.get("from", ""))
            if from_pid:
                players[from_pid]["passes"] += 1
                players[from_pid]["passes_reussies"] += 1
                players[from_pid]["xa_total"] += e.get("xA", 0)

        # ── PROGRESSIVE PASS
        elif t == "progressive_pass":
            players[pid]["progressive_passes"] += 1

        # ── KEY PASS
        elif t == "key_pass":
            players[pid]["key_passes"] += 1

        # ── SHOT
        elif t == "shot":
            players[pid]["tirs"] += 1
            players[pid]["xg_total"] += e.get("xg", 0)
            players[pid]["danger_total"] += e.get("danger", 0)

        # ── GOAL
        elif t in ["goal", "score"]:
            players[pid]["buts"] += 1

        # ── DEFENSE
        elif t == "interception":
            players[pid]["interceptions"] += 1

        # ── DRIBBLE
        elif t == "dribble":
            players[pid]["dribbles"] += 1

        # ── LONG PASS
        elif t == "long_pass":
            players[pid]["long_passes"] += 1

        # ── RUN
        elif t == "progressive_run":
            players[pid]["progressive_runs"] += 1

    # 🔥 FILTRE INTELLIGENT
    return {
        pid: stats
        for pid, stats in players.items()
        if stats["touches"] >= 5
    }