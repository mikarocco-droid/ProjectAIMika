# analysis/post_processing.py
# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# DÉLAIS MINIMUM PAR TYPE D'EVENT
# Basés sur la réalité du jeu :
# - goal      : 150s  — après un but, remise en jeu + célébration
# - shot      :   3s  — on ne peut pas tirer 2x en 3s
# - dribble   :   1s  — possible d'enchaîner mais pas trop vite
# - interception: 2s  — une interception = action ponctuelle
# - fast_break:   3s  — une contre-attaque dure plusieurs secondes
# - progressive_run: 1s — course vers l'avant
# - pass      :   0.3s — les passes s'enchaînent rapidement
# - possession:   0.1s — très fréquent, délai minimal
# - long_pass :   1s  — moins fréquent qu'une passe courte
# - under_pressure: 0.5s
# - build_up  :   2s
# ─────────────────────────────────────────
MIN_DELTA = {
    "goal":            150.0,
    "score":           150.0,
    "shot":              3.0,
    "dribble":           1.0,
    "interception":      2.0,
    "fast_break":        3.0,
    "progressive_run":   1.0,
    "pass":              0.3,
    "long_pass":         1.0,
    "possession":        0.1,
    "under_pressure":    0.5,
    "build_up":          2.0,
    "build_up_play":     2.0,
    "default":           1.0,
}


# ─────────────────────────────────────────
# FILTRE TEMPOREL PAR TYPE
# ─────────────────────────────────────────
def temporal_filter(events, min_delta=None):
    """
    Filtre les events trop rapprochés dans le temps.
    min_delta ignoré — on utilise MIN_DELTA par type.
    """
    filtered          = []
    last_time_by_type = {}

    for e in sorted(events, key=lambda x: x.get("time", 0)):
        t     = e.get("time", 0)
        etype = e.get("type", "default")
        delta = MIN_DELTA.get(etype, MIN_DELTA["default"])
        last  = last_time_by_type.get(etype, -999)

        if t - last >= delta:
            filtered.append(e)
            last_time_by_type[etype] = t

    return filtered


# ─────────────────────────────────────────
# FILTRE GOALS ANTI-DOUBLON
# FIX — fenêtre large : après un but, 150s de cooldown
#        (célébration + remise en jeu + reprise)
# ─────────────────────────────────────────
def filter_goals(events, window=150.0):
    goals          = []
    last_goal_time = -999

    for e in sorted(events, key=lambda x: x.get("time", 0)):
        if e.get("type") in ["goal", "score"]:
            t = e.get("time", 0)
            if t - last_goal_time > window:
                goals.append(e)
                last_goal_time = t

    others = [e for e in events if e.get("type") not in ["goal", "score"]]
    return others + goals


# ─────────────────────────────────────────
# MERGE TRACKER IDS
# Réduit les doublons de joueurs (76 → ~22)
# ─────────────────────────────────────────
def merge_players(events, distance_thresh=80):
    player_positions = defaultdict(list)

    for e in events:
        pid = e.get("player")
        if pid and "x" in e and "y" in e:
            player_positions[pid].append((e["x"], e["y"]))

    centroids = {}
    for pid, pts in player_positions.items():
        if pts:
            centroids[pid] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts)
            )

    merged_map = {}
    used       = set()

    for p1, c1 in centroids.items():
        if p1 in used:
            continue
        merged_map[p1] = p1
        used.add(p1)

        for p2, c2 in centroids.items():
            if p2 in used:
                continue
            dx   = c1[0] - c2[0]
            dy   = c1[1] - c2[1]
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < distance_thresh:
                merged_map[p2] = p1
                used.add(p2)

    for e in events:
        pid = e.get("player")
        if pid in merged_map:
            e["player"] = merged_map[pid]

    return events