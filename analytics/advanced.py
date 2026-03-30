# analytics/advanced.py
# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# SCORE D'ACTION (pour highlights & IA)
# ─────────────────────────────────────────
def compute_action_score(e):
    t = e.get("type")

    if t == "goal":
        return 10

    if t == "shot":
        return 5 + e.get("xg", 0) * 5

    if t == "progressive_pass":
        return 4

    if t == "progressive_run":
        return 3

    if t == "key_pass":
        return 6

    if t == "pass":
        return 2

    if t == "dribble":
        return 2

    if t == "interception":
        return 3

    return 1


# ─────────────────────────────────────────
# PASS NETWORK
# ─────────────────────────────────────────
def build_pass_network(events):
    net = defaultdict(int)

    for e in events:
        if e.get("type") == "pass":
            p1 = str(e.get("from"))
            p2 = str(e.get("to"))

            if p1 and p2:
                net[(p1, p2)] += 1

    return dict(net)


# ─────────────────────────────────────────
# xA (EXPECTED ASSIST)
# ─────────────────────────────────────────
def compute_xa(events):
    """
    Associe xA = xG du tir suivant (dans une fenêtre courte)
    """
    for i in range(len(events) - 1):
        e = events[i]
        nxt = events[i + 1]

        if e.get("type") == "pass" and nxt.get("type") == "shot":
            e["xA"] = nxt.get("xg", 0)

    return events


# ─────────────────────────────────────────
# CHAÎNES DE PASSES (BUILD-UP)
# ─────────────────────────────────────────
def extract_pass_sequences(events, min_length=3):
    sequences = []
    current = []

    for e in events:
        if e.get("type") == "pass":
            current.append(e)
        else:
            if len(current) >= min_length:
                sequences.append(current)
            current = []

    if len(current) >= min_length:
        sequences.append(current)

    return sequences


# ─────────────────────────────────────────
# OFFSIDE (simple heuristic)
# ─────────────────────────────────────────
def detect_offside(events, frame_w=1280):
    offsides = []

    for e in events:
        if e.get("type") == "pass":
            if e.get("x", 0) > frame_w * 0.9:
                offsides.append(e)

    return offsides


# ─────────────────────────────────────────
# TEAM DOMINANCE
# ─────────────────────────────────────────
def compute_team_dominance(events):
    teams = defaultdict(lambda: {
        "passes": 0,
        "shots": 0,
        "possession": 0
    })

    for e in events:
        team = e.get("team")
        if team is None:
            continue

        if e["type"] == "pass":
            teams[team]["passes"] += 1

        if e["type"] == "shot":
            teams[team]["shots"] += 1

        if e["type"] == "possession":
            teams[team]["possession"] += 1

    return dict(teams)