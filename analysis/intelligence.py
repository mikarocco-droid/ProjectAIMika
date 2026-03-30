# analysis/intelligence.py
# -*- coding: utf-8 -*-

import math

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


# ─────────────────────────────────────────
# xA (EXPECTED ASSIST)
# ─────────────────────────────────────────
def compute_xa(event, next_event):
    if event["type"] != "pass":
        return None

    if next_event and next_event["type"] == "shot":
        return round(0.3, 2)  # base simple (upgrade possible)

    return 0.05


# ─────────────────────────────────────────
# PROGRESSION
# ─────────────────────────────────────────
def is_progressive(prev_x, new_x, frame_w):
    return (new_x - prev_x) > frame_w * 0.15


# ─────────────────────────────────────────
# DANGEROUS ACTION SCORE
# ─────────────────────────────────────────
def compute_danger(event):
    score = 0

    if event["type"] == "shot":
        score += 5
    if event["type"] == "goal":
        score += 10
    if event["type"] == "assist":
        score += 6
    if event["type"] == "progressive_run":
        score += 2
    if event["type"] == "counter_attack":
        score += 4

    return score


# ─────────────────────────────────────────
# BUILD-UP PLAY
# ─────────────────────────────────────────
def detect_build_up(sequence, frame_w):
    if len(sequence) < 5:
        return False

    start_x = sequence[0][0]
    end_x   = sequence[-1][0]

    return (end_x - start_x) > frame_w * 0.4


# ─────────────────────────────────────────
# TEAM DOMINANCE
# ─────────────────────────────────────────
def compute_team_dominance(events):
    team_stats = {}

    for e in events:
        t = e.get("team")
        if t is None:
            continue

        team_stats.setdefault(t, 0)
        team_stats[t] += compute_danger(e)

    return team_stats