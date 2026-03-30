# analysis/intelligence.py
# -*- coding: utf-8 -*-

import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ─────────────────────────────────────────
# xA
# ─────────────────────────────────────────
def compute_xa(pass_event, shot_event):
    if pass_event.get("type") != "pass":
        return 0.0

    if shot_event and shot_event.get("type") == "shot":
        return round(shot_event.get("xg", 0.1) * 0.7, 3)

    return 0.05


# ─────────────────────────────────────────
# PROGRESSIVE RUN
# ─────────────────────────────────────────
def is_progressive(prev_x, new_x, frame_w):
    return (new_x - prev_x) > frame_w * 0.12


# ─────────────────────────────────────────
# DANGER SCORE (🔥 CORE IA)
# ─────────────────────────────────────────
def compute_danger(event):
    t = event.get("type")

    if t in ["goal", "score"]:
        return 10

    if t == "shot":
        return 5 + event.get("xg", 0.1) * 5

    if t == "assist":
        return 7

    if t == "key_pass":
        return 5

    if t == "progressive_run":
        return 2

    if t == "interception":
        return 3

    if t == "counter_attack":
        return 6

    return 1


# ─────────────────────────────────────────
# BUILD-UP PLAY
# ─────────────────────────────────────────
def detect_build_up(sequence, frame_w):
    if len(sequence) < 5:
        return False

    start_x = sequence[0][0]
    end_x   = sequence[-1][0]

    return (end_x - start_x) > frame_w * 0.35


# ─────────────────────────────────────────
# DOMINANCE
# ─────────────────────────────────────────
def compute_team_dominance(events):
    stats = {}

    for e in events:
        t = e.get("team")
        if t is None:
            continue

        stats.setdefault(t, 0)
        stats[t] += compute_danger(e)

    return stats