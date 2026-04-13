# analysis/post_processing.py
# -*- coding: utf-8 -*-

from goal_posthoc import detect_fast_goals_from_ball
from utils_goals import deduplicate_goals


def infer_shots_from_goals(events):
    shots = []

    for e in events:
        if e.get("type") == "goal":
            shots.append({
                "type": "shot",
                "time": e["time"] - 1.0,
                "xg": 0.3,
                "synthetic": True
            })

    return events + shots


def filter_goals(events):
    clean = []

    for e in events:
        if e.get("type") != "goal":
            clean.append(e)
            continue

        # filtre simple
        if e.get("confidence", 0) < 0.5:
            continue

        clean.append(e)

    return clean


def post_process(events, frames_data):
    # 1. posthoc
    events = detect_fast_goals_from_ball(frames_data, events)

    # 2. shots synthétiques AVANT filtre
    events = infer_shots_from_goals(events)

    # 3. filtrage
    events = filter_goals(events)

    # 4. déduplication
    events = deduplicate_goals(events)

    return events