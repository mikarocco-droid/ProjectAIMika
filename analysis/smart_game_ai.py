# analysis/smart_game_ai.py
# -*- coding: utf-8 -*-
# FIX — renommé depuis smart_game.py pour correspondre aux imports

from collections import defaultdict
import math


def compute_possession(events):
    possession = defaultdict(float)

    for i in range(1, len(events)):
        e1 = events[i-1]
        e2 = events[i]
        dt = max(0, e2.get("time", 0) - e1.get("time", 0))
        team = e1.get("team")
        if team:
            possession[team] += dt

    total = sum(possession.values()) or 1
    return {
        team: round(100 * t / total, 1)
        for team, t in possession.items()
    }


def clean_events_smart(events):
    """
    Anti-spam par type d'event.
    FIX — min_delta adapté par type pour ne pas tuer les passes.
    """
    cleaned         = []
    last_event_time = {}

    # Délais minimum par type (secondes)
    MIN_DELTA = {
        "goal":           45.0,
        "shot":            3.0,
        "interception":    2.0,
        "dribble":         1.0,
        "fast_break":      2.0,
        "progressive_run": 1.0,
        "pass":            0.2,   # passes : délai très court
        "possession":      0.1,
        "default":         1.0,
    }

    for e in events:
        etype = e.get("type", "default")
        t     = e.get("time", 0)
        delta = MIN_DELTA.get(etype, MIN_DELTA["default"])
        last  = last_event_time.get(etype, -999)

        if t - last >= delta:
            cleaned.append(e)
            last_event_time[etype] = t

    return cleaned


def cluster_events(events, eps=50):
    """Regroupe les events proches spatialement."""
    clusters = []

    for e in events:
        added = False
        for c in clusters:
            for ce in c:
                dx = e.get("x", 0) - ce.get("x", 0)
                dy = e.get("y", 0) - ce.get("y", 0)
                if math.sqrt(dx*dx + dy*dy) < eps:
                    c.append(e)
                    added = True
                    break
            if added:
                break
        if not added:
            clusters.append([e])

    return clusters