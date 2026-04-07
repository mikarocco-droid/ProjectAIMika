from collections import defaultdict
import math


# ─────────────────────────────────────────
# POSSESSION RÉELLE
# ─────────────────────────────────────────
def compute_possession(events):
    possession = defaultdict(float)

    for i in range(1, len(events)):
        e1 = events[i-1]
        e2 = events[i]

        t1 = e1.get("time", 0)
        t2 = e2.get("time", 0)

        dt = max(0, t2 - t1)

        team = e1.get("team")
        if team:
            possession[team] += dt

    total = sum(possession.values()) or 1

    return {
        team: round(100 * t / total, 1)
        for team, t in possession.items()
    }


# ─────────────────────────────────────────
# VALIDATION EVENTS INTELLIGENTE
# ─────────────────────────────────────────
def clean_events_smart(events):
    cleaned = []
    last_event_time = {}

    for e in events:
        etype = e.get("type")
        t = e.get("time", 0)

        # anti spam global
        if etype in last_event_time:
            if t - last_event_time[etype] < 1.5:
                continue

        last_event_time[etype] = t
        cleaned.append(e)

    return cleaned


# ─────────────────────────────────────────
# CLUSTER ACTIONS (STYLE DBSCAN)
# ─────────────────────────────────────────
def cluster_events(events, eps=50):
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