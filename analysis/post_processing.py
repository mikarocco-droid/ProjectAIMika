# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# FILTRE TEMPOREL GLOBAL
# ─────────────────────────────────────────
def temporal_filter(events, min_delta=2.0):
    filtered = []
    last_time_by_type = {}

    for e in sorted(events, key=lambda x: x.get("time", 0)):
        t = e.get("time", 0)
        etype = e.get("type")

        last_t = last_time_by_type.get(etype, -999)

        if t - last_t >= min_delta:
            filtered.append(e)
            last_time_by_type[etype] = t

    return filtered


# ─────────────────────────────────────────
# FILTRE GOALS (ANTI DOUBLON)
# ─────────────────────────────────────────
def filter_goals(events, window=5.0):
    goals = []
    last_goal_time = -999

    for e in sorted(events, key=lambda x: x.get("time", 0)):
        if e.get("type") == "goal":
            t = e.get("time", 0)

            if t - last_goal_time > window:
                goals.append(e)
                last_goal_time = t

    others = [e for e in events if e.get("type") != "goal"]
    return others + goals


# ─────────────────────────────────────────
# MERGE TRACKER IDS
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
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            centroids[pid] = (sum(xs)/len(xs), sum(ys)/len(ys))

    merged_map = {}
    used = set()

    for p1, c1 in centroids.items():
        if p1 in used:
            continue

        merged_map[p1] = p1
        used.add(p1)

        for p2, c2 in centroids.items():
            if p2 in used:
                continue

            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist = (dx*dx + dy*dy) ** 0.5

            if dist < distance_thresh:
                merged_map[p2] = p1
                used.add(p2)

    # appliquer fusion
    for e in events:
        pid = e.get("player")
        if pid in merged_map:
            e["player"] = merged_map[pid]

    return events