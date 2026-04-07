# analysis/post_processing.py
# -*- coding: utf-8 -*-

from collections import defaultdict


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


def temporal_filter(events, min_delta=None):
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
# FILTRE GOALS
# Règles :
# 1. Cooldown 200s entre deux buts
# 2. Tir précédent obligatoire dans les 12s
#    (sauf si Gemini a explicitement validé)
# 3. xG=0 + pas Gemini + danger faible → rejeté
# ─────────────────────────────────────────
def filter_goals(events, window=200.0, shot_before_goal_window=12.0):
    all_sorted     = sorted(events, key=lambda x: x.get("time", 0))
    goals_raw      = [e for e in all_sorted if e.get("type") in ["goal", "score"]]
    others         = [e for e in all_sorted if e.get("type") not in ["goal", "score"]]
    shot_times     = [e.get("time", 0) for e in all_sorted if e.get("type") == "shot"]

    validated      = []
    last_goal_time = -999

    for g in goals_raw:
        t    = g.get("time", 0)
        xg   = g.get("xg", 0) or 0
        gem  = g.get("gemini_validated", False)
        dang = g.get("danger", 0) or 0

        # ── Règle 1 : cooldown ──
        if t - last_goal_time <= window:
            print(f"  filter_goals : but à {t:.0f}s rejeté (cooldown)")
            continue

        # ── Règle 2 : tir précédent obligatoire ──
        if not gem:
            shot_before = any(
                0 < t - st <= shot_before_goal_window
                for st in shot_times
            )
            if not shot_before:
                print(f"  filter_goals : but à {t:.0f}s rejeté "
                      f"(aucun tir dans les {shot_before_goal_window:.0f}s "
                      f"précédentes)")
                continue

        # ── Règle 3 : xG + danger ──
        if xg == 0 and not gem and dang < 5:
            print(f"  filter_goals : but à {t:.0f}s rejeté "
                  f"(xG=0, non validé Gemini, danger={dang})")
            continue

        validated.append(g)
        last_goal_time = t

    if len(validated) != len(goals_raw):
        print(f"  filter_goals : {len(goals_raw)} buts bruts → "
              f"{len(validated)} retenus")

    return others + validated


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