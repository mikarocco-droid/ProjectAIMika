# analysis/post_processing.py
# -*- coding: utf-8 -*-

from collections import defaultdict


MIN_DELTA = {
    "goal":            150.0,
    "score":           150.0,
    "shot":              3.0,
    "dribble":           1.5,
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
# INFÉRENCE SHOT DEPUIS BUT
# ─────────────────────────────────────────
def infer_shots_from_goals(events):
    """
    Pour chaque but sans tir précédent dans les 15s,
    crée un shot synthétique depuis le dribble précédent.
    """
    all_sorted = sorted(events, key=lambda x: x.get("time", 0))
    shot_times = {e.get("time", 0) for e in all_sorted if e.get("type") == "shot"}
    injected   = []

    for g in all_sorted:
        if g.get("type") not in ["goal", "score"]:
            continue
        t = g.get("time", 0)

        has_shot = any(0 < t - st <= 15.0 for st in shot_times)
        if has_shot:
            continue

        candidates = [
            e for e in all_sorted
            if e.get("type") == "dribble"
            and 0 < t - e.get("time", 0) <= 5.0
        ]
        if not candidates:
            continue

        best = min(candidates, key=lambda e: t - e.get("time", 0))

        shot = {
            "type":     "shot",
            "time":     best.get("time", t - 1.0),
            "frame":    best.get("frame", g.get("frame", 0)),
            "player":   best.get("player", g.get("player")),
            "team":     best.get("team",   g.get("team")),
            "x":        best.get("x",      g.get("x", 0)),
            "y":        best.get("y",      g.get("y", 0)),
            "xg":       0.35,
            "danger":   6.0,
            "inferred": True,
        }
        injected.append(shot)
        shot_times.add(shot["time"])
        print(f"  infer_shots : shot synthétique injecté à "
              f"t={shot['time']:.1f}s avant but à {t:.1f}s")

    return events + injected


# ─────────────────────────────────────────
# FILTRE GOALS
# ─────────────────────────────────────────
def filter_goals(
    events,
    window                 = 200.0,
    shot_before_goal_window = 15.0,
    frame_w                = 1920,    # FIX position
):
    """
    Filtre les faux buts :
    1. Cooldown 200s entre deux buts
    2. Tir précédent obligatoire dans les 15s
       Exception : Gemini valide ET danger >= 8
    3. xG=0 + pas Gemini + danger faible → rejeté
    4. FIX position : but trop loin des cages (dégagement de tête, etc.)
    """
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
        x    = g.get("x", 0) or 0

        # ── Règle 1 : cooldown ──
        if t - last_goal_time <= window:
            print(f"  filter_goals : but à {t:.0f}s rejeté (cooldown)")
            continue

        # ── Règle 2 : tir précédent ──
        shot_before   = any(0 < t - st <= shot_before_goal_window for st in shot_times)
        gemini_strong = gem and dang >= 8.0

        if not shot_before and not gemini_strong:
            print(f"  filter_goals : but à {t:.0f}s rejeté "
                  f"(pas de tir, gemini={gem}, danger={dang:.1f})")
            continue

        # ── Règle 3 : xG + danger ──
        if xg == 0 and not gem and dang < 5:
            print(f"  filter_goals : but à {t:.0f}s rejeté "
                  f"(xG=0, non validé, danger={dang:.1f})")
            continue

        # ── Règle 4 : FIX position ──
        # Un but doit être dans la zone des cages (x < 18% ou x > 82%)
        # Exception : Gemini a validé avec confiance élevée
        x_pct       = x / frame_w if frame_w > 0 else 0.5
        near_goal   = x_pct < 0.18 or x_pct > 0.82
        gemini_conf = g.get("gemini_conf", 0) or 0

        if not near_goal and not (gem and gemini_conf >= 0.9):
            print(f"  filter_goals : but à {t:.0f}s rejeté "
                  f"(position trop loin des cages : x={x:.0f} "
                  f"soit {x_pct*100:.1f}% du terrain)")
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