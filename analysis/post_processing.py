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
    crée un shot synthétique.
    Priorité :
      1. Dribble dans les 5s avant le but
      2. shot_blocked dans les 10s avant (tir contré → but)
      3. N'importe quel event significatif dans les 3s avant
      4. Fallback direct depuis la position du but
    """
    all_sorted = sorted(events, key=lambda x: x.get("time", 0))
    shot_times = {e.get("time", 0) for e in all_sorted if e.get("type") == "shot"}
    injected   = []

    for g in all_sorted:
        if g.get("type") not in ["goal", "score"]:
            continue
        t = g.get("time", 0)

        # Déjà un tir dans les 15s avant → pas besoin d'inférer
        has_shot = any(0 < t - st <= 15.0 for st in shot_times)
        if has_shot:
            continue

        shot = None

        # ── Priorité 1 : dribble dans les 5s ──
        dribbles = [
            e for e in all_sorted
            if e.get("type") == "dribble"
            and 0 < t - e.get("time", 0) <= 5.0
        ]
        if dribbles:
            best = min(dribbles, key=lambda e: t - e.get("time", 0))
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
            print(f"  infer_shots : shot depuis dribble à "
                  f"t={shot['time']:.1f}s avant but à {t:.1f}s")

        # ── Priorité 2 : shot_blocked dans les 10s ──
        # FIX — tir contré suivi d'un but = séquence réelle
        if shot is None:
            blocked = [
                e for e in all_sorted
                if e.get("type") == "shot_blocked"
                and 0 < t - e.get("time", 0) <= 10.0
            ]
            if blocked:
                best = min(blocked, key=lambda e: t - e.get("time", 0))
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
                    "from_blocked": True,
                }
                print(f"  infer_shots : shot depuis shot_blocked à "
                      f"t={shot['time']:.1f}s avant but à {t:.1f}s")

        # ── Priorité 3 : n'importe quel event dans les 3s ──
        if shot is None:
            near = [
                e for e in all_sorted
                if e.get("type") not in ["goal", "score", "possession",
                                          "under_pressure", "build_up"]
                and 0 < t - e.get("time", 0) <= 3.0
            ]
            if near:
                best = min(near, key=lambda e: t - e.get("time", 0))
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
                print(f"  infer_shots : shot depuis event '{best.get('type')}' à "
                      f"t={shot['time']:.1f}s avant but à {t:.1f}s")

        # ── Priorité 4 : fallback direct depuis le but ──
        if shot is None:
            shot = {
                "type":     "shot",
                "time":     max(0.0, t - 1.0),
                "frame":    g.get("frame", 0),
                "player":   g.get("player"),
                "team":     g.get("team"),
                "x":        g.get("x", 0),
                "y":        g.get("y", 0),
                "xg":       0.35,
                "danger":   6.0,
                "inferred": True,
            }
            print(f"  infer_shots : shot fallback direct à "
                  f"t={shot['time']:.1f}s depuis but à {t:.1f}s")

        injected.append(shot)
        shot_times.add(shot["time"])

    return events + injected


# ─────────────────────────────────────────
# FILTRE GOALS
# ─────────────────────────────────────────
def filter_goals(
    events,
    window                  = 200.0,
    shot_before_goal_window = 15.0,
    frame_w                 = 1920,
):
    """
    Filtre les faux buts :
    1. Cooldown 200s entre deux buts
    2. Tir précédent obligatoire dans les 15s
       Exception : Gemini valide ET danger >= 8
    3. xG=0 + pas Gemini + danger faible → rejeté
    4. Position trop loin des cages → rejeté TOUJOURS
       FIX — Gemini ne peut plus bypasser cette règle
       car les frames CPU sont moins nettes et Gemini
       peut valider des faux positifs visuels
       Seuil élargi à 20% pour ne pas rater les vrais buts
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

        # ── Règle 4 : position STRICTE ──
        # FIX — seuil paramétrable selon type vidéo
        # match complet : 20% | résumé : 35%
        x_pct     = x / frame_w if frame_w > 0 else 0.5
        near_goal = x_pct < position_threshold or x_pct > (1.0 - position_threshold)

        if not near_goal:
            print(f"  filter_goals : but à {t:.0f}s rejeté "
                  f"(position trop loin des cages : x={x:.0f} "
                  f"soit {x_pct*100:.1f}% du terrain, "
                  f"gemini={gem})")
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