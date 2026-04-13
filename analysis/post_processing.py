# analysis/post_processing.py
# -*- coding: utf-8 -*-

"""
POST PROCESSING — VERSION PRO (centralisée)

Pipeline interne :
    1. Deduplicate goals (confidence-based)
    2. Merge players (ID proches)
    3. Temporal filter (anti spam events)
    4. Infer shots from goals (si absent)
    5. Goal cooldown (anti doublons longs)

⚠️ IMPORTANT :
Ce module remplace :
    - temporal_filter
    - filter_goals
    - merge_players
    - infer_shots_from_goals
"""

# ─────────────────────────────────────────────────────────────
# DÉDUPLICATION BUTS (confidence-aware)
# ─────────────────────────────────────────────────────────────

def deduplicate_goals(events, window=3.0):
    goals = sorted(
        [e for e in events if e.get("type") in ("goal", "score")],
        key=lambda x: x.get("time", 0)
    )
    others = [e for e in events if e.get("type") not in ("goal", "score")]

    kept = []
    for g in goals:
        if kept and abs(g["time"] - kept[-1]["time"]) < window:
            if g.get("confidence", 0) > kept[-1].get("confidence", 0):
                kept[-1] = g
        else:
            kept.append(g)

    return sorted(others + kept, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────────────────────────
# MERGE PLAYERS
# ─────────────────────────────────────────────────────────────

def merge_players(events, time_window=2.0):
    """
    Fusionne événements proches avec même joueur approximatif
    """
    merged = []
    for e in events:
        if not merged:
            merged.append(e)
            continue

        prev = merged[-1]

        if (
            e.get("player") == prev.get("player")
            and abs(e.get("time", 0) - prev.get("time", 0)) < time_window
            and e.get("type") == prev.get("type")
        ):
            # garder le plus confiant
            if e.get("confidence", 0) > prev.get("confidence", 0):
                merged[-1] = e
        else:
            merged.append(e)

    return merged


# ─────────────────────────────────────────────────────────────
# FILTRE TEMPOREL
# ─────────────────────────────────────────────────────────────

def temporal_filter(events, min_delta=2.0):
    """
    Supprime événements trop rapprochés (bruit)
    """
    filtered = []
    last_time = {}

    for e in events:
        t = e.get("time", 0)
        etype = e.get("type")

        if etype not in last_time or t - last_time[etype] >= min_delta:
            filtered.append(e)
            last_time[etype] = t

    return filtered


# ─────────────────────────────────────────────────────────────
# INFÉRER SHOTS DEPUIS GOALS
# ─────────────────────────────────────────────────────────────

def infer_shots_from_goals(events):
    """
    Si un but n’a pas de tir associé → en créer un synthétique
    """
    new_events = []
    for e in events:
        new_events.append(e)

        if e.get("type") in ("goal", "score"):
            has_shot = any(
                ev.get("type") == "shot"
                and abs(ev.get("time", 0) - e.get("time", 0)) < 3.0
                for ev in events
            )

            if not has_shot:
                new_events.append({
                    "type": "shot",
                    "time": e.get("time"),
                    "x": e.get("x"),
                    "y": e.get("y"),
                    "player": e.get("player"),
                    "team": e.get("team"),
                    "xg": e.get("xg", 0.1),
                    "synthetic": True,
                    "confidence": 0.6,
                })

    return sorted(new_events, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────────────────────────
# FILTRE BUTS (COOLDOWN + POSITION)
# ─────────────────────────────────────────────────────────────

def filter_goals(events, window=150.0, frame_w=1920, position_threshold=0.2):
    """
    Supprime doublons lointains (même action détectée plusieurs fois)
    """
    filtered = []
    last_goal_time = -999

    for e in events:
        if e.get("type") not in ("goal", "score"):
            filtered.append(e)
            continue

        t = e.get("time", 0)

        if abs(t - last_goal_time) < window:
            continue

        # Vérification position (évite faux positifs loin du but)
        x = e.get("x", 0)
        if not (x < frame_w * position_threshold or x > frame_w * (1 - position_threshold)):
            continue

        filtered.append(e)
        last_goal_time = t

    return filtered


# ─────────────────────────────────────────────────────────────
# ORCHESTRATEUR GLOBAL
# ─────────────────────────────────────────────────────────────

def post_process_events(
    events,
    frames_data=None,
    fps=25,
    frame_w=1920,
    frame_h=1080,
):
    """
    Pipeline complet post-processing.

    ⚠️ Appel unique depuis pipeline.py
    """

    if not events:
        return events

    print(f"  post_processing : START ({len(events)} events)")

    # ── 1. Deduplicate goals ──────────────────────────────────
    n_before = len(events)
    events = deduplicate_goals(events)
    print(f"  deduplicate_goals : {n_before} → {len(events)}")

    # ── 2. Merge players ──────────────────────────────────────
    events = merge_players(events)
    print(f"  merge_players : OK")

    # ── 3. Temporal filter ────────────────────────────────────
    events = temporal_filter(events)
    print(f"  temporal_filter : OK")

    # ── 4. Infer shots ────────────────────────────────────────
    events = infer_shots_from_goals(events)
    print(f"  infer_shots : OK")

    # ── 5. Goal filtering (cooldown + zone) ───────────────────
    video_duration = (
        (len(frames_data) / fps) if frames_data else 600
    )

    is_summary = video_duration < 480

    goal_cooldown = 10.0 if is_summary else 150.0
    position_threshold = 0.35 if is_summary else 0.20

    events = filter_goals(
        events,
        window=goal_cooldown,
        frame_w=frame_w,
        position_threshold=position_threshold
    )

    print(f"  filter_goals : cooldown={goal_cooldown}s")

    # ── SORT FINAL ────────────────────────────────────────────
    events = sorted(events, key=lambda x: x.get("time", 0))

    n_goals = sum(1 for e in events if e.get("type") in ("goal", "score"))
    print(f"  post_processing : DONE | {len(events)} events | {n_goals} buts")

    return events