# analysis/post_processing.py
# -*- coding: utf-8 -*-
#
# POST PROCESSING — VERSION PRO
#
# Fixes v2 :
#   - merge_players moins agressif (fenêtre réduite + condition same_type strict)
#     → évite la fusion de 16 joueurs en 5
#   - filter_goals avec position_threshold cohérent avec résolution réelle
#   - log explicite à chaque étape

from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# DÉDUPLICATION BUTS (confidence-aware)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_goals(events, window=3.0):
    goals  = sorted(
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


# ─────────────────────────────────────────────────────────────────────────────
# MERGE PLAYERS
# ─────────────────────────────────────────────────────────────────────────────

def merge_players(events, time_window=0.5):
    """
    Fusionne uniquement les événements STRICTEMENT identiques :
      - même joueur (ID exact)
      - même type exact
      - fenêtre très courte (0.5s au lieu de 2s)
      - position proche (< 50px)

    FIX : l'ancienne version fusionnait trop agressivement
    → 16 joueurs devenaient 5 dans les stats.
    """
    if not events:
        return events

    merged = []
    for e in events:
        if not merged:
            merged.append(e)
            continue

        prev = merged[-1]

        # Conditions strictes pour fusion
        same_player = (e.get("player") is not None and
                       str(e.get("player")) == str(prev.get("player")))
        same_type   = e.get("type") == prev.get("type")
        close_time  = abs(e.get("time", 0) - prev.get("time", 0)) < time_window

        # Distance position (évite fusion de joueurs différents au même moment)
        ex, ey = e.get("x", 0), e.get("y", 0)
        px, py = prev.get("x", 0), prev.get("y", 0)
        close_pos = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5 < 50

        if same_player and same_type and close_time and close_pos:
            # Garder le plus confiant
            if e.get("confidence", 0) > prev.get("confidence", 0):
                merged[-1] = e
        else:
            merged.append(e)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# FILTRE TEMPOREL
# ─────────────────────────────────────────────────────────────────────────────

def temporal_filter(events, min_delta=2.0):
    """
    Supprime événements trop rapprochés du même type.
    Les buts et tirs ont un delta plus permissif (0.5s).
    """
    filtered  = []
    last_time = {}

    for e in events:
        t     = e.get("time", 0)
        etype = e.get("type")

        # Buts et tirs : pas de filtre temporel agressif
        if etype in ("goal", "score", "shot"):
            filtered.append(e)
            continue

        if etype not in last_time or t - last_time[etype] >= min_delta:
            filtered.append(e)
            last_time[etype] = t

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# INFÉRER SHOTS DEPUIS GOALS
# ─────────────────────────────────────────────────────────────────────────────

def infer_shots_from_goals(events):
    """
    Si un but n'a pas de tir associé dans les 5s → créer un tir synthétique.
    Utile pour le learning xG et les highlights.
    """
    new_events = list(events)

    for e in events:
        if e.get("type") not in ("goal", "score"):
            continue
        if e.get("shot_linked"):
            continue

        has_shot = any(
            ev.get("type") == "shot"
            and abs(ev.get("time", 0) - e.get("time", 0)) < 5.0
            for ev in events
        )

        if not has_shot:
            new_events.append({
                "type":       "shot",
                "time":       e.get("time"),
                "x":          e.get("x"),
                "y":          e.get("y"),
                "player":     e.get("player"),
                "team":       e.get("team"),
                "xg":         e.get("xg", 0.1),
                "synthetic":  True,
                "confidence": 0.6,
                "on_target":  True,
            })

    return sorted(new_events, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────────────────────────────────────────
# FILTRE BUTS (COOLDOWN + POSITION)
# ─────────────────────────────────────────────────────────────────────────────

def filter_goals(events, window=150.0, frame_w=1920, position_threshold=0.2):
    """
    Supprime les buts trop proches dans le temps (cooldown)
    et ceux détectés hors zone de but (position aberrante).

    position_threshold : fraction de frame_w autorisée depuis chaque bord.
    Exemple : 0.2 → le but doit être dans les 20% gauche ou 20% droite.
    """
    filtered       = []
    last_goal_time = -999

    for e in events:
        if e.get("type") not in ("goal", "score"):
            filtered.append(e)
            continue

        t = e.get("time", 0)

        # Cooldown
        if abs(t - last_goal_time) < window:
            continue

        # Vérification position
        x = e.get("x", 0)
        if frame_w > 0 and x > 0:
            if not (x < frame_w * position_threshold or
                    x > frame_w * (1 - position_threshold)):
                continue

        filtered.append(e)
        last_goal_time = t

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

def post_process_events(
    events,
    frames_data    = None,
    fps            = 25,
    frame_w        = 1920,
    frame_h        = 1080,
    goal_cooldown  = 150.0,
    is_summary     = False,
):
    """
    Pipeline complet post-processing.
    Appelé une seule fois depuis pipeline.py après goal_posthoc.
    """
    if not events:
        return events

    print(f"  post_processing : START ({len(events)} events)")

    # ── 1. Deduplicate goals ──────────────────────────────────────────────────
    n_before = len(events)
    events   = deduplicate_goals(events)
    print(f"  deduplicate_goals : {n_before} → {len(events)}")

    # ── 2. Merge players (moins agressif) ─────────────────────────────────────
    events = merge_players(events, time_window=0.5)
    print(f"  merge_players : OK")

    # ── 3. Temporal filter ────────────────────────────────────────────────────
    events = temporal_filter(events)
    print(f"  temporal_filter : OK")

    # ── 4. Infer shots ────────────────────────────────────────────────────────
    events = infer_shots_from_goals(events)
    print(f"  infer_shots : OK")

    # ── 5. Filter goals (cooldown + zone) ─────────────────────────────────────
    position_threshold = 0.35 if is_summary else 0.20

    events = filter_goals(
        events,
        window             = goal_cooldown,
        frame_w            = frame_w,
        position_threshold = position_threshold,
    )
    print(f"  filter_goals : cooldown={goal_cooldown}s")

    events  = sorted(events, key=lambda x: x.get("time", 0))
    n_goals = sum(1 for e in events if e.get("type") in ("goal", "score"))
    print(f"  post_processing : DONE | {len(events)} events | {n_goals} buts")

    return events