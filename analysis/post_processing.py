# analysis/post_processing.py
# -*- coding: utf-8 -*-
#
# POST PROCESSING — VERSION PRO
#
# Fix v3 (critique) :
#   - filter_goals : garde le but avec la meilleure confidence dans chaque
#     fenêtre, PAS le premier arrivé (l'ancien code laissait passer des faux
#     buts précoces qui bloquaient les vrais buts dans la fenêtre de cooldown)
#   - cooldown réduit à 30s (150s = trop long, bloquait 02:14 et 09:44)
#   - merge_players : fenêtre 0.5s + contrainte position (évite 16→5 joueurs)
#   - temporal_filter : buts et tirs exemptés du filtre temporel


# ─────────────────────────────────────────────────────────────────────────────
# DÉDUPLICATION BUTS (confidence-aware, fenêtre courte)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_goals(events, window=3.0):
    """
    Dans une fenêtre de 3s, garde le but avec la meilleure confidence.
    """
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
# MERGE PLAYERS (non-agressif)
# ─────────────────────────────────────────────────────────────────────────────

def merge_players(events, time_window=0.5):
    """
    Fusionne uniquement les événements strictement identiques :
    même joueur + même type + < 0.5s + position < 50px.
    Évite la fusion 16 joueurs → 5.
    """
    if not events:
        return events

    merged = []
    for e in events:
        if not merged:
            merged.append(e)
            continue

        prev = merged[-1]

        same_player = (e.get("player") is not None and
                       str(e.get("player")) == str(prev.get("player")))
        same_type   = e.get("type") == prev.get("type")
        close_time  = abs(e.get("time", 0) - prev.get("time", 0)) < time_window

        ex, ey = e.get("x", 0), e.get("y", 0)
        px, py = prev.get("x", 0), prev.get("y", 0)
        close_pos = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5 < 50

        if same_player and same_type and close_time and close_pos:
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
    Buts et tirs sont exemptés (ne pas filtrer des vrais buts rapprochés).
    """
    filtered  = []
    last_time = {}

    for e in events:
        t     = e.get("time", 0)
        etype = e.get("type")

        # Buts et tirs : jamais filtrés temporellement ici
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
# FILTRE BUTS — PAR CONFIDENCE, PAS PAR ORDRE D'ARRIVÉE
# ─────────────────────────────────────────────────────────────────────────────

def filter_goals(events, window=30.0, frame_w=1920, position_threshold=0.15):
    """
    FIX CRITIQUE v3 :
    Dans chaque fenêtre de `window` secondes, garde le but avec la
    MEILLEURE CONFIDENCE — pas le premier arrivé.

    L'ancienne logique (premier arrivé) laissait passer des faux buts
    précoces (ex: 00:06 avec conf=0.68) qui bloquaient les vrais buts
    (02:14, 09:44) dans la fenêtre de cooldown.

    Paramètres :
      window             : durée de la fenêtre anti-doublon (30s par défaut)
                           150s était beaucoup trop long pour un match réel
      position_threshold : fraction de frame_w depuis chaque bord
                           0.15 = les 15% gauche + 15% droite (zone filet)
    """
    if not events:
        return events

    # Séparer buts et autres events
    goals  = sorted(
        [e for e in events if e.get("type") in ("goal", "score")],
        key=lambda x: x.get("time", 0)
    )
    others = [e for e in events if e.get("type") not in ("goal", "score")]

    if not goals:
        return events

    # Regrouper les buts par fenêtres temporelles
    # Méthode : scan linéaire, regrouper si < window secondes du dernier groupe
    groups   = []
    current  = [goals[0]]

    for g in goals[1:]:
        # Comparer avec le PREMIER but du groupe (ancrage fixe), Ça crée des groupes glissants, beaucoup plus réalistes
        if abs(g["time"] - current[-1]["time"]) < window:
            current.append(g)
        else:
            groups.append(current)
            current = [g]
    groups.append(current)

    # Dans chaque groupe, garder le but avec la meilleure confidence
    kept = []
    for group in groups:
        # Filtrer d'abord par position (zone de but)
        in_zone = [
            g for g in group
            if frame_w > 0 and (
                g.get("x", 0) < frame_w * position_threshold or
                g.get("x", 0) > frame_w * (1 - position_threshold)
            )
        ]

        candidates = in_zone if in_zone else group

        # Prioriser : gemini_validated=True > confidence > score
        def _sort_key(g):
            gemini_ok = 1 if g.get("gemini_validated") else 0
            conf      = g.get("confidence", 0)
            score     = g.get("score", 0)
            return (gemini_ok, conf, score)

        best = max(candidates, key=_sort_key)
        kept.append(best)

    return sorted(others + kept, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

def post_process_events(
    events,
    frames_data    = None,
    fps            = 25,
    frame_w        = 1920,
    frame_h        = 1080,
    goal_cooldown  = None,
    is_summary     = False,
):
    """
    Pipeline complet post-processing.
    Appelé une seule fois depuis pipeline.py après goal_posthoc.
    """
    if not events:
        return events

    print(f"  post_processing : START ({len(events)} events)")

    # ── 1. Deduplicate goals (fenêtre courte 3s) ──────────────────────────────
    n_before = len(events)
    events   = deduplicate_goals(events, window=3.0)
    print(f"  deduplicate_goals : {n_before} → {len(events)}")

    # ── 2. Merge players (non-agressif) ───────────────────────────────────────
    events = merge_players(events, time_window=0.5)
    print(f"  merge_players : OK")

    # ── 3. Temporal filter (buts exemptés) ───────────────────────────────────
    events = temporal_filter(events)
    print(f"  temporal_filter : OK")

    # ── 4. Infer shots ────────────────────────────────────────────────────────
    events = infer_shots_from_goals(events)
    print(f"  infer_shots : OK")

    # ── 5. Filter goals par confidence dans fenêtre ───────────────────────────
    if goal_cooldown is None:
        if frames_data:
            video_duration = len(frames_data) / fps
            is_summary     = video_duration < 480
        goal_cooldown = 10.0 if is_summary else 30.0  # FIX : 150s → 30s

    position_threshold = 0.25 if is_summary else 0.15  # FIX : zone but stricte

    events = filter_goals(
        events,
        window             = goal_cooldown,
        frame_w            = frame_w,
        position_threshold = position_threshold,
    )
    print(f"  filter_goals : cooldown={goal_cooldown}s | sélection par confidence")

    events  = sorted(events, key=lambda x: x.get("time", 0))
    n_goals = sum(1 for e in events if e.get("type") in ("goal", "score"))
    print(f"  post_processing : DONE | {len(events)} events | {n_goals} buts")

    return events