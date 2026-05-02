# analysis/post_processing.py
# -*- coding: utf-8 -*-
#
# POST PROCESSING — VERSION PROPRE
#
# Responsabilité de ce module :
#   merge_players, temporal_filter, infer_shots_from_goals, filter_goals
#
# Ce module NE gère PAS :
#   - deduplicate_goals  → géré dans pipeline.py (source unique)
#   - goal_cooldown      → calculé dans pipeline.py, passé en paramètre
#   - position_threshold → calculé dans pipeline.py, passé en paramètre
#
# Règle : pipeline.py est le chef d'orchestre.
# post_processing.py est un module de transformation pure, sans décision métier.


# ─────────────────────────────────────────────────────────────────────────────
# MERGE PLAYERS (non-agressif)
# ─────────────────────────────────────────────────────────────────────────────

def merge_players(events, time_window=0.5):
    """
    Fusionne uniquement les événements strictement identiques :
      même joueur + même type + < 0.5s + position < 50px.

    Les buts ne sont jamais fusionnés pour ne pas perdre de détections.
    """
    if not events:
        return events

    merged = []
    for e in events:
        if not merged:
            merged.append(e)
            continue

        prev = merged[-1]

        # Buts protégés de toute fusion
        if e.get("type") in ("goal", "score"):
            merged.append(e)
            continue

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
    Buts et tirs sont TOUJOURS exemptés.
    """
    filtered  = []
    last_time = {}

    for e in events:
        t     = e.get("time", 0)
        etype = e.get("type")

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
# FILTRE BUTS — paramètres TOUJOURS fournis par pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def filter_goals(events, window, frame_w, position_threshold):
    """
    Dans chaque fenêtre de `window` secondes, garde le but avec la
    meilleure confidence.

    Pas de valeurs par défaut : tous les paramètres viennent de pipeline.py
    pour éviter tout conflit silencieux.

    Priorité : gemini_validated=True > confidence > score
    """
    if not events:
        return events

    goals  = sorted(
        [e for e in events if e.get("type") in ("goal", "score")],
        key=lambda x: x.get("time", 0)
    )
    others = [e for e in events if e.get("type") not in ("goal", "score")]

    if not goals:
        return events

    # V9.7 — regroupement en 2 passes :
    # Passe 1 : fenêtre serrée (5s) pour regrouper les doublons immédiats
    # Passe 2 : cooldown large (window) pour écarter les buts légitimes distincts
    DEDUP_WINDOW = 5.0

    groups  = []
    current = [goals[0]]
    for g in goals[1:]:
        if abs(g["time"] - current[-1]["time"]) < DEDUP_WINDOW:
            current.append(g)
        else:
            groups.append(current)
            current = [g]
    groups.append(current)

    # Log des groupes pour debug
    print(f"  filter_goals groupes ({len(groups)}) : "
          + " | ".join(f"[{','.join(f'{g["time"]:.1f}s' for g in grp)}]"
                       for grp in groups[:10]))

    # Sélection du meilleur dans chaque groupe de doublons
    candidates_per_group = []
    for group in groups:
        in_zone = [
            g for g in group
            if frame_w > 0 and (
                g.get("x", 0) < frame_w * position_threshold or
                g.get("x", 0) > frame_w * (1 - position_threshold)
            )
        ]
        candidates = in_zone if in_zone else group

        def _sort_key(g):
            return (
                1 if g.get("gemini_validated") else 0,
                g.get("confidence", 0),
                g.get("score", 0),
            )

        candidates_per_group.append(max(candidates, key=_sort_key))

    # Passe 2 — anti-doublon court seulement (10s max)
    # V9.7+ : le vrai cooldown long se fait APRÈS validation Gemini dans pipeline.py
    # Un candidat rejeté par Gemini ne doit pas bloquer le candidat suivant
    ANTI_DEDUP = min(window, 10.0)
    kept = []
    for g in sorted(candidates_per_group, key=lambda x: x.get("time", 0)):
        if not kept or abs(g["time"] - kept[-1]["time"]) >= ANTI_DEDUP:
            kept.append(g)
        else:
            # Dans la fenêtre → garder le meilleur
            if (g.get("confidence", 0), g.get("score", 0)) > \
               (kept[-1].get("confidence", 0), kept[-1].get("score", 0)):
                kept[-1] = g

    return sorted(others + kept, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR — appelé depuis pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def post_process_events(
    events,
    goal_cooldown,
    position_threshold,
    frame_w     = 1920,
    frame_h     = 1080,
    frames_data = None,
    fps         = 25,
):
    """
    Transformations post-tracking.

    Paramètres obligatoires (décidés par pipeline.py) :
      goal_cooldown      : fenêtre anti-doublon en secondes
      position_threshold : fraction de frame_w pour la zone de but

    Ce module n'a AUCUNE valeur par défaut pour ces deux paramètres :
    ils sont toujours fournis depuis pipeline.py pour garantir la cohérence.
    """
    if not events:
        return events

    print(f"  post_processing : START ({len(events)} events)")

    events = merge_players(events, time_window=0.5)
    print(f"  merge_players : OK")

    events = temporal_filter(events)
    print(f"  temporal_filter : OK")

    events = infer_shots_from_goals(events)
    print(f"  infer_shots : OK")

    events = filter_goals(
        events,
        window             = goal_cooldown,
        frame_w            = frame_w,
        position_threshold = position_threshold,
    )
    print(f"  filter_goals : cooldown={goal_cooldown}s | "
          f"position={position_threshold * 100:.0f}% | sélection par confidence")

    events  = sorted(events, key=lambda x: x.get("time", 0))
    n_goals = sum(1 for e in events if e.get("type") in ("goal", "score"))
    print(f"  post_processing : DONE | {len(events)} events | {n_goals} buts")

    return events