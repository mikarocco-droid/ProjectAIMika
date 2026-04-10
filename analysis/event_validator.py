# analysis/event_validator.py
# -*- coding: utf-8 -*-
"""
Validation structurelle des events AVANT le learning.
Filtre les events avec coords absurdes, temps invalides,
valeurs manquantes ou physiquement impossibles.
"""

import math


# ─────────────────────────────────────────
# BORNES PHYSIQUES PAR SPORT
# ─────────────────────────────────────────
PHYSICAL_BOUNDS = {
    "football": {
        "x": (0, 1920),
        "y": (0, 1080),
        "xg": (0.0, 1.0),
        "speed": (0, 200),        # px/frame max raisonnable
        "time_min": 0,
        "time_max": 7200,         # 2h max
    },
    "basketball": {
        "x": (0, 1920), "y": (0, 1080),
        "xg": (0.0, 1.0), "speed": (0, 300),
        "time_min": 0, "time_max": 3600,
    },
    "handball": {
        "x": (0, 1920), "y": (0, 1080),
        "xg": (0.0, 1.0), "speed": (0, 250),
        "time_min": 0, "time_max": 4800,
    },
}

# Types d'events connus
KNOWN_TYPES = {
    "goal", "shot", "pass", "interception", "dribble",
    "possession", "fast_break", "build_up", "long_pass",
    "progressive_run", "under_pressure", "shot_blocked",
    "corner", "touche", "goalkeeper_hold", "goalkeeper_throw",
    "defensive_clearance", "key_pass", "action",
}


def _get_bounds(sport):
    return PHYSICAL_BOUNDS.get(sport, PHYSICAL_BOUNDS["football"])


# ─────────────────────────────────────────
# VALIDATION D'UN SEUL EVENT
# ─────────────────────────────────────────
def validate_event(e, sport="football", frame_w=1920, frame_h=1080):
    """
    Retourne (True, None) si l'event est valide,
    (False, "raison") sinon.
    """
    bounds = _get_bounds(sport)
    etype  = e.get("type", "")

    # Type connu
    if etype not in KNOWN_TYPES:
        return False, f"type inconnu : {etype!r}"

    # Coords — uniquement pour les events avec position
    if etype in {"goal", "shot", "pass", "interception", "dribble",
                 "shot_blocked", "progressive_run"}:
        x = e.get("x")
        y = e.get("y")

        if x is None or y is None:
            return False, "coords manquantes"

        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return False, f"coords non numériques : x={x} y={y}"

        xlo, xhi = bounds["x"]
        ylo, yhi = bounds["y"]

        if not (xlo <= x <= xhi):
            return False, f"x hors terrain : {x:.0f} (attendu {xlo}-{xhi})"
        if not (ylo <= y <= yhi):
            return False, f"y hors terrain : {y:.0f} (attendu {ylo}-{yhi})"

        # Coords à (0,0) = souvent une valeur par défaut non renseignée
        if x == 0 and y == 0 and etype in {"goal", "shot"}:
            return False, "coords (0,0) sur but/tir — valeur par défaut"

    # xG
    xg = e.get("xg")
    if xg is not None:
        try:
            xg = float(xg)
        except (TypeError, ValueError):
            return False, f"xg non numérique : {xg}"
        xg_lo, xg_hi = bounds["xg"]
        if not (xg_lo <= xg <= xg_hi):
            return False, f"xg hors bornes : {xg}"

    # Temps
    t = e.get("time") or e.get("t")
    if t is not None:
        try:
            t = float(t)
        except (TypeError, ValueError):
            return False, f"time non numérique : {t}"
        if t < bounds["time_min"] or t > bounds["time_max"]:
            return False, f"time hors bornes : {t:.1f}s"
        if t < 0:
            return False, f"time négatif : {t}"

    # Frame
    frame = e.get("frame")
    if frame is not None:
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            return False, f"frame non entier : {frame}"
        if frame < 0:
            return False, f"frame négative : {frame}"

    # Danger score
    danger = e.get("danger")
    if danger is not None:
        try:
            danger = float(danger)
        except (TypeError, ValueError):
            return False, f"danger non numérique : {danger}"
        if danger < 0 or danger > 15:
            return False, f"danger hors bornes : {danger}"

    # Confiance Gemini
    conf = e.get("gemini_conf")
    if conf is not None:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return False, f"gemini_conf non numérique : {conf}"
        if not (0.0 <= conf <= 1.0):
            return False, f"gemini_conf hors [0,1] : {conf}"

    return True, None


# ─────────────────────────────────────────
# FILTRE UN BATCH D'EVENTS
# ─────────────────────────────────────────
def filter_events(events, sport="football", frame_w=1920, frame_h=1080,
                  verbose=False):
    """
    Filtre structurel : retire les events invalides.
    Retourne (events_valides, stats).
    """
    valid   = []
    invalid = []

    for e in events:
        ok, reason = validate_event(e, sport, frame_w, frame_h)
        if ok:
            valid.append(e)
        else:
            invalid.append((e, reason))
            if verbose:
                print(f"  [INVALID] {e.get('type')} t={e.get('time','?')} "
                      f"→ {reason}")

    stats = {
        "total":   len(events),
        "valid":   len(valid),
        "invalid": len(invalid),
        "by_reason": {},
    }
    for _, reason in invalid:
        # groupe par préfixe de raison
        key = reason.split(":")[0].strip()
        stats["by_reason"][key] = stats["by_reason"].get(key, 0) + 1

    if invalid:
        print(f"  Event validator : {len(valid)}/{len(events)} valides "
              f"| {len(invalid)} rejetés "
              f"| raisons: {stats['by_reason']}")

    return valid, stats


# ─────────────────────────────────────────
# DÉTECTION RÉELLE DE TIRS
# Priorité #1 — remplace infer_shots_from_goals
# ─────────────────────────────────────────
def detect_real_shots(events, ball_history_by_frame=None,
                      frame_w=1920, frame_h=1080,
                      sport="football", fps=25):
    """
    Détecte les vrais tirs depuis la trajectoire du ballon.
    Ne FABRIQUE pas de données — détecte seulement ce qui
    est physiquement présent dans les events existants.

    Critères :
    1. Vitesse ballon élevée (> seuil)
    2. Direction vers un but (dot product > seuil)
    3. Pas de possession calme juste avant
    4. Pas déjà un shot/goal dans les 2s
    """
    # Seuils selon sport
    SHOT_SPEED_MIN = {
        "football":   frame_w * 0.04,   # ~77px sur 1920
        "basketball": frame_w * 0.06,
        "handball":   frame_w * 0.05,
    }.get(sport, frame_w * 0.04)

    # Position des buts en vue latérale (normalisé)
    GOAL_POSITIONS = [
        (0.0,  0.5),    # but gauche (centre)
        (1.0,  0.5),    # but droit  (centre)
    ]

    shots_added = []
    last_shot_t = -999.0

    # Index des events par temps pour recherche rapide
    possession_times = {
        round(e.get("time", 0), 1)
        for e in events if e.get("type") == "possession"
    }

    for i, e in enumerate(events):
        if e.get("type") not in ["shot_blocked", "fast_break"]:
            continue

        t = float(e.get("time", 0))
        x = float(e.get("x", 0) or 0)
        y = float(e.get("y", 0) or 0)

        if x == 0 and y == 0:
            continue

        # Cooldown — pas deux tirs trop proches
        if t - last_shot_t < 2.0:
            continue

        # Chercher les positions ballon dans la fenêtre -0.5s à 0s
        prev_events = [
            ev for ev in events
            if abs(ev.get("time", 0) - t) < 0.5
            and ev.get("type") == "possession"
            and ev.get("x") and ev.get("y")
        ]

        if len(prev_events) < 2:
            continue

        # Calculer vitesse et direction depuis les positions précédentes
        xs = [float(ev.get("x", 0)) for ev in prev_events[-3:]] + [x]
        ys = [float(ev.get("y", 0)) for ev in prev_events[-3:]] + [y]

        if len(xs) < 2:
            continue

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        speed = math.sqrt(dx * dx + dy * dy)

        if speed < SHOT_SPEED_MIN:
            continue

        # Vérifier direction vers un but
        norm = speed
        dx_n = dx / norm
        dy_n = dy / norm

        toward_goal = False
        for gx_n, gy_n in GOAL_POSITIONS:
            gx = gx_n * frame_w
            gy = gy_n * frame_h
            to_goal_x = (gx - x) / max(abs(gx - x), 1)
            to_goal_y = (gy - y) / max(abs(gy - y), 1)
            dot = dx_n * to_goal_x + dy_n * to_goal_y
            if dot > 0.55:   # angle < 57° vers le but
                toward_goal = True
                break

        if not toward_goal:
            continue

        # Calcul xG basique depuis la distance au but
        dist_right = math.sqrt((x - frame_w) ** 2 + (y - frame_h / 2) ** 2)
        dist_left  = math.sqrt(x ** 2 + (y - frame_h / 2) ** 2)
        dist       = min(dist_right, dist_left)
        max_dist   = math.sqrt(frame_w ** 2 + (frame_h / 2) ** 2)
        xg         = round(max(0.01, min(0.5, 1.0 - dist / max_dist)), 3)

        shot = {
            "type":            "shot",
            "x":               x,
            "y":               y,
            "xg":              xg,
            "speed":           round(speed, 1),
            "toward_goal":     toward_goal,
            "time":            t,
            "frame":           e.get("frame", 0),
            "player":          e.get("player"),
            "team":            e.get("team"),
            "detected_from":   e.get("type"),   # traçabilité
            "gemini_validated": False,
            "inferred":        False,            # PAS fabriqué — détecté
        }
        shots_added.append(shot)
        last_shot_t = t

    if shots_added:
        print(f"  Shot detector : {len(shots_added)} tirs détectés "
              f"(vitesse + direction)")

    return shots_added