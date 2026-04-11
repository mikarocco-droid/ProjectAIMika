# analysis/player_rating.py
# -*- coding: utf-8 -*-
#
# Ranking joueur basé sur xG + xA + efficacité
# Inspiré StatsBomb / Opta
#
# Score = (xG * 1.0) + (xA * 1.2) + (efficiency * 0.5)
#
# xG  = expected goals (tirs)
# xA  = expected assists (passes clés avant tir)
# efficiency = goals - xG  (sur/sous-performance)


# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────
WEIGHT_XG         = 1.0   # poids xG
WEIGHT_XA         = 1.2   # poids xA (création légèrement plus valorisée)
WEIGHT_EFFICIENCY = 0.5   # poids sur/sous-performance
WEIGHT_DRIBBLE    = 0.15  # dribble réussi
WEIGHT_INTERCEPT  = 0.10  # interception
WEIGHT_PROG_RUN   = 0.08  # progressive run


# ─────────────────────────────────────────
# COMPUTE PLAYER RANKINGS
# ─────────────────────────────────────────
def compute_player_ratings(events, jersey_map=None, fps=25):
    """
    Calcule le score de chaque joueur depuis les events du pipeline.

    Paramètres :
        events     : liste d'events (dict) avec au minimum :
                     player, type, xg, time, frame
        jersey_map : dict {player_id: jersey_number} pour affichage
        fps        : frames par seconde (pour minutes jouées)

    Retourne :
        dict {player_id: stats_dict} trié par score décroissant
    """
    if jersey_map is None:
        jersey_map = {}

    players = {}

    for e in events:
        pid = str(e.get("player", "")) if e.get("player") is not None else None
        if not pid:
            continue

        if pid not in players:
            players[pid] = {
                "xg":           0.0,
                "xa":           0.0,
                "goals":        0,
                "shots":        0,
                "key_passes":   0,
                "dribbles":     0,
                "interceptions": 0,
                "progressive_runs": 0,
                "touches":      0,
                "frames":       [],   # pour minutes jouées
                "label":        jersey_map.get(pid) or jersey_map.get(int(pid) if pid.isdigit() else pid),
            }

        p = players[pid]
        etype = e.get("type", "")
        xg    = float(e.get("xg", 0.0) or 0.0)
        frame = e.get("frame", 0) or 0

        if frame > 0:
            p["frames"].append(frame)

        p["touches"] += 1

        # ── SHOT ──────────────────────────
        if etype == "shot":
            p["xg"]    += xg
            p["shots"] += 1

        # ── GOAL ──────────────────────────
        elif etype in ["goal", "score"]:
            p["goals"] += 1
            # Un but = aussi un shot s'il n'y en a pas déjà un juste avant
            p["shots"] += 1
            p["xg"]    += max(xg, 0.5)   # minimum xG 0.5 pour un but marqué

        # ── KEY PASS → xA ─────────────────
        elif etype in ["key_pass", "pass"]:
            # Passe clé = passe qui précède directement un tir/but
            # On vérifie si dans les 5s suivantes il y a un shot/goal
            if e.get("is_key_pass") or e.get("key_pass"):
                p["xa"]          += xg  # xA = xG du tir suivant
                p["key_passes"]  += 1

        # ── DRIBBLE ───────────────────────
        elif etype == "dribble":
            p["dribbles"] += 1

        # ── INTERCEPTION ──────────────────
        elif etype == "interception":
            p["interceptions"] += 1

        # ── PROGRESSIVE RUN ───────────────
        elif etype == "progressive_run":
            p["progressive_runs"] += 1

    # ─────────────────────────────────────
    # CALCUL SCORE FINAL
    # ─────────────────────────────────────
    ratings = {}

    for pid, p in players.items():
        xg         = p["xg"]
        xa         = p["xa"]
        goals      = p["goals"]
        efficiency = goals - xg   # positif = sur-performe, négatif = sous-performe

        # Score de base
        score = (
            (xg         * WEIGHT_XG)         +
            (xa         * WEIGHT_XA)         +
            (efficiency * WEIGHT_EFFICIENCY) +
            (p["dribbles"]          * WEIGHT_DRIBBLE)   +
            (p["interceptions"]     * WEIGHT_INTERCEPT) +
            (p["progressive_runs"]  * WEIGHT_PROG_RUN)
        )

        # Minutes jouées (approx via écart frame min/max)
        minutes_played = 0.0
        if p["frames"] and fps > 0:
            span_frames  = max(p["frames"]) - min(p["frames"])
            minutes_played = span_frames / fps / 60.0

        # Score par 90 minutes (si assez de temps de jeu)
        score_per_90 = 0.0
        if minutes_played >= 5.0:
            score_per_90 = round(score / minutes_played * 90, 3)

        # Label joueur
        label = p.get("label")
        if label:
            label = f"#{label}"
        else:
            label = f"ID-{pid}"

        ratings[pid] = {
            "label":            label,
            "score":            round(max(0.0, score), 3),
            "score_per_90":     score_per_90,
            "xg":               round(xg,         3),
            "xa":               round(xa,          3),
            "goals":            goals,
            "shots":            p["shots"],
            "efficiency":       round(efficiency,  3),
            "key_passes":       p["key_passes"],
            "dribbles":         p["dribbles"],
            "interceptions":    p["interceptions"],
            "progressive_runs": p["progressive_runs"],
            "touches":          p["touches"],
            "minutes_played":   round(minutes_played, 1),
        }

    # Tri par score décroissant
    ratings = dict(
        sorted(ratings.items(), key=lambda kv: kv[1]["score"], reverse=True)
    )

    return ratings


# ─────────────────────────────────────────
# GET MVP
# ─────────────────────────────────────────
def get_mvp(ratings):
    """
    Retourne l'ID du joueur avec le score le plus élevé.
    """
    if not ratings:
        return None
    return next(iter(ratings))


# ─────────────────────────────────────────
# DETECT KEY PASSES
# ─────────────────────────────────────────
def tag_key_passes(events, window_sec=5.0):
    """
    Marque les passes qui précèdent un tir dans les window_sec secondes.
    Modifie les events in-place.
    """
    shot_times = [
        e.get("time", 0)
        for e in events
        if e.get("type") in ["shot", "goal", "score"]
    ]

    for e in events:
        if e.get("type") not in ["pass", "long_pass"]:
            continue
        t = e.get("time", 0)
        # Vérifie si un tir arrive dans les window_sec suivantes
        for st in shot_times:
            if 0 < st - t <= window_sec:
                e["is_key_pass"] = True
                break

    return events


# ─────────────────────────────────────────
# FORMAT RANKING (pour affichage PDF / UI)
# ─────────────────────────────────────────
def format_ranking(ratings, top_n=10):
    """
    Retourne une liste triée des top N joueurs avec stats clés.
    """
    result = []
    for i, (pid, stats) in enumerate(list(ratings.items())[:top_n]):
        result.append({
            "rank":         i + 1,
            "player_id":    pid,
            "label":        stats["label"],
            "score":        stats["score"],
            "score_per_90": stats["score_per_90"],
            "xg":           stats["xg"],
            "xa":           stats["xa"],
            "goals":        stats["goals"],
            "efficiency":   stats["efficiency"],
            "minutes":      stats["minutes_played"],
        })
    return result