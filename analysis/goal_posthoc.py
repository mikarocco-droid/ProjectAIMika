# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-
#
# Détecteur de buts rapides basé sur la physique du ballon
#
# Principe :
#   Un vrai but rapide = ballon accélère → entre dans zone but → disparaît durablement
#   Signature physique :
#     1. vitesse normalisée élevée (indépendante résolution)
#     2. trajectoire vers une des deux zones de but
#     3. ballon absent pendant N frames consécutives (pas d'occlusion courte)
#     4. tir réel ou synthétique présent dans les 3s précédentes
#
# Anti-faux-positifs :
#   - disparition longue (12 frames) élimine les occlusions
#   - filtre tir récent élimine les dégagements
#   - filtre doublon évite les conflits avec goals déjà détectés

import math


def detect_fast_goals_from_ball(
    frames_data,
    events,
    fps                  = 25,
    frame_w              = 1920,
    frame_h              = 1080,
    disappearance_frames = 12,    # ballon absent pendant ~0.5s → vrai but
    speed_threshold      = 0.012, # vitesse normalisée min (12% de frame_w par frame)
    shot_window          = 3.0,   # tir doit être dans les 3s précédentes
    xg_min               = 0.20,  # xG min du tir pour valider
):
    """
    Détecte les buts rapides depuis la trajectoire physique du ballon.

    Paramètres :
        frames_data          : liste de frames avec ball.x, ball.y
        events               : events déjà détectés (pour filtre tir récent)
        fps                  : frames par seconde
        frame_w / frame_h    : résolution pour normalisation
        disappearance_frames : nombre de frames d'absence pour valider disparition
        speed_threshold      : vitesse normalisée minimum (frame_w units)
        shot_window          : fenêtre en secondes pour chercher un tir précédent
        xg_min               : xG minimum du tir associé
    """
    goals = []

    if not frames_data or len(frames_data) < disappearance_frames + 5:
        return goals

    # Index des goals déjà détectés pour éviter les doublons
    existing_goal_times = [
        e.get("time", 0) for e in events
        if e.get("type") in ["goal", "score"]
    ]

    # Index des tirs pour le filtre
    shot_events = [
        e for e in events
        if e.get("type") == "shot"
    ]

    for i in range(2, len(frames_data) - disappearance_frames - 1):
        f_prev = frames_data[i - 1]
        f_now  = frames_data[i]

        ball_prev = f_prev.get("ball")
        ball_now  = f_now.get("ball")

        if not ball_prev or not ball_now:
            continue

        bx_prev = ball_prev.get("x") or ball_prev.get("center", [0, 0])[0]
        by_prev = ball_prev.get("y") or ball_prev.get("center", [0, 0])[1]
        bx_now  = ball_now.get("x")  or ball_now.get("center",  [0, 0])[0]
        by_now  = ball_now.get("y")  or ball_now.get("center",  [0, 0])[1]

        if not bx_now or not bx_prev:
            continue

        # ── Vitesse normalisée (indépendante résolution + frame_skip) ──
        dx    = (bx_now - bx_prev) / max(frame_w, 1)
        dy    = (by_now - by_prev) / max(frame_h, 1)
        speed = math.hypot(dx, dy)

        if speed < speed_threshold:
            continue

        # ── Direction vers un but ──
        toward_goal = (
            bx_now > frame_w * 0.70 or
            bx_now < frame_w * 0.30
        )
        if not toward_goal:
            continue

        # ── Zone proche du but (<15% de chaque côté) ──
        near_goal = (
            bx_now > frame_w * 0.85 or
            bx_now < frame_w * 0.15
        )
        if not near_goal:
            continue

        # ── Disparition longue (robustesse occlusions) ──
        # Le ballon doit être absent pendant disappearance_frames consécutives
        disappeared_long = True
        for j in range(1, disappearance_frames + 1):
            if i + j >= len(frames_data):
                break
            future_ball = frames_data[i + j].get("ball")
            # Accepter aussi les frames avec ballon interpolé (peu fiables)
            if future_ball is not None and not future_ball.get("interpolated", False):
                disappeared_long = False
                break

        if not disappeared_long:
            continue

        frame_id  = f_now.get("frame", i)
        goal_time = round(frame_id / fps, 2)

        # ── Anti-doublon avec goals déjà détectés ──
        if any(abs(gt - goal_time) < 5.0 for gt in existing_goal_times):
            continue

        # ── Filtre tir récent (vrai ou synthétique) ──
        recent_shot = None
        for e in reversed(shot_events):
            e_time = e.get("time", 0)
            if e_time < goal_time - shot_window:
                break
            if 0 <= goal_time - e_time <= shot_window:
                recent_shot = e
                break

        if recent_shot is None:
            continue

        # ── Filtre xG minimum ──
        shot_xg = recent_shot.get("xg", 0) or 0
        if shot_xg < xg_min:
            continue

        # ── Valider ! ──
        goals.append({
            "type":           "goal",
            "time":           goal_time,
            "frame":          frame_id,
            "x":              bx_now,
            "y":              by_now,
            "player":         recent_shot.get("player"),
            "team":           recent_shot.get("team"),
            "xg":             shot_xg,
            "danger":         8.0,
            "detected_from":  "ball_physics_v2",
            "shot_linked":    True,
            "gemini_validated": False,
        })

        # Ajouter ce but à l'index pour éviter doublons dans la même passe
        existing_goal_times.append(goal_time)

        mins = int(goal_time // 60)
        secs = int(goal_time % 60)
        print(f"  goal_posthoc : but détecté à {mins:02d}:{secs:02d} "
              f"(speed={speed:.3f} | shot_xg={shot_xg:.2f} | "
              f"x={bx_now:.0f} | disparu {disappearance_frames} frames)")

    return goals