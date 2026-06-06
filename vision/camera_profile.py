# vision/camera_profile.py
# -*- coding: utf-8 -*-
"""
Sprint 1 — Calibration géométrique de la caméra.

LECTURE SEULE. Ne modifie aucun seuil, aucune décision.
Produit un profil et le logue. C'est tout.

Usage dans pipeline.py (après tracking, avant goal_posthoc) :

    from vision.camera_profile import build_camera_profile
    _camera_profile = build_camera_profile(
        frames_data = frames_data,
        fps         = fps,
        frame_w     = _frame_w,
        frame_h     = _frame_h,
    )

Le profil est loggé mais PAS utilisé pour modifier les seuils.
Sprint 2 connectera goal_posthoc.
Sprint 3 connectera is_shot_candidate et toward_goal.
"""

import math


def build_camera_profile(
    frames_data: list,
    fps: float = 25.0,
    frame_w: int = 1920,
    frame_h: int = 1080,
    calib_seconds: int = 99999,  # par défaut : tout le match
) -> dict:
    """
    Analyse les positions du ballon sur les premières secondes
    et produit un profil géométrique de la caméra.

    Ne modifie rien. Log uniquement.
    """
    profile = {
        "calibrated":   False,
        "camera_type":  "unknown",
        "frame_w":      frame_w,
        "frame_h":      frame_h,
    }

    # ── Collecter les positions ballon sur les N premières secondes ───────
    ball_positions = []
    n_calib = int(calib_seconds * fps)

    for fd in frames_data[:n_calib]:
        ball = fd.get("ball")
        if not ball:
            continue
        center = ball.get("center")
        if center and center[0] is not None and center[1] is not None:
            cx, cy = float(center[0]), float(center[1])
            # Exclure les positions aberrantes : ballon hors image ou tracking perdu
            if cx < 10 or cy < 10:
                continue
            if cx > frame_w - 10 or cy > frame_h - 10:
                continue
            ball_positions.append((cx, cy))

    if len(ball_positions) < 50:
        print(f"  [CAMERA_PROFILE] Pas assez de positions ({len(ball_positions)}) "
              f"→ profil non disponible")
        return profile

    xs = sorted(p[0] for p in ball_positions)
    ys = sorted(p[1] for p in ball_positions)
    n  = len(xs)

    def pct(arr, p):
        return arr[max(0, min(len(arr)-1, int(len(arr) * p / 100)))]

    # Percentiles x
    x_p01 = pct(xs, 1)
    x_p05 = pct(xs, 5)
    x_p50 = pct(xs, 50)
    x_p95 = pct(xs, 95)
    x_p99 = pct(xs, 99)

    # Percentiles y
    y_p05 = pct(ys, 5)
    y_p50 = pct(ys, 50)
    y_p95 = pct(ys, 95)

    terrain_width_px = x_p99 - x_p01
    x_coverage       = terrain_width_px / frame_w

    # ── Détection du type de caméra ───────────────────────────────────────
    # Caméra haute : terrain visible sur >80% de la largeur, ballon
    # atteint les bords (x_p01 < 10% et x_p99 > 90%)
    # Caméra basse : terrain moins large, buts très proches des bords
    x_reach_left  = x_p01 / frame_w
    x_reach_right = x_p99 / frame_w
    y_coverage    = (y_p95 - y_p05) / frame_h

    if x_coverage > 0.82 and y_coverage > 0.40:
        camera_type = "high_side"
    elif x_coverage < 0.70:
        camera_type = "low_side_zoom"
    else:
        camera_type = "low_side"

    # ── Estimation des zones de but ───────────────────────────────────────
    # Hypothèse : le ballon ne dépasse p01/p99 que dans les buts (ou très rarement).
    # On ajoute une petite marge pour être conservateur.
    margin_px = max(20, int(terrain_width_px * 0.015))
    est_goal_left_px  = int(x_p01) + margin_px
    est_goal_right_px = int(x_p99) - margin_px

    est_goal_left_pct  = est_goal_left_px  / frame_w * 100
    est_goal_right_pct = est_goal_right_px / frame_w * 100

    # ── Vitesse de tir estimée ────────────────────────────────────────────
    # 15% du terrain visible / seconde = seuil minimal
    est_shot_speed_px  = terrain_width_px * 0.15
    est_shot_speed_factor = est_shot_speed_px / frame_w
    current_factor = 1.5   # valeur actuelle dans _dynamic_shot_threshold

    profile.update({
        "calibrated":           True,
        "camera_type":          camera_type,
        "n_positions":          len(ball_positions),

        # Mesures brutes
        "x_p01":                round(x_p01),
        "x_p05":                round(x_p05),
        "x_p50":                round(x_p50),
        "x_p95":                round(x_p95),
        "x_p99":                round(x_p99),
        "y_p05":                round(y_p05),
        "y_p50":                round(y_p50),
        "y_p95":                round(y_p95),

        # Dérivées
        "terrain_width_px":     round(terrain_width_px),
        "x_coverage_pct":       round(x_coverage * 100, 1),
        "y_coverage_pct":       round(y_coverage * 100, 1),
        "x_reach_left_pct":     round(x_reach_left * 100, 1),
        "x_reach_right_pct":    round(x_reach_right * 100, 1),

        # Estimations (pas encore utilisées)
        "est_goal_left_px":     est_goal_left_px,
        "est_goal_right_px":    est_goal_right_px,
        "est_goal_left_pct":    round(est_goal_left_pct, 1),
        "est_goal_right_pct":   round(est_goal_right_pct, 1),
        "est_shot_speed_factor": round(est_shot_speed_factor, 3),
        "current_shot_speed_factor": current_factor,
    })

    # ── LOG ───────────────────────────────────────────────────────────────
    print(f"\n  [CAMERA_PROFILE] {'='*50}")
    print(f"  [CAMERA_PROFILE] type          = {camera_type}")
    print(f"  [CAMERA_PROFILE] n_positions   = {len(ball_positions)} (sur {calib_seconds}s)")
    print(f"  [CAMERA_PROFILE] terrain_width = {terrain_width_px:.0f}px "
          f"({x_coverage*100:.1f}% du cadre)")
    print(f"  [CAMERA_PROFILE] ball_x        : "
          f"p01={x_p01:.0f} p05={x_p05:.0f} p50={x_p50:.0f} "
          f"p95={x_p95:.0f} p99={x_p99:.0f}")
    print(f"  [CAMERA_PROFILE] ball_y        : "
          f"p05={y_p05:.0f} p50={y_p50:.0f} p95={y_p95:.0f}")
    print(f"  [CAMERA_PROFILE] est_goal_left  = {est_goal_left_px}px "
          f"({est_goal_left_pct:.1f}%)  [actuel: {frame_w*0.06:.0f}px (6.0%)]")
    print(f"  [CAMERA_PROFILE] est_goal_right = {est_goal_right_px}px "
          f"({est_goal_right_pct:.1f}%)  [actuel: {frame_w*0.94:.0f}px (94.0%)]")
    print(f"  [CAMERA_PROFILE] est_shot_speed = ×{est_shot_speed_factor:.2f}  "
          f"[actuel: ×{current_factor:.2f}]")
    print(f"  [CAMERA_PROFILE] {'='*50}\n")

    return profile