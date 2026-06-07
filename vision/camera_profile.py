# vision/camera_profile.py
# -*- coding: utf-8 -*-
"""
Sprint 2 — Calibration géométrique par détection de lignes de terrain.

Détecte les lignes blanches du terrain (ligne de but, touche) pour
positionner précisément les buts, indépendamment des positions du ballon.

Sprint 1 : positions ballon uniquement (biais si jeu concentré d'un côté)
Sprint 2 : lignes blanches du terrain (ancres géométriques fixes)

Usage dans pipeline.py :

    from vision.camera_profile import build_camera_profile
    _camera_profile = build_camera_profile(
        frames_data = frames_data,
        fps         = fps,
        frame_w     = _frame_w,
        frame_h     = _frame_h,
        video_path  = video_path,   # Sprint 2 : lecture directe des frames
    )
"""

import math


def _detect_goal_lines_from_video(video_path, frame_w, frame_h, fps=25.0):
    """
    Lit 20 frames espacées entre t=5s et t=90s.
    Pour chaque frame, détecte les colonnes denses de blanc sur fond vert.
    La ligne de but = pic de blanc dans x < 20% (gauche) ou x > 80% (droite).

    Retourne (goal_left_x, goal_right_x) en pixels résolution originale,
    ou (None, None) si détection échoue.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps      = cap.get(cv2.CAP_PROP_FPS) or fps
        vid_fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prendre 20 frames entre 5s et 90s
        t_start = int(5  * vid_fps)
        t_end   = int(min(90 * vid_fps, total_frames - 1))
        if t_end <= t_start:
            cap.release()
            return None, None

        sample_frames = np.linspace(t_start, t_end, 20, dtype=int)

        left_votes  = []
        right_votes = []

        for fid in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fh, fw = frame.shape[:2]

            # Réduire à 480p pour vitesse
            scale = 480 / fh
            small = cv2.resize(frame, (int(fw * scale), 480))
            sh, sw = small.shape[:2]

            # Masque herbe verte
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (35, 35, 35), (90, 255, 255))

            # Pixels blancs sur fond vert (lignes de terrain)
            # Blanc = tous canaux > 170 ET herbe visible dans le voisinage
            white_mask = cv2.inRange(small, (160, 160, 160), (255, 255, 255))

            # Garder seulement la zone terrain (y > 15% de hauteur)
            terrain_top = int(sh * 0.15)
            white_mask[:terrain_top, :] = 0

            # Dilater le masque vert pour avoir le voisinage
            kernel = np.ones((15, 1), np.uint8)
            green_dilated = cv2.dilate(green_mask, kernel)
            white_on_field = cv2.bitwise_and(white_mask, green_dilated)

            # Somme par colonne
            col_sum = white_on_field.sum(axis=0).astype(float) / 255

            # Lisser
            col_smooth = np.convolve(col_sum, np.ones(7)/7, mode='same')

            # Zones de but : x < 18% (gauche) et x > 82% (droite)
            left_end  = int(sw * 0.18)
            right_beg = int(sw * 0.82)

            left_zone  = col_smooth[:left_end]
            right_zone = col_smooth[right_beg:]

            # Seuil = 40% de la hauteur de frame (lignes doivent couvrir >40% verticalement)
            min_coverage = sh * 0.40

            if left_zone.max() > min_coverage:
                # Pic le plus à droite dans la zone gauche
                # (poteau droit du but gauche = limite intérieure)
                peaks = np.where(left_zone > min_coverage)[0]
                if len(peaks) > 0:
                    x_small = peaks.max()  # bord intérieur du but gauche
                    x_orig = int(x_small / scale)
                    left_votes.append(x_orig)

            if right_zone.max() > min_coverage:
                peaks = np.where(right_zone > min_coverage)[0]
                if len(peaks) > 0:
                    x_small = right_beg + peaks.min()  # bord intérieur but droit
                    x_orig = int(x_small / scale)
                    right_votes.append(x_orig)

        cap.release()

        # Médiane des votes (robuste aux outliers)
        goal_left_x  = None
        goal_right_x = None

        if len(left_votes) >= 5:
            goal_left_x = int(sorted(left_votes)[len(left_votes)//2])

        if len(right_votes) >= 5:
            goal_right_x = int(sorted(right_votes)[len(right_votes)//2])

        return goal_left_x, goal_right_x

    except Exception as e:
        print(f"  [CAMERA_PROFILE] Détection lignes échouée : {e}")
        return None, None


def build_camera_profile(
    frames_data: list,
    fps: float = 25.0,
    frame_w: int = 1920,
    frame_h: int = 1080,
    calib_seconds: int = 99999,
    video_path: str = None,   # Sprint 2 : détection lignes
) -> dict:
    """
    Produit un profil géométrique de la caméra.

    Sprint 2 : si video_path fourni, détecte les lignes blanches du terrain
    pour positionner les buts avec précision.
    Fallback : estimation par positions du ballon (Sprint 1).
    """
    profile = {
        "calibrated":   False,
        "camera_type":  "unknown",
        "frame_w":      frame_w,
        "frame_h":      frame_h,
    }

    # ── Collecter les positions ballon (Sprint 1 — fallback) ──────────────
    ball_positions = []
    n_calib = int(calib_seconds * fps)

    for fd in frames_data[:n_calib]:
        ball = fd.get("ball")
        if not ball:
            continue
        center = ball.get("center")
        if center and center[0] is not None and center[1] is not None:
            cx, cy = float(center[0]), float(center[1])
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

    x_p01 = pct(xs, 1)
    x_p05 = pct(xs, 5)
    x_p50 = pct(xs, 50)
    x_p95 = pct(xs, 95)
    x_p99 = pct(xs, 99)
    y_p05 = pct(ys, 5)
    y_p50 = pct(ys, 50)
    y_p95 = pct(ys, 95)

    terrain_width_px = x_p99 - x_p01
    x_coverage       = terrain_width_px / frame_w
    x_reach_left     = x_p01 / frame_w
    x_reach_right    = x_p99 / frame_w
    y_coverage       = (y_p95 - y_p05) / frame_h

    if x_coverage > 0.82 and y_coverage > 0.40:
        camera_type = "high_side"
    elif x_coverage < 0.70:
        camera_type = "low_side_zoom"
    else:
        camera_type = "low_side"

    # ── Sprint 1 : estimation par positions ballon ────────────────────────
    margin_px = max(20, int(terrain_width_px * 0.015))
    est_goal_left_ball  = int(x_p01) + margin_px
    est_goal_right_ball = int(x_p99) - margin_px
    goal_method = "ball_positions"

    # ── Sprint 2 : détection lignes de terrain ────────────────────────────
    est_goal_left_px  = est_goal_left_ball
    est_goal_right_px = est_goal_right_ball

    if video_path:
        gl_lines, gr_lines = _detect_goal_lines_from_video(
            video_path, frame_w, frame_h, fps
        )
        if gl_lines is not None or gr_lines is not None:
            # Fusionner : prendre la détection lignes quand disponible
            if gl_lines is not None:
                est_goal_left_px = gl_lines
            if gr_lines is not None:
                est_goal_right_px = gr_lines
            goal_method = "line_detection"
            if gl_lines is not None and gr_lines is not None:
                goal_method = "line_detection_both"
            elif gl_lines is not None:
                goal_method = "line_detection_left_only"
            else:
                goal_method = "line_detection_right_only"

    est_goal_left_pct  = est_goal_left_px  / frame_w * 100
    est_goal_right_pct = est_goal_right_px / frame_w * 100

    est_shot_speed_px     = terrain_width_px * 0.15
    est_shot_speed_factor = est_shot_speed_px / frame_w
    current_factor        = 1.5

    profile.update({
        "calibrated":           True,
        "camera_type":          camera_type,
        "goal_method":          goal_method,
        "n_positions":          len(ball_positions),

        "x_p01":                round(x_p01),
        "x_p05":                round(x_p05),
        "x_p50":                round(x_p50),
        "x_p95":                round(x_p95),
        "x_p99":                round(x_p99),
        "y_p05":                round(y_p05),
        "y_p50":                round(y_p50),
        "y_p95":                round(y_p95),

        "terrain_width_px":     round(terrain_width_px),
        "x_coverage_pct":       round(x_coverage * 100, 1),
        "y_coverage_pct":       round(y_coverage * 100, 1),
        "x_reach_left_pct":     round(x_reach_left * 100, 1),
        "x_reach_right_pct":    round(x_reach_right * 100, 1),

        "est_goal_left_px":     est_goal_left_px,
        "est_goal_right_px":    est_goal_right_px,
        "est_goal_left_pct":    round(est_goal_left_pct, 1),
        "est_goal_right_pct":   round(est_goal_right_pct, 1),
        "est_goal_left_ball":   est_goal_left_ball,
        "est_goal_right_ball":  est_goal_right_ball,
        "est_shot_speed_factor": round(est_shot_speed_factor, 3),
        "current_shot_speed_factor": current_factor,
    })

    # ── LOG ───────────────────────────────────────────────────────────────
    print(f"\n  [CAMERA_PROFILE] {'='*50}")
    print(f"  [CAMERA_PROFILE] type          = {camera_type}")
    print(f"  [CAMERA_PROFILE] goal_method   = {goal_method}")
    print(f"  [CAMERA_PROFILE] n_positions   = {len(ball_positions)} (sur {calib_seconds}s)")
    print(f"  [CAMERA_PROFILE] terrain_width = {terrain_width_px:.0f}px "
          f"({x_coverage*100:.1f}% du cadre)")
    print(f"  [CAMERA_PROFILE] ball_x        : "
          f"p01={x_p01:.0f} p05={x_p05:.0f} p50={x_p50:.0f} "
          f"p95={x_p95:.0f} p99={x_p99:.0f}")
    print(f"  [CAMERA_PROFILE] ball_y        : "
          f"p05={y_p05:.0f} p50={y_p50:.0f} p95={y_p95:.0f}")
    print(f"  [CAMERA_PROFILE] est_goal_left  = {est_goal_left_px}px "
          f"({est_goal_left_pct:.1f}%)  [ball: {est_goal_left_ball}px]")
    print(f"  [CAMERA_PROFILE] est_goal_right = {est_goal_right_px}px "
          f"({est_goal_right_pct:.1f}%)  [ball: {est_goal_right_ball}px]")
    print(f"  [CAMERA_PROFILE] est_shot_speed = ×{est_shot_speed_factor:.2f}  "
          f"[actuel: ×{current_factor:.2f}]")
    print(f"  [CAMERA_PROFILE] {'='*50}\n")

    return profile