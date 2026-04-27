# detect_goal_box.py — détection automatique de la position des buts
# Utilisé par goal_posthoc pour adapter les zones de détection à l'angle caméra

import cv2
import numpy as np


def detect_goal_box(video_path, n_frames=60, fps=25):
    """
    Analyse les n_frames premières frames pour détecter la position
    des buts (poteaux blancs verticaux) dans l'image.

    Retourne :
        dict avec :
            "left"  : {"x_min", "x_max", "y_min", "y_max"} ou None
            "right" : {"x_min", "x_max", "y_min", "y_max"} ou None
            "frame_w" : largeur frame
            "frame_h" : hauteur frame
            "method"  : "vision" ou "fallback"
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return _fallback_goal_box(1920, 1080)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Accumuler les détections de lignes verticales blanches
        left_candidates  = []   # x des poteaux gauche
        right_candidates = []   # x des poteaux droit
        frames_analyzed  = 0

        step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // n_frames)

        for frame_idx in range(0, n_frames * step, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            posts = _detect_goalposts_in_frame(frame, frame_w, frame_h)
            if posts:
                left_candidates.extend(posts.get("left", []))
                right_candidates.extend(posts.get("right", []))
            frames_analyzed += 1

        cap.release()

        if frames_analyzed == 0:
            return _fallback_goal_box(frame_w, frame_h)

        # Construire les bounding boxes finales
        result = {
            "frame_w": frame_w,
            "frame_h": frame_h,
            "method":  "vision",
            "left":    None,
            "right":   None,
        }

        goal_h_min = int(frame_h * 0.25)
        goal_h_max = int(frame_h * 0.75)

        if left_candidates:
            lx = int(np.median(left_candidates))
            result["left"] = {
                "x_min": max(0, lx - int(frame_w * 0.06)),
                "x_max": min(frame_w, lx + int(frame_w * 0.06)),
                "y_min": goal_h_min,
                "y_max": goal_h_max,
                "x_center": lx,
            }

        if right_candidates:
            rx = int(np.median(right_candidates))
            result["right"] = {
                "x_min": max(0, rx - int(frame_w * 0.06)),
                "x_max": min(frame_w, rx + int(frame_w * 0.06)),
                "y_min": goal_h_min,
                "y_max": goal_h_max,
                "x_center": rx,
            }

        # Valider : les deux buts doivent être de chaque côté du terrain
        if result["left"] and result["right"]:
            lx = result["left"]["x_center"]
            rx = result["right"]["x_center"]
            if lx > frame_w * 0.3 or rx < frame_w * 0.7:
                # Incohérent → fallback
                print(f"  [GOAL_BOX] Positions incohérentes (L={lx} R={rx}) → fallback")
                return _fallback_goal_box(frame_w, frame_h)

        if result["left"] or result["right"]:
            lx_str = result["left"]["x_center"]  if result["left"]  else "?"
            rx_str = result["right"]["x_center"] if result["right"] else "?"
            print(f"  [GOAL_BOX] Détecté : but_gauche x={lx_str} | but_droit x={rx_str}")
            return result

        # Aucune détection → fallback
        print("  [GOAL_BOX] Aucun poteau détecté → fallback")
        return _fallback_goal_box(frame_w, frame_h)

    except Exception as e:
        print(f"  [GOAL_BOX] Erreur : {e} → fallback")
        return _fallback_goal_box(1920, 1080)


def _detect_goalposts_in_frame(frame, frame_w, frame_h):
    """
    Détecte les poteaux de but (lignes verticales blanches) dans une frame.
    Retourne dict {"left": [x, ...], "right": [x, ...]}
    """
    # Redimensionner pour accélérer
    scale = 0.5
    small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))

    # Zone de jeu : moitié inférieure (terrain visible)
    h_start = int(small.shape[0] * 0.15)
    h_end   = int(small.shape[0] * 0.85)
    roi = small[h_start:h_end, :]

    # Masque blanc (poteaux)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Blanc = faible saturation + haute luminosité
    lower_white = np.array([0,   0, 180])
    upper_white = np.array([180, 50, 255])
    mask_white  = cv2.inRange(hsv, lower_white, upper_white)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    # Détection de lignes verticales via HoughLines
    edges = cv2.Canny(mask_white, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=int(frame_h * 0.12 * scale),
        maxLineGap=10,
    )

    if lines is None:
        return None

    left_xs  = []
    right_xs = []
    mid_x = small.shape[1] // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Vertical : dy >> dx
        if dy < 20 or dx > dy * 0.4:
            continue

        x_center = (x1 + x2) / 2 * (1 / scale)  # rescale vers full res

        # But gauche : x < 30% de la frame
        if x_center < frame_w * 0.30:
            left_xs.append(x_center)
        # But droit : x > 70% de la frame
        elif x_center > frame_w * 0.70:
            right_xs.append(x_center)

    return {"left": left_xs, "right": right_xs}


def _fallback_goal_box(frame_w, frame_h):
    """
    Fallback : zones calculées par pourcentage fixe (comportement actuel).
    Garantit que goal_posthoc fonctionne même sans détection.
    """
    goal_h_min = int(frame_h * 0.25)
    goal_h_max = int(frame_h * 0.75)
    margin     = int(frame_w * 0.06)

    return {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "method":  "fallback",
        "left": {
            "x_min":    0,
            "x_max":    int(frame_w * 0.08),
            "y_min":    goal_h_min,
            "y_max":    goal_h_max,
            "x_center": int(frame_w * 0.04),
        },
        "right": {
            "x_min":    int(frame_w * 0.92),
            "x_max":    frame_w,
            "y_min":    goal_h_min,
            "y_max":    goal_h_max,
            "x_center": int(frame_w * 0.96),
        },
    }


def goal_box_to_posthoc_params(goal_box, margin_pct=0.08):
    """
    Convertit goal_box en paramètres pour goal_posthoc :
    - GOAL_X_LEFT  : x max du but gauche (le ballon doit être <= cette valeur)
    - GOAL_X_RIGHT : x min du but droit  (le ballon doit être >= cette valeur)
    - GOAL_DISAPPEAR_ZONES : zones pour goal_posthoc_disappear
    """
    frame_w = goal_box["frame_w"]
    margin  = int(frame_w * margin_pct)

    params = {
        "frame_w": frame_w,
        "frame_h": goal_box["frame_h"],
    }

    if goal_box.get("left"):
        lbox = goal_box["left"]
        params["GOAL_X_LEFT"] = lbox["x_max"] + margin
        params["DISAPPEAR_X_LEFT_MIN"]  = max(0, lbox["x_center"] - int(frame_w * 0.15))
        params["DISAPPEAR_X_LEFT_MAX"]  = min(frame_w, lbox["x_center"] + int(frame_w * 0.15))
    else:
        params["GOAL_X_LEFT"] = int(frame_w * 0.06)
        params["DISAPPEAR_X_LEFT_MIN"]  = 0
        params["DISAPPEAR_X_LEFT_MAX"]  = int(frame_w * 0.20)

    if goal_box.get("right"):
        rbox = goal_box["right"]
        params["GOAL_X_RIGHT"] = rbox["x_min"] - margin
        params["DISAPPEAR_X_RIGHT_MIN"] = max(0, rbox["x_center"] - int(frame_w * 0.15))
        params["DISAPPEAR_X_RIGHT_MAX"] = min(frame_w, rbox["x_center"] + int(frame_w * 0.15))
    else:
        params["GOAL_X_RIGHT"] = int(frame_w * 0.94)
        params["DISAPPEAR_X_RIGHT_MIN"] = int(frame_w * 0.80)
        params["DISAPPEAR_X_RIGHT_MAX"] = frame_w

    return params