# detect_goal_box.py — détection automatique de la position des buts
# V2 : détection par structure rectangulaire (2 verticales + 1 horizontale)
# Invariant au zoom, au cadrage, et à la position du but dans le frame.

import cv2
import numpy as np

_ANGLE_TOL_DEG   = 18
_MIN_VERT_LEN    = 0.08
_MIN_GOAL_WIDTH  = 0.10
_MAX_GOAL_WIDTH  = 0.65
_MIN_GOAL_HEIGHT = 0.08
_MAX_GOAL_HEIGHT = 0.55
_BAR_Y_TOL       = 0.06
_N_FRAMES        = 80


def detect_goal_box(video_path, n_frames=_N_FRAMES, fps=25):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return _fallback_goal_box(1920, 1080)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step    = max(1, total // n_frames)

        candidates = []

        for frame_idx in range(0, min(n_frames * step, total), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            result = _detect_goal_in_frame(frame, frame_w, frame_h)
            if result:
                candidates.append(result)

        cap.release()

        if not candidates:
            print("  [GOAL_BOX] Aucune structure détectée → fallback")
            return _fallback_goal_box(frame_w, frame_h)

        candidates.sort(key=lambda c: c[0], reverse=True)

        # Filtre fréquence : garder seulement les structures vues souvent
        # Un but est stable → visible sur >20% des frames
        # Une barrière en arrière-plan → apparaît de façon intermittente
        freq_threshold = max(3, len(candidates) * 0.20)
        if len(candidates) < freq_threshold:
            print(f"  [GOAL_BOX] Structure trop rare ({len(candidates)} frames) → fallback")
            return _fallback_goal_box(frame_w, frame_h)

        # Clustering : regrouper les candidats proches (même but)
        # Un but doit avoir x_left stable sur plusieurs frames
        clustered = _cluster_candidates(candidates, frame_w)
        if not clustered:
            print("  [GOAL_BOX] Pas de cluster stable → fallback")
            return _fallback_goal_box(frame_w, frame_h)

        top = clustered

        x_left   = int(np.median([c[1] for c in top]))
        x_right  = int(np.median([c[2] for c in top]))
        y_top    = int(np.median([c[3] for c in top]))
        y_bottom = int(np.median([c[4] for c in top]))
        score    = float(np.mean([c[0] for c in top]))

        goal_w = x_right - x_left
        if goal_w < frame_w * _MIN_GOAL_WIDTH or goal_w > frame_w * _MAX_GOAL_WIDTH:
            print(f"  [GOAL_BOX] Largeur incohérente ({goal_w}px) → fallback")
            return _fallback_goal_box(frame_w, frame_h)

        print(f"  [GOAL_BOX] Détecté : but_gauche x={x_left} | but_droit x={x_right}")
        print(f"  [GOAL_BOX] Poteaux détectés via vision (score={score:.2f}, n={len(top)}/{len(candidates)})")

        return {
            "frame_w":    frame_w,
            "frame_h":    frame_h,
            "method":     "vision",
            "score":      score,
            "but_gauche": x_left,
            "but_droit":  x_right,
            "but_top":    y_top,
            "but_bottom": y_bottom,
            "left":  {"x_center": x_left,  "x_min": x_left  - 30, "x_max": x_left  + 30, "y_min": y_top, "y_max": y_bottom},
            "right": {"x_center": x_right, "x_min": x_right - 30, "x_max": x_right + 30, "y_min": y_top, "y_max": y_bottom},
        }

    except Exception as e:
        print(f"  [GOAL_BOX] Erreur : {e} → fallback")
        return _fallback_goal_box(1920, 1080)


def _detect_goal_in_frame(frame, frame_w, frame_h):
    scale = 0.5
    small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
    sh, sw = small.shape[:2]

    roi_y1 = int(sh * 0.20)
    roi_y2 = int(sh * 0.80)
    roi = small[roi_y1:roi_y2, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    mask_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v)
    mask_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h)
    mask_combined = cv2.bitwise_or(mask_v, mask_h)

    edges = cv2.Canny(mask_combined, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=20,
        minLineLength=int(sh * _MIN_VERT_LEN * 0.7), maxLineGap=15,
    )

    if lines is None:
        return None

    verticals   = []
    horizontals = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1 += roi_y1
        y2 += roi_y1
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = np.sqrt(dx*dx + dy*dy)
        if length < 5:
            continue
        angle = np.degrees(np.arctan2(dy, dx + 1e-6))

        if angle > (90 - _ANGLE_TOL_DEG):
            xc = (x1 + x2) / 2 / scale
            verticals.append((xc, min(y1,y2)/scale, max(y1,y2)/scale, length/scale))
        elif angle < _ANGLE_TOL_DEG:
            yc = (y1 + y2) / 2 / scale
            horizontals.append((yc, min(x1,x2)/scale, max(x1,x2)/scale, length/scale))

    if len(verticals) < 2:
        return None

    min_w = frame_w * _MIN_GOAL_WIDTH
    max_w = frame_w * _MAX_GOAL_WIDTH
    min_h = frame_h * _MIN_GOAL_HEIGHT
    max_h = frame_h * _MAX_GOAL_HEIGHT

    best_score  = 0
    best_result = None

    for i in range(len(verticals)):
        for j in range(i + 1, len(verticals)):
            v1, v2 = verticals[i], verticals[j]
            x_left  = min(v1[0], v2[0])
            x_right = max(v1[0], v2[0])
            goal_w  = x_right - x_left

            if goal_w < min_w or goal_w > max_w:
                continue

            y_top_v  = min(v1[1], v2[1])
            y_bot_v  = max(v1[2], v2[2])
            goal_h_v = y_bot_v - y_top_v

            if goal_h_v < min_h or goal_h_v > max_h:
                continue

            # Score ratio largeur/hauteur (but standard ~3:1)
            ratio = goal_w / max(goal_h_v, 1)
            ratio_score = max(0, 1.0 - abs(ratio - 3.0) / 3.0)

            # Score longueur des poteaux
            len_score = min(1.0, (v1[3] + v2[3]) / (2 * frame_h * 0.3))

            # Score barre transversale — garder la meilleure (pas break)
            bar_score = 0.0
            bar_y     = None
            for h in horizontals:
                hy, hx1, hx2 = h[0], h[1], h[2]
                if hy > y_bot_v:
                    continue
                overlap = (min(hx2, x_right) - max(hx1, x_left)) / max(goal_w, 1)
                if overlap > 0.7 and overlap > bar_score:
                    bar_score = min(1.0, overlap)
                    bar_y = hy

            score = ratio_score * 0.3 + len_score * 0.3 + bar_score * 0.4

            if score > best_score:
                best_score  = score
                y_top_final = int(bar_y if bar_y else y_top_v)
                best_result = (score, int(x_left), int(x_right), y_top_final, int(y_bot_v))

    return best_result if best_score > 0.15 else None


def _cluster_candidates(candidates, frame_w, x_tol_pct=0.08):
    """
    Regroupe les candidats proches (même but détecté sur plusieurs frames).
    Retourne le cluster le plus fréquent et stable.
    Un but est stable → x_left varie peu entre frames.
    Une barrière → position plus variable ou cluster plus petit.
    """
    if not candidates:
        return None

    x_tol = frame_w * x_tol_pct
    clusters = []

    for cand in candidates:
        score, xl, xr, yt, yb = cand
        placed = False
        for cluster in clusters:
            # Comparer avec le premier élément du cluster
            ref_xl = cluster[0][1]
            ref_xr = cluster[0][2]
            if abs(xl - ref_xl) < x_tol and abs(xr - ref_xr) < x_tol:
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])

    if not clusters:
        return None

    # Garder le cluster le plus fréquent (>= 3 détections)
    clusters.sort(key=lambda c: len(c), reverse=True)
    best = clusters[0]

    if len(best) < 3:
        return None

    # Validation variance — le but doit être stable (faible écart-type)
    xl_std = float(np.std([c[1] for c in best]))
    xr_std = float(np.std([c[2] for c in best]))
    if xl_std > 50 or xr_std > 50:  # > 50px de variance → instable
        print(f"  [GOAL_BOX] Cluster instable (std_xl={xl_std:.0f} std_xr={xr_std:.0f}) → fallback")
        return None

    print(f"  [GOAL_BOX] Cluster principal : {len(best)} frames | "
          f"x_left≈{int(best[0][1])} x_right≈{int(best[0][2])} | "
          f"std=({xl_std:.0f},{xr_std:.0f})px")

    return best


def _fallback_goal_box(frame_w, frame_h):
    goal_h_min = int(frame_h * 0.25)
    goal_h_max = int(frame_h * 0.75)
    return {
        "frame_w":    frame_w,
        "frame_h":    frame_h,
        "method":     "fallback",
        "score":      0.0,
        "but_gauche": int(frame_w * 0.25),
        "but_droit":  int(frame_w * 0.75),
        "but_top":    goal_h_min,
        "but_bottom": goal_h_max,
        "left":  {"x_center": int(frame_w * 0.25), "x_min": int(frame_w * 0.20), "x_max": int(frame_w * 0.30), "y_min": goal_h_min, "y_max": goal_h_max},
        "right": {"x_center": int(frame_w * 0.75), "x_min": int(frame_w * 0.70), "x_max": int(frame_w * 0.80), "y_min": goal_h_min, "y_max": goal_h_max},
    }


def goal_box_to_posthoc_params(goal_box, margin_pct=0.05):
    frame_w = goal_box["frame_w"]
    frame_h = goal_box["frame_h"]
    margin  = int(frame_w * margin_pct)

    if "but_gauche" in goal_box and "but_droit" in goal_box:
        x_left  = goal_box["but_gauche"]
        x_right = goal_box["but_droit"]
    elif goal_box.get("left") and goal_box.get("right"):
        x_left  = goal_box["left"]["x_center"]
        x_right = goal_box["right"]["x_center"]
    else:
        return {
            "frame_w": frame_w, "frame_h": frame_h,
            "GOAL_X_LEFT": int(frame_w * 0.06), "GOAL_X_RIGHT": int(frame_w * 0.94),
            "DISAPPEAR_X_LEFT_MIN": 0, "DISAPPEAR_X_LEFT_MAX": int(frame_w * 0.12),
            "DISAPPEAR_X_RIGHT_MIN": int(frame_w * 0.88), "DISAPPEAR_X_RIGHT_MAX": frame_w,
            "but_gauche": int(frame_w * 0.04), "but_droit": int(frame_w * 0.96),
        }

    return {
        "frame_w":               frame_w,
        "frame_h":               frame_h,
        "GOAL_X_LEFT":           x_left  + margin,
        "GOAL_X_RIGHT":          x_right - margin,
        "DISAPPEAR_X_LEFT_MIN":  max(0,       x_left  - int(frame_w * 0.06)),
        "DISAPPEAR_X_LEFT_MAX":  min(frame_w, x_left  + int(frame_w * 0.06)),
        "DISAPPEAR_X_RIGHT_MIN": max(0,       x_right - int(frame_w * 0.06)),
        "DISAPPEAR_X_RIGHT_MAX": min(frame_w, x_right + int(frame_w * 0.06)),
        "but_gauche":            x_left,
        "but_droit":             x_right,
    }