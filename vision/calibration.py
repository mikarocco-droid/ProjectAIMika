# vision/calibration.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np


# ─────────────────────────────────────────
# DÉTECTION ANGLE CAMÉRA
# ─────────────────────────────────────────
def detect_camera_angle(frame):
    """
    Analyse la première frame pour déterminer
    l'angle de prise de vue.

    Retourne :
        "side"    → caméra de côté (vue latérale)
        "corner"  → caméra en coin
        "top"     → caméra en hauteur (drone/nacelle)
        "behind"  → caméra derrière un but
    """
    h, w = frame.shape[:2]

    # Détecter les lignes du terrain
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blur, 50, 150)
    lines   = cv2.HoughLinesP(
        edges,
        rho         = 1,
        theta       = np.pi / 180,
        threshold   = 80,
        minLineLength = int(w * 0.15),
        maxLineGap  = 20
    )

    if lines is None:
        return "side"  # fallback

    # Analyser les angles des lignes détectées
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            angles.append(angle)

    if not angles:
        return "side"

    angles      = np.array(angles)
    horiz_lines = np.sum((angles < 15) | (angles > 165))
    vert_lines  = np.sum((angles > 75) & (angles < 105))
    diag_lines  = len(angles) - horiz_lines - vert_lines

    # Ratio lignes horizontales vs diagonales
    total = len(angles)
    horiz_ratio = horiz_lines / total
    diag_ratio  = diag_lines  / total

    if horiz_ratio > 0.5:
        return "side"
    elif diag_ratio > 0.5:
        return "corner"
    elif vert_lines > horiz_lines:
        return "behind"
    else:
        return "top"


# ─────────────────────────────────────────
# DÉTECTION COULEUR DU TERRAIN
# ─────────────────────────────────────────
def detect_pitch_color(frame):
    """
    Détecte la couleur dominante du terrain
    pour distinguer gazon, parquet, glace, etc.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Vert → gazon (football, rugby, hockey gazon)
    mask_green = cv2.inRange(hsv,
        np.array([30,  30,  30]),
        np.array([90, 255, 255])
    )

    # Marron/beige → parquet (basketball, handball)
    mask_wood = cv2.inRange(hsv,
        np.array([10,  20,  80]),
        np.array([30, 120, 220])
    )

    # Blanc/gris → glace (hockey glace)
    mask_ice = cv2.inRange(hsv,
        np.array([0,   0, 180]),
        np.array([180, 30, 255])
    )

    # Bleu → court (tennis, padel)
    mask_blue = cv2.inRange(hsv,
        np.array([90,  50,  50]),
        np.array([130, 255, 255])
    )

    total_px  = frame.shape[0] * frame.shape[1]
    green_pct = np.sum(mask_green > 0) / total_px
    wood_pct  = np.sum(mask_wood  > 0) / total_px
    ice_pct   = np.sum(mask_ice   > 0) / total_px
    blue_pct  = np.sum(mask_blue  > 0) / total_px

    scores = {
        "green": green_pct,
        "wood":  wood_pct,
        "ice":   ice_pct,
        "blue":  blue_pct
    }

    dominant = max(scores, key=scores.get)

    print(f"  Couleur terrain : {dominant} "
          f"(vert={green_pct:.1%} bois={wood_pct:.1%} "
          f"glace={ice_pct:.1%} bleu={blue_pct:.1%})")

    return dominant, scores


# ─────────────────────────────────────────
# DÉTECTION ZONE DE JEU
# ─────────────────────────────────────────
def detect_play_zone(frame, pitch_color):
    """
    Détecte automatiquement la zone de jeu
    en isolant la couleur du terrain.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_frame, w_frame = frame.shape[:2]

    # Masque selon couleur détectée
    if pitch_color == "green":
        mask = cv2.inRange(hsv,
            np.array([30,  30,  30]),
            np.array([90, 255, 255])
        )
    elif pitch_color == "wood":
        mask = cv2.inRange(hsv,
            np.array([10,  20,  80]),
            np.array([30, 120, 220])
        )
    elif pitch_color == "ice":
        mask = cv2.inRange(hsv,
            np.array([0,   0, 180]),
            np.array([180, 30, 255])
        )
    elif pitch_color == "blue":
        mask = cv2.inRange(hsv,
            np.array([90,  50,  50]),
            np.array([130, 255, 255])
        )
    else:
        # Fallback — zone large
        return {
            "x_min": 0.02, "x_max": 0.98,
            "y_min": 0.05, "y_max": 0.95
        }

    # Nettoyage morphologique
    kernel = np.ones((20, 20), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Trouver le plus grand contour (= le terrain)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "x_min": 0.02, "x_max": 0.98,
            "y_min": 0.05, "y_max": 0.95
        }

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Convertir en ratios + marge de sécurité
    margin = 0.02
    zone = {
        "x_min": max(0.0,  (x / w_frame)           - margin),
        "x_max": min(1.0,  ((x + w) / w_frame)     + margin),
        "y_min": max(0.0,  (y / h_frame)            - margin),
        "y_max": min(1.0,  ((y + h) / h_frame)      + margin),
    }

    print(f"  Zone detectee : x=[{zone['x_min']:.2f}, {zone['x_max']:.2f}] "
          f"y=[{zone['y_min']:.2f}, {zone['y_max']:.2f}]")

    return zone


# ─────────────────────────────────────────
# ZONES DE TIR PAR SPORT ET ANGLE
# ─────────────────────────────────────────
def compute_shot_zones(sport, camera_angle, play_zone):
    """
    Calcule les zones de tir dynamiquement
    selon le sport, l'angle de caméra et la zone de jeu.

    Retourne un dict de seuils pour detect_events_v5.
    """
    x_min = play_zone["x_min"]
    x_max = play_zone["x_max"]
    y_min = play_zone["y_min"]
    y_max = play_zone["y_max"]

    # Largeur et hauteur effectives de la zone de jeu (en ratio)
    pw = x_max - x_min
    ph = y_max - y_min

    if sport in ["football", "mini-foot"]:
        if camera_angle == "side":
            # But à droite et à gauche — 15% de chaque côté
            return {
                "axis":         "x",
                "threshold_hi": x_min + pw * 0.85,  # zone but droite
                "threshold_lo": x_min + pw * 0.15,  # zone but gauche
                "y_min":        y_min + ph * 0.25,
                "y_max":        y_min + ph * 0.75,
            }
        elif camera_angle == "corner":
            return {
                "axis":         "x",
                "threshold_hi": x_min + pw * 0.80,
                "threshold_lo": x_min + pw * 0.20,
                "y_min":        y_min + ph * 0.20,
                "y_max":        y_min + ph * 0.80,
            }
        else:
            return {
                "axis":         "x",
                "threshold_hi": x_min + pw * 0.85,
                "threshold_lo": x_min + pw * 0.15,
                "y_min":        y_min,
                "y_max":        y_max,
            }

    elif sport == "basketball":
        if camera_angle == "side":
            # Paniers à gauche et à droite
            return {
                "axis":         "x",
                "threshold_hi": x_min + pw * 0.88,  # panier droit
                "threshold_lo": x_min + pw * 0.12,  # panier gauche
                "y_min":        y_min + ph * 0.30,
                "y_max":        y_min + ph * 0.70,
            }
        else:
            return {
                "axis":         "y",
                "threshold_hi": y_min + ph * 0.20,  # panier haut
                "threshold_lo": y_min + ph * 0.80,  # panier bas
                "y_min":        y_min,
                "y_max":        y_max,
            }

    elif sport == "handball":
        return {
            "axis":         "x",
            "threshold_hi": x_min + pw * 0.82,
            "threshold_lo": x_min + pw * 0.18,
            "y_min":        y_min + ph * 0.20,
            "y_max":        y_min + ph * 0.80,
        }

    elif sport == "rugby":
        return {
            "axis":         "x",
            "threshold_hi": x_min + pw * 0.90,
            "threshold_lo": x_min + pw * 0.10,
            "y_min":        y_min,
            "y_max":        y_max,
        }

    else:
        # Fallback générique
        return {
            "axis":         "x",
            "threshold_hi": x_min + pw * 0.85,
            "threshold_lo": x_min + pw * 0.15,
            "y_min":        y_min,
            "y_max":        y_max,
        }


# ─────────────────────────────────────────
# CALIBRATION COMPLÈTE
# ─────────────────────────────────────────
def calibrate(video_path, sport):
    """
    Point d'entrée principal.
    Analyse les premières frames et retourne
    la configuration calibrée.

    Retourne :
        dict {
            "camera_angle",
            "pitch_color",
            "play_zone",
            "shot_zones"
        }
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir : {video_path}")

    # Analyser 30 frames réparties sur les 60 premières secondes
    # Évite les frames atypiques (boue, ombre, joueur en gros plan)
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_end = min(int(fps_cap * 60), total_frames)  # 60s max
    step = max(1, sample_end // 30)

    frames_sample = []
    for i in range(0, sample_end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames_sample.append(frame)
        if len(frames_sample) >= 30:
            break

    cap.release()

    if not frames_sample:
        raise ValueError("Impossible de lire les frames")

    # Utiliser la frame du milieu de l'échantillon
    frame = frames_sample[len(frames_sample) // 2]

    print("\n  Calibration en cours...")

    # 1. Angle caméra
    camera_angle = detect_camera_angle(frame)
    print(f"  Angle camera : {camera_angle}")

    # 2. Couleur terrain — médiane sur plusieurs frames
    # Plus robuste que frame unique sur terrains boueux/hivernaux
    color_scores_all = {"green": [], "wood": [], "ice": [], "blue": []}
    for f in frames_sample:
        _, scores = detect_pitch_color(f)
        for k, v in scores.items():
            color_scores_all[k].append(v)
    # Médiane de chaque couleur → plus robuste aux outliers
    median_scores = {k: float(np.median(v)) for k, v in color_scores_all.items()}
    pitch_color = max(median_scores, key=median_scores.get)
    print(f"  Couleur terrain (médiane) : {pitch_color} "
          f"(vert={median_scores['green']:.1%} bois={median_scores['wood']:.1%} "
          f"glace={median_scores['ice']:.1%} bleu={median_scores['blue']:.1%})")

    # 3. Zone de jeu
    play_zone = detect_play_zone(frame, pitch_color)

    # 4. Zones de tir
    shot_zones = compute_shot_zones(sport, camera_angle, play_zone)

    result = {
        "camera_angle": camera_angle,
        "pitch_color":  pitch_color,
        "play_zone":    play_zone,
        "shot_zones":   shot_zones
    }

    print(f"  Calibration OK : {camera_angle} | {pitch_color}")
    print(f"  Shot zones : {shot_zones}")

    return result