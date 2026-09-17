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
    # Plage élargie pour terrains hivernaux/boueux (olive, kaki, vert terne)
    mask_green_bright = cv2.inRange(hsv,
        np.array([30,  30,  30]),
        np.array([90, 255, 255])
    )
    # Vert désaturé / olive hivernal : saturation faible, teinte 25-95
    mask_green_dull = cv2.inRange(hsv,
        np.array([25,  10,  30]),
        np.array([95,  80, 200])
    )
    mask_green = cv2.bitwise_or(mask_green_bright, mask_green_dull)

    # Marron/beige → parquet (basketball, handball)
    # Seuils resserrés pour éviter confusion avec gazon boueux
    mask_wood = cv2.inRange(hsv,
        np.array([10,  40,  80]),
        np.array([25, 160, 220])
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
        mask_bright = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
        mask_dull   = cv2.inRange(hsv, np.array([25, 10, 30]), np.array([95,  80, 200]))
        mask = cv2.bitwise_or(mask_bright, mask_dull)
    elif pitch_color == "wood":
        mask = cv2.inRange(hsv,
            np.array([10,  40,  80]),
            np.array([25, 160, 220])
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
    Analyse des frames représentatives du match et retourne
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

    # V5.2 (17/09/2026) FIX : échantillonnait auparavant les 10 TOUTES
    # PREMIÈRES frames de la vidéo (t=0), sans jamais avancer dans le
    # temps - potentiellement pré-match, terrain vide, échauffement,
    # écran noir/logo d'intro. Ce shot_zones calibré ici est ensuite
    # utilisé EN DIRECT pour tout le match (detect_events(), voir
    # pipeline.py) - contrairement au systeme plus robuste
    # (vision/camera_profile.py, base sur 50+ positions ballon reelles
    # accumulees), qui lui n'intervient qu'en post-traitement (BC4,
    # goal_posthoc), trop tard pour influencer cette calibration
    # initiale. Ne CONNAIT PAS encore l'heure exacte de KO1 a ce stade
    # (frames_data pas encore construit) - heuristique : avancer a
    # t=90s (raisonnablement au-dela d'un intro/echauffement typique,
    # sans supposer un KO1 precis), et etaler les 10 echantillons sur
    # une fenetre de 10s plutot que des frames consecutives (~0,33s a
    # 30fps) - plus robuste a un instant malchanceux isole (ballon hors
    # cadre, joueurs groupes de façon inhabituelle).
    fps_natif = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _t_debut_s = 90.0
    frame_debut = int(_t_debut_s * fps_natif)
    frame_fin   = int((_t_debut_s + 10.0) * fps_natif)
    frame_fin   = min(frame_fin, total_frames - 1)

    frames_sample = []
    if frame_debut < total_frames:
        indices = [int(x) for x in
                   (frame_debut + i * (frame_fin - frame_debut) / 9
                    for i in range(10))]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_sample.append(frame)
    else:
        # Vidéo plus courte que 90s (rare, mais possible sur un
        # extrait court) - repli sur le comportement d'origine.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frames_sample.append(frame)

    cap.release()

    if not frames_sample:
        raise ValueError("Impossible de lire les frames")

    # Utiliser la frame du milieu de l'échantillon
    frame = frames_sample[len(frames_sample) // 2]

    print("\n  Calibration en cours...")

    # 1. Angle caméra
    camera_angle = detect_camera_angle(frame)
    print(f"  Angle camera : {camera_angle}")

    # 2. Couleur terrain
    pitch_color, _ = detect_pitch_color(frame)

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