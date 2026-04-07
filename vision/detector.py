# vision/detector.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
from vision.ball import BallDetector
import config


PLAY_ZONES = {
    "football":         {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "mini-foot":        {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "basketball":       {"x_min": 0.02, "x_max": 0.98, "y_min": 0.10, "y_max": 0.92},
    "handball":         {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "rugby":            {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "hockey sur glace": {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "hockey sur gazon": {"x_min": 0.02, "x_max": 0.98, "y_min": 0.05, "y_max": 0.95},
    "tennis":           {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
    "tennis de table":  {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
    "padel":            {"x_min": 0.05, "x_max": 0.95, "y_min": 0.10, "y_max": 0.90},
}

MIN_PLAYER_W = 0.015
MIN_PLAYER_H = 0.04
MAX_PLAYER_W = 0.25
MAX_PLAYER_H = 0.70
MIN_RATIO    = 1.2
MAX_RATIO    = 4.0

# ─────────────────────────────────────────
# MODÈLES — séparés joueurs / ballon
# yolo11m  pour les joueurs  : rapide, suffisant
# yolo11x  en fallback si m  indisponible
# ─────────────────────────────────────────
PLAYER_MODELS = [
    "yolo11m.pt",   # priorité — medium, ~3x plus rapide que x
    "yolo11l.pt",   # fallback grand
    "yolo11x.pt",   # fallback max
    "yolov8m.pt",   # fallback YOLOv8
    "yolov8n.pt",   # fallback minimal
]


def load_player_model(sport):
    """Charge le modèle joueurs — priorité yolo11m."""
    import os

    specialized = {
        "football":   "models/yolov8_football.pt",
        "mini-foot":  "models/yolov8_football.pt",
        "basketball": "models/yolov8_basketball.pt",
        "handball":   "models/yolov8_handball.pt",
    }
    sport_model = specialized.get(sport)
    if sport_model and os.path.exists(sport_model):
        print(f"  Modele specialise joueurs : {sport_model}")
        return YOLO(sport_model), sport_model

    for name in PLAYER_MODELS:
        try:
            print(f"  Chargement {name}...")
            m = YOLO(name)
            print(f"  Modele joueurs : {name}")
            return m, name
        except Exception as e:
            print(f"  {name} indisponible : {e}")

    raise RuntimeError("Aucun modèle YOLO disponible")


# ─────────────────────────────────────────
# DÉTECTEUR BALLON HSV — indépendant de YOLO
# Détecte les objets ronds et blancs/clairs
# Beaucoup plus fiable que yolo11m sur les petits objets
# ─────────────────────────────────────────
class BallHSVDetector:
    """
    Détection ballon par couleur HSV + circularité.
    Fonctionne bien pour football (ballon blanc/noir),
    basketball (ballon orange), handball (ballon jaune/orange).
    """

    # Plages HSV par sport
    HSV_RANGES = {
        "football":   [
            # blanc
            ((0,   0, 180), (180,  40, 255)),
            # noir/sombre (partie noire du ballon)
            ((0,   0,   0), (180,  50,  80)),
        ],
        "basketball": [
            # orange
            ((5,  100, 100), (25, 255, 255)),
        ],
        "handball":   [
            # jaune/orange
            ((15,  80, 100), (35, 255, 255)),
        ],
        "default":    [
            ((0,   0, 160), (180,  50, 255)),   # blanc/clair
            ((5,  100, 100), (25, 255, 255)),   # orange
        ],
    }

    def __init__(self, sport="football"):
        self.sport  = sport
        self.ranges = self.HSV_RANGES.get(sport, self.HSV_RANGES["default"])

    def detect(self, frame, last_pos=None, search_radius=200):
        """
        Cherche le ballon dans la frame.
        Si last_pos connu, cherche dans un rayon réduit (plus rapide).
        Retourne dict {bbox, center, conf} ou None.
        """
        h, w = frame.shape[:2]

        # Zone de recherche réduite si position précédente connue
        if last_pos is not None:
            h_f, w_f = frame.shape[:2]
            cx, cy  = last_pos
            x1 = max(0, int(cx - search_radius))
            y1 = max(0, int(cy - search_radius))
            x2 = min(w_f, int(cx + search_radius))
            y2 = min(h_f, int(cy + search_radius))
            # FIX — si le crop est vide (last_pos hors frame) → fallback frame entière
            if x2 <= x1 or y2 <= y1:
                roi    = frame
                offset = (0, 0)
            else:
                roi    = frame[y1:y2, x1:x2]
                offset = (x1, y1)
        else:
            roi    = frame
            offset = (0, 0)

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lo, hi) in self.ranges:
            m    = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.bitwise_or(mask, m)

        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best      = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Taille min/max ballon en pixels (adapté résolution 1080p)
            if area < 30 or area > 3000:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue

            # Circularité : 1.0 = cercle parfait
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < 0.5:   # filtre les formes non rondes
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Rapport largeur/hauteur proche de 1 (carré = rond)
            if bw == 0 or bh == 0:
                continue
            ratio = bw / bh
            if ratio < 0.5 or ratio > 2.0:
                continue

            # Score = circularité × taille normalisée
            score = circularity * min(area / 200, 1.0)

            # Bonus si proche de la dernière position
            if last_pos is not None:
                cx_b = x + bw // 2 + offset[0]
                cy_b = y + bh // 2 + offset[1]
                dist = np.hypot(cx_b - last_pos[0], cy_b - last_pos[1])
                score += max(0, 1.0 - dist / search_radius) * 0.5

            if score > best_score:
                best_score = score
                best       = (x, y, bw, bh, offset)

        if best is None:
            return None

        x, y, bw, bh, (ox, oy) = best
        cx = x + bw // 2 + ox
        cy = y + bh // 2 + oy
        x1b = x + ox
        y1b = y + oy

        return {
            "bbox":   [x1b, y1b, x1b + bw, y1b + bh],
            "center": [cx, cy],
            "conf":   round(best_score, 2),
            "method": "hsv"
        }


# ─────────────────────────────────────────
# DÉTECTEUR PRINCIPAL
# ─────────────────────────────────────────
class Detector:

    def __init__(self, sport="football"):
        self.sport        = sport
        self.zone         = PLAY_ZONES.get(sport, PLAY_ZONES["football"])
        self.model, self.model_name = load_player_model(sport)

        # Détecteur ballon — HSV en priorité + BallDetector en fallback
        self.hsv_ball    = BallHSVDetector(sport=sport)
        self.ball_backup = BallDetector(method=config.BALL_METHOD)
        self._last_ball_pos = None   # mémorise dernière position ballon

        self.player_cls = 0    # COCO : person
        self.ball_cls   = 32   # COCO : sports ball

        print(f"  Detector pret : {self.model_name} | sport={sport}")

    def set_sport(self, sport):
        if sport == self.sport:
            return
        self.sport    = sport
        self.zone     = PLAY_ZONES.get(sport, PLAY_ZONES["football"])
        self.hsv_ball = BallHSVDetector(sport=sport)
        new_model, new_name = load_player_model(sport)
        if new_name != self.model_name:
            self.model      = new_model
            self.model_name = new_name
            print(f"  Modele mis a jour : {new_name}")

    def _in_play_zone(self, center, frame_w, frame_h):
        cx, cy = center
        z      = self.zone
        return (z["x_min"] * frame_w <= cx <= z["x_max"] * frame_w and
                z["y_min"] * frame_h <= cy <= z["y_max"] * frame_h)

    def _valid_size(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:            return False
        if w < MIN_PLAYER_W * frame_w:  return False
        if h < MIN_PLAYER_H * frame_h:  return False
        if w > MAX_PLAYER_W * frame_w:  return False
        if h > MAX_PLAYER_H * frame_h:  return False
        ratio = h / w
        return MIN_RATIO <= ratio <= MAX_RATIO

    def _detect_ball(self, frame, yolo_ball, last_pos_override=None):
        """
        Stratégie ballon à 3 niveaux :
        1. YOLO a trouvé le ballon → on l'utilise + on met à jour last_pos
        2. HSV trouve quelque chose de circulaire → on l'utilise
        3. Fallback BallDetector (méthode config)

        last_pos_override : position en coordonnées de `frame` (pas originales)
                            Utilisé quand frame est une version réduite.
        """
        # 1. YOLO
        if yolo_ball is not None:
            self._last_ball_pos = yolo_ball["center"]
            return yolo_ball

        # Position de recherche HSV — override si fourni (frame réduite)
        search_pos = last_pos_override if last_pos_override is not None \
                     else self._last_ball_pos

        # 2. HSV avec recherche guidée
        hsv_result = self.hsv_ball.detect(
            frame,
            last_pos      = search_pos,
            search_radius = 250
        )
        if hsv_result is not None:
            self._last_ball_pos = hsv_result["center"]
            return hsv_result

        # 3. Fallback BallDetector
        fallback = self.ball_backup.get_position(frame, None)
        if fallback is not None:
            self._last_ball_pos = fallback.get("center")
        return fallback

    def detect(self, frame):
        """
        Détecte joueurs et ballon sur une frame.
        Utilisé hors batch (calibration, sport_detector...).
        Retourne :
            players : list de dicts {bbox, center, conf}
            ball    : dict {bbox, center, conf} ou None
        """
        h_frame, w_frame = frame.shape[:2]

        results = self.model(
            frame,
            conf    = config.YOLO_CONFIDENCE,
            verbose = False,
            imgsz   = 960    # FIX: 1280 → 960 cohérent avec batch
        )[0]

        players   = []
        yolo_ball = None

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox   = [x1, y1, x2, y2]

            if cls == self.player_cls:
                if not self._in_play_zone(center, w_frame, h_frame):
                    continue
                if not self._valid_size(bbox, w_frame, h_frame):
                    continue
                players.append({
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                })
            elif cls == self.ball_cls:
                yolo_ball = {
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                }

        ball = self._detect_ball(frame, yolo_ball)
        return players, ball