# vision/detector.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
from vision.ball import BallDetector
from vision.model_manager import manager, MODEL_CLASSES, COCO_CLASSES
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

# Filtres plus stricts
MIN_PLAYER_W = 0.015  # était 0.01
MIN_PLAYER_H = 0.04   # était 0.02
MAX_PLAYER_W = 0.25   # était 0.35
MAX_PLAYER_H = 0.70   # était 0.80
MIN_RATIO    = 1.2    # était 0.8 — joueur plus haut que large
MAX_RATIO    = 4.0    # était 5.0

# Modèles YOLO11 disponibles — du plus léger au plus lourd
YOLO11_MODELS = [
    "yolo11x.pt",   # meilleur — 110MB
    "yolo11l.pt",   # grand    — 50MB
    "yolo11m.pt",   # medium   — 20MB
    "yolov8x.pt",   # fallback YOLOv8 grand
    "yolov8n.pt",   # fallback minimal
]


def load_best_model(sport):
    """
    Charge le meilleur modèle disponible.
    Priorité : modèle spécialisé sport > YOLO11x > fallback
    """
    # 1. Modèle spécialisé sport
    specialized = {
        "football":   "models/yolov8_football.pt",
        "mini-foot":  "models/yolov8_football.pt",
        "basketball": "models/yolov8_basketball.pt",
        "handball":   "models/yolov8_handball.pt",
    }

    import os
    sport_model = specialized.get(sport)
    if sport_model and os.path.exists(sport_model):
        print(f"  Modele specialise : {sport_model}")
        return YOLO(sport_model), "specialized"

    # 2. Meilleur YOLO11 disponible
    for model_name in YOLO11_MODELS:
        try:
            print(f"  Chargement {model_name}...")
            model = YOLO(model_name)
            print(f"  Modele charge : {model_name}")
            return model, model_name
        except Exception as e:
            print(f"  {model_name} non disponible : {e}")
            continue

    # 3. Fallback minimal
    print("  Fallback : yolov8n.pt")
    return YOLO("yolov8n.pt"), "yolov8n.pt"


class Detector:

    def __init__(self, sport="football"):
        self.sport         = sport
        self.zone          = PLAY_ZONES.get(sport, PLAY_ZONES["football"])
        self.ball_detector = BallDetector(method=config.BALL_METHOD)
        self.model, self.model_name = load_best_model(sport)

        # Classes selon modèle
        self.player_cls  = 0   # COCO : person
        self.ball_cls    = 32  # COCO : sports ball
        self.referee_cls = -1  # pas de classe arbitre sur modèle COCO

        print(f"  Detector pret : {self.model_name} | sport={sport}")

    def set_sport(self, sport):
        if sport == self.sport:
            return

        self.sport = sport
        self.zone  = PLAY_ZONES.get(sport, PLAY_ZONES["football"])

        # Recharger si modèle spécialisé disponible
        new_model, new_name = load_best_model(sport)
        if new_name != self.model_name:
            self.model      = new_model
            self.model_name = new_name
            print(f"  Modele mis a jour : {new_name}")

    def _in_play_zone(self, center, frame_w, frame_h):
        cx, cy = center
        z      = self.zone
        x_ok   = (z["x_min"] * frame_w) <= cx <= (z["x_max"] * frame_w)
        y_ok   = (z["y_min"] * frame_h) <= cy <= (z["y_max"] * frame_h)
        return x_ok and y_ok

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
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            return False

        return True

    def detect(self, frame):
        """
        Détecte joueurs et ballon sur une frame.

        Retourne :
            players  : list de dicts {bbox, center, conf}
            ball     : dict {bbox, center, conf} ou None
        """
        h_frame, w_frame = frame.shape[:2]

        results = self.model(
            frame,
            conf    = config.YOLO_CONFIDENCE,
            verbose = False,
            imgsz   = 1280    # résolution maximale pour YOLO11
        )[0]

        players   = []
        yolo_ball = None

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox   = [x1, y1, x2, y2]

            # ── Joueurs ─────────────────────
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

            # ── Ballon ──────────────────────
            elif cls == self.ball_cls:
                yolo_ball = {
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                }

        # Détection ballon hybride
        ball = self.ball_detector.get_position(frame, yolo_ball)

        return players, ball