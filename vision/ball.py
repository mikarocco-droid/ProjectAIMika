# vision/ball.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class BallDetector:
    """
    Détection du ballon en complément de YOLO.
    YOLO (classe 32) rate souvent le ballon en mouvement rapide —
    ce module prend le relais avec une détection par couleur + forme.
    """

    # V5.2 (17/09/2026) FIX : MAX_LOST=8 etait un nombre de frames fixe
    # ("~0.3s a 25fps" suppose), jamais mis a l'echelle du fps effectif
    # reel (frame_skip reduit le rythme d'appel) - meme motif systemique
    # deja corrige ailleurs ce soir (ball_tracker.py max_lost,
    # player_reid.py TTL). Ce detecteur est un fallback tertiaire
    # (rarement sollicite - utilise seulement si YOLO ET HSV echouent
    # tous les deux), impact probablement faible, mais corrige par
    # coherence. self.fps doit etre fixe par l'appelant apres
    # construction si le fps effectif reel differe de la reference.
    _MAX_LOST_BASE = 8       # a 25fps de reference : ~0.3s
    _FPS_REFERENCE = 25.0

    def __init__(self, method="hybrid", fps=25.0):
        self.method     = method
        self.last_known = None
        self.lost_count = 0   # frames consécutives sans détection réelle
        self.fps        = fps

    @property
    def MAX_LOST(self):
        return max(1, round(self._MAX_LOST_BASE * self._FPS_REFERENCE / max(1.0, self.fps)))

    # ─────────────────────────────────────────
    # DÉTECTION PAR COULEUR (HSV)
    # ─────────────────────────────────────────
    def _detect_by_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_white = cv2.inRange(hsv,
            np.array([0,   0,   200]),
            np.array([180,  40, 255])
        )
        mask_yellow = cv2.inRange(hsv,
            np.array([20,  100, 100]),
            np.array([35,  255, 255])
        )
        mask = cv2.bitwise_or(mask_white, mask_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best       = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 8000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.55:
                continue

            score = circularity * area
            if score > best_score:
                best_score = score
                best       = cnt

        if best is None:
            return None

        x, y, w, h = cv2.boundingRect(best)
        cx = x + w // 2
        cy = y + h // 2

        return {
            "bbox":   [x, y, x + w, y + h],
            "center": [cx, cy],
            "conf":   round(best_score / 10000, 3),
            "method": "color"
        }

    # ─────────────────────────────────────────
    # DÉTECTION HYBRIDE
    # ─────────────────────────────────────────
    def detect(self, frame, yolo_ball=None):
        ball = None

        if self.method == "yolo":
            ball = yolo_ball

        elif self.method == "color":
            ball = self._detect_by_color(frame)

        elif self.method == "hybrid":
            if yolo_ball:
                ball = yolo_ball
                ball["method"] = "yolo"
            else:
                ball = self._detect_by_color(frame)

        if ball:
            self.last_known = ball
            self.lost_count = 0
        else:
            self.lost_count += 1

        return ball

    # ─────────────────────────────────────────
    # INTERPOLATION LIMITÉE
    # FIX — retourne None après MAX_LOST frames sans détection
    # pour que ball_speed soit recalculé correctement dès
    # que le ballon réapparaît
    # ─────────────────────────────────────────
    def get_position(self, frame, yolo_ball=None):
        ball = self.detect(frame, yolo_ball)

        if ball is None and self.last_known is not None:
            if self.lost_count <= self.MAX_LOST:
                # Interpolation courte — position connue récente
                ball = dict(self.last_known)
                ball["interpolated"] = True
            else:
                # Trop longtemps perdu → on coupe l'interpolation
                # events.py recevra None et ne calculera pas de tir
                ball = None

        return ball

    def reset(self):
        self.last_known = None
        self.lost_count = 0