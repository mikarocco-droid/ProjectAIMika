# vision/ball.py

import numpy as np
import cv2


class BallDetector:
    """
    Détection du ballon en complément de YOLO.
    YOLO (classe 32) rate souvent le ballon en mouvement rapide —
    ce module prend le relais avec une détection par couleur + forme.
    """

    def __init__(self, method="hybrid"):
        """
        method :
            "yolo"   → uniquement résultat YOLO passé en paramètre
            "color"  → détection HSV (ballon blanc/jaune)
            "hybrid" → YOLO en priorité, fallback color si non détecté
        """
        self.method = method
        self.last_known = None  # dernière position connue

    # ─────────────────────────────────────────
    # DÉTECTION PAR COULEUR (HSV)
    # ─────────────────────────────────────────
    def _detect_by_color(self, frame):
        """
        Cherche un objet rond de couleur blanche ou jaune
        correspondant à un ballon.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masque ballon blanc
        mask_white = cv2.inRange(hsv,
            np.array([0,   0,   200]),
            np.array([180, 40,  255])
        )

        # Masque ballon jaune (ex: futsal, basket)
        mask_yellow = cv2.inRange(hsv,
            np.array([20,  100, 100]),
            np.array([35,  255, 255])
        )

        mask = cv2.bitwise_or(mask_white, mask_yellow)

        # Nettoyage morphologique
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filtrer trop petit ou trop grand
            if area < 80 or area > 8000:
                continue

            # Circularité (1.0 = cercle parfait)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < 0.55:  # pas assez rond
                continue

            score = circularity * area

            if score > best_score:
                best_score = score
                best = cnt

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
        """
        Point d'entrée principal.

        frame     : image BGR (numpy array)
        yolo_ball : résultat YOLO existant (dict ou None)

        Retourne un dict ballon ou None.
        """

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

        # Mise à jour dernière position connue
        if ball:
            self.last_known = ball
        
        return ball

    # ─────────────────────────────────────────
    # INTERPOLATION (ballon perdu quelques frames)
    # ─────────────────────────────────────────
    def get_position(self, frame, yolo_ball=None, max_lost=10):
        """
        Comme detect() mais retourne la dernière position connue
        si le ballon est temporairement perdu (évite les trous
        dans la timeline d'events).
        """
        ball = self.detect(frame, yolo_ball)

        if ball is None and self.last_known is not None:
            ball = dict(self.last_known)
            ball["interpolated"] = True

        return ball