# vision/ball_tracker.py
# -*- coding: utf-8 -*-

import math
from collections import deque


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────
def distance(p1, p2):
    if p1 is None or p2 is None:
        return 9999
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def smooth_position(pos, alpha=0.7):
    """
    Lissage exponentiel
    """
    if pos is None:
        return None

    if isinstance(pos, tuple):
        return pos

    # historique (liste)
    if len(pos) < 2:
        return pos[-1]

    x = 0
    y = 0
    total = 0

    for i, (px, py) in enumerate(reversed(pos)):
        w = alpha ** i
        x += px * w
        y += py * w
        total += w

    return (int(x / total), int(y / total))


def is_valid_ball(x, y, frame_w=1920, frame_h=1080):
    """
    filtre positions absurdes
    """
    if x < 0 or y < 0:
        return False
    if x > frame_w or y > frame_h:
        return False
    return True


# ─────────────────────────────────────────
# BALL TRACKER V12
# ─────────────────────────────────────────
class BallTracker:
    def __init__(self, max_history=30):
        self.history = deque(maxlen=max_history)
        self.frames  = deque(maxlen=max_history)

        self.last_position = None
        self.velocity      = (0, 0)

        self.lost_frames   = 0
        self.max_lost      = 15  # tolérance perte balle

    # ─────────────────────────────────────
    # UPDATE (détection YOLO)
    # ─────────────────────────────────────
    def update(self, position, frame_id):
        """
        position = (x, y)
        """

        if position is None:
            return

        if self.last_position:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]

            # 🔥 filtrage saut improbable
            if abs(dx) > 200 or abs(dy) > 200:
                return

            self.velocity = (dx, dy)

        self.last_position = position

        self.history.append(position)
        self.frames.append(frame_id)

        self.lost_frames = 0

    # ─────────────────────────────────────
    # PREDICTION (si balle perdue)
    # ─────────────────────────────────────
    def predict(self, frame_id):
        if self.last_position is None:
            return None

        self.lost_frames += 1

        # trop longtemps perdu → reset
        if self.lost_frames > self.max_lost:
            self.reset()
            return None

        # 🔥 prédiction simple + inertie
        vx, vy = self.velocity

        # amortissement (évite explosion)
        vx *= 0.9
        vy *= 0.9

        px = int(self.last_position[0] + vx)
        py = int(self.last_position[1] + vy)

        self.last_position = (px, py)

        self.history.append(self.last_position)
        self.frames.append(frame_id)

        self.velocity = (vx, vy)

        return self.last_position

    # ─────────────────────────────────────
    # POSITION ACTUELLE
    # ─────────────────────────────────────
    def get_position(self):
        if not self.history:
            return None
        return list(self.history)

    # ─────────────────────────────────────
    # VITESSE
    # ─────────────────────────────────────
    def get_speed(self):
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    # ─────────────────────────────────────
    # DETECT SHOT (tir)
    # ─────────────────────────────────────
    def is_fast_movement(self, threshold=25):
        """
        détecte tir / passe rapide
        """
        return self.get_speed() > threshold

    # ─────────────────────────────────────
    # RESET
    # ─────────────────────────────────────
    def reset(self):
        self.history.clear()
        self.frames.clear()
        self.last_position = None
        self.velocity = (0, 0)
        self.lost_frames = 0