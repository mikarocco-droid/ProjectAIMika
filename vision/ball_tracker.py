# vision/ball_tracker.py
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def is_valid_ball(ball, frame_w, frame_h):
    if ball is None:
        return False
    x, y, w, h = ball
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return False
    if w <= 2 or h <= 2:
        return False
    if w > frame_w * 0.2:
        return False
    return True


class SimpleKalman:
    """
    Kalman simplifié avec réactivité améliorée.
    FIX — coefficient velocity 0.6 → 0.9 pour mieux suivre
    les ballons rapides sans trop lisser.
    """
    def __init__(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])

    def update(self, measurement):
        if measurement is None:
            if self.state is not None:
                # FIX — on applique la vélocité mais on la réduit
                # progressivement pour éviter la dérive
                self.state    = self.state + self.velocity
                self.velocity = self.velocity * 0.7
            return self.state

        m = np.array(measurement, dtype=float)
        if self.state is None:
            self.state    = m
            self.velocity = np.array([0.0, 0.0])
            return self.state

        # FIX — coefficient 0.6 → 0.9 : suit mieux les mouvements rapides
        self.velocity = (m - self.state) * 0.9
        self.state    = m   # on prend la mesure directement (pas de lissage)
        return self.state

    def reset(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])


class BallTracker:
    """
    FIX — max_lost réduit de 15 → 5 frames
    Au-delà de 5 frames sans détection, on retourne None
    plutôt qu'une position Kalman extrapolée.
    Ça permet à events.py de calculer une vraie ball_speed
    dès que le ballon réapparaît.
    """

    def __init__(self, max_history=30):
        self.history     = deque(maxlen=max_history)
        self.kalman      = SimpleKalman()
        self.last_seen   = 0
        self.frame_id    = 0
        self.lost_frames = 0
        # FIX : 15 → 5 frames max d'interpolation (~0.2s à 25fps)
        self.max_lost    = 5

    def select_best_ball(self, balls, last_pos):
        if not balls:
            return None
        if last_pos is None:
            return balls[0]
        best       = None
        best_score = 1e9
        for b in balls:
            x, y, w, h = b
            cx = x + w // 2
            cy = y + h // 2
            d            = distance((cx, cy), last_pos)
            size_penalty = w * h * 0.001
            score        = d + size_penalty
            if score < best_score:
                best_score = score
                best       = b
        return best

    def update(self, detected_balls, frame_w, frame_h):
        self.frame_id += 1

        detected_balls = [
            b for b in detected_balls
            if is_valid_ball(b, frame_w, frame_h)
        ]

        last_pos = self.history[-1] if self.history else None
        best     = self.select_best_ball(detected_balls, last_pos)

        if best is not None:
            x, y, w, h = best
            cx = x + w // 2
            cy = y + h // 2
            self.history.append((cx, cy))
            self.last_seen   = self.frame_id
            self.lost_frames = 0
            pos = self.kalman.update((cx, cy))
            return self.get_ball_bbox(pos), False   # interpolated=False

        else:
            self.lost_frames += 1
            if self.lost_frames < self.max_lost:
                # Interpolation courte — Kalman prédit
                pos = self.kalman.update(None)
                if pos is not None:
                    self.history.append(tuple(pos.astype(int)))
                    return self.get_ball_bbox(pos), True  # interpolated=True
            else:
                # Trop longtemps perdu → on coupe
                self.history.clear()
                self.kalman.reset()

            return None, True  # pas de position fiable

    def get_ball_bbox(self, pos):
        if pos is None:
            return None
        x, y = int(pos[0]), int(pos[1])
        size = 10
        return (x - size, y - size, size * 2, size * 2)

    def get_trajectory(self):
        return list(self.history)

    def get_speed(self):
        if len(self.history) < 2:
            return 0.0
        return distance(self.history[-1], self.history[-2])

    def closest_player(self, players):
        if not self.history:
            return None
        ball_pos  = self.history[-1]
        best      = None
        best_dist = 9999
        for p in players:
            x1, y1, x2, y2 = p["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            d  = distance((cx, cy), ball_pos)
            if d < best_dist:
                best_dist = d
                best      = p
        if best_dist < 80:
            return best
        return None

    def reset(self):
        self.history.clear()
        self.kalman.reset()
        self.last_seen   = 0
        self.frame_id    = 0
        self.lost_frames = 0