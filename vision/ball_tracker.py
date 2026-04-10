# vision/ball_tracker.py
# -*- coding: utf-8 -*-
"""
FIX V23 — BallBuffer timestampé (x, y, t) au lieu de (x, y)
Permet de calculer vitesse en px/seconde, robuste avec frame skip.
"""

import numpy as np
from collections import deque
import math


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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


# ─────────────────────────────────────────
# BALL BUFFER TIMESTAMPÉ — apport V23
# Stocke (x, y, t) pour vitesse en px/s
# robuste face au frame skip variable
# ─────────────────────────────────────────
class BallBuffer:
    """
    Buffer circulaire de positions ballon avec timestamp.
    Remplace le simple deque (x,y) par (x, y, t).
    """
    def __init__(self, size=30):
        self._buf = deque(maxlen=size)

    def add(self, x, y, t):
        self._buf.append((float(x), float(y), float(t)))

    def get(self):
        return list(self._buf)

    def last_pos(self):
        if not self._buf:
            return None
        x, y, _ = self._buf[-1]
        return (x, y)

    def speed_px_per_sec(self):
        """
        Vitesse en px/seconde sur les 3 dernières positions.
        Beaucoup plus fiable que px/frame avec frame skip.
        """
        pts = list(self._buf)
        if len(pts) < 2:
            return 0.0

        # Moyenne glissante sur max 3 derniers segments
        n      = min(len(pts), 4)
        recent = pts[-n:]
        speeds = []

        for i in range(1, len(recent)):
            x1, y1, t1 = recent[i-1]
            x2, y2, t2 = recent[i]
            dt = max(t2 - t1, 1e-4)
            d  = math.hypot(x2 - x1, y2 - y1)
            speeds.append(d / dt)

        return sum(speeds) / len(speeds) if speeds else 0.0

    def speed_px_per_frame(self):
        """
        Compatibilité arrière — vitesse px/frame entre les 2 derniers points.
        """
        pts = list(self._buf)
        if len(pts) < 2:
            return 0.0
        x1, y1, _ = pts[-2]
        x2, y2, _ = pts[-1]
        return math.hypot(x2 - x1, y2 - y1)

    def direction(self):
        """
        Vecteur direction normalisé depuis les 3 derniers points.
        Plus stable qu'un simple vecteur 2 points.
        """
        pts = list(self._buf)
        if len(pts) < 2:
            return (0.0, 0.0)

        # Régression linéaire simple sur les 4 derniers points
        recent = pts[-4:] if len(pts) >= 4 else pts
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        norm = math.hypot(dx, dy) + 1e-6
        return (dx / norm, dy / norm)

    def toward_goal(self, frame_w, frame_h, threshold=0.55):
        """
        Retourne True si le ballon se dirige vers un but.
        Vérifie les deux buts (gauche et droite) en vue latérale.
        Utilise shot_zones si disponible, sinon heuristique.
        """
        if not self._buf:
            return False, None

        bx, by, _ = self._buf[-1]
        dx, dy    = self.direction()

        goal_centers = [
            (0.0,         frame_h * 0.5),   # but gauche
            (frame_w,     frame_h * 0.5),   # but droit
        ]

        for i, (gx, gy) in enumerate(goal_centers):
            to_goal_x = gx - bx
            to_goal_y = gy - by
            norm      = math.hypot(to_goal_x, to_goal_y) + 1e-6
            to_goal_n = (to_goal_x / norm, to_goal_y / norm)
            dot       = dx * to_goal_n[0] + dy * to_goal_n[1]
            if dot > threshold:
                return True, i   # i=0 but gauche, i=1 but droit

        return False, None

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)


# ─────────────────────────────────────────
# KALMAN SIMPLIFIÉ
# ─────────────────────────────────────────
class SimpleKalman:
    def __init__(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])

    def update(self, measurement):
        if measurement is None:
            if self.state is not None:
                self.state    = self.state + self.velocity
                self.velocity = self.velocity * 0.7
            return self.state

        m = np.array(measurement, dtype=float)
        if self.state is None:
            self.state    = m
            self.velocity = np.array([0.0, 0.0])
            return self.state

        self.velocity = (m - self.state) * 0.9
        self.state    = m
        return self.state

    def reset(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])


# ─────────────────────────────────────────
# BALL TRACKER PRINCIPAL
# ─────────────────────────────────────────
class BallTracker:
    """
    Tracker ballon avec BallBuffer timestampé.
    - vitesse en px/seconde (robuste frame skip)
    - direction vers but intégrée
    - compatibilité arrière totale avec le pipeline
    """

    def __init__(self, max_history=30, fps=25):
        self.fps         = fps
        self.ball_buffer = BallBuffer(size=max_history)   # FIX V23
        self.kalman      = SimpleKalman()
        self.last_seen   = 0
        self.frame_id    = 0
        self.lost_frames = 0
        self.max_lost    = 5   # ~0.2s à 25fps

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

    def update(self, detected_balls, frame_w, frame_h, timestamp=None):
        """
        FIX V23 — accepte un timestamp optionnel (en secondes).
        Si absent, calcule depuis frame_id et fps.
        """
        self.frame_id += 1

        # Timestamp en secondes
        t = timestamp if timestamp is not None else self.frame_id / self.fps

        detected_balls = [
            b for b in detected_balls
            if is_valid_ball(b, frame_w, frame_h)
        ]

        last_pos = self.ball_buffer.last_pos()
        best     = self.select_best_ball(detected_balls, last_pos)

        if best is not None:
            x, y, w, h = best
            cx = x + w // 2
            cy = y + h // 2

            self.ball_buffer.add(cx, cy, t)   # FIX V23 — avec timestamp
            self.last_seen   = self.frame_id
            self.lost_frames = 0

            pos = self.kalman.update((cx, cy))
            return self.get_ball_bbox(pos), False   # interpolated=False

        else:
            self.lost_frames += 1
            if self.lost_frames < self.max_lost:
                pos = self.kalman.update(None)
                if pos is not None:
                    cx, cy = int(pos[0]), int(pos[1])
                    self.ball_buffer.add(cx, cy, t)   # FIX V23
                    return self.get_ball_bbox(pos), True
            else:
                self.ball_buffer.clear()
                self.kalman.reset()

            return None, True

    def get_ball_bbox(self, pos):
        if pos is None:
            return None
        x, y = int(pos[0]), int(pos[1])
        size = 10
        return (x - size, y - size, size * 2, size * 2)

    # ─────────────────────────────────────────
    # GETTERS — interface enrichie V23
    # ─────────────────────────────────────────

    def get_trajectory(self):
        """Compatibilité arrière — retourne liste de (x, y)."""
        return [(x, y) for x, y, _ in self.ball_buffer.get()]

    def get_trajectory_with_time(self):
        """Nouveau — retourne liste de (x, y, t)."""
        return self.ball_buffer.get()

    def get_speed(self):
        """Compatibilité arrière — px/frame entre les 2 derniers points."""
        return self.ball_buffer.speed_px_per_frame()

    def get_speed_per_second(self):
        """Nouveau V23 — vitesse en px/seconde, robuste frame skip."""
        return self.ball_buffer.speed_px_per_sec()

    def get_direction(self):
        """Nouveau V23 — vecteur direction normalisé."""
        return self.ball_buffer.direction()

    def is_shot_candidate(self, frame_w, frame_h,
                          speed_threshold_px_per_sec=None):
        """
        Nouveau V23 — combine vitesse px/s + direction vers but.
        Utilisé par events.py pour la détection de tirs.
        """
        if speed_threshold_px_per_sec is None:
            # ~77px/frame à 25fps sur 1920px = ~1925 px/s
            speed_threshold_px_per_sec = frame_w * 1.0

        speed       = self.get_speed_per_second()
        toward, _   = self.ball_buffer.toward_goal(frame_w, frame_h)

        return speed > speed_threshold_px_per_sec and toward

    def closest_player(self, players):
        last = self.ball_buffer.last_pos()
        if last is None:
            return None
        ball_pos  = last
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
        self.ball_buffer.clear()
        self.kalman.reset()
        self.last_seen   = 0
        self.frame_id    = 0
        self.lost_frames = 0