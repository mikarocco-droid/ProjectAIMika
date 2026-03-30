# vision/ball_tracker.py

import numpy as np


class BallTracker:
    def __init__(self, max_missing=10):
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])
        self.missing_frames = 0
        self.max_missing = max_missing

    # ─────────────────────────────
    # UPDATE PRINCIPAL
    # ─────────────────────────────
    def update(self, detection):
        """
        detection = (x, y) ou None
        """

        # 🎯 CAS 1 : balle détectée
        if detection is not None:
            detection = np.array(detection, dtype=float)

            if self.last_position is not None:
                # calcul vitesse
                self.velocity = detection - self.last_position

            self.last_position = detection
            self.missing_frames = 0

            return tuple(self.last_position.astype(int))

        # ❌ CAS 2 : balle non détectée
        else:
            self.missing_frames += 1

            if self.last_position is None:
                return None

            # 💡 prédiction (inertie)
            predicted = self.last_position + self.velocity

            # limiter dérive
            if self.missing_frames > self.max_missing:
                return None

            self.last_position = predicted

            return tuple(predicted.astype(int))


# ─────────────────────────────
# FILTRE VALIDATION
# ─────────────────────────────
def is_valid_ball(x, y, width=1280, height=720):
    return 0 <= x <= width and 0 <= y <= height


# ─────────────────────────────
# SMOOTHING (anti jitter)
# ─────────────────────────────
def smooth_position(prev, current, alpha=0.7):
    if prev is None:
        return current

    x = int(alpha * current[0] + (1 - alpha) * prev[0])
    y = int(alpha * current[1] + (1 - alpha) * prev[1])
    return (x, y)