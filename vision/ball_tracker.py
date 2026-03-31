# vision/ball_tracker.py
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque


def smooth_position(history, window=5):
    if len(history) == 0:
        return None
    pts = list(history)[-window:]
    x = int(np.mean([p[0] for p in pts]))
    y = int(np.mean([p[1] for p in pts]))
    return (x, y)


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
    def __init__(self):
        self.state    = None
        self.velocity = np.array([0.0, 0.0])

    def update(self, measurement):
        if measurement is None:
            if self.state is not None:
                self.state = self.state + self.velocity
            return self.state
        m = np.array(measurement)
        if self.state is None:
            self.state = m
            return self.state
        self.velocity = (m - self.state) * 0.6
        self.state    = self.state + self.velocity
        return self.state


class BallTracker:

    def __init__(self, max_history=30):
        self.history    = deque(maxlen=max_history)
        self.kalman     = SimpleKalman()
        self.last_seen  = 0
        self.frame_id   = 0
        self.lost_frames = 0
        self.max_lost   = 15

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
        else:
            self.lost_frames += 1
            if self.lost_frames < self.max_lost:
                pos = self.kalman.update(None)
                if pos is not None:
                    self.history.append(tuple(pos.astype(int)))
            else:
                pos = None
                self.history.clear()

        return self.get_ball_bbox(pos)

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