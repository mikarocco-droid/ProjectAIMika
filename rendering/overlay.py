# rendering/overlay.py (V15)

import cv2
import numpy as np
from collections import defaultdict
import config


FALLBACK_COLORS = {
    0:    (0, 200, 255),
    1:    (255, 80, 80),
    None: (180, 180, 180),
}

TEXT_COLOR   = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


class TeamColorDetector:

    def __init__(self, sample_frames=60):
        self.sample_frames = sample_frames
        self.frame_count   = 0
        self.locked        = False
        self._samples      = defaultdict(list)
        self.team_colors   = dict(FALLBACK_COLORS)

    def _dominant_color(self, patch):
        h, w = patch.shape[:2]
        if h < 20 or w < 10:
            return None

        roi = patch[int(h*0.2):int(h*0.6), :]
        roi = cv2.resize(roi, (20, 20))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv,
            np.array([35, 40, 40]),
            np.array([85, 255, 255])
        )
        mask = cv2.bitwise_not(mask_green)

        pixels = roi[mask > 0]
        if len(pixels) < 15:
            return None

        return tuple(np.median(pixels, axis=0).astype(int))

    def update(self, frame, players):
        if self.locked:
            return

        self.frame_count += 1

        for p in players:
            team = p.get("team")
            if team is None:
                continue

            x1, y1, x2, y2 = map(int, p["bbox"])
            patch = frame[y1:y2, x1:x2]

            color = self._dominant_color(patch)
            if color:
                self._samples[team].append(color)

        if self.frame_count >= self.sample_frames:
            self._lock_colors()

    def _lock_colors(self):
        for team, samples in self._samples.items():
            if len(samples) < 10:
                continue

            arr = np.array(samples)

            # nettoyage robuste
            low  = np.percentile(arr, 10, axis=0)
            high = np.percentile(arr, 90, axis=0)
            arr  = arr[(arr >= low).all(axis=1) & (arr <= high).all(axis=1)]

            if len(arr) == 0:
                continue

            self.team_colors[team] = tuple(np.mean(arr, axis=0).astype(int))

        self.locked = True
        print("🎨 Couleurs détectées:", self.team_colors)

    def get_color(self, team):
        return self.team_colors.get(team, FALLBACK_COLORS[None])

    def reset(self):
        self.__init__(self.sample_frames)


# ─────────────────────────────────────────
# DRAW UTILS
# ─────────────────────────────────────────

def draw_text(frame, text, pos, scale=0.5, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    cv2.putText(frame, text, (x+1,y+1), font, scale, SHADOW_COLOR, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,y), font, scale, TEXT_COLOR, thick, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)


# ─────────────────────────────────────────
# PLAYERS
# ─────────────────────────────────────────

def draw_players(frame, players, detector, jersey_map=None):
    for p in players:
        x1,y1,x2,y2 = map(int, p["bbox"])
        team  = p.get("team")
        pid   = p.get("id")
        color = detector.get_color(team)

        draw_box(frame, x1,y1,x2,y2, color)

        jersey = p.get("jersey") or (jersey_map or {}).get(pid)
        label  = f"#{jersey}" if jersey else f"P{pid}"

        draw_text(frame, label, (x1, y1-5))

    return frame


# ─────────────────────────────────────────
# BALL
# ─────────────────────────────────────────

def draw_ball(frame, ball):
    if not ball:
        return frame

    cx, cy = map(int, ball["center"])

    cv2.circle(frame, (cx,cy), 10, (0,255,0), 2)
    cv2.circle(frame, (cx,cy), 3,  (0,255,0), -1)

    return frame


# ─────────────────────────────────────────
# SCOREBOARD V15
# ─────────────────────────────────────────

def draw_scoreboard(frame, frame_id, fps, score):

    h,w = frame.shape[:2]
    sec = int(frame_id / fps)
    chrono = f"{sec//60:02d}:{sec%60:02d}"

    text = f"{score['A']} - {score['B']}   {chrono}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (w-220, 10), (w-10, 60), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    draw_text(frame, text, (w-200, 40), 0.8, 2)

    return frame


# ─────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────

class Overlay:

    def __init__(self, fps=None):
        self.fps = fps or config.FPS
        self.score = {"A":0,"B":0}
        self.detector = TeamColorDetector()

    def update_score(self, events):
        for e in events:
            if e.get("type") in ["goal","score"]:
                if e.get("team")==0: self.score["A"]+=1
                if e.get("team")==1: self.score["B"]+=1

    def render(self, frame, players, ball, events, frame_id, jersey_map=None):

        self.detector.update(frame, players)
        self.update_score(events)

        frame = draw_players(frame, players, self.detector, jersey_map)
        frame = draw_ball(frame, ball)
        frame = draw_scoreboard(frame, frame_id, self.fps, self.score)

        return frame