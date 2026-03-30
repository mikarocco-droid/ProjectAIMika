# main.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from vision.detector import Detector
from vision.tracker import Tracker
from vision.ocr import OCRReader
from vision.ball_tracker import BallTracker

from analysis.events import process_match, detect_events
from rendering.overlay import Overlay, TeamColorDetector

import config


# ─────────────────────────────────────────
# INIT MODULES
# ─────────────────────────────────────────
detector = Detector()
tracker  = Tracker()
ocr      = OCRReader(min_confidence=0.6, ocr_every_n_frames=30)

# 🔥 V15 BALL TRACKER (UNIQUE)
ball_tracker = BallTracker(max_history=30)


def default_progress(pct):
    print(f"  {pct}%", end="\r")


# ─────────────────────────────────────────
# TEAM ASSIGNMENT
# ─────────────────────────────────────────
def assign_teams(frame, tracked, color_detector):
    color_detector.update(frame, tracked)

    h_f, w_f = frame.shape[:2]

    for p in tracked:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_f, x2)
        y2 = min(h_f, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        patch = frame[y1:y2, x1:x2]
        color = color_detector._dominant_color(patch)

        if color:
            b, g, r = color

            if g > r and g > b:
                p["team"] = 0
            elif r > g and r > b:
                p["team"] = 1
            else:
                p["team"] = 0

    return tracked


# ─────────────────────────────────────────
# PIPELINE VIDEO V15 GOD MODE
# ─────────────────────────────────────────
def process_video(
    video_path,
    sport             = "football",
    progress_callback = None,
    save_annotated    = False,
    annotated_path    = None,
    shot_zones        = None,
    return_frames     = False    # ← ajouter
):

    if progress_callback is None:
        progress_callback = default_progress

    detector.set_sport(sport)

    color_detector = TeamColorDetector(sample_frames=60)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video : {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or config.FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video : {video_path}")
    print(f"  {total_frames} frames | {fps:.1f} fps | {total_frames / fps:.1f}s")
    print(f"  Resolution : {w}x{h}")
    print(f"  Sport : {sport}")

    overlay = Overlay(fps=fps) if save_annotated else None
    writer  = None

    if save_annotated and annotated_path:
        writer = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

    frames_data = []
    frame_id    = 0
    last_pct    = -1

    # 🔥 STATE GLOBAL POUR EVENTS (ULTRA IMPORTANT)
    state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── DETECTION ─────────────────────
        players, balls = detector.detect(frame)

        # ── TRACKING JOUEURS ──────────────
        tracked = tracker.update(players, frame)

        # ── TEAM COLOR ────────────────────
        tracked = assign_teams(frame, tracked, color_detector)

        # ── OCR ───────────────────────────
        tracked = ocr.read_all(frame, tracked, frame_id=frame_id)

        # ── BALL TRACKING (V15 CLEAN) 🔥
        ball = ball_tracker.update(
            detected_balls = balls,
            frame_w = w,
            frame_h = h
        )

        # ── EVENTS (STATEFUL) 🔥🔥🔥
        frame_events, state = detect_events(
            players    = tracked,
            ball       = ball,
            sport      = sport,
            state      = state,
            shot_zones = shot_zones,
            frame_w    = w,
            frame_h    = h
        )

        for e in frame_events:
            e["frame"] = frame_id

        frames_data.append({
            "players": tracked,
            "ball":    ball,
            "frame":   frame_id,
            "frame_w": w,
            "frame_h": h,
            "events":  frame_events
        })

        # ── OVERLAY ───────────────────────
        if writer and overlay:
            annotated = overlay.render(
                frame, tracked, ball, frame_events, frame_id
            )
            writer.write(annotated)

        # ── PROGRESS ──────────────────────
        if total_frames > 0:
            pct = int((frame_id / total_frames) * 100)
            if pct % 5 == 0 and pct != last_pct:
                progress_callback(pct)
                last_pct = pct

        frame_id += 1

    cap.release()
    if writer:
        writer.release()

    print(f"\n  {frame_id} frames traitees")

    jersey_map = ocr.get_jersey_map()
    ocr.reset()

    # ── EVENTS GLOBALS ───────────────────
    events = process_match(frames_data, sport, shot_zones=shot_zones)

    print(f"  {len(events)} events detectes")
    print(f"  {len(jersey_map)} maillots identifies")
    
    if return_frames:
        return events, jersey_map, fps, total_frames, frames_data
    else:
        return events, jersey_map, fps, total_frames

# ─────────────────────────────────────────
# TEST LOCAL
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_PATH
    sport = sys.argv[2] if len(sys.argv) > 2 else "football"

    events, jersey_map, fps, total_frames = process_video(
        video_path = video,
        sport      = sport
    )

    print(f"\nEvents : {len(events)}")
    for e in events[:10]:
        print(f"  {e}")