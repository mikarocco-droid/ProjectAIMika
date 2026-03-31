# main.py
# -*- coding: utf-8 -*-

import cv2
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
detector     = Detector()
tracker      = Tracker()
ocr          = OCRReader(min_confidence=0.6, ocr_every_n_frames=30)
ball_tracker = BallTracker(max_history=30)


def default_progress(pct):
    print(f"  {pct}%", end="\r")


# ─────────────────────────────────────────
# ASSIGNATION ÉQUIPE PAR COULEUR
# ─────────────────────────────────────────
def assign_teams(frame, tracked, color_detector):
    color_detector.update(frame, tracked)

    for p in tracked:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        h_f, w_f = frame.shape[:2]

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_f, x2); y2 = min(h_f, y2)

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
            elif b > r and b > g:
                p["team"] = 1
            else:
                p["team"] = 0

    return tracked


# ─────────────────────────────────────────
# CONVERSION DICT BALL → TUPLE (x,y,w,h)
# ─────────────────────────────────────────
def ball_dict_to_tuple(ball_dict):
    """
    Convertit un dict ballon {bbox, center, conf}
    en tuple (x, y, w, h) pour BallTracker.
    """
    if ball_dict is None:
        return None
    bbox = ball_dict.get("bbox")
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


# ─────────────────────────────────────────
# CONVERSION TUPLE → DICT BALL
# ─────────────────────────────────────────
def ball_tuple_to_dict(ball_tuple, interpolated=False):
    """
    Convertit un tuple (x, y, w, h) de BallTracker
    en dict ballon {bbox, center} pour le pipeline.
    """
    if ball_tuple is None:
        return None
    x, y, w, h = ball_tuple
    cx = x + w // 2
    cy = y + h // 2
    return {
        "bbox":         [x, y, x + w, y + h],
        "center":       [cx, cy],
        "conf":         1.0,
        "interpolated": interpolated
    }


# ─────────────────────────────────────────
# PIPELINE FRAME PAR FRAME
# ─────────────────────────────────────────
def process_video(
    video_path,
    sport             = "football",
    progress_callback = None,
    save_annotated    = False,
    annotated_path    = None,
    shot_zones        = None,
    return_frames     = False
):
    if progress_callback is None:
        progress_callback = default_progress

    detector.set_sport(sport)
    color_detector = TeamColorDetector(sample_frames=60)

    # Reset ball tracker entre vidéos
    ball_tracker.__init__(max_history=30)

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

    # Convertir shot_zones ratios → pixels si nécessaire
    if shot_zones:
        hi = shot_zones.get("threshold_hi", 0.85)
        lo = shot_zones.get("threshold_lo", 0.15)
        if hi <= 1.0:
            shot_zones = {
                "axis":         shot_zones.get("axis", "x"),
                "threshold_hi": hi * w,
                "threshold_lo": lo * w,
                "y_min":        shot_zones.get("y_min", 0.25) * h,
                "y_max":        shot_zones.get("y_max", 0.75) * h,
            }
            print(f"  Shot zones (px) : "
                  f"x>[{shot_zones['threshold_hi']:.0f}] "
                  f"x<[{shot_zones['threshold_lo']:.0f}] "
                  f"y=[{shot_zones['y_min']:.0f}, {shot_zones['y_max']:.0f}]")

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Détection YOLO ───────────────
        players, yolo_ball = detector.detect(frame)

        # ── Tracking joueurs ─────────────
        tracked = tracker.update(players, frame)

        # ── Assignation équipes ──────────
        tracked = assign_teams(frame, tracked, color_detector)

        # ── OCR maillots (1/30) ──────────
        tracked = ocr.read_all(frame, tracked, frame_id=frame_id)

        # ── Ball Tracker ─────────────────
        # Convertir le dict YOLO en tuple pour BallTracker
        yolo_ball_tuple = ball_dict_to_tuple(yolo_ball)
        balls_list      = [yolo_ball_tuple] if yolo_ball_tuple else []

        ball_result = ball_tracker.update(
            detected_balls = balls_list,
            frame_w        = w,
            frame_h        = h
        )

        # Reconvertir en dict pour le pipeline
        was_interpolated = (yolo_ball_tuple is None and ball_result is not None)
        ball = ball_tuple_to_dict(ball_result, interpolated=was_interpolated)

        # ── Events de cette frame ─────────
        frame_events, _ = detect_events(
            players    = tracked,
            ball       = ball,
            sport      = sport,
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

        # ── Vidéo annotée ─────────────────
        if writer and overlay:
            annotated = overlay.render(
                frame, tracked, ball, frame_events, frame_id
            )
            writer.write(annotated)

        # ── Progression ───────────────────
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