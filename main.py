# main.py
# -*- coding: utf-8 -*-

import cv2
from vision.detector import Detector
from vision.tracker import Tracker
from vision.ocr import OCRReader
from analysis.events import process_match, detect_events_v5
from rendering.overlay import Overlay
import config


# ─────────────────────────────────────────
# INIT MODULES
# ─────────────────────────────────────────
detector = Detector()
tracker  = Tracker()
ocr      = OCRReader(min_confidence=0.6)


def default_progress(pct):
    print(f"  {pct}%", end="\r")


# ─────────────────────────────────────────
# PIPELINE FRAME PAR FRAME
# ─────────────────────────────────────────
def process_video(
    video_path,
    sport             = "football",
    progress_callback = None,
    save_annotated    = False,
    annotated_path    = None,
    shot_zones        = None
):
    if progress_callback is None:
        progress_callback = default_progress

    detector.set_sport(sport)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection
        players, ball = detector.detect(frame)

        # Tracking
        tracked = tracker.update(players, frame)

        # OCR
        tracked = ocr.read_all(frame, tracked)

        # Events de cette frame
        frame_events, _ = detect_events_v5(
            players    = tracked,
            ball       = ball,
            sport      = sport,
            shot_zones = shot_zones,
            frame_w    = w,      # ← passer la vraie résolution
            frame_h    = h       # ← passer la vraie résolution
        )
        for e in frame_events:
            e["frame"] = frame_id

        frames_data.append({
            "players": tracked,
            "ball":    ball,
            "frame":   frame_id,
            "frame_w": w,        # ← stocker la résolution
            "frame_h": h,        # ← stocker la résolution
            "events":  frame_events
        })

        # Vidéo annotée
        if writer and overlay:
            annotated = overlay.render(
                frame, tracked, ball, frame_events, frame_id
            )
            writer.write(annotated)

        # Progression
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