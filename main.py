# main.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from vision.detector import Detector
from vision.tracker import Tracker
from vision.ocr import OCRReader
from analysis.events import process_match, detect_events
from rendering.overlay import Overlay, TeamColorDetector
import config

from config import FRAME_SKIP_EVERY, YOLO_BATCH_SIZE

PROCESS_W = 960
PROCESS_H = 540


def default_progress(pct):
    print(f"  {pct}%", end="\r")


def assign_teams_by_color(frame, tracked, color_detector):
    color_detector.update(frame, tracked)

    for p in tracked:
        if p.get("team") is not None:
            continue

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


def rescale_detections(players, yolo_ball, scale_x, scale_y):
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        p["bbox"]   = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
        p["center"] = [(p["bbox"][0]+p["bbox"][2])/2,
                       (p["bbox"][1]+p["bbox"][3])/2]

    if yolo_ball:
        x1, y1, x2, y2 = yolo_ball["bbox"]
        yolo_ball["bbox"]   = [x1*scale_x, y1*scale_y,
                                x2*scale_x, y2*scale_y]
        yolo_ball["center"] = [(yolo_ball["bbox"][0]+yolo_ball["bbox"][2])/2,
                                (yolo_ball["bbox"][1]+yolo_ball["bbox"][3])/2]

    return players, yolo_ball


def ball_dict_to_tuple(ball_dict):
    if ball_dict is None:
        return None
    bbox = ball_dict.get("bbox")
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def ball_tuple_to_dict(ball_tuple, interpolated=False):
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


def process_batch(
    batch_frames,
    detector,
    tracker,
    ocr,
    color_detector,
    ball_tracker,
    sport,
    shot_zones,
    w, h,
    scale_x, scale_y,
    analyzed_offset,
    fps          = 25,
    events_state = None,
    b_size       = None,   # ← taille effective du batch
):
    if not batch_frames:
        return [], events_state

    # Utilise b_size effectif pour imgsz (impacte la qualité de détection YOLO)
    effective_batch = b_size if b_size is not None else YOLO_BATCH_SIZE

    small_frames  = [bf[2] for bf in batch_frames]
    batch_results = detector.model(
        small_frames,
        conf    = config.YOLO_CONFIDENCE,
        verbose = False,
        imgsz   = effective_batch * 240
    )

    batch_data = []

    for i, (frame_id, frame_orig, frame_small) in enumerate(batch_frames):
        result   = batch_results[i]
        analyzed = analyzed_offset + i

        players   = []
        yolo_ball = None

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox   = [x1, y1, x2, y2]

            if cls == detector.player_cls:
                if not detector._in_play_zone(center, PROCESS_W, PROCESS_H):
                    continue
                if not detector._valid_size(bbox, PROCESS_W, PROCESS_H):
                    continue
                players.append({
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                })
            elif cls == detector.ball_cls:
                yolo_ball = {
                    "bbox":   bbox,
                    "center": [center[0], center[1]],
                    "conf":   conf
                }

        last_pos_small = None
        if detector._last_ball_pos is not None:
            lx, ly = detector._last_ball_pos
            last_pos_small = (lx / scale_x, ly / scale_y)

        yolo_ball = detector._detect_ball(
            frame_small, yolo_ball,
            last_pos_override=last_pos_small
        )

        players, yolo_ball = rescale_detections(
            players, yolo_ball, scale_x, scale_y
        )

        tracked = tracker.update(players, frame_orig)
        tracked = assign_teams_by_color(frame_orig, tracked, color_detector)
        tracked = ocr.read_all(frame_orig, tracked, frame_id=analyzed)

        if ball_tracker is not None:
            yolo_ball_tuple = ball_dict_to_tuple(yolo_ball)
            balls_list      = [yolo_ball_tuple] if yolo_ball_tuple else []
            ball_result, was_interpolated = ball_tracker.update(
                detected_balls = balls_list,
                frame_w        = w,
                frame_h        = h
            )
            ball = ball_tuple_to_dict(ball_result, interpolated=was_interpolated)
        else:
            ball = yolo_ball

        frame_events, events_state = detect_events(
            players    = tracked,
            ball       = ball,
            sport      = sport,
            state      = events_state,
            shot_zones = shot_zones,
            frame_w    = w,
            frame_h    = h,
            fps        = fps,
        )
        for e in frame_events:
            e["frame"] = frame_id
            if e.get("team") is None:
                pid = e.get("player")
                if pid:
                    match = next(
                        (t for t in tracked if str(t.get("id")) == str(pid)),
                        None
                    )
                    if match and match.get("team") is not None:
                        e["team"] = match["team"]

        batch_data.append({
            "players":     tracked,
            "ball":        ball,
            "frame":       frame_id,
            "frame_w":     w,
            "frame_h":     h,
            "fps":         fps,
            "events":      frame_events,
            "_frame_orig": frame_orig,
        })

    return batch_data, events_state


def process_video(
    video_path,
    sport             = "football",
    progress_callback = None,
    save_annotated    = False,
    annotated_path    = None,
    shot_zones        = None,
    return_frames     = False,
    frame_skip_every  = None,
    batch_size        = None,
):
    if progress_callback is None:
        progress_callback = default_progress

    skip_every = frame_skip_every if frame_skip_every is not None else FRAME_SKIP_EVERY
    b_size     = batch_size       if batch_size       is not None else YOLO_BATCH_SIZE

    detector       = Detector(sport=sport)
    tracker        = Tracker()
    ocr            = OCRReader(min_confidence=0.6, ocr_every_n_frames=30)
    color_detector = TeamColorDetector(sample_frames=60)

    ball_tracker = None
    try:
        from vision.ball_tracker import BallTracker
        ball_tracker = BallTracker(max_history=30)
        print("  BallTracker : OK")
    except Exception as e:
        print(f"  BallTracker indisponible : {e} — fallback YOLO")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video : {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or config.FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_x = w / PROCESS_W
    scale_y = h / PROCESS_H

    analyzed_count = total_frames - (total_frames // skip_every)

    print(f"Video : {video_path}")
    print(f"  {total_frames} frames | {fps:.1f} fps | {total_frames / fps:.1f}s")
    print(f"  Resolution : {w}x{h} → traitement {PROCESS_W}x{PROCESS_H}")
    print(f"  Sport : {sport}")
    print(f"  Frame skip  : 2/{skip_every} → ~{analyzed_count} frames analysées "
          f"({analyzed_count * 100 // total_frames}%)")
    print(f"  YOLO batch  : {b_size} frames/passe | imgsz={b_size * 240}")

    if shot_zones:
        hi = shot_zones.get("threshold_hi", 0.85)
        lo = shot_zones.get("threshold_lo", 0.15)
        if isinstance(hi, float) and hi <= 1.0:
            shot_zones = {
                "axis":         shot_zones.get("axis", "x"),
                "threshold_hi": hi * w,
                "threshold_lo": lo * w,
                "y_min":        shot_zones.get("y_min", 0.25) * h,
                "y_max":        shot_zones.get("y_max", 0.75) * h,
            }
            print(f"  Shot zones (px) : "
                  f"hi={shot_zones['threshold_hi']:.0f} "
                  f"lo={shot_zones['threshold_lo']:.0f} "
                  f"y=[{shot_zones['y_min']:.0f}, {shot_zones['y_max']:.0f}]")

    overlay = Overlay(fps=fps) if save_annotated else None
    writer  = None
    if save_annotated and annotated_path:
        out_fps = fps * (2 / skip_every)
        writer  = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps, (w, h)
        )

    frames_data   = []
    frame_id      = 0
    analyzed      = 0
    last_pct      = -1
    current_batch = []
    events_state  = None

    # ──────────────────────────────────────────────────────────
    # PIPELINE 2 VITESSES
    # Mode léger  : skip_every normal (config)
    # Mode action : skip_every=1 si ballon rapide ou près du but
    # ──────────────────────────────────────────────────────────
    _skip_base       = skip_every   # skip configuré (2 ou 4)
    _skip_action     = 1            # skip en mode action
    _action_mode     = False
    _action_ttl      = 0
    ACTION_TTL       = 75           # ~3s @25fps avant retour léger
    ACTION_SPD_THR   = 80.0         # px/frame seuil vitesse ballon
    ACTION_ZONE_THR  = 0.18         # % frame — zone de but gauche/droite
    _n_action        = 0            # compteur passages mode action
    _last_ball_data  = {}

    def _check_action_mode(ball_dict):
        nonlocal _action_mode, _action_ttl, _n_action, _last_ball_data
        if not ball_dict:
            if _action_ttl > 0:
                _action_ttl -= 1
                if _action_ttl == 0:
                    _action_mode = False
            return
        _last_ball_data = ball_dict
        bx = ball_dict.get("x") or ball_dict.get("cx") or ball_dict.get("center", [None])[0]
        bs = float(ball_dict.get("speed", 0) or 0)

        in_goal_zone = (
            bx is not None and w > 0
            and (bx / w < ACTION_ZONE_THR or bx / w > (1 - ACTION_ZONE_THR))
        )
        was_action = _action_mode

        if bs > ACTION_SPD_THR or in_goal_zone:
            _action_mode = True
            _action_ttl  = ACTION_TTL
            if not was_action:
                _n_action += 1
        elif _action_ttl > 0:
            _action_ttl -= 1
            if _action_ttl == 0:
                _action_mode = False

    def flush_batch(batch, analyzed_so_far):
        nonlocal events_state
        if not batch:
            return
        data, events_state = process_batch(
            batch, detector, tracker, ocr,
            color_detector, ball_tracker,
            sport, shot_zones, w, h, scale_x, scale_y,
            analyzed_offset = analyzed_so_far - len(batch) + 1,
            fps             = fps,
            events_state    = events_state,
            b_size          = b_size,
        )
        for fd in data:
            if writer and overlay:
                orig = fd.pop("_frame_orig", None)
                if orig is not None:
                    ann = overlay.render(
                        orig, fd["players"], fd["ball"],
                        fd["events"], fd["frame"]
                    )
                    writer.write(ann)
            else:
                fd.pop("_frame_orig", None)
            frames_data.append(fd)
            # Mettre à jour le mode action après chaque frame traitée
            _check_action_mode(fd.get("ball"))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip dynamique selon le mode courant
        _cur_skip = _skip_action if _action_mode else _skip_base
        if frame_id % _cur_skip == (_cur_skip - 1):
            frame_id += 1
            continue

        frame_small = cv2.resize(
            frame, (PROCESS_W, PROCESS_H),
            interpolation=cv2.INTER_LINEAR
        )

        current_batch.append((frame_id, frame, frame_small))

        if len(current_batch) >= b_size:
            flush_batch(current_batch, analyzed)
            current_batch = []

        if total_frames > 0:
            pct = int((frame_id / total_frames) * 100)
            if pct % 5 == 0 and pct != last_pct:
                progress_callback(pct)
                last_pct = pct

        frame_id += 1
        analyzed += 1

    flush_batch(current_batch, analyzed)

    cap.release()
    if writer:
        writer.release()

    print(f"\n  {frame_id} frames lues | {analyzed} analysées"
          f" | batches de {b_size}")
    print(f"  2-speed : {_n_action} passages mode action ({_skip_action}→{_skip_base})")

    jersey_map = ocr.get_jersey_map()
    ocr.reset()

    events = process_match(frames_data, sport, shot_zones=shot_zones)

    print(f"  {len(events)} events detectes")
    print(f"  {len(jersey_map)} maillots identifies")

    if return_frames:
        return events, jersey_map, fps, total_frames, frames_data
    else:
        return events, jersey_map, fps, total_frames


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