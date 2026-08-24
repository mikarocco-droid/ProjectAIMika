import os
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
    # Supprimer le team déjà assigné pour que color_detector.update()
    # puisse réassigner avec les centroides calibrés.
    # Sans ça, les joueurs avec team=0 (défaut) ne sont jamais corrigés.
    for p in tracked:
        if p.get("team") is not None:
            p["team"] = None

    color_detector.update(frame, tracked)

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
        imgsz   = int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ))
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

        # PATCH : injecter le frame courant dans ball pour que detect_events
        # puisse calculer current_time = frame / fps correctement
        # Sans ça, ball.get("frame", 0) retourne 0 → tous les logs t=0.0s
        if ball is not None:
            ball["frame"] = analyzed

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
    print(f"  Frame skip  : {skip_every} → ~{analyzed_count} frames analysées "
          f"({analyzed_count * 100 // total_frames}%)")
    print(f"  YOLO batch  : {b_size} frames/passe | imgsz={int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ))}")

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
            b_size          = b_size,   # ← propagé jusqu'à imgsz
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_every == (skip_every - 1):
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

    jersey_map_tesseract = ocr.get_jersey_map()
    ocr.reset()

    # V5.1 : lecture des maillots via Gemini, en PRIORITE sur Tesseract.
    # Reutilise read_jersey_numbers() deja existant (regroupe jusqu'a 20
    # joueurs par appel - PAS un appel par joueur par frame, cout modere :
    # ~10 appels Gemini pour tout un match de ~180-200 track_id). Repli
    # sur Tesseract pour tout track_id que Gemini n'a pas resolu.
    jersey_map = dict(jersey_map_tesseract)
    try:
        from ai.gemini_validator import read_jersey_numbers

        # Un representant par track_id : la frame ou sa bbox est la plus
        # grande (meilleure chance de numero lisible)
        meilleure_frame_par_tid = {}
        for fd in frames_data:
            for p in fd.get("players", []):
                tid = p.get("id")
                bbox = p.get("bbox")
                if tid is None or not bbox:
                    continue
                aire = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                prec = meilleure_frame_par_tid.get(tid)
                if prec is None or aire > prec["aire"]:
                    meilleure_frame_par_tid[tid] = {
                        "id": tid, "frame_id": fd.get("frame"),
                        "bbox": bbox, "aire": aire,
                    }

        players_with_frames = list(meilleure_frame_par_tid.values())
        print(f"  [GEMINI JERSEYS] {len(players_with_frames)} track_id à lire "
              f"({(len(players_with_frames) + 19) // 20} appels prévus)")

        jersey_map_gemini = {}
        TAILLE_LOT = 20
        for i in range(0, len(players_with_frames), TAILLE_LOT):
            lot = players_with_frames[i:i + TAILLE_LOT]
            resultat_lot = read_jersey_numbers(video_path, lot, fps=fps, max_players=TAILLE_LOT)
            jersey_map_gemini.update(resultat_lot)

        print(f"  [GEMINI JERSEYS] {len(jersey_map_gemini)} numéros lus avec succès "
              f"(sur {len(players_with_frames)} track_id tentés)")

        # Gemini prioritaire, Tesseract en repli pour ce que Gemini n'a pas resolu
        jersey_map = {**jersey_map_tesseract, **jersey_map_gemini}
    except Exception as _e_gemini_jersey:
        print(f"  [GEMINI JERSEYS] échec, repli intégral sur Tesseract : {_e_gemini_jersey}")

    events = process_match(frames_data, sport, shot_zones=shot_zones)

    print(f"  {len(events)} events detectes")
    print(f"  {len(jersey_map)} maillots identifies")

    # V5.1 DIAGNOSTIC - SAUVEGARDE EN FICHIER (pas juste console !) la
    # distribution reelle de abs(d0-d1), pour calibrer le seuil d'ambiguite
    # team (actuellement 15.0, suspecte trop large). Meme dossier que
    # frames_data.pkl / jersey_map_brut.json (pipeline.py), pour tout
    # retrouver au meme endroit sans jamais avoir a relancer un run pour
    # recuperer une info deja calculee. A retirer une fois le seuil corrige.
    try:
        import json as _json_diag
        _diag_dir = "outputs/test/audit_identite"
        os.makedirs(_diag_dir, exist_ok=True)
        _diag_stats = tracker.reid.stats()
        with open(os.path.join(_diag_dir, "reid_diag.json"), "w", encoding="utf-8") as _f_diag:
            _json_diag.dump(_diag_stats, _f_diag, indent=2)
        print(f"  [DIAG PlayerReID] sauvegardé dans {_diag_dir}/reid_diag.json : {_diag_stats}")
    except Exception as _e_diag:
        print(f"  [DIAG PlayerReID] indisponible : {_e_diag}")

    # V5.1 DIAGNOSTIC - zippe le dossier de crops OCR (bruts + traites)
    # en un seul fichier, plus simple a telecharger depuis Kaggle qu'un
    # dossier de dizaines d'images individuelles.
    try:
        import shutil as _shutil_diag
        _crops_dir = "outputs/test/audit_identite/ocr_crops_diag"
        if os.path.isdir(_crops_dir) and os.listdir(_crops_dir):
            _zip_path_sans_ext = "outputs/test/audit_identite/ocr_crops_diag"
            _shutil_diag.make_archive(_zip_path_sans_ext, "zip", _crops_dir)
            _taille_ko = os.path.getsize(_zip_path_sans_ext + ".zip") / 1024
            print(f"  [DIAG PlayerReID] crops zippés : {_zip_path_sans_ext}.zip ({_taille_ko:.0f} Ko)")
        else:
            print(f"  [DIAG PlayerReID] aucun crop a zipper ({_crops_dir} vide ou absent)")
    except Exception as _e_zip:
        print(f"  [DIAG PlayerReID] zip des crops échoué : {_e_zip}")

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