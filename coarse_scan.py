# -*- coding: utf-8 -*-
"""
coarse_scan.py — Pass 1 : scan léger pour détecter les segments chauds

Objectif : analyser tout le match rapidement (imgsz=640, frame_skip=5)
et retourner uniquement les segments temporels qui méritent une analyse profonde.

Output : [(start_sec, end_sec), ...]
"""

import cv2
import math


# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
COARSE_IMGSZ       = 640    # résolution réduite
COARSE_FRAME_SKIP  = 5      # 1 frame sur 5 → ~20% des frames
COARSE_BATCH_SIZE  = 8

BALL_SPEED_HOT     = 8.0    # px/frame → mouvement rapide
BALL_NEAR_GOAL_PCT = 0.20   # % de la largeur frame → proche du but
DENSITY_HOT        = 4      # nb joueurs dans la zone but → pression

SEGMENT_MARGIN_S   = 3.0    # secondes ajoutées avant/après un event
SEGMENT_MERGE_GAP  = 5.0    # fusion si deux segments à < 5s d'écart
SEGMENT_MIN_LEN    = 4.0    # durée minimale d'un segment


def _ball_center(ball):
    if not ball:
        return None
    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    if ball.get("x") is not None:
        return float(ball["x"]), float(ball.get("y", 0))
    return None


def merge_segments(segments, gap=SEGMENT_MERGE_GAP, min_len=SEGMENT_MIN_LEN):
    """Fusionne les segments proches et filtre les trop courts."""
    if not segments:
        return []
    segs = sorted(segments)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def run_coarse_scan(video_path, sport="football", fps_override=None):
    """
    Scan léger du match complet.
    Retourne : (candidate_segments, coarse_stats)
      candidate_segments = [(start_sec, end_sec), ...]
      coarse_stats = {coverage_pct, n_segments, hot_seconds, fps, total_frames}
    """
    print("\n[COARSE SCAN] Démarrage scan léger...")

    try:
        from main import process_video
    except ImportError:
        print("  [COARSE] process_video non disponible — skip scan")
        return [], {}

    # ── Tracking léger ───────────────────────────────────────────────
    try:
        events, jersey_map, fps, total_frames, frames_data = process_video(
            video_path        = video_path,
            sport             = sport,
            save_annotated    = False,
            annotated_path    = None,
            shot_zones        = None,
            return_frames     = True,
            batch_size        = COARSE_BATCH_SIZE,
            frame_skip_every  = COARSE_FRAME_SKIP,
            imgsz             = COARSE_IMGSZ,
            # Mode léger — désactive les modules lourds
            disable_reid      = True,
            disable_ocr       = True,
            lightweight       = True,
        )
    except Exception as e:
        print(f"  [COARSE] Erreur tracking : {e}")
        return [], {}

    fps = fps or 25.0
    duration = total_frames / fps

    print(f"  [COARSE] {total_frames} frames | {duration:.0f}s | "
          f"{len(frames_data)} analysées | {len(events)} events bruts")

    # ── Détection segments chauds ─────────────────────────────────────
    hot_times = set()

    # Source 1 : events détectés (shots, goals)
    for e in events:
        etype = e.get("type", "")
        t     = e.get("time", 0)
        if etype in ("shot", "goal", "corner", "free_kick", "key_pass"):
            for dt in range(-int(SEGMENT_MARGIN_S), int(SEGMENT_MARGIN_S)+1):
                hot_times.add(round(t + dt, 1))

    # Source 2 : ballon rapide ou proche du but
    prev_ball = None
    frame_w   = 1920
    frame_h   = 1080
    if frames_data:
        frame_w = int(frames_data[0].get("frame_w") or 1920)
        frame_h = int(frames_data[0].get("frame_h") or 1080)

    for fd in frames_data:
        ball = fd.get("ball")
        c    = _ball_center(ball)
        t    = fd.get("frame", 0) / fps

        if c and prev_ball:
            speed = math.sqrt(
                (c[0] - prev_ball[0])**2 +
                (c[1] - prev_ball[1])**2
            )
            near_goal = (c[0] < frame_w * BALL_NEAR_GOAL_PCT or
                         c[0] > frame_w * (1 - BALL_NEAR_GOAL_PCT))

            if speed > BALL_SPEED_HOT or near_goal:
                for dt_t in [t - 2, t - 1, t, t + 1, t + 2]:
                    if 0 <= dt_t <= duration:
                        hot_times.add(round(dt_t, 1))

        # Source 3 : densité joueurs près du but
        players = fd.get("players") or []
        near_players = sum(
            1 for p in players
            if p.get("x") is not None and (
                p["x"] < frame_w * BALL_NEAR_GOAL_PCT or
                p["x"] > frame_w * (1 - BALL_NEAR_GOAL_PCT)
            )
        )
        if near_players >= DENSITY_HOT:
            for dt_t in [t - 1, t, t + 1]:
                if 0 <= dt_t <= duration:
                    hot_times.add(round(dt_t, 1))

        prev_ball = c

    # ── Construire segments ───────────────────────────────────────────
    if not hot_times:
        print("  [COARSE] Aucun segment chaud détecté → analyse complète")
        return [(0, duration)], {
            "coverage_pct": 100,
            "n_segments":   1,
            "hot_seconds":  int(duration),
            "fps":          fps,
            "total_frames": total_frames,
        }

    sorted_times = sorted(hot_times)
    raw_segs = []
    seg_start = sorted_times[0]
    seg_end   = sorted_times[0]

    for t in sorted_times[1:]:
        if t - seg_end <= 1.5:
            seg_end = t
        else:
            raw_segs.append((max(0, seg_start - SEGMENT_MARGIN_S),
                             min(duration, seg_end + SEGMENT_MARGIN_S)))
            seg_start = t
            seg_end   = t
    raw_segs.append((max(0, seg_start - SEGMENT_MARGIN_S),
                     min(duration, seg_end + SEGMENT_MARGIN_S)))

    segments = merge_segments(raw_segs)

    hot_seconds = sum(e - s for s, e in segments)
    coverage    = hot_seconds / duration * 100

    print(f"  [COARSE] {len(segments)} segments chauds | "
          f"{hot_seconds:.0f}s / {duration:.0f}s ({coverage:.1f}%)")
    for i, (s, e) in enumerate(segments):
        print(f"    Segment {i+1} : {int(s//60):02d}:{int(s%60):02d} → "
              f"{int(e//60):02d}:{int(e%60):02d} ({e-s:.0f}s)")

    stats = {
        "coverage_pct":  round(coverage, 1),
        "n_segments":    len(segments),
        "hot_seconds":   int(hot_seconds),
        "fps":           fps,
        "total_frames":  total_frames,
        "coarse_events": len(events),
    }

    return segments, stats


def filter_frames_to_segments(frames_data, segments, fps):
    """
    Filtre frames_data pour ne garder que les frames dans les segments chauds.
    Optimisé : O(n log s) via lookup rapide sur segments triés.
    """
    if not segments:
        return frames_data
    # Pré-trier les segments (normalement déjà triés)
    segs = sorted(segments)
    result = []
    for fd in frames_data:
        t = fd.get("frame", 0) / fps
        # Recherche rapide : est-ce que t est dans un segment ?
        in_seg = False
        for s, e in segs:
            if s > t:
                break  # segments triés → inutile de continuer
            if t <= e:
                in_seg = True
                break
        if in_seg:
            result.append(fd)
    return result
