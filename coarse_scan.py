# -*- coding: utf-8 -*-
"""
coarse_scan.py — Pass 1 : scan léger pour détecter les segments chauds

Objectif : analyser tout le match rapidement (imgsz=640, frame_skip=5)
et retourner uniquement les segments temporels qui méritent une analyse profonde.

Output : [(start_sec, end_sec), ...]
"""

import cv2
import math
import subprocess
import struct


# ─────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────
# Pass 1 — coarse scan : rapide, léger
COARSE_IMGSZ       = 640    # résolution réduite
COARSE_FRAME_SKIP  = 5      # 1 frame sur 5 → ~20% des frames
COARSE_BATCH_SIZE  = 8

# Pass 2 — deep analysis : précis, uniquement sur segments chauds
DEEP_IMGSZ         = 960    # résolution complète
DEEP_FRAME_SKIP    = 2      # 1 frame sur 2 → analyse fine
DEEP_BATCH_SIZE    = 8

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


# ─────────────────────────────────────────
# DÉTECTEURS SUPPLÉMENTAIRES
# ─────────────────────────────────────────
def detect_audio_spikes(video_path, fps=25.0, threshold_db=0.7, window_s=1.0):
    """
    Détecte les pics audio (réactions foule = souvent buts/occasions).
    Utilise ffmpeg pour extraire l'amplitude RMS.
    Retourne : [timestamp_s, ...] des pics
    """
    spikes = []
    try:
        # Extraire amplitudes RMS via ffmpeg
        cmd = [
            "ffmpeg", "-i", video_path,
            "-af", f"asetnsamples={int(fps*window_s)},astats=metadata=1:reset=1",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        current_t = 0.0
        rms_values = []

        for line in result.stderr.splitlines():
            if 'pts_time' in line:
                try:
                    current_t = float(line.split('pts_time:')[1].split()[0])
                except Exception:
                    pass
            if 'RMS level dB' in line:
                try:
                    db = float(line.split('RMS level dB:')[1].strip())
                    rms_values.append((current_t, db))
                except Exception:
                    pass

        if rms_values:
            # Normaliser et trouver les pics
            max_db  = max(v for _, v in rms_values if v > -100)
            min_db  = min(v for _, v in rms_values if v > -100)
            range_db = max_db - min_db if max_db != min_db else 1

            for t, db in rms_values:
                if db > -100:
                    norm = (db - min_db) / range_db
                    if norm > threshold_db:
                        spikes.append(t)

    except Exception as e:
        pass  # audio non disponible → silencieux

    return spikes


def detect_replay_segments(video_path, fps=25.0, sample_every=30):
    """
    Détecte les segments replay/ralenti en cherchant les répétitions visuelles.
    Heuristique simple : frames très similaires consécutives = ralenti.
    Retourne : [(start_s, end_s), ...] à EXCLURE
    """
    replays = []
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_frame = None
        slow_start = None

        for i in range(0, total, sample_every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            t = i / fps
            small = cv2.resize(frame, (64, 36))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame).mean()
                # Diff très faible = ralenti ou freeze
                if diff < 1.5:
                    if slow_start is None:
                        slow_start = t
                else:
                    if slow_start is not None and t - slow_start > 2.0:
                        replays.append((slow_start, t))
                    slow_start = None

            prev_frame = gray

        cap.release()
        if slow_start is not None:
            cap2 = cv2.VideoCapture(video_path)
            end_t = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            cap2.release()
            if end_t - slow_start > 2.0:
                replays.append((slow_start, end_t))

    except Exception:
        pass

    return replays


def filter_out_replays(segments, replay_segs, overlap_threshold=0.5):
    """Supprime les segments qui chevauchent trop avec des replays."""
    if not replay_segs:
        return segments
    result = []
    for s, e in segments:
        dur = e - s
        overlap = sum(
            min(e, re) - max(s, rs)
            for rs, re in replay_segs
            if min(e, re) > max(s, rs)
        )
        if dur > 0 and overlap / dur < overlap_threshold:
            result.append((s, e))
    return result


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

    # ── Source 4 : pics audio (réactions foule) ──────────────────────
    audio_spikes = detect_audio_spikes(video_path, fps=fps)
    if audio_spikes:
        print(f"  [COARSE] {len(audio_spikes)} pics audio détectés")
    for t in audio_spikes:
        if 0 <= t <= duration:
            for dt in [-1, 0, 1, 2, 3]:
                hot_times.add(round(t + dt, 1))

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

    # ── Exclure les segments replay/ralenti ──────────────────────────
    replay_segs = detect_replay_segments(video_path, fps=fps)
    if replay_segs:
        before = len(segments)
        segments = filter_out_replays(segments, replay_segs)
        print(f"  [COARSE] Replays filtrés : {before} → {len(segments)} segments")

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
        # Paramètres recommandés pour le pass 2
        "deep_imgsz":      DEEP_IMGSZ,
        "deep_frame_skip": DEEP_FRAME_SKIP,
        "deep_batch_size": DEEP_BATCH_SIZE,
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