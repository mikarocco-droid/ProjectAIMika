"""
zone_analyzer.py — Analyse dense des zones détectées par terminal_events

Principe :
  1. terminal_events détecte ~9 zones dangereuses sur 15 min
  2. On relit la vidéo en skip=1 UNIQUEMENT sur ces zones (~94s / 900s)
  3. On applique une détection physique précise (crossing + stuck + fast_disappear)
  4. On retourne des candidats haute confiance pour Gemini

Avantages vs posthoc global :
  - 4× plus de frames par zone (skip=1 vs skip=2)
  - Pas de bruit du reste du match
  - Trajectoires ballon plus propres → moins de FP
  - GPU : ~2.5 min supplémentaires seulement

Usage :
    from zone_analyzer import analyze_dense_zones

    dense_candidates = analyze_dense_zones(
        video_path       = video_path,
        terminal_events  = terminal_events,   # [{type, time, confidence}]
        fps              = fps,
        sport            = "football",
    )
"""

from __future__ import annotations
import cv2
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────
# Fenêtres de relecture par type d'événement (secondes)
# ─────────────────────────────────────────────────────────────
ZONE_REWIND = {
    "goal":             8,   # rewind 8s avant le signal → capture le tir
    "goalkeeper_save":  6,
    "clearance":        5,
}
ZONE_FORWARD = {
    "goal":             6,
    "goalkeeper_save":  4,
    "clearance":        4,
}

# ─────────────────────────────────────────────────────────────
# Paramètres détection physique (coordonnées normalisées 0-1)
# ─────────────────────────────────────────────────────────────
GOAL_X_LINE    = 0.05   # ligne de but = 5% depuis chaque bord
LINE_MARGIN    = 0.02
GOAL_Y_MIN     = 0.45
GOAL_Y_MAX     = 0.85
STUCK_MIN      = 4      # frames minimum dans zone but (tir lent)
FAST_SHOT_SPD  = 0.08   # vitesse normalisée = tir puissant
SPEED_MIN      = 0.04   # vitesse minimum pour valider crossing
REBOUND_DROP   = 0.4    # chute de vitesse = filet absorbe
SCORE_MIN      = 4.5    # score composite minimum pour candidat


def _ball_pos(fd: Dict) -> tuple | None:
    """Retourne (bx, by) normalisés depuis un frame_data, ou None."""
    ball = fd.get("ball")
    if not ball:
        return None
    fw = fd.get("frame_w") or 1920
    fh = fd.get("frame_h") or 1080
    x = ball.get("x") or ball.get("cx")
    y = ball.get("y") or ball.get("cy")
    if x is None or y is None:
        return None
    return (float(x) / fw, float(y) / fh)


def _read_zone_frames(
    video_path: str,
    t_start: float,
    t_end: float,
    fps: float,
    detector,
    tracker,
    ball_tracker,
    frame_w: int = 960,
    frame_h: int = 540,
) -> List[Dict]:
    """
    Relit la vidéo entre t_start et t_end en skip=1 (toutes les frames).
    Retourne une liste de frame_data avec positions ballon et joueurs.
    """
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # Seeking direct ffmpeg-style via OpenCV
    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = orig_w / frame_w
    scale_y = orig_h / frame_h

    frames_data = []
    frame_id    = int(t_start * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_t > t_end:
            break

        # Resize pour traitement
        small = cv2.resize(frame, (frame_w, frame_h))

        # Détection YOLO
        try:
            results = detector.model(
                small, imgsz=640, conf=0.25, verbose=False
            )
        except Exception:
            frame_id += 1
            continue

        ball   = None
        players = []

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cls == detector.ball_cls and conf > 0.20:
                    # Rescale vers résolution originale
                    ball = {
                        "x": cx * scale_x, "y": cy * scale_y,
                        "cx": cx, "cy": cy,
                        "conf": conf,
                    }
                elif cls == detector.player_cls and conf > 0.30:
                    players.append({
                        "cx": cx, "cy": cy,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf,
                    })

        # BallTracker pour lisser les détections
        if ball_tracker is not None:
            try:
                tracked_ball = ball_tracker.update(
                    ball, small, frame_id
                )
                if tracked_ball:
                    ball = tracked_ball
            except Exception:
                pass

        frames_data.append({
            "frame":   frame_id,
            "time":    current_t,
            "ball":    ball,
            "players": players,
            "frame_w": orig_w,
            "frame_h": orig_h,
        })
        frame_id += 1

    cap.release()
    return frames_data


def _detect_goals_in_frames(
    frames: List[Dict],
    fps: float,
    zone_t: float,
    zone_type: str,
) -> List[Dict]:
    """
    Détection physique précise sur frames denses (skip=1).
    Même logique que terminal_events._detect_goal + fast_disappear.
    """
    if len(frames) < 5:
        return []

    events = []

    # Vitesses horizontales normalisées
    speeds = [0.0]
    for k in range(1, len(frames)):
        p1 = _ball_pos(frames[k])
        p0 = _ball_pos(frames[k - 1])
        if p1 and p0:
            speeds.append(abs(p1[0] - p0[0]))
        else:
            speeds.append(0.0)

    last_goal_t = -999.0
    COOLDOWN    = 20.0

    i = 5
    while i < len(frames) - 5:
        fd  = frames[i]
        t   = fd.get("time", 0)
        pos = _ball_pos(fd)

        if pos is None:
            i += 1
            continue

        bx, by = pos

        # Filtre hauteur filet
        if not (GOAL_Y_MIN <= by <= GOAL_Y_MAX):
            i += 1
            continue

        pos_prev = _ball_pos(frames[i - 1])
        if pos_prev is None:
            i += 1
            continue

        bx_prev = pos_prev[0]

        # CROSSING
        cross_left  = (bx_prev > GOAL_X_LINE + LINE_MARGIN
                       and bx <= GOAL_X_LINE)
        cross_right = (bx_prev < (1 - GOAL_X_LINE - LINE_MARGIN)
                       and bx >= (1 - GOAL_X_LINE))

        if not (cross_left or cross_right):
            i += 1
            continue

        # VITESSE pic avant crossing
        peak_speed = max(speeds[max(0, i - 8):i + 1])
        if peak_speed < SPEED_MIN:
            i += 1
            continue

        # STUCK : ballon reste dans zone
        stuck = 0
        for j in range(i, min(i + 15, len(frames))):
            pj = _ball_pos(frames[j])
            if pj is None:
                break
            in_left  = pj[0] <= GOAL_X_LINE + LINE_MARGIN
            in_right = pj[0] >= (1 - GOAL_X_LINE - LINE_MARGIN)
            if (cross_left and in_left) or (cross_right and in_right):
                stuck += 1
            else:
                break

        # CAS SPÉCIAL : tir rapide → disparition immédiate
        next_none = sum(
            1 for j in range(i + 1, min(i + 4, len(frames)))
            if _ball_pos(frames[j]) is None
        )
        fast_disappear = (
            peak_speed >= FAST_SHOT_SPD
            and next_none >= 2
            and stuck <= 2
        )
        stuck_min_eff = 2 if fast_disappear else STUCK_MIN

        if stuck < stuck_min_eff:
            i += 1
            continue

        # REBOUND
        pre_spd  = max(speeds[max(0, i - 5):i]) if i > 0 else 0
        post_spd = max(speeds[i:min(i + 5, len(speeds))])
        rebound  = (pre_spd > 1e-4 and post_spd / pre_spd < REBOUND_DROP)

        # SCORE COMPOSITE
        score = 3.0
        if stuck >= 6:   score += 2.0
        elif stuck >= 4: score += 1.0
        if rebound:      score += 1.5
        if peak_speed > SPEED_MIN * 2: score += 0.5
        if fast_disappear: score += 2.0

        if score < SCORE_MIN:
            i += 1
            continue

        # Cooldown
        if t - last_goal_t < COOLDOWN:
            i += 1
            continue

        last_goal_t = t
        conf = min(0.72 + score * 0.04, 0.92)
        side = "gauche" if cross_left else "droite"
        mm, ss = int(t // 60), int(t % 60)

        print(f"  [DENSE] goal à {mm:02d}:{ss:02d} "
              f"| {side} bx={bx:.2f} by={by:.2f} "
              f"stuck={stuck}f peak={peak_speed:.3f} "
              f"fast={fast_disappear} score={score:.1f} conf={conf:.2f} "
              f"(zone terminal={zone_type} @{zone_t:.0f}s)")

        events.append({
            "type":            "goal",
            "time":            t,
            "frame":           int(t * fps),
            "confidence":      conf,
            "source":          "zone_analyzer",
            "detected_from":   "zone_analyzer_dense",
            "score":           score,
            "stuck":           stuck,
            "rebound":         rebound,
            "fast_disappear":  fast_disappear,
            "zone_type":       zone_type,
        })

        i += max(stuck, 8)

    return events


def analyze_dense_zones(
    video_path:      str,
    terminal_events: List[Dict],
    fps:             float,
    sport:           str = "football",
    frame_w:         int = 960,
    frame_h:         int = 540,
) -> List[Dict]:
    """
    Point d'entrée principal.
    Relit la vidéo en skip=1 sur les zones terminal_events.
    Retourne des candidats buts haute confiance.
    """
    if not terminal_events:
        return []

    # Import des détecteurs (même stack que process_video)
    try:
        from vision.detector import Detector
        from vision.tracker import Tracker
        detector     = Detector(sport=sport)
        tracker      = Tracker()
    except Exception as e:
        print(f"  [ZONE ANALYZER] Erreur init détecteurs : {e}")
        return []

    ball_tracker = None
    try:
        from vision.ball_tracker import BallTracker
        ball_tracker = BallTracker(max_history=30)
    except Exception:
        pass

    # Fusionner les fenêtres qui se chevauchent
    windows = []
    for ev in terminal_events:
        ev_type = ev.get("type", "goal")
        t       = float(ev.get("time", 0))
        rewind  = ZONE_REWIND.get(ev_type, 6)
        forward = ZONE_FORWARD.get(ev_type, 4)
        t0 = max(0.0, t - rewind)
        t1 = t + forward
        windows.append((t0, t1, ev_type, t))

    # Tri et fusion des fenêtres qui se chevauchent
    windows.sort(key=lambda w: w[0])
    merged = []
    for w in windows:
        if merged and w[0] <= merged[-1][1]:
            # Fusion : étendre la fenêtre existante
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], w[1]), prev[2], prev[3])
        else:
            merged.append(list(w))

    total_duration = sum(w[1] - w[0] for w in merged)
    print(f"\n[ZONE ANALYZER] {len(merged)} zone(s) dense(s) "
          f"| durée totale = {total_duration:.0f}s "
          f"({total_duration / (fps * 900) * fps * 100:.1f}% du match)")

    all_candidates = []

    for w in merged:
        t0, t1, zone_type, zone_t = w
        mm0, ss0 = int(t0 // 60), int(t0 % 60)
        mm1, ss1 = int(t1 // 60), int(t1 % 60)
        print(f"  → Zone {mm0:02d}:{ss0:02d}–{mm1:02d}:{ss1:02d} "
              f"({t1-t0:.0f}s) type={zone_type}")

        frames = _read_zone_frames(
            video_path  = video_path,
            t_start     = t0,
            t_end       = t1,
            fps         = fps,
            detector    = detector,
            tracker     = tracker,
            ball_tracker= ball_tracker,
            frame_w     = frame_w,
            frame_h     = frame_h,
        )

        if not frames:
            print(f"    ⚠️  Aucune frame lue pour cette zone")
            continue

        print(f"    {len(frames)} frames lues (skip=1)")

        candidates = _detect_goals_in_frames(
            frames    = frames,
            fps       = fps,
            zone_t    = zone_t,
            zone_type = zone_type,
        )
        all_candidates.extend(candidates)

    # Déduplication finale ±5s
    all_candidates.sort(key=lambda e: e["time"])
    deduped = []
    for c in all_candidates:
        if not deduped or abs(c["time"] - deduped[-1]["time"]) > 5.0:
            deduped.append(c)

    print(f"\n[ZONE ANALYZER] {len(deduped)} candidat(s) but détecté(s)")
    for c in deduped:
        mm, ss = int(c["time"] // 60), int(c["time"] % 60)
        print(f"  ⚽ {mm:02d}:{ss:02d} conf={c['confidence']:.2f} "
              f"score={c['score']:.1f} fast={c['fast_disappear']}")

    return deduped
