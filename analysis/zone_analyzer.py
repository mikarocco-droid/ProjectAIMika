"""
zone_analyzer.py — Analyse dense des zones détectées par terminal_events

Principe :
  1. terminal_events détecte ~9 zones dangereuses sur 15 min
  2. On calcule dynamiquement la fenêtre de chaque zone depuis frames_data
     (pas de durées fixes — on remonte jusqu'au ballon hors zone dangereuse)
  3. On relit la vidéo en skip=1 UNIQUEMENT sur ces zones
  4. On applique une détection physique précise sur données 4× plus denses
  5. On retourne des candidats haute confiance pour Gemini

Fenêtres dynamiques :
  - Début : on remonte dans frames_data jusqu'à trouver le ballon
    hors de la zone dangereuse (bx > DANGER_ZONE_X), plafond MAX_REWIND
  - Fin   : on avance jusqu'à la prochaine pause de jeu ou MAX_FORWARD
  - Résultat : fenêtres adaptées à chaque action (3s pour contre-attaque,
    12s pour corner, etc.)
"""

from __future__ import annotations
import cv2
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────
# Paramètres fenêtres dynamiques
# ─────────────────────────────────────────────────────────────
DANGER_ZONE_X  = 0.30   # bx < 0.30 ou > 0.70 = zone dangereuse
MAX_REWIND     = 15.0   # plafond rewind (s) — évite fenêtres infinies
MAX_FORWARD    =  6.0   # plafond forward (s) après l'événement terminal
MIN_REWIND     =  3.0   # rewind minimum même si ballon déjà loin

# ─────────────────────────────────────────────────────────────
# Paramètres détection physique (coordonnées normalisées 0-1)
# ─────────────────────────────────────────────────────────────
GOAL_X_LINE    = 0.05
LINE_MARGIN    = 0.02
GOAL_Y_MIN     = 0.45
GOAL_Y_MAX     = 0.85
STUCK_MIN      = 4
FAST_SHOT_SPD  = 0.08
SPEED_MIN      = 0.04
REBOUND_DROP   = 0.4
SCORE_MIN      = 4.5
COOLDOWN       = 20.0


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ball_pos(fd: Dict) -> Optional[tuple]:
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


def _frame_time(fd: Dict, fps: float) -> float:
    """Retourne le timestamp en secondes d'un frame_data."""
    t = fd.get("time")
    if t is not None:
        return float(t)
    return float(fd.get("frame", 0)) / fps if fps > 0 else 0.0


def _is_in_danger_zone(bx: float) -> bool:
    """Ballon dans la zone dangereuse (proche d'un but)."""
    return bx < DANGER_ZONE_X or bx > (1.0 - DANGER_ZONE_X)


# ─────────────────────────────────────────────────────────────
# Calcul dynamique des fenêtres
# ─────────────────────────────────────────────────────────────

def compute_dynamic_window(
    event_t:     float,
    frames_data: List[Dict],
    fps:         float,
    next_event_t: Optional[float] = None,
) -> tuple:
    """
    Calcule (t_start, t_end) dynamiquement depuis frames_data.

    Début : remonte depuis event_t jusqu'à trouver le ballon
    hors zone dangereuse (bx entre 0.30 et 0.70), plafond MAX_REWIND.

    Fin : avance jusqu'à la prochaine pause de jeu (ballon absent
    plusieurs frames consécutives) ou jusqu'au prochain event terminal,
    plafond MAX_FORWARD.
    """
    if not frames_data:
        return (max(0.0, event_t - MAX_REWIND), event_t + MAX_FORWARD)

    # Indexer les frames par timestamp
    frames_near = [
        fd for fd in frames_data
        if abs(_frame_time(fd, fps) - event_t) <= MAX_REWIND + MAX_FORWARD
    ]
    frames_near.sort(key=lambda fd: _frame_time(fd, fps))

    # ── Calcul du début dynamique ──────────────────────────
    # On remonte depuis event_t : on cherche le premier moment
    # où le ballon quitte la zone dangereuse en allant vers l'arrière
    frames_before = [
        fd for fd in frames_near
        if _frame_time(fd, fps) <= event_t
    ]
    frames_before.sort(key=lambda fd: _frame_time(fd, fps), reverse=True)

    t_start = max(0.0, event_t - MAX_REWIND)  # fallback
    consecutive_safe = 0

    for fd in frames_before:
        t = _frame_time(fd, fps)
        if event_t - t > MAX_REWIND:
            break

        pos = _ball_pos(fd)
        if pos is None:
            consecutive_safe += 1
            if consecutive_safe >= 3:
                # Ballon absent 3+ frames = pause = début d'action
                t_start = max(0.0, t - 0.5)
                break
            continue

        consecutive_safe = 0
        bx, _ = pos

        if not _is_in_danger_zone(bx):
            # Ballon hors zone dangereuse → c'est le début de l'action
            t_start = max(0.0, t - 0.5)  # -0.5s de marge
            break

    # Garantir rewind minimum
    t_start = min(t_start, event_t - MIN_REWIND)
    t_start = max(0.0, t_start)

    # ── Calcul de la fin dynamique ─────────────────────────
    # On avance depuis event_t : fin = prochaine pause de jeu
    # (ballon absent 4+ frames consécutives) ou prochain event terminal
    frames_after = [
        fd for fd in frames_near
        if _frame_time(fd, fps) >= event_t
    ]
    frames_after.sort(key=lambda fd: _frame_time(fd, fps))

    t_end = event_t + MAX_FORWARD  # fallback
    absent_streak = 0

    for fd in frames_after:
        t = _frame_time(fd, fps)
        if t - event_t > MAX_FORWARD:
            break

        # Stopper au prochain event terminal si proche
        if next_event_t is not None and t >= next_event_t - 1.0:
            t_end = t
            break

        pos = _ball_pos(fd)
        if pos is None:
            absent_streak += 1
            if absent_streak >= 4:
                # Pause de jeu → fin de l'action
                t_end = t
                break
        else:
            absent_streak = 0

    duration = t_end - t_start
    return (t_start, t_end, duration)


# ─────────────────────────────────────────────────────────────
# Relecture dense d'une zone
# ─────────────────────────────────────────────────────────────

def _read_zone_frames(
    video_path:  str,
    t_start:     float,
    t_end:       float,
    fps:         float,
    detector,
    ball_tracker,
    frame_w:     int = 960,
    frame_h:     int = 540,
) -> List[Dict]:
    """
    Relit la vidéo entre t_start et t_end en skip=1 (toutes les frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)

    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x  = orig_w / frame_w
    scale_y  = orig_h / frame_h
    frame_id = int(t_start * fps)

    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_t > t_end:
            break

        small = cv2.resize(frame, (frame_w, frame_h))

        # Détection YOLO
        ball    = None
        players = []
        try:
            results = detector.model(small, imgsz=640, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if cls == detector.ball_cls and conf > 0.20:
                        ball = {
                            "x": cx * scale_x, "y": cy * scale_y,
                            "cx": cx, "cy": cy, "conf": conf,
                        }
                    elif cls == detector.player_cls and conf > 0.30:
                        players.append({
                            "cx": cx, "cy": cy,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        })
        except Exception:
            pass

        # BallTracker
        if ball_tracker is not None:
            try:
                tracked = ball_tracker.update(ball, small, frame_id)
                if tracked:
                    ball = tracked
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


# ─────────────────────────────────────────────────────────────
# Détection physique sur frames denses
# ─────────────────────────────────────────────────────────────

def _detect_goals_in_frames(
    frames:    List[Dict],
    fps:       float,
    zone_t:    float,
    zone_type: str,
) -> List[Dict]:
    """
    Détection physique précise sur frames denses (skip=1).
    Crossing + stuck + fast_disappear.
    """
    if len(frames) < 5:
        return []

    events = []

    # Vitesses horizontales normalisées
    speeds = [0.0]
    for k in range(1, len(frames)):
        p1 = _ball_pos(frames[k])
        p0 = _ball_pos(frames[k - 1])
        speeds.append(abs(p1[0] - p0[0]) if p1 and p0 else 0.0)

    last_goal_t = -999.0
    i = 5

    while i < len(frames) - 5:
        fd  = frames[i]
        t   = fd.get("time", 0)
        pos = _ball_pos(fd)

        if pos is None:
            i += 1
            continue

        bx, by = pos

        if not (GOAL_Y_MIN <= by <= GOAL_Y_MAX):
            i += 1
            continue

        pos_prev = _ball_pos(frames[i - 1])
        if pos_prev is None:
            i += 1
            continue

        bx_prev = pos_prev[0]

        # CROSSING
        cross_left  = bx_prev > GOAL_X_LINE + LINE_MARGIN and bx <= GOAL_X_LINE
        cross_right = bx_prev < (1 - GOAL_X_LINE - LINE_MARGIN) and bx >= (1 - GOAL_X_LINE)

        if not (cross_left or cross_right):
            i += 1
            continue

        # PIC DE VITESSE
        peak_speed = max(speeds[max(0, i - 8):i + 1])
        if peak_speed < SPEED_MIN:
            i += 1
            continue

        # STUCK
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
        fast_disappear    = peak_speed >= FAST_SHOT_SPD and next_none >= 2 and stuck <= 2
        stuck_min_eff     = 2 if fast_disappear else STUCK_MIN

        if stuck < stuck_min_eff:
            i += 1
            continue

        # REBOUND
        pre_spd  = max(speeds[max(0, i - 5):i]) if i > 0 else 0
        post_spd = max(speeds[i:min(i + 5, len(speeds))])
        rebound  = pre_spd > 1e-4 and post_spd / pre_spd < REBOUND_DROP

        # SCORE
        score = 3.0
        if stuck >= 6:          score += 2.0
        elif stuck >= 4:        score += 1.0
        if rebound:             score += 1.5
        if peak_speed > SPEED_MIN * 2: score += 0.5
        if fast_disappear:      score += 2.0

        if score < SCORE_MIN:
            i += 1
            continue

        if t - last_goal_t < COOLDOWN:
            i += 1
            continue

        last_goal_t = t
        conf = min(0.72 + score * 0.04, 0.92)
        side = "gauche" if cross_left else "droite"
        mm, ss = int(t // 60), int(t % 60)

        print(f"  [DENSE] goal à {mm:02d}:{ss:02d} | {side} "
              f"bx={bx:.2f} stuck={stuck}f peak={peak_speed:.3f} "
              f"fast={fast_disappear} score={score:.1f} conf={conf:.2f} "
              f"(zone={zone_type} @{zone_t:.0f}s)")

        events.append({
            "type":           "goal",
            "time":           t,
            "frame":          int(t * fps),
            "confidence":     conf,
            "source":         "zone_analyzer",
            "detected_from":  "zone_analyzer_dense",
            "score":          score,
            "stuck":          stuck,
            "rebound":        rebound,
            "fast_disappear": fast_disappear,
            "zone_type":      zone_type,
        })
        i += max(stuck, 8)

    return events


# ─────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────

def analyze_dense_zones(
    video_path:      str,
    terminal_events: List[Dict],
    fps:             float,
    frames_data:     Optional[List[Dict]] = None,
    sport:           str = "football",
    frame_w:         int = 960,
    frame_h:         int = 540,
) -> List[Dict]:
    """
    Relit la vidéo en skip=1 sur les zones terminal_events.
    Les fenêtres sont calculées dynamiquement depuis frames_data du pass 1.
    Retourne des candidats buts haute confiance.
    """
    if not terminal_events:
        return []

    # Init détecteurs
    try:
        from vision.detector import Detector
        detector = Detector(sport=sport)
    except Exception as e:
        print(f"  [ZONE ANALYZER] Erreur init détecteur : {e}")
        return []

    ball_tracker = None
    try:
        from vision.ball_tracker import BallTracker
        ball_tracker = BallTracker(max_history=30)
    except Exception:
        pass

    # ── Calcul des fenêtres dynamiques ────────────────────────────────────
    events_sorted = sorted(terminal_events, key=lambda e: e.get("time", 0))
    windows = []

    for idx, ev in enumerate(events_sorted):
        ev_type = ev.get("type", "goal")
        t       = float(ev.get("time", 0))

        # Prochain event terminal (pour borner la fin)
        next_t = float(events_sorted[idx + 1]["time"]) if idx + 1 < len(events_sorted) else None

        if frames_data:
            t0, t1, dur = compute_dynamic_window(
                event_t      = t,
                frames_data  = frames_data,
                fps          = fps,
                next_event_t = next_t,
            )
        else:
            # Fallback si frames_data absent
            t0, t1, dur = max(0.0, t - 8.0), t + 5.0, 13.0

        windows.append((t0, t1, ev_type, t, dur))
        mm0, ss0 = int(t0 // 60), int(t0 % 60)
        mm1, ss1 = int(t1 // 60), int(t1 % 60)
        print(f"  [ZONE] {mm0:02d}:{ss0:02d}–{mm1:02d}:{ss1:02d} "
              f"({dur:.1f}s dynamique) type={ev_type}")

    # ── Fusion des fenêtres qui se chevauchent ────────────────────────────
    windows.sort(key=lambda w: w[0])
    merged = []
    for w in windows:
        if merged and w[0] <= merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], w[1]), prev[2], prev[3],
                          max(prev[1], w[1]) - prev[0])
        else:
            merged.append(list(w))

    total_dur = sum(w[1] - w[0] for w in merged)
    video_dur = (frames_data[-1].get("time", 900) if frames_data else 900)
    print(f"\n[ZONE ANALYZER] {len(merged)} zone(s) | "
          f"durée totale = {total_dur:.0f}s "
          f"({total_dur / video_dur * 100:.1f}% du match)")

    # ── Analyse dense zone par zone ───────────────────────────────────────
    all_candidates = []

    for w in merged:
        t0, t1, zone_type, zone_t = w[0], w[1], w[2], w[3]
        mm0, ss0 = int(t0 // 60), int(t0 % 60)
        mm1, ss1 = int(t1 // 60), int(t1 % 60)
        print(f"\n  → Lecture dense {mm0:02d}:{ss0:02d}–{mm1:02d}:{ss1:02d} "
              f"({t1-t0:.1f}s)")

        frames = _read_zone_frames(
            video_path   = video_path,
            t_start      = t0,
            t_end        = t1,
            fps          = fps,
            detector     = detector,
            ball_tracker = ball_tracker,
            frame_w      = frame_w,
            frame_h      = frame_h,
        )

        if not frames:
            print(f"    ⚠️  Aucune frame lue")
            continue

        print(f"    {len(frames)} frames (skip=1 vs ~{int((t1-t0)*fps/2)} en skip=2)")

        candidates = _detect_goals_in_frames(
            frames    = frames,
            fps       = fps,
            zone_t    = zone_t,
            zone_type = zone_type,
        )
        all_candidates.extend(candidates)

    # ── Déduplication finale ±5s ──────────────────────────────────────────
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
