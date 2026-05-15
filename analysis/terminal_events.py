"""
terminal_events.py — Détecteur d'événements terminaux football

Détecte les fins naturelles d'actions dangereuses :
  - Arrêt gardien (save)
  - Ballon capté par le gardien
  - Corner
  - Six mètres (goal kick)
  - Dégagement d'urgence depuis la surface

Chaque événement terminal génère une fenêtre candidate [-15s, +2s]
qui sera validée par Gemini.

Usage :
    from analysis.terminal_events import detect_terminal_events, build_candidate_windows

    terminal = detect_terminal_events(frames_data, fps, frame_w, frame_h, goal_box)
    windows  = build_candidate_windows(terminal, rewind_sec=15)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────

# Zone de but : % de la largeur frame depuis chaque bord
GOAL_ZONE_PCT      = 0.10   # 10% = ~192px sur 1920

# Zone surface de réparation : % largeur depuis chaque bord
BOX_ZONE_PCT       = 0.20   # 20% = ~384px sur 1920

# Hauteur de jeu valide (éviter le bas de l'image = publicités)
PLAY_Y_MIN_PCT     = 0.35   # ignorer le dessus (score, ciel)
PLAY_Y_MAX_PCT     = 0.95

# Seuils vitesse ballon
SPEED_FAST         = 120.0  # px/frame — tir ou dégagement fort
SPEED_CLEARANCE    = 140.0  # px/frame — dégagement d'urgence

# Gardien : distance ballon/joueur pour "possession"
GK_BALL_DIST_MAX   = 80     # px
GK_POSSESSION_SEC  = 0.8    # secondes de possession minimum

# Cooldown entre deux événements du même type (éviter doublons)
COOLDOWN_SEC       = 8.0

# Fenêtre candidate
REWIND_SEC         = 15.0   # remonter avant l'événement terminal
WINDOW_END_SEC     = 2.0    # garder un peu après


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ball_pos(fd: Dict) -> Optional[tuple]:
    """Retourne (bx, by) normalisé [0-1] ou None."""
    ball = fd.get("ball") or {}
    bx = ball.get("x") or ball.get("cx")
    by = ball.get("y") or ball.get("cy")
    w  = fd.get("frame_w") or 1920
    h  = fd.get("frame_h") or 1080
    if bx is None or by is None:
        return None
    return (float(bx) / w, float(by) / h)


def _ball_speed(fd: Dict, frame_w: int = 1920) -> float:
    ball = fd.get("ball") or {}
    spd  = ball.get("speed") or 0
    return float(spd)


def _frame_time(fd: Dict, fps: float) -> float:
    return fd.get("frame", 0) / max(fps, 1)


def _is_in_goal_zone(bx_norm: float) -> bool:
    return bx_norm < GOAL_ZONE_PCT or bx_norm > (1 - GOAL_ZONE_PCT)


def _is_in_box_zone(bx_norm: float) -> bool:
    return bx_norm < BOX_ZONE_PCT or bx_norm > (1 - BOX_ZONE_PCT)


def _is_valid_y(by_norm: float) -> bool:
    return PLAY_Y_MIN_PCT <= by_norm <= PLAY_Y_MAX_PCT


def _find_goalkeeper(players: List[Dict], frame_w: int, frame_h: int) -> Optional[Dict]:
    """Trouve le gardien = joueur le plus proche des buts (bords latéraux)."""
    gks = [p for p in players if p.get("role") == "goalkeeper"
           or p.get("label", "").lower() in ("gk", "goalkeeper")]
    if gks:
        return gks[0]

    # Fallback : joueur le plus près d'un bord latéral
    best = None
    best_dist = float("inf")
    for p in players:
        px = float(p.get("x", p.get("cx", frame_w / 2)))
        dist = min(px, frame_w - px)
        if dist < best_dist:
            best_dist = dist
            best = p
    return best if best_dist < frame_w * 0.12 else None


def _dist_player_ball(player: Dict, ball: Dict, frame_w: int, frame_h: int) -> float:
    px = float(player.get("x", player.get("cx", 0)))
    py = float(player.get("y", player.get("cy", 0)))
    bx = float(ball.get("x", ball.get("cx", -999)))
    by = float(ball.get("y", ball.get("cy", -999)))
    return ((px - bx) ** 2 + (py - by) ** 2) ** 0.5


def _cooldown_ok(last_t: Dict[str, float], event_type: str, t: float) -> bool:
    return t - last_t.get(event_type, -999) > COOLDOWN_SEC


# ─────────────────────────────────────────────────────────────
# Détecteurs individuels
# ─────────────────────────────────────────────────────────────

def _detect_goalkeeper_save(
    frames: List[Dict],
    fps: float,
    last_t: Dict[str, float],
) -> List[Dict]:
    """
    Détecte : ballon rapide → ralentit brutalement près du gardien dans la surface.
    Signal : speed drop fort + gardien proche du ballon.
    """
    events = []
    window = 8  # frames à regarder en arrière

    for i in range(window, len(frames)):
        fd = frames[i]
        t  = _frame_time(fd, fps)
        w  = fd.get("frame_w") or 1920
        h  = fd.get("frame_h") or 1080

        ball = fd.get("ball") or {}
        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)

        if pos is None:
            continue
        bx_n, by_n = pos
        if not _is_in_box_zone(bx_n) or not _is_valid_y(by_n):
            continue

        # Vitesse actuelle faible
        if spd > 30:
            continue

        # Vitesse était forte dans la fenêtre précédente
        speeds_before = [_ball_speed(frames[j], w) for j in range(i - window, i)]
        max_spd_before = max(speeds_before) if speeds_before else 0
        if max_spd_before < SPEED_FAST:
            continue

        # Gardien proche du ballon
        players = fd.get("players") or []
        gk = _find_goalkeeper(players, w, h)
        if gk is None:
            continue

        dist = _dist_player_ball(gk, ball, w, h)
        if dist > GK_BALL_DIST_MAX * 1.5:
            continue

        if not _cooldown_ok(last_t, "goalkeeper_save", t):
            continue

        last_t["goalkeeper_save"] = t
        events.append({
            "type":       "goalkeeper_save",
            "time":       t,
            "confidence": min(0.95, 0.60 + (max_spd_before / 400)),
            "ball_speed_before": max_spd_before,
            "ball_speed_after":  spd,
            "gk_dist":    dist,
        })

    return events


def _detect_ball_caught(
    frames: List[Dict],
    fps: float,
    last_t: Dict[str, float],
) -> List[Dict]:
    """
    Détecte : ballon immobile + gardien dessus pendant > GK_POSSESSION_SEC.
    """
    events = []
    min_frames = max(1, int(GK_POSSESSION_SEC * fps / 4))  # /4 car skip=4
    possession_start = None
    possession_count = 0

    for i, fd in enumerate(frames):
        t  = _frame_time(fd, fps)
        w  = fd.get("frame_w") or 1920
        h  = fd.get("frame_h") or 1080
        ball = fd.get("ball") or {}
        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)

        if pos is None:
            possession_count = 0
            possession_start = None
            continue

        bx_n, by_n = pos
        if not _is_in_box_zone(bx_n) or not _is_valid_y(by_n):
            possession_count = 0
            possession_start = None
            continue

        if spd > 25:
            possession_count = 0
            possession_start = None
            continue

        players = fd.get("players") or []
        gk = _find_goalkeeper(players, w, h)
        if gk is None:
            possession_count = 0
            possession_start = None
            continue

        dist = _dist_player_ball(gk, ball, w, h)
        if dist > GK_BALL_DIST_MAX:
            possession_count = 0
            possession_start = None
            continue

        # Gardien près d'un ballon lent → possession
        if possession_start is None:
            possession_start = t
        possession_count += 1

        if possession_count >= min_frames:
            if _cooldown_ok(last_t, "ball_caught", t):
                last_t["ball_caught"] = t
                events.append({
                    "type":       "ball_caught",
                    "time":       possession_start,
                    "confidence": 0.80,
                    "duration":   t - possession_start,
                })
            possession_count = 0
            possession_start = None

    return events


def _detect_clearance(
    frames: List[Dict],
    fps: float,
    last_t: Dict[str, float],
) -> List[Dict]:
    """
    Détecte : ballon dans la surface → accélération brutale vers l'extérieur.
    Signal : balle en zone box + speed > SPEED_CLEARANCE + direction away from goal.
    """
    events = []

    for i in range(1, len(frames)):
        fd   = frames[i]
        fd_p = frames[i - 1]
        t    = _frame_time(fd, fps)
        w    = fd.get("frame_w") or 1920
        h    = fd.get("frame_h") or 1080

        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)
        pos_p = _ball_pos(fd_p)

        if pos is None or pos_p is None:
            continue
        bx_n, by_n   = pos
        bx_p_n, _    = pos_p

        if not _is_valid_y(by_n):
            continue

        # Vitesse forte
        if spd < SPEED_CLEARANCE:
            continue

        # Vient d'une zone dangereuse
        if not _is_in_box_zone(bx_p_n):
            continue

        # Direction : s'éloigne du but (vers le centre)
        center = 0.5
        moving_to_center = abs(bx_n - center) < abs(bx_p_n - center)
        if not moving_to_center:
            continue

        if not _cooldown_ok(last_t, "clearance", t):
            continue

        last_t["clearance"] = t
        events.append({
            "type":       "clearance",
            "time":       t,
            "confidence": min(0.90, 0.55 + spd / 600),
            "ball_speed": spd,
        })

    return events


def _detect_corner_or_goalkick(
    frames: List[Dict],
    fps: float,
    last_t: Dict[str, float],
) -> List[Dict]:
    """
    Détecte : ballon sort par la ligne de but (disparaît près du bord).
    Corner ou six mètres selon qui touche en dernier.
    Approximation : ballon disparaît dans zone de but + était en jeu juste avant.
    """
    events = []
    consecutive_missing = 0
    last_seen_pos = None
    last_seen_t   = None

    for fd in frames:
        t   = _frame_time(fd, fps)
        w   = fd.get("frame_w") or 1920
        h   = fd.get("frame_h") or 1080
        pos = _ball_pos(fd)

        if pos is not None:
            bx_n, by_n = pos
            if _is_valid_y(by_n):
                last_seen_pos = pos
                last_seen_t   = t
                consecutive_missing = 0
            else:
                consecutive_missing += 1
        else:
            consecutive_missing += 1

        # Ballon disparu 3+ frames consécutives alors qu'il était en zone de but
        if consecutive_missing >= 3 and last_seen_pos is not None:
            bx_last, by_last = last_seen_pos
            if _is_in_goal_zone(bx_last) and _is_valid_y(by_last):
                event_t = last_seen_t
                if _cooldown_ok(last_t, "corner_or_goalkick", event_t):
                    last_t["corner_or_goalkick"] = event_t
                    events.append({
                        "type":       "corner_or_goalkick",
                        "time":       event_t,
                        "confidence": 0.70,
                        "ball_last_x": bx_last,
                    })
                last_seen_pos = None  # reset
                consecutive_missing = 0

    return events


# ─────────────────────────────────────────────────────────────
# Interface principale
# ─────────────────────────────────────────────────────────────

def detect_terminal_events(
    frames_data: List[Dict],
    fps:         float      = 25.0,
    frame_w:     int        = 1920,
    frame_h:     int        = 1080,
    goal_box:    Any        = None,
) -> List[Dict]:
    """
    Détecte tous les événements terminaux dans frames_data.

    Retourne une liste d'événements triés par temps :
    [{"type": str, "time": float, "confidence": float, ...}, ...]
    """
    if not frames_data:
        return []

    last_t: Dict[str, float] = {}  # cooldown par type

    all_events = []
    all_events += _detect_goalkeeper_save(frames_data, fps, last_t)
    all_events += _detect_ball_caught(frames_data, fps, last_t)
    all_events += _detect_clearance(frames_data, fps, last_t)
    all_events += _detect_corner_or_goalkick(frames_data, fps, last_t)

    all_events.sort(key=lambda e: e["time"])

    print(f"  [TERMINAL EVENTS] détectés : {len(all_events)}")
    by_type: Dict[str, int] = {}
    for e in all_events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t:25} : {n}")

    return all_events


def build_candidate_windows(
    terminal_events: List[Dict],
    rewind_sec:      float = REWIND_SEC,
    end_sec:         float = WINDOW_END_SEC,
    existing_candidates: Optional[List[Dict]] = None,
    merge_radius_sec:    float = 5.0,
) -> List[Dict]:
    """
    Transforme les événements terminaux en fenêtres candidates.

    Fusionne avec les candidats existants (goal_posthoc) si fournis.
    Évite les doublons à ±merge_radius_sec.

    Retourne une liste de fenêtres :
    [{"time": float, "window_start": float, "window_end": float,
      "source": str, "confidence": float}, ...]
    """
    windows = []

    for ev in terminal_events:
        t = ev["time"]
        windows.append({
            "time":         t,
            "window_start": max(0, t - rewind_sec),
            "window_end":   t + end_sec,
            "source":       f"terminal_{ev['type']}",
            "confidence":   ev["confidence"],
            "terminal_type": ev["type"],
        })

    # Fusionner avec candidats existants
    if existing_candidates:
        for cand in existing_candidates:
            t = cand.get("time", 0)
            # Vérifier overlap avec les fenêtres terminal
            overlap = any(
                abs(t - w["time"]) < merge_radius_sec
                for w in windows
            )
            if not overlap:
                windows.append({
                    "time":         t,
                    "window_start": max(0, t - rewind_sec),
                    "window_end":   t + end_sec,
                    "source":       cand.get("source", "posthoc"),
                    "confidence":   cand.get("confidence", 0.5),
                })

    windows.sort(key=lambda w: w["time"])

    print(f"  [CANDIDATE WINDOWS] total : {len(windows)}")
    for w in windows:
        mm = int(w["time"] // 60)
        ss = int(w["time"] % 60)
        print(f"    {mm:02d}:{ss:02d}  [{w['source']:30}]  conf={w['confidence']:.2f}")

    return windows
