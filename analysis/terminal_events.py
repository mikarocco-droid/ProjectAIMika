"""
terminal_events.py — Détecteur d'événements terminaux multi-sport

Détecte les fins naturelles d'actions dangereuses.
Logique uniquement positionnelle — pas besoin de vitesse.

FOOTBALL   : goalkeeper_save, ball_caught, corner_or_goalkick, clearance
BASKETBALL : 3pt_made, 2pt_made, block, defensive_rebound
HANDBALL   : goal, goalkeeper_save, 7m_throw, fast_break
HOCKEY     : goalkeeper_save, powerplay_goal, penalty_shot, icing
RUGBY      : try, conversion, penalty_kick, tackle_in_22

Usage :
    from analysis.terminal_events import detect_terminal_events, build_candidate_windows

    terminal = detect_terminal_events(frames_data, fps, frame_w, frame_h, goal_box, sport="football")
    windows  = build_candidate_windows(terminal, frames_data=frames_data, fps=fps)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# Registre des événements terminaux par sport
# ─────────────────────────────────────────────────────────────

TERMINAL_EVENTS_BY_SPORT = {
    "football":   ["goalkeeper_save", "ball_caught", "corner_or_goalkick", "clearance"],
    "basketball": ["3pt_made", "2pt_made", "block", "defensive_rebound"],
    "handball":   ["goal", "goalkeeper_save", "7m_throw", "fast_break"],
    "hockey":     ["goalkeeper_save", "powerplay_goal", "penalty_shot", "icing"],
    "rugby":      ["try", "conversion", "penalty_kick", "tackle_in_22"],
}


# ─────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────

GOAL_ZONE_PCT    = 0.08   # 8% depuis chaque bord latéral = zone but
BOX_ZONE_PCT     = 0.22   # 22% = surface de réparation approximative
PLAY_Y_MIN_PCT   = 0.30   # ignorer le haut (score, ciel)
PLAY_Y_MAX_PCT   = 0.97

GK_BALL_DIST_MAX = 100    # px — gardien considéré "sur le ballon"
GK_POSSESS_FRAMES = 3     # nb frames consécutives gardien+ballon immobile

MISSING_FRAMES_CORNER = 5  # nb frames sans ballon pour déclencher corner/6m
CLEARANCE_ZONE_IN  = 0.25  # ballon vient de la zone < 25% du bord
CLEARANCE_ZONE_OUT = 0.40  # ballon va vers le centre > 40%

COOLDOWN_SEC     = 8.0
REWIND_SEC       = 15.0
WINDOW_END_SEC   = 2.0


# ─────────────────────────────────────────────────────────────
# Helpers positionnels
# ─────────────────────────────────────────────────────────────

def _ball_center(fd: Dict):
    """Retourne (cx, cy) en pixels ou None."""
    ball = fd.get("ball") or {}
    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return None


def _ball_norm(fd: Dict):
    """Retourne (bx_norm, by_norm) en [0-1] ou None."""
    c = _ball_center(fd)
    if c is None:
        return None
    w = fd.get("frame_w") or 1920
    h = fd.get("frame_h") or 1080
    return c[0] / w, c[1] / h


def _frame_t(fd: Dict, fps: float) -> float:
    return fd.get("frame", 0) / max(fps, 1)


def _in_goal_zone(bx_n: float) -> bool:
    return bx_n < GOAL_ZONE_PCT or bx_n > (1 - GOAL_ZONE_PCT)


def _in_box_zone(bx_n: float) -> bool:
    return bx_n < BOX_ZONE_PCT or bx_n > (1 - BOX_ZONE_PCT)


def _valid_y(by_n: float) -> bool:
    return PLAY_Y_MIN_PCT <= by_n <= PLAY_Y_MAX_PCT


def _gk_center(fd: Dict):
    """Trouve le gardien = joueur le plus proche d'un bord latéral."""
    players = fd.get("players") or []
    w = fd.get("frame_w") or 1920

    # Priorité : joueur avec role=goalkeeper
    for p in players:
        if p.get("role") == "goalkeeper" or p.get("label", "").lower() in ("gk", "goalkeeper"):
            bbox = p.get("bbox") or []
            if len(bbox) == 4:
                return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    # Fallback : joueur le plus près d'un bord latéral
    best, best_d = None, float("inf")
    for p in players:
        bbox = p.get("bbox") or []
        if len(bbox) < 4:
            continue
        px = (bbox[0] + bbox[2]) / 2.0
        d  = min(px, w - px)
        if d < best_d:
            best_d = d
            best   = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    return best if best_d < w * 0.12 else None


def _dist(p1, p2) -> float:
    if p1 is None or p2 is None:
        return float("inf")
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def _cooldown_ok(last_t: Dict, etype: str, t: float) -> bool:
    return t - last_t.get(etype, -999) > COOLDOWN_SEC


# ─────────────────────────────────────────────────────────────
# Détecteurs FOOTBALL
# ─────────────────────────────────────────────────────────────

def _detect_gk_possession(frames: List[Dict], fps: float, last_t: Dict) -> List[Dict]:
    """
    Gardien + corner ou goalkeeper_save :
    Ballon dans la surface ET gardien proche du ballon pendant GK_POSSESS_FRAMES frames.
    Signal purement positionnel — pas de vitesse.
    """
    events = []
    streak = 0
    streak_start_t = None

    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)
        bc  = _ball_center(fd)

        if pos is None or bc is None:
            streak = 0
            streak_start_t = None
            continue

        bx_n, by_n = pos
        if not _in_box_zone(bx_n) or not _valid_y(by_n):
            streak = 0
            streak_start_t = None
            continue

        gk = _gk_center(fd)
        if gk is None or _dist(gk, bc) > GK_BALL_DIST_MAX:
            streak = 0
            streak_start_t = None
            continue

        # Gardien sur le ballon dans la surface
        if streak == 0:
            streak_start_t = t
        streak += 1

        if streak >= GK_POSSESS_FRAMES:
            if _cooldown_ok(last_t, "goalkeeper_save", streak_start_t):
                last_t["goalkeeper_save"] = streak_start_t
                mm = int(streak_start_t // 60); ss = int(streak_start_t % 60)
                print(f"  [TERMINAL] goalkeeper_save à {mm:02d}:{ss:02d} "
                      f"| gk_dist={_dist(gk,bc):.0f}px streak={streak}f conf=0.82")
                events.append({
                    "type":       "goalkeeper_save",
                    "time":       streak_start_t,
                    "confidence": 0.82,
                })
            streak = 0
            streak_start_t = None

    return events


def _detect_corner_goalkick(frames: List[Dict], fps: float, last_t: Dict) -> List[Dict]:
    """
    Ballon disparaît (None) 3+ frames consécutives alors qu'il était
    dans la zone but (bx < 10% ou bx > 90%).
    → Corner ou six mètres.
    """
    events = []
    missing_count  = 0
    last_seen_pos  = None
    last_seen_t    = None

    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)

        if pos is not None:
            bx_n, by_n = pos
            if _valid_y(by_n):
                last_seen_pos = pos
                last_seen_t   = t
                missing_count = 0
            else:
                missing_count += 1
        else:
            missing_count += 1

        if missing_count >= MISSING_FRAMES_CORNER and last_seen_pos is not None:
            bx_last, by_last = last_seen_pos
            if _in_goal_zone(bx_last) and _valid_y(by_last):
                if _cooldown_ok(last_t, "corner_or_goalkick", last_seen_t):
                    last_t["corner_or_goalkick"] = last_seen_t
                    mm = int(last_seen_t // 60); ss = int(last_seen_t % 60)
                    print(f"  [TERMINAL] corner_or_goalkick à {mm:02d}:{ss:02d} "
                          f"| bx_last={bx_last:.2f} missing={missing_count}f conf=0.72")
                    events.append({
                        "type":       "corner_or_goalkick",
                        "time":       last_seen_t,
                        "confidence": 0.72,
                    })
                last_seen_pos = None
                missing_count = 0

    return events


def _detect_clearance(frames: List[Dict], fps: float, last_t: Dict) -> List[Dict]:
    """
    Dégagement : ballon passe de la zone surface (<25% bord)
    vers le centre (>40%) en 1-2 frames.
    Signal positionnel pur — pas de vitesse nécessaire.
    """
    events = []
    prev_pos = None

    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)

        if pos is None:
            prev_pos = None
            continue

        bx_n, by_n = pos

        if prev_pos is not None:
            bx_prev, _ = prev_pos
            w = fd.get("frame_w") or 1920

            # Vient d'une zone proche du but (surface)
            in_danger_before = bx_prev < CLEARANCE_ZONE_IN or bx_prev > (1 - CLEARANCE_ZONE_IN)
            # Va vers le centre
            going_center = (CLEARANCE_ZONE_OUT < bx_n < (1 - CLEARANCE_ZONE_OUT))

            if in_danger_before and going_center and _valid_y(by_n):
                if _cooldown_ok(last_t, "clearance", t):
                    last_t["clearance"] = t
                    mm = int(t // 60); ss = int(t % 60)
                    print(f"  [TERMINAL] clearance à {mm:02d}:{ss:02d} "
                          f"| bx {bx_prev:.2f}→{bx_n:.2f} conf=0.68")
                    events.append({
                        "type":       "clearance",
                        "time":       t,
                        "confidence": 0.68,
                    })

        prev_pos = pos

    return events



def _detect_goal(frames: List[Dict], fps: float, last_t: Dict) -> List[Dict]:
    """
    But : ballon dans la zone but (bx < 10% ou bx > 90%)
    ET dans la zone filet (hauteur entre 35% et 80% de l'image)
    pendant 2+ frames consécutives.
    Signal positionnel pur.
    """
    events = []
    streak = 0
    streak_start_t = None

    # Zone filet stricte : < 5% des bords latéraux ET hauteur filet réelle
    # Sur caméra latérale, le filet est très proche du bord
    GOAL_X_STRICT = 0.05   # 5% = ~96px sur 1920
    GOAL_Y_MIN    = 0.45   # pas trop haut (éviter ciel)
    GOAL_Y_MAX    = 0.85   # pas trop bas (éviter sol)

    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)

        if pos is None:
            streak = 0
            streak_start_t = None
            continue

        bx_n, by_n = pos

        # Zone très stricte : vraiment dans le filet
        in_goal_zone = (bx_n < GOAL_X_STRICT or bx_n > (1 - GOAL_X_STRICT))
        in_goal = in_goal_zone and GOAL_Y_MIN <= by_n <= GOAL_Y_MAX

        if in_goal:
            if streak == 0:
                streak_start_t = t
            streak += 1
        else:
            streak = 0
            streak_start_t = None
            continue

        if streak >= 2:
            if _cooldown_ok(last_t, "goal", streak_start_t):
                last_t["goal"] = streak_start_t
                mm = int(streak_start_t // 60); ss = int(streak_start_t % 60)
                print(f"  [TERMINAL] goal à {mm:02d}:{ss:02d} "
                      f"| bx={bx_n:.2f} by={by_n:.2f} streak={streak}f conf=0.88")
                events.append({
                    "type":       "goal",
                    "time":       streak_start_t,
                    "confidence": 0.88,
                })
            streak = 0
            streak_start_t = None

    return events

# ─────────────────────────────────────────────────────────────
# Stubs autres sports (à implémenter)
# ─────────────────────────────────────────────────────────────

def _stub(*args, **kwargs):
    return []


# ─────────────────────────────────────────────────────────────
# Registre des détecteurs
# ─────────────────────────────────────────────────────────────

_DETECTORS = {
    "football":   [_detect_goal, _detect_gk_possession, _detect_corner_goalkick, _detect_clearance],
    "basketball": [_stub],
    "handball":   [_stub],
    "hockey":     [_stub],
    "rugby":      [_stub],
}


# ─────────────────────────────────────────────────────────────
# Détection dynamique du début d'action
# ─────────────────────────────────────────────────────────────

def detect_action_start(
    frames_data:    List[Dict],
    event_time:     float,
    fps:            float,
    max_rewind_sec: float = 25.0,
    min_rewind_sec: float = 5.0,
) -> float:
    """
    Remonte depuis event_time pour trouver le timestamp exact du début d'action.
    Utilise uniquement les positions du ballon — pas de vitesse.

    Signaux par priorité :
      1. Ballon absent 3+ frames = remise en jeu
      2. Calme positionnel : ballon ne bouge plus pendant 1.5s
      3. Changement d'équipe en possession
      4. Fallback : event_time - 15s
    """
    if not frames_data or fps <= 0:
        return max(0, event_time - 15.0)

    search = [fd for fd in frames_data
              if event_time - max_rewind_sec <= _frame_t(fd, fps) < event_time]
    if not search:
        return max(0, event_time - 15.0)

    search_rev = list(reversed(search))

    # Signal 1 : ballon absent = remise en jeu
    consecutive_missing = 0
    for fd in search_rev:
        t = _frame_t(fd, fps)
        if _ball_norm(fd) is None:
            consecutive_missing += 1
        else:
            if consecutive_missing >= 3:
                action_start = t + 0.5
                action_start = max(event_time - max_rewind_sec,
                                   min(action_start, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] remise_en_jeu → {mm:02d}:{ss:02d} "
                      f"(absent {consecutive_missing}f)")
                return max(0, action_start)
            consecutive_missing = 0

    # Signal 2 : calme positionnel (ballon quasi immobile 1.5s)
    calm_positions = []
    calm_start_t   = None
    prev_c         = None

    for fd in search_rev:
        t  = _frame_t(fd, fps)
        c  = _ball_center(fd)
        w  = fd.get("frame_w") or 1920

        if c is None:
            calm_positions = []
            calm_start_t   = None
            prev_c         = None
            continue

        if prev_c is not None:
            move = _dist(c, prev_c) / w  # mouvement normalisé
            if move < 0.015:             # moins de 1.5% de la largeur
                if calm_start_t is None:
                    calm_start_t = t
                calm_positions.append(t)
            else:
                if calm_start_t is not None and (t - calm_start_t) >= 1.5:
                    action_start = max(event_time - max_rewind_sec,
                                       min(calm_start_t, event_time - min_rewind_sec))
                    mm = int(action_start//60); ss = int(action_start%60)
                    print(f"    [ACTION_START] calme_positionnel → {mm:02d}:{ss:02d} "
                          f"({t - calm_start_t:.1f}s)")
                    return max(0, action_start)
                calm_positions = []
                calm_start_t   = None

        prev_c = c

    # Signal 3 : changement d'équipe
    prev_team = None
    for fd in search_rev:
        t = _frame_t(fd, fps)
        players = fd.get("players") or []
        bc = _ball_center(fd)
        if bc is None:
            continue
        closest_team, closest_d = None, float("inf")
        for p in players:
            bbox = p.get("bbox") or []
            if len(bbox) < 4:
                continue
            pc = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            d  = _dist(pc, bc)
            if d < closest_d:
                closest_d    = d
                closest_team = p.get("team")
        if closest_team is not None and closest_d < 80:
            if prev_team is not None and closest_team != prev_team:
                action_start = max(event_time - max_rewind_sec,
                                   min(t, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] changement_equipe → {mm:02d}:{ss:02d} "
                      f"({prev_team}→{closest_team})")
                return max(0, action_start)
            prev_team = closest_team

    # Fallback
    fallback_t = max(0, event_time - 15.0)
    mm = int(fallback_t//60); ss = int(fallback_t%60)
    print(f"    [ACTION_START] fallback → {mm:02d}:{ss:02d}")
    return fallback_t


def detect_action_start_for_event(frames_data, terminal_event, fps) -> float:
    """Wrapper avec limites par type d'événement."""
    LIMITS = {
        "goal":              (5.0, 20.0),
        "goalkeeper_save":   (5.0, 12.0),
        "ball_caught":       (5.0, 20.0),
        "clearance":         (5.0, 18.0),
        "corner_or_goalkick":(5.0, 22.0),
        "3pt_made":          (3.0, 8.0),
        "2pt_made":          (3.0, 8.0),
        "block":             (2.0, 6.0),
        "fast_break":        (3.0, 6.0),
        "try":               (5.0, 25.0),
    }
    etype = terminal_event.get("type", "")
    et    = terminal_event.get("time", 0)
    min_r, max_r = LIMITS.get(etype, (5.0, 15.0))

    mm = int(et//60); ss = int(et%60)
    print(f"  [ACTION_START] '{etype}' à {mm:02d}:{ss:02d} | fenêtre [{min_r:.0f}s–{max_r:.0f}s]")

    return detect_action_start(frames_data, et, fps, max_r, min_r)


# ─────────────────────────────────────────────────────────────
# Interface principale
# ─────────────────────────────────────────────────────────────

def detect_terminal_events(
    frames_data,
    fps      = 25.0,
    frame_w  = 1920,
    frame_h  = 1080,
    goal_box = None,
    sport    = "football",
) -> List[Dict]:
    """Détecte tous les événements terminaux pour le sport donné."""
    if not frames_data:
        return []

    sport_key = sport.lower().strip()
    detectors = _DETECTORS.get(sport_key, _DETECTORS["football"])
    last_t    = {}

    all_events = []
    for detector in detectors:
        try:
            all_events += detector(frames_data, fps, last_t)
        except Exception as e:
            print(f"  [TERMINAL] {detector.__name__} ignoré : {e}")

    all_events.sort(key=lambda e: e["time"])

    print(f"  [TERMINAL EVENTS] sport={sport} | détectés={len(all_events)}")
    by_type: Dict[str, int] = {}
    for e in all_events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    for tp, n in sorted(by_type.items()):
        print(f"    {tp:30} : {n}")

    return all_events


def build_candidate_windows(
    terminal_events,
    rewind_sec          = REWIND_SEC,
    end_sec             = WINDOW_END_SEC,
    existing_candidates = None,
    merge_radius_sec    = 5.0,
    frames_data         = None,
    fps                 = 25.0,
) -> List[Dict]:
    """
    Transforme les événements terminaux en fenêtres candidates.
    Rewind dynamique si frames_data fourni, fixe sinon.
    """
    windows = []

    for ev in terminal_events:
        t = ev["time"]
        if frames_data is not None:
            action_start = detect_action_start_for_event(frames_data, ev, fps)
        else:
            action_start = max(0, t - rewind_sec)

        windows.append({
            "time":          t,
            "window_start":  action_start,
            "window_end":    t + end_sec,
            "source":        f"terminal_{ev['type']}",
            "confidence":    ev["confidence"],
            "terminal_type": ev["type"],
            "rewind_sec":    t - action_start,
        })

    if existing_candidates:
        for cand in existing_candidates:
            ct = cand.get("time", 0)
            if not any(abs(ct - w["time"]) < merge_radius_sec for w in windows):
                windows.append({
                    "time":         ct,
                    "window_start": max(0, ct - rewind_sec),
                    "window_end":   ct + end_sec,
                    "source":       cand.get("source", "posthoc"),
                    "confidence":   cand.get("confidence", 0.5),
                    "rewind_sec":   rewind_sec,
                })

    windows.sort(key=lambda w: w["time"])

    print(f"  [CANDIDATE WINDOWS] total={len(windows)}")
    for w in windows:
        mm = int(w["time"]//60); ss = int(w["time"]%60)
        print(f"    {mm:02d}:{ss:02d}  {w['source']:35} "
              f"conf={w['confidence']:.2f}  rewind={w['rewind_sec']:.0f}s")

    return windows
