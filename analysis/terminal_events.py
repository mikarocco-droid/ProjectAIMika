"""
terminal_events.py — Détecteur d'événements terminaux multi-sport

Chaque sport définit ses "fins naturelles d'action dangereuse".
Le pipeline est identique pour tous les sports — seuls les détecteurs changent.

FOOTBALL   : save, ball_caught, clearance, corner, goalkick
BASKETBALL : 3pt_made, 2pt_made, block, defensive_rebound
HANDBALL   : goal, goalkeeper_save, 7m_throw, fast_break
HOCKEY     : save, powerplay_goal, penalty_shot
RUGBY      : try, conversion, penalty_kick, tackle_in_22

Usage :
    from analysis.terminal_events import detect_terminal_events, build_candidate_windows

    terminal = detect_terminal_events(frames_data, fps, frame_w, frame_h, goal_box, sport="football")
    windows  = build_candidate_windows(terminal, rewind_sec=15)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# Registre des événements terminaux par sport
# ─────────────────────────────────────────────────────────────

TERMINAL_EVENTS_BY_SPORT = {
    "football": [
        "goalkeeper_save",      # ballon rapide → s'arrête près du gardien
        "ball_caught",          # gardien capte le ballon
        "clearance",            # dégagement d'urgence depuis la surface
        "corner_or_goalkick",   # ballon sort par la ligne de but
    ],
    "basketball": [
        "3pt_made",             # tir à 3 points marqué
        "2pt_made",             # beau tir à 2 points
        "block",                # contre
        "defensive_rebound",    # rebond défensif (fin d'attaque)
    ],
    "handball": [
        "goal",                 # but marqué
        "goalkeeper_save",      # arrêt du gardien
        "7m_throw",             # jet de 7m
        "fast_break",           # contre-attaque rapide
    ],
    "hockey": [
        "goalkeeper_save",      # arrêt du gardien
        "powerplay_goal",       # but en supériorité numérique
        "penalty_shot",         # tir de pénalité
        "icing",                # dégagement interdit
    ],
    "rugby": [
        "try",                  # essai
        "conversion",           # transformation
        "penalty_kick",         # coup de pied de pénalité
        "tackle_in_22",         # plaquage dans les 22m
    ],
    "tennis": [
        "ace",                  # ace
        "double_fault",         # double faute
        "winner",               # coup gagnant
        "net_point",            # point au filet
    ],
    "volleyball": [
        "spike_point",          # smash gagnant
        "block_point",          # contre gagnant
        "service_ace",          # ace au service
        "dig_save",             # réception difficile
    ],
}


# ─────────────────────────────────────────────────────────────
# Paramètres globaux
# ─────────────────────────────────────────────────────────────

GOAL_ZONE_PCT      = 0.10
BOX_ZONE_PCT       = 0.20
PLAY_Y_MIN_PCT     = 0.35
PLAY_Y_MAX_PCT     = 0.95
SPEED_FAST         = 120.0
SPEED_CLEARANCE    = 140.0
GK_BALL_DIST_MAX   = 80
GK_POSSESSION_SEC  = 0.8
COOLDOWN_SEC       = 8.0
REWIND_SEC         = 15.0
WINDOW_END_SEC     = 2.0


# ─────────────────────────────────────────────────────────────
# Helpers communs
# ─────────────────────────────────────────────────────────────

def _ball_pos(fd):
    ball = fd.get("ball") or {}
    bx = ball.get("x") or ball.get("cx")
    by = ball.get("y") or ball.get("cy")
    w  = fd.get("frame_w") or 1920
    h  = fd.get("frame_h") or 1080
    if bx is None or by is None:
        return None
    return (float(bx) / w, float(by) / h)

def _ball_speed(fd, frame_w=1920):
    return float((fd.get("ball") or {}).get("speed") or 0)

def _frame_time(fd, fps):
    return fd.get("frame", 0) / max(fps, 1)

def _is_in_goal_zone(bx_norm):
    return bx_norm < GOAL_ZONE_PCT or bx_norm > (1 - GOAL_ZONE_PCT)

def _is_in_box_zone(bx_norm):
    return bx_norm < BOX_ZONE_PCT or bx_norm > (1 - BOX_ZONE_PCT)

def _is_valid_y(by_norm):
    return PLAY_Y_MIN_PCT <= by_norm <= PLAY_Y_MAX_PCT

def _find_goalkeeper(players, frame_w, frame_h):
    gks = [p for p in players if p.get("role") == "goalkeeper"
           or p.get("label", "").lower() in ("gk", "goalkeeper")]
    if gks:
        return gks[0]
    best, best_dist = None, float("inf")
    for p in players:
        px = float(p.get("x", p.get("cx", frame_w / 2)))
        dist = min(px, frame_w - px)
        if dist < best_dist:
            best_dist = dist
            best = p
    return best if best_dist < frame_w * 0.12 else None

def _dist_player_ball(player, ball, frame_w, frame_h):
    px = float(player.get("x", player.get("cx", 0)))
    py = float(player.get("y", player.get("cy", 0)))
    bx = float(ball.get("x", ball.get("cx", -999)))
    by = float(ball.get("y", ball.get("cy", -999)))
    return ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

def _cooldown_ok(last_t, event_type, t):
    return t - last_t.get(event_type, -999) > COOLDOWN_SEC


# ─────────────────────────────────────────────────────────────
# Détecteurs FOOTBALL
# ─────────────────────────────────────────────────────────────

def _detect_goalkeeper_save(frames, fps, last_t):
    events = []
    window = 8
    for i in range(window, len(frames)):
        fd  = frames[i]
        t   = _frame_time(fd, fps)
        w   = fd.get("frame_w") or 1920
        h   = fd.get("frame_h") or 1080
        ball = fd.get("ball") or {}
        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)
        if pos is None:
            continue
        bx_n, by_n = pos
        if not _is_in_box_zone(bx_n) or not _is_valid_y(by_n):
            continue
        if spd > 30:
            continue
        speeds_before = [_ball_speed(frames[j], w) for j in range(i - window, i)]
        max_spd_before = max(speeds_before) if speeds_before else 0
        if max_spd_before < SPEED_FAST:
            continue
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
        conf = min(0.95, 0.60 + (max_spd_before / 400))
        mm = int(t//60); ss = int(t%60)
        print(f"  [TERMINAL] goalkeeper_save à {mm:02d}:{ss:02d} | spd_before={max_spd_before:.0f}px spd_after={spd:.0f}px gk_dist={dist:.0f}px conf={conf:.2f}")
        events.append({
            "type":       "goalkeeper_save",
            "time":       t,
            "confidence": conf,
        })
    return events


def _detect_ball_caught(frames, fps, last_t):
    events = []
    min_frames = max(1, int(GK_POSSESSION_SEC * fps / 4))
    possession_start = None
    possession_count = 0
    for fd in frames:
        t    = _frame_time(fd, fps)
        w    = fd.get("frame_w") or 1920
        h    = fd.get("frame_h") or 1080
        ball = fd.get("ball") or {}
        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)
        if pos is None or spd > 25:
            possession_count = 0; possession_start = None; continue
        bx_n, by_n = pos
        if not _is_in_box_zone(bx_n) or not _is_valid_y(by_n):
            possession_count = 0; possession_start = None; continue
        players = fd.get("players") or []
        gk = _find_goalkeeper(players, w, h)
        if gk is None:
            possession_count = 0; possession_start = None; continue
        dist = _dist_player_ball(gk, ball, w, h)
        if dist > GK_BALL_DIST_MAX:
            possession_count = 0; possession_start = None; continue
        if possession_start is None:
            possession_start = t
        possession_count += 1
        if possession_count >= min_frames:
            if _cooldown_ok(last_t, "ball_caught", t):
                last_t["ball_caught"] = t
                mm = int(possession_start//60); ss = int(possession_start%60)
                print(f"  [TERMINAL] ball_caught à {mm:02d}:{ss:02d} | durée={t-possession_start:.1f}s conf=0.80")
                events.append({"type": "ball_caught", "time": possession_start, "confidence": 0.80})
            possession_count = 0; possession_start = None
    return events


def _detect_clearance(frames, fps, last_t):
    events = []
    for i in range(1, len(frames)):
        fd   = frames[i]
        fd_p = frames[i - 1]
        t    = _frame_time(fd, fps)
        w    = fd.get("frame_w") or 1920
        spd  = _ball_speed(fd, w)
        pos  = _ball_pos(fd)
        pos_p = _ball_pos(fd_p)
        if pos is None or pos_p is None or spd < SPEED_CLEARANCE:
            continue
        bx_n, by_n   = pos
        bx_p_n, _    = pos_p
        if not _is_valid_y(by_n) or not _is_in_box_zone(bx_p_n):
            continue
        if abs(bx_n - 0.5) >= abs(bx_p_n - 0.5):
            continue
        if not _cooldown_ok(last_t, "clearance", t):
            continue
        last_t["clearance"] = t
        conf = min(0.90, 0.55 + spd / 600)
        mm = int(t//60); ss = int(t%60)
        print(f"  [TERMINAL] clearance à {mm:02d}:{ss:02d} | spd={spd:.0f}px conf={conf:.2f}")
        events.append({"type": "clearance", "time": t, "confidence": conf})
    return events


def _detect_corner_or_goalkick(frames, fps, last_t):
    events = []
    consecutive_missing = 0
    last_seen_pos = None
    last_seen_t   = None
    for fd in frames:
        t   = _frame_time(fd, fps)
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
        if consecutive_missing >= 3 and last_seen_pos is not None:
            bx_last, by_last = last_seen_pos
            if _is_in_goal_zone(bx_last) and _is_valid_y(by_last):
                if _cooldown_ok(last_t, "corner_or_goalkick", last_seen_t):
                    last_t["corner_or_goalkick"] = last_seen_t
                    mm = int(last_seen_t//60); ss = int(last_seen_t%60)
                    print(f"  [TERMINAL] corner_or_goalkick à {mm:02d}:{ss:02d} | bx_last={bx_last:.2f} conf=0.70")
                    events.append({"type": "corner_or_goalkick", "time": last_seen_t,
                                   "confidence": 0.70})
                last_seen_pos = None
                consecutive_missing = 0
    return events


# ─────────────────────────────────────────────────────────────
# Détecteurs BASKETBALL (stubs — à implémenter)
# ─────────────────────────────────────────────────────────────

def _detect_basketball_3pt(frames, fps, last_t):
    """
    TODO : détecter tir à 3 points marqué.
    Signal : joueur derrière la ligne des 3pts + ballon disparaît dans zone panier
             + regroupement joueurs autour du cercle.
    """
    return []

def _detect_basketball_2pt(frames, fps, last_t):
    """
    TODO : détecter beau tir à 2 points.
    Signal : ballon en cloche vers panier + disparition + rebond
    """
    return []

def _detect_basketball_block(frames, fps, last_t):
    """
    TODO : détecter contre.
    Signal : ballon rapide vers panier → déviation violente vers l'extérieur
    """
    return []

def _detect_basketball_defensive_rebound(frames, fps, last_t):
    """
    TODO : détecter rebond défensif.
    Signal : ballon rebondit haut + récupéré par défense sous le panier
    """
    return []


# ─────────────────────────────────────────────────────────────
# Détecteurs HANDBALL (stubs — à implémenter)
# ─────────────────────────────────────────────────────────────

def _detect_handball_goal(frames, fps, last_t):
    """
    TODO : ballon franchit la ligne de but + gardien ne bloque pas
    """
    return []

def _detect_handball_goalkeeper_save(frames, fps, last_t):
    """
    TODO : tir rapide + arrêt dans les 6m
    """
    return []

def _detect_handball_7m(frames, fps, last_t):
    """
    TODO : joueur isolé face au gardien depuis les 7m
    """
    return []

def _detect_handball_fast_break(frames, fps, last_t):
    """
    TODO : ballon progresse rapidement vers le but en 1-2 passes
    """
    return []


# ─────────────────────────────────────────────────────────────
# Détecteurs HOCKEY (stubs — à implémenter)
# ─────────────────────────────────────────────────────────────

def _detect_hockey_save(frames, fps, last_t):
    """TODO"""
    return []

def _detect_hockey_powerplay_goal(frames, fps, last_t):
    """TODO"""
    return []

def _detect_hockey_penalty_shot(frames, fps, last_t):
    """TODO"""
    return []

def _detect_hockey_icing(frames, fps, last_t):
    """TODO"""
    return []


# ─────────────────────────────────────────────────────────────
# Détecteurs RUGBY (stubs — à implémenter)
# ─────────────────────────────────────────────────────────────

def _detect_rugby_try(frames, fps, last_t):
    """TODO : ballon aplatit derrière la ligne d'en-but"""
    return []

def _detect_rugby_conversion(frames, fps, last_t):
    """TODO : tir au but après essai"""
    return []

def _detect_rugby_penalty_kick(frames, fps, last_t):
    """TODO : coup de pied de pénalité"""
    return []

def _detect_rugby_tackle_in_22(frames, fps, last_t):
    """TODO : plaquage dans les 22m adverses"""
    return []



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
    Remonte dans frames_data depuis event_time pour trouver
    le timestamp exact du début de l'action.

    Retourne un TIMESTAMP (pas un offset) — le moment précis
    où l'action a commencé, quelle que soit la durée.

    Signaux détectés par ordre de priorité :
      1. Ballon absent 3+ frames  → remise en jeu = début naturel
      2. Calme prolongé >1.5s     → possession placée avant action
      3. Transition équipe        → début de contre-attaque
      4. Fallback                 → event_time - 15s
    """
    if not frames_data or fps <= 0:
        return max(0, event_time - 15.0)

    # Construire la fenêtre de recherche (frames avant event_time)
    search_frames = []
    for fd in frames_data:
        t = _frame_time(fd, fps)
        if event_time - max_rewind_sec <= t < event_time:
            search_frames.append(fd)

    if not search_frames:
        return max(0, event_time - 15.0)

    frame_w = search_frames[0].get("frame_w") or 1920

    # Inverser : on remonte du plus récent vers le plus ancien
    search_reversed = list(reversed(search_frames))

    # ── Signal 1 : ballon absent 3+ frames = remise en jeu ──
    # Le début de l'action = la frame JUSTE APRÈS la remise en jeu
    consecutive_missing = 0
    last_present_t = event_time
    for fd in search_reversed:
        t   = _frame_time(fd, fps)
        pos = _ball_pos(fd)
        if pos is None:
            consecutive_missing += 1
        else:
            if consecutive_missing >= 3:
                # t = dernier instant où le ballon était absent
                # → action commence juste après = t + 0.5s
                action_start = t + 0.5
                action_start = max(event_time - max_rewind_sec,
                                   min(action_start, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] signal=remise_en_jeu → {mm:02d}:{ss:02d} (ballon absent {consecutive_missing} frames)")
                return max(0, action_start)
            consecutive_missing = 0
            last_present_t = t

    # ── Signal 2 : calme prolongé >1.5s = fin de phase précédente ──
    # Remonter jusqu'au dernier moment calme continu
    # Le début de l'action = quand le calme se termine (vitesse repart)
    calm_start_t  = None
    calm_duration = 0.0
    prev_t        = None

    for fd in search_reversed:
        t   = _frame_time(fd, fps)
        spd = _ball_speed(fd, frame_w)
        dt  = abs(t - prev_t) if prev_t is not None else (1.0 / fps)

        if spd < 25:
            # Ballon lent : on accumule le calme
            if calm_start_t is None:
                calm_start_t = t
            calm_duration += dt
        else:
            # Ballon rapide : fin du calme
            if calm_duration >= 1.5 and calm_start_t is not None:
                # L'action a commencé quand le calme s'est terminé
                # = calm_start_t (en remontant) = fin du calme en temps réel
                action_start = calm_start_t
                action_start = max(event_time - max_rewind_sec,
                                   min(action_start, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] signal=calme_prolongé → {mm:02d}:{ss:02d} (calme={calm_duration:.1f}s)")
                return max(0, action_start)
            calm_start_t  = None
            calm_duration = 0.0

        prev_t = t

    # ── Signal 3 : changement d'équipe en possession ──
    prev_team = None
    for fd in search_reversed:
        t       = _frame_time(fd, fps)
        players = fd.get("players") or []
        # Trouver l'équipe qui a le ballon (joueur le plus proche)
        ball    = fd.get("ball") or {}
        bx      = ball.get("x") or ball.get("cx")
        by      = ball.get("y") or ball.get("cy")
        if bx is None:
            continue
        closest_team = None
        closest_dist = float("inf")
        for p in players:
            px   = float(p.get("x", p.get("cx", 0)))
            py   = float(p.get("y", p.get("cy", 0)))
            dist = ((px - float(bx))**2 + (py - float(by))**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_team = p.get("team")
        if closest_team is not None and closest_dist < 100:
            if prev_team is not None and closest_team != prev_team:
                # Changement d'équipe = début de contre-attaque
                action_start = t
                action_start = max(event_time - max_rewind_sec,
                                   min(action_start, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] signal=changement_équipe → {mm:02d}:{ss:02d} ({prev_team}→{closest_team})")
                return max(0, action_start)
            prev_team = closest_team

    # ── Fallback ──
    fallback_t = max(0, event_time - 15.0)
    mm = int(fallback_t // 60); ss = int(fallback_t % 60)
    print(f"    [ACTION_START] fallback → {mm:02d}:{ss:02d} (aucun signal trouvé)")
    return fallback_t


def detect_action_start_for_event(
    frames_data: List[Dict],
    terminal_event: Dict,
    fps: float,
) -> float:
    """
    Wrapper qui adapte max_rewind selon le type d'événement terminal.

    Logique métier par type :
      - goalkeeper_save   : action courte (tir direct) → max 12s
      - ball_caught       : action peut être longue    → max 20s
      - clearance         : dégagement = fin d'action  → max 18s
      - corner_or_goalkick: peut venir de loin         → max 22s
      - basketball 3pt    : action rapide              → max 8s
      - handball fast_break: très rapide               → max 6s
    """
    event_type = terminal_event.get("type", "")
    event_time = terminal_event.get("time", 0)

    # Limites par type d'événement
    REWIND_LIMITS = {
        # Football
        "goalkeeper_save":   (5.0, 12.0),
        "ball_caught":       (5.0, 20.0),
        "clearance":         (5.0, 18.0),
        "corner_or_goalkick":(5.0, 22.0),
        # Basketball
        "3pt_made":          (3.0, 8.0),
        "2pt_made":          (3.0, 8.0),
        "block":             (2.0, 6.0),
        "defensive_rebound": (3.0, 8.0),
        # Handball
        "goal":              (4.0, 12.0),
        "goalkeeper_save":   (4.0, 10.0),
        "7m_throw":          (3.0, 8.0),
        "fast_break":        (3.0, 6.0),
        # Hockey
        "powerplay_goal":    (5.0, 20.0),
        "penalty_shot":      (3.0, 8.0),
        "icing":             (2.0, 5.0),
        # Rugby
        "try":               (5.0, 25.0),
        "conversion":        (3.0, 8.0),
        "penalty_kick":      (3.0, 8.0),
        "tackle_in_22":      (5.0, 15.0),
    }

    min_r, max_r = REWIND_LIMITS.get(event_type, (5.0, 15.0))
    ev_mm = int(event_time//60); ev_ss = int(event_time%60)
    print(f"  [ACTION_START] cherche début de '{event_type}' à {ev_mm:02d}:{ev_ss:02d} | fenêtre [{min_r:.0f}s–{max_r:.0f}s]")

    # Retourne un timestamp précis (pas un offset)
    action_start_t = detect_action_start(
        frames_data    = frames_data,
        event_time     = event_time,
        fps            = fps,
        max_rewind_sec = max_r,
        min_rewind_sec = min_r,
    )

    return action_start_t

# ─────────────────────────────────────────────────────────────
# Registre des fonctions par sport
# ─────────────────────────────────────────────────────────────

_DETECTORS = {
    "football": [
        _detect_goalkeeper_save,
        _detect_ball_caught,
        _detect_clearance,
        _detect_corner_or_goalkick,
    ],
    "basketball": [
        _detect_basketball_3pt,
        _detect_basketball_2pt,
        _detect_basketball_block,
        _detect_basketball_defensive_rebound,
    ],
    "handball": [
        _detect_handball_goal,
        _detect_handball_goalkeeper_save,
        _detect_handball_7m,
        _detect_handball_fast_break,
    ],
    "hockey": [
        _detect_hockey_save,
        _detect_hockey_powerplay_goal,
        _detect_hockey_penalty_shot,
        _detect_hockey_icing,
    ],
    "rugby": [
        _detect_rugby_try,
        _detect_rugby_conversion,
        _detect_rugby_penalty_kick,
        _detect_rugby_tackle_in_22,
    ],
}


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
):
    """
    Détecte tous les événements terminaux pour le sport donné.
    Retourne une liste d'événements triés par temps.
    """
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
    by_type = {}
    for e in all_events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t:30} : {n}")

    return all_events


def build_candidate_windows(
    terminal_events,
    rewind_sec           = REWIND_SEC,
    end_sec              = WINDOW_END_SEC,
    existing_candidates  = None,
    merge_radius_sec     = 5.0,
    frames_data          = None,
    fps                  = 25.0,
):
    """
    Transforme les événements terminaux en fenêtres candidates.
    Fusionne avec les candidats existants (goal_posthoc) si fournis.
    Évite les doublons à ±merge_radius_sec.

    Si frames_data + fps fournis : détection dynamique du début d'action.
    Sinon : rewind fixe = rewind_sec.
    """
    windows = []

    for ev in terminal_events:
        t = ev["time"]

        # Détection dynamique du début d'action si données disponibles
        if frames_data is not None:
            # action_start_t = timestamp précis du début de l'action
            action_start_t = detect_action_start_for_event(frames_data, ev, fps)
        else:
            action_start_t = max(0, t - rewind_sec)

        actual_rewind = t - action_start_t

        windows.append({
            "time":          t,
            "window_start":  action_start_t,
            "window_end":    t + end_sec,
            "source":        f"terminal_{ev['type']}",
            "confidence":    ev["confidence"],
            "terminal_type": ev["type"],
            "rewind_sec":    actual_rewind,   # pour les logs : durée réelle remontée
        })

    if existing_candidates:
        for cand in existing_candidates:
            t = cand.get("time", 0)
            overlap = any(abs(t - w["time"]) < merge_radius_sec for w in windows)
            if not overlap:
                windows.append({
                    "time":         t,
                    "window_start": max(0, t - rewind_sec),
                    "window_end":   t + end_sec,
                    "source":       cand.get("source", "posthoc"),
                    "confidence":   cand.get("confidence", 0.5),
                })

    windows.sort(key=lambda w: w["time"])

    print(f"  [CANDIDATE WINDOWS] total={len(windows)}")
    for w in windows:
        mm = int(w["time"] // 60)
        ss = int(w["time"] % 60)
        rewind = w.get("rewind_sec", REWIND_SEC)
        print(f"    {mm:02d}:{ss:02d}  {w['source']:35} conf={w['confidence']:.2f}  rewind={rewind:.0f}s")

    return windows
