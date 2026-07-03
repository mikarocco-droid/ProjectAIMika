"""
terminal_events.py — Détecteur d'événements terminaux multi-sport
PATCH v2 (2026-05-24) :
  - SPEED_MIN      : 0.04  → 0.025  (vidéo longue : ball tracker drift sur 85 min)
  - FAST_SHOT_SPEED: 0.08  → 0.06
  - score < 4.5    → 4.0
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

TERMINAL_EVENTS_BY_SPORT = {
    "football":   ["goalkeeper_save", "ball_caught", "corner_or_goalkick", "clearance"],
    "basketball": ["3pt_made", "2pt_made", "block", "defensive_rebound"],
    "handball":   ["goal", "goalkeeper_save", "7m_throw", "fast_break"],
    "hockey":     ["goalkeeper_save", "powerplay_goal", "penalty_shot", "icing"],
    "rugby":      ["try", "conversion", "penalty_kick", "tackle_in_22"],
}

GOAL_ZONE_PCT     = 0.08
BOX_ZONE_PCT      = 0.22
PLAY_Y_MIN_PCT    = 0.30
PLAY_Y_MAX_PCT    = 0.97
GK_BALL_DIST_MAX  = 100
GK_POSSESS_FRAMES = 3
MISSING_FRAMES_CORNER = 8
CLEARANCE_ZONE_IN  = 0.25
CLEARANCE_ZONE_OUT = 0.40
COOLDOWN_SEC  = 20.0
REWIND_SEC    = 15.0
WINDOW_END_SEC = 2.0

def _ball_center(fd):
    ball = fd.get("ball") or {}
    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0
    return None

def _ball_norm(fd):
    c = _ball_center(fd)
    if c is None: return None
    w = fd.get("frame_w") or 1920
    h = fd.get("frame_h") or 1080
    return c[0]/w, c[1]/h

def _frame_t(fd, fps):
    return fd.get("frame", 0) / max(fps, 1)

def _in_goal_zone(bx_n):
    return bx_n < GOAL_ZONE_PCT or bx_n > (1 - GOAL_ZONE_PCT)

def _in_box_zone(bx_n):
    return bx_n < BOX_ZONE_PCT or bx_n > (1 - BOX_ZONE_PCT)

def _valid_y(by_n):
    return PLAY_Y_MIN_PCT <= by_n <= PLAY_Y_MAX_PCT

def _gk_center(fd):
    players = fd.get("players") or []
    w = fd.get("frame_w") or 1920
    for p in players:
        if p.get("role") == "goalkeeper" or p.get("label","").lower() in ("gk","goalkeeper"):
            bbox = p.get("bbox") or []
            if len(bbox) == 4:
                return (bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0
    best, best_d = None, float("inf")
    for p in players:
        bbox = p.get("bbox") or []
        if len(bbox) < 4: continue
        px = (bbox[0]+bbox[2])/2.0
        d  = min(px, w-px)
        if d < best_d:
            best_d = d
            best   = ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)
    return best if best_d < w*0.12 else None

def _dist(p1, p2):
    if p1 is None or p2 is None: return float("inf")
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def _cooldown_ok(last_t, etype, t, cooldown=None):
    return t - last_t.get(etype, -999) > (cooldown if cooldown is not None else COOLDOWN_SEC)


def _detect_gk_possession(frames, fps, last_t):
    events = []
    streak = 0
    streak_start_t = None
    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)
        bc  = _ball_center(fd)
        if pos is None or bc is None:
            streak = 0; streak_start_t = None; continue
        bx_n, by_n = pos
        if not _in_box_zone(bx_n) or not _valid_y(by_n):
            streak = 0; streak_start_t = None; continue
        gk = _gk_center(fd)
        if gk is None or _dist(gk, bc) > GK_BALL_DIST_MAX:
            streak = 0; streak_start_t = None; continue
        if streak == 0: streak_start_t = t
        streak += 1
        if streak >= GK_POSSESS_FRAMES:
            if _cooldown_ok(last_t, "goalkeeper_save", streak_start_t):
                last_t["goalkeeper_save"] = streak_start_t
                mm = int(streak_start_t//60); ss = int(streak_start_t%60)
                print(f"  [TERMINAL] goalkeeper_save à {mm:02d}:{ss:02d} | gk_dist={_dist(gk,bc):.0f}px streak={streak}f conf=0.82")
                events.append({"type":"goalkeeper_save","time":streak_start_t,"confidence":0.82})
            streak = 0; streak_start_t = None
    return events


def _detect_corner_goalkick(frames, fps, last_t):
    events = []
    missing_count = 0
    last_seen_pos = None
    last_seen_t   = None
    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)
        if pos is not None:
            bx_n, by_n = pos
            if _valid_y(by_n):
                last_seen_pos = pos; last_seen_t = t; missing_count = 0
            else:
                missing_count += 1
        else:
            missing_count += 1
        if missing_count >= MISSING_FRAMES_CORNER and last_seen_pos is not None:
            bx_last, by_last = last_seen_pos
            if _in_goal_zone(bx_last) and _valid_y(by_last):
                if _cooldown_ok(last_t, "corner_or_goalkick", last_seen_t):
                    last_t["corner_or_goalkick"] = last_seen_t
                    mm = int(last_seen_t//60); ss = int(last_seen_t%60)
                    print(f"  [TERMINAL] corner_or_goalkick à {mm:02d}:{ss:02d} | bx_last={bx_last:.2f} missing={missing_count}f conf=0.72")
                    events.append({"type":"corner_or_goalkick","time":last_seen_t,"confidence":0.72})
            last_seen_pos = None; missing_count = 0
    return events


def _detect_clearance(frames, fps, last_t):
    events = []
    prev_pos = None
    for fd in frames:
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)
        if pos is None:
            prev_pos = None; continue
        bx_n, by_n = pos
        if prev_pos is not None:
            bx_prev, _ = prev_pos
            in_danger_before = bx_prev < CLEARANCE_ZONE_IN or bx_prev > (1-CLEARANCE_ZONE_IN)
            going_center     = CLEARANCE_ZONE_OUT < bx_n < (1-CLEARANCE_ZONE_OUT)
            if in_danger_before and going_center and _valid_y(by_n):
                if _cooldown_ok(last_t, "clearance", t):
                    last_t["clearance"] = t
                    mm = int(t//60); ss = int(t%60)
                    print(f"  [TERMINAL] clearance à {mm:02d}:{ss:02d} | bx {bx_prev:.2f}→{bx_n:.2f} conf=0.68")
                    events.append({"type":"clearance","time":t,"confidence":0.68})
        prev_pos = pos
    return events


def _detect_goal(frames, fps, last_t):
    """
    PATCH v2 :
      SPEED_MIN      0.04  → 0.025   (drift ball tracker sur vidéo longue 85 min)
      FAST_SHOT_SPEED 0.08 → 0.06
      score < 4.5    → 4.0
    """
    events = []

    GOAL_X_LINE    = 0.05
    GOAL_Y_MIN     = 0.45
    GOAL_Y_MAX     = 0.85
    LINE_MARGIN    = 0.02
    STUCK_MIN      = 4
    SPEED_MIN      = 0.025   # PATCH v2 : était 0.04
    REBOUND_DROP   = 0.4
    FAST_SHOT_SPEED = 0.06   # PATCH v2 : était 0.08

    # ── DIAGNOSTIC PENALTY ────────────────────────────────────────────────────
    # Trace ciblée sur la zone temporelle du penalty (t_abs ≈ 382–386s)
    # Permet de comprendre pourquoi _detect_goal ne produit pas de terminal_goal.
    # À retirer une fois le bug identifié.
    _TRACE_T_MIN = 380.0   # t_abs (frame_id/fps)
    _TRACE_T_MAX = 392.0
    def _should_trace(fd):
        t_abs = fd.get("frame", fd.get("frame_id", 0)) / max(fps, 1)
        return _TRACE_T_MIN <= t_abs <= _TRACE_T_MAX
    # ──────────────────────────────────────────────────────────────────────────

    speeds_n = [0.0]
    for k in range(1, len(frames)):
        p1 = _ball_norm(frames[k])
        p0 = _ball_norm(frames[k-1])
        if p1 and p0:
            speeds_n.append(abs(p1[0]-p0[0]))
        else:
            speeds_n.append(0.0)

    i = 10
    while i < len(frames) - 8:
        fd  = frames[i]
        t   = _frame_t(fd, fps)
        pos = _ball_norm(fd)

        # ── Trace ball brut (toutes frames de la fenêtre, ball=None inclus) ──
        if _should_trace(fd):
            t_abs    = fd.get("frame", fd.get("frame_id", 0)) / max(fps, 1)
            prev_pos = _ball_norm(frames[i-1]) if i > 0 else None
            bx_prev  = f"{prev_pos[0]:.3f}" if prev_pos else "None"
            bx_str   = f"{pos[0]:.3f}" if pos else "None"
            by_str   = f"{pos[1]:.3f}" if pos else "None"
            in_y_str = f"{GOAL_Y_MIN <= pos[1] <= GOAL_Y_MAX}" if pos else "—"
            if pos and prev_pos:
                cl = prev_pos[0] > GOAL_X_LINE + LINE_MARGIN and pos[0] <= GOAL_X_LINE
                cr = prev_pos[0] < (1-GOAL_X_LINE-LINE_MARGIN) and pos[0] >= (1-GOAL_X_LINE)
            else:
                cl = cr = False
            print(f"  [GOAL TRACE] t_abs={t_abs:.2f}s "
                  f"bx_prev={bx_prev} bx={bx_str} by={by_str} "
                  f"speed={speeds_n[i]:.3f} in_y={in_y_str} "
                  f"cross_L={cl} cross_R={cr}")

        if pos is None: i += 1; continue

        bx_n, by_n = pos
        if not (GOAL_Y_MIN <= by_n <= GOAL_Y_MAX): i += 1; continue

        pos_prev = _ball_norm(frames[i-1])
        if pos_prev is None: i += 1; continue
        bx_prev = pos_prev[0]

        cross_left  = (bx_prev > GOAL_X_LINE + LINE_MARGIN and bx_n <= GOAL_X_LINE)
        cross_right = (bx_prev < (1-GOAL_X_LINE-LINE_MARGIN) and bx_n >= (1-GOAL_X_LINE))
        if not (cross_left or cross_right): i += 1; continue

        peak_speed = max(speeds_n[max(0, i-8):i+1])
        if peak_speed < SPEED_MIN: i += 1; continue

        stuck = 0
        for j in range(i, min(i+15, len(frames))):
            pj = _ball_norm(frames[j])
            if pj is None: break
            in_left  = pj[0] <= GOAL_X_LINE + LINE_MARGIN
            in_right = pj[0] >= (1-GOAL_X_LINE-LINE_MARGIN)
            if (cross_left and in_left) or (cross_right and in_right):
                stuck += 1
            else:
                break

        next_frames_none = sum(
            1 for j in range(i+1, min(i+4, len(frames)))
            if _ball_norm(frames[j]) is None
        )
        fast_disappear = (peak_speed >= FAST_SHOT_SPEED and next_frames_none >= 2 and stuck <= 2)
        stuck_min_effective = 2 if fast_disappear else STUCK_MIN

        # ── Trace décision (uniquement si cross détecté dans la fenêtre) ─────
        if _should_trace(fd):
            t_abs = fd.get("frame", fd.get("frame_id", 0)) / max(fps, 1)
            pre_speed_  = max(speeds_n[max(0, i-5):i]) if i > 0 else 0
            post_speed_ = max(speeds_n[i:min(i+5, len(speeds_n))])
            rebound_    = (pre_speed_ > 1e-4 and post_speed_/pre_speed_ < REBOUND_DROP)
            score_      = (3.0
                           + (2.0 if stuck >= 6 else 1.0 if stuck >= 4 else 0)
                           + (1.5 if rebound_ else 0)
                           + (0.5 if peak_speed > SPEED_MIN * 2 else 0)
                           + (2.0 if fast_disappear else 0))
            reject_     = ("stuck<min" if stuck < stuck_min_effective else
                           "score<4.0" if score_ < 4.0 else
                           "cooldown"  if not _cooldown_ok(last_t, "goal", t) else
                           "→TERMINAL")
            print(f"  [GOAL DECISION] t_abs={t_abs:.2f}s "
                  f"cross_L={cross_left} cross_R={cross_right} "
                  f"peak={peak_speed:.3f} stuck={stuck} stuck_min={stuck_min_effective} "
                  f"next_none={next_frames_none} fast_disappear={fast_disappear} "
                  f"score={score_:.1f} → {reject_}")

        if stuck < stuck_min_effective: i += 1; continue

        pre_speed  = max(speeds_n[max(0, i-5):i]) if i > 0 else 0
        post_speed = max(speeds_n[i:min(i+5, len(speeds_n))])
        rebound    = (pre_speed > 1e-4 and post_speed/pre_speed < REBOUND_DROP)

        score = 3.0
        if stuck >= 6:   score += 2.0
        elif stuck >= 4: score += 1.0
        if rebound:      score += 1.5
        if peak_speed > SPEED_MIN * 2: score += 0.5
        if fast_disappear: score += 2.0

        if score < 4.0:  # PATCH v2 : était 4.5
            i += 1; continue

        bx_n, by_n = pos
        if not (GOAL_Y_MIN <= by_n <= GOAL_Y_MAX): i += 1; continue

        pos_prev = _ball_norm(frames[i-1])
        if pos_prev is None: i += 1; continue
        bx_prev = pos_prev[0]

        cross_left  = (bx_prev > GOAL_X_LINE + LINE_MARGIN and bx_n <= GOAL_X_LINE)
        cross_right = (bx_prev < (1-GOAL_X_LINE-LINE_MARGIN) and bx_n >= (1-GOAL_X_LINE))
        if not (cross_left or cross_right): i += 1; continue

        peak_speed = max(speeds_n[max(0, i-8):i+1])
        if peak_speed < SPEED_MIN: i += 1; continue

        stuck = 0
        for j in range(i, min(i+15, len(frames))):
            pj = _ball_norm(frames[j])
            if pj is None: break
            in_left  = pj[0] <= GOAL_X_LINE + LINE_MARGIN
            in_right = pj[0] >= (1-GOAL_X_LINE-LINE_MARGIN)
            if (cross_left and in_left) or (cross_right and in_right):
                stuck += 1
            else:
                break

        next_frames_none = sum(
            1 for j in range(i+1, min(i+4, len(frames)))
            if _ball_norm(frames[j]) is None
        )
        fast_disappear = (peak_speed >= FAST_SHOT_SPEED and next_frames_none >= 2 and stuck <= 2)
        stuck_min_effective = 2 if fast_disappear else STUCK_MIN

        if stuck < stuck_min_effective: i += 1; continue

        pre_speed  = max(speeds_n[max(0, i-5):i]) if i > 0 else 0
        post_speed = max(speeds_n[i:min(i+5, len(speeds_n))])
        rebound    = (pre_speed > 1e-4 and post_speed/pre_speed < REBOUND_DROP)

        score = 3.0
        if stuck >= 6:   score += 2.0
        elif stuck >= 4: score += 1.0
        if rebound:      score += 1.5
        if peak_speed > SPEED_MIN * 2: score += 0.5
        if fast_disappear: score += 2.0

        if score < 4.0:  # PATCH v2 : était 4.5
            i += 1; continue

        if _cooldown_ok(last_t, "goal", t):
            last_t["goal"] = t
            conf = min(0.70 + score*0.04, 0.90)
            mm = int(t//60); ss = int(t%60)
            side = "gauche" if cross_left else "droite"
            print(f"  [TERMINAL] goal à {mm:02d}:{ss:02d} | {side} bx={bx_n:.2f} by={by_n:.2f} stuck={stuck}f peak={peak_speed:.3f} rebound={rebound} conf={conf:.2f}")
            events.append({"type":"goal","time":t,"confidence":conf})
            i += max(stuck, 8)
        else:
            i += 1

    return events


def _detect_celebration(frames, fps, last_t):
    events = []
    CELEB_ZONE_X   = 0.20
    CELEB_PLAYERS  = 3
    CELEB_FRAMES   = 8
    CELEB_COOLDOWN = 20.0
    streak = 0
    streak_start_t = None
    for fd in frames:
        t       = _frame_t(fd, fps)
        players = fd.get("players", [])
        if not players:
            streak = 0; streak_start_t = None; continue
        frame_w = fd.get("frame_w", 1920)
        in_left  = sum(1 for p in players if p.get("center") and p["center"][0]/frame_w < CELEB_ZONE_X)
        in_right = sum(1 for p in players if p.get("center") and p["center"][0]/frame_w > (1-CELEB_ZONE_X))
        in_zone  = max(in_left, in_right) >= CELEB_PLAYERS
        if in_zone:
            if streak == 0: streak_start_t = t
            streak += 1
        else:
            streak = 0; streak_start_t = None; continue
        if streak >= CELEB_FRAMES:
            if _cooldown_ok(last_t, "celebration", streak_start_t, cooldown=CELEB_COOLDOWN):
                last_t["celebration"] = streak_start_t
                mm = int(streak_start_t//60); ss = int(streak_start_t%60)
                side = "gauche" if in_left >= CELEB_PLAYERS else "droite"
                print(f"  [TERMINAL] celebration à {mm:02d}:{ss:02d} | {side} players={max(in_left,in_right)} streak={streak}f conf=0.75")
                events.append({"type":"celebration","time":streak_start_t,"confidence":0.75})
            streak = 0; streak_start_t = None
    return events


def _stub(*args, **kwargs):
    return []


_DETECTORS = {
    "football":   [_detect_goal, _detect_gk_possession, _detect_clearance],
    "basketball": [_stub],
    "handball":   [_stub],
    "hockey":     [_stub],
    "rugby":      [_stub],
}


def detect_action_start(frames_data, event_time, fps, max_rewind_sec=25.0, min_rewind_sec=5.0):
    if not frames_data or fps <= 0:
        return max(0, event_time - 15.0)
    search = [fd for fd in frames_data if event_time - max_rewind_sec <= _frame_t(fd, fps) < event_time]
    if not search:
        return max(0, event_time - 15.0)
    search_rev = list(reversed(search))

    consecutive_missing = 0
    for fd in search_rev:
        t = _frame_t(fd, fps)
        if _ball_norm(fd) is None:
            consecutive_missing += 1
        else:
            if consecutive_missing >= 3:
                action_start = max(event_time - max_rewind_sec, min(t + 0.5, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] remise_en_jeu → {mm:02d}:{ss:02d} (absent {consecutive_missing}f)")
                return max(0, action_start)
            consecutive_missing = 0

    calm_positions = []; calm_start_t = None; prev_c = None
    for fd in search_rev:
        t = _frame_t(fd, fps)
        c = _ball_center(fd)
        w = fd.get("frame_w") or 1920
        if c is None:
            calm_positions = []; calm_start_t = None; prev_c = None; continue
        if prev_c is not None:
            move = _dist(c, prev_c) / w
            if move < 0.015:
                if calm_start_t is None: calm_start_t = t
                calm_positions.append(t)
            else:
                if calm_start_t is not None and (t - calm_start_t) >= 1.5:
                    action_start = max(event_time - max_rewind_sec, min(calm_start_t, event_time - min_rewind_sec))
                    mm = int(action_start//60); ss = int(action_start%60)
                    print(f"    [ACTION_START] calme_positionnel → {mm:02d}:{ss:02d} ({t-calm_start_t:.1f}s)")
                    return max(0, action_start)
                calm_positions = []; calm_start_t = None
        prev_c = c

    prev_team = None
    for fd in search_rev:
        t = _frame_t(fd, fps)
        players = fd.get("players") or []
        bc = _ball_center(fd)
        if bc is None: continue
        closest_team, closest_d = None, float("inf")
        for p in players:
            bbox = p.get("bbox") or []
            if len(bbox) < 4: continue
            pc = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            d  = _dist(pc, bc)
            if d < closest_d:
                closest_d = d; closest_team = p.get("team")
        if closest_team is not None and closest_d < 80:
            if prev_team is not None and closest_team != prev_team:
                action_start = max(event_time - max_rewind_sec, min(t, event_time - min_rewind_sec))
                mm = int(action_start//60); ss = int(action_start%60)
                print(f"    [ACTION_START] changement_equipe → {mm:02d}:{ss:02d} ({prev_team}→{closest_team})")
                return max(0, action_start)
            prev_team = closest_team

    fallback_t = max(0, event_time - 15.0)
    mm = int(fallback_t//60); ss = int(fallback_t%60)
    print(f"    [ACTION_START] fallback → {mm:02d}:{ss:02d}")
    return fallback_t


def detect_action_start_for_event(frames_data, terminal_event, fps):
    LIMITS = {
        "goal":               (5.0, 20.0),
        "celebration":        (8.0, 18.0),
        "goalkeeper_save":    (5.0, 12.0),
        "ball_caught":        (5.0, 20.0),
        "clearance":          (5.0, 18.0),
        "corner_or_goalkick": (5.0, 22.0),
        "3pt_made":           (3.0,  8.0),
        "2pt_made":           (3.0,  8.0),
        "block":              (2.0,  6.0),
        "fast_break":         (3.0,  6.0),
        "try":                (5.0, 25.0),
    }
    etype = terminal_event.get("type", "")
    et    = terminal_event.get("time", 0)
    min_r, max_r = LIMITS.get(etype, (5.0, 15.0))
    mm = int(et//60); ss = int(et%60)
    print(f"  [ACTION_START] '{etype}' à {mm:02d}:{ss:02d} | fenêtre [{min_r:.0f}s–{max_r:.0f}s]")
    return detect_action_start(frames_data, et, fps, max_r, min_r)


def detect_terminal_events(frames_data, fps=25.0, frame_w=1920, frame_h=1080, goal_box=None, sport="football"):
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
    for tp, n in sorted(by_type.items()):
        print(f"    {tp:30} : {n}")
    return all_events


def build_candidate_windows(terminal_events, rewind_sec=REWIND_SEC, end_sec=WINDOW_END_SEC,
                             existing_candidates=None, merge_radius_sec=5.0, frames_data=None, fps=25.0):
    windows = []
    for ev in terminal_events:
        t = ev["time"]
        action_start = detect_action_start_for_event(frames_data, ev, fps) if frames_data is not None else max(0, t - rewind_sec)
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
    MERGE_RADIUS = 8.0
    merged = []
    for w in windows:
        if merged and abs(w["time"] - merged[-1]["time"]) <= MERGE_RADIUS:
            prev = merged[-1]
            if w["confidence"] > prev["confidence"]:
                prev["confidence"]    = w["confidence"]
                prev["source"]        = w["source"]
                prev["terminal_type"] = w.get("terminal_type", prev.get("terminal_type"))
            prev["window_start"] = min(prev["window_start"], w["window_start"])
            prev["window_end"]   = max(prev["window_end"],   w["window_end"])
            prev["rewind_sec"]   = prev["time"] - prev["window_start"]
        else:
            merged.append(dict(w))
    print(f"  [CANDIDATE WINDOWS] {len(windows)} → fusionnées={len(merged)}")
    for w in merged:
        mm = int(w["time"]//60); ss = int(w["time"]%60)
        print(f"    {mm:02d}:{ss:02d}  {w['source']:35} conf={w['confidence']:.2f}  rewind={w['rewind_sec']:.0f}s")
    return merged