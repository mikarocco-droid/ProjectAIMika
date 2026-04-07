# analysis/events.py
# -*- coding: utf-8 -*-

from collections import deque
import math

from analysis.intelligence import (
    compute_xa,
    compute_danger,
    is_progressive,
    detect_build_up
)

# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def speed(a, b):
    return distance(a, b)

def get_closest_player(players, ball):
    closest  = None
    min_dist = float("inf")
    for p in players:
        d = distance(p["center"], ball["center"])
        if d < min_dist:
            min_dist = d
            closest  = p
    return closest, min_dist


# ─────────────────────────────────────────
# SHOT ZONES MULTI SPORT
# FIX v2 — seuils assouplis + zone centrale incluse
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):
    if shot_zones:
        axis  = shot_zones.get("axis", "x")
        hi    = shot_zones.get("threshold_hi", frame_w * 0.85)
        lo    = shot_zones.get("threshold_lo", frame_w * 0.15)
        y_min = shot_zones.get("y_min", 0)
        y_max = shot_zones.get("y_max", frame_h)

        if axis == "x":
            y_tol     = (y_max - y_min) * 0.30          # FIX: 0.20→0.30 tolérance Y
            in_y_shot = (y_min - y_tol <= y <= y_max + y_tol)
            in_y_goal = (y_min <= y <= y_max)
            # FIX: zone tir élargie de 85% à 80% du bord
            in_zone   = (x > hi or x < lo) and in_y_shot
            # FIX: zone but à 88% au lieu de 92%
            in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
            return in_zone or in_goal, in_goal
        else:
            in_zone = (y < hi or y > lo)
            in_goal = (y < hi * 0.88 or y > lo * 1.12)
            return in_zone, in_goal

    if sport == "football":
        in_y_shot = (frame_h * 0.25 <= y <= frame_h * 0.75)   # FIX: élargi 0.30→0.25
        in_y_goal = (frame_h * 0.30 <= y <= frame_h * 0.70)   # FIX: élargi 0.35→0.30
        # FIX: seuil tir 0.80 (était 0.92) — inclut les tirs depuis l'intérieur
        in_zone   = (x > frame_w * 0.80 or x < frame_w * 0.20) and in_y_shot
        # FIX: zone but 0.88 (était 0.92)
        in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
        return in_zone or in_goal, in_goal

    if sport == "basketball":
        in_zone = (x > frame_w * 0.88 or x < frame_w * 0.12)
        in_goal = (x > frame_w * 0.94 or x < frame_w * 0.06)
        return in_zone, in_goal

    if sport == "handball":
        in_y_shot = (frame_h * 0.20 <= y <= frame_h * 0.80)
        in_y_goal = (frame_h * 0.25 <= y <= frame_h * 0.75)
        in_zone   = (x > frame_w * 0.82 or x < frame_w * 0.18) and in_y_shot
        in_goal   = (x > frame_w * 0.90 or x < frame_w * 0.10) and in_y_goal
        return in_zone or in_goal, in_goal

    in_zone = (x > frame_w * 0.85 or x < frame_w * 0.15)
    in_goal = (x > frame_w * 0.90 or x < frame_w * 0.10)
    return in_zone or in_goal, in_goal


# ─────────────────────────────────────────
# CALCUL xG
# ─────────────────────────────────────────
def compute_xg(x, y, frame_w=1280, frame_h=720, learner=None):
    """xG depuis le modèle appris si disponible, sinon formule géométrique."""
    if learner and learner.xg_model.get("n_samples", 0) >= 10:
        return learner.predict_xg(x, y, frame_w, frame_h)
    dist_right = math.hypot(x - frame_w, y - frame_h / 2)
    dist_left  = math.hypot(x,           y - frame_h / 2)
    dist       = min(dist_right, dist_left)
    max_dist   = math.hypot(frame_w, frame_h / 2)
    xg         = round(max(0.0, 1.0 - dist / max_dist), 3)
    return min(xg, 0.5)


# ─────────────────────────────────────────
# INIT STATE
# ─────────────────────────────────────────
def init_state(learner=None):
    """Initialise l'état depuis les seuils appris si disponibles."""
    thr = learner.get_thresholds() if learner else {}

    # Conversion secondes → frames (25fps)
    fps = 25
    shot_cd_frames = int(thr.get("shot_cooldown", 3.0)  * fps)
    goal_cd_frames = int(thr.get("goal_cooldown", 150.0) * fps)
    ball_speed_min = thr.get("ball_speed_min",    0.02)   # FIX: 0.03→0.02
    player_near    = thr.get("player_near_goal",  0.15)
    goal_frames    = int(thr.get("goal_frames_min", 12))

    return {
        "last_player":         None,
        "last_ball_pos":       None,
        "last_team":           None,
        "sequence":            deque(maxlen=30),
        "events_buffer":       deque(maxlen=10),
        "shot_cd":             0,
        "goal_cd":             0,
        "possession_time":     0,
        "team_possession":     {0: 0, 1: 0},
        "pressing":            False,
        "turnover_window":     0,
        "ball_in_goal_zone":   0,
        # Seuils dynamiques
        "_shot_cd_max":        shot_cd_frames,
        "_goal_cd_max":        goal_cd_frames,
        "_ball_speed_min":     ball_speed_min,
        "_player_near_goal":   player_near,
        "_goal_frames_min":    goal_frames,
    }


# ─────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────
def detect_events(
    players,
    ball,
    sport      = "football",
    state      = None,
    shot_zones = None,
    frame_w    = 1280,
    frame_h    = 720,
    learner    = None,
):
    if state is None:
        state = init_state(learner)

    events = []

    state["shot_cd"] = max(0, state["shot_cd"] - 1)
    state["goal_cd"] = max(0, state["goal_cd"] - 1)

    if not players or not ball:
        state["ball_in_goal_zone"] = 0
        return events, state

    # ── POSSESSION ───────────────────────
    closest, dist = get_closest_player(players, ball)
    threshold     = frame_w * 0.06
    current       = closest if dist < threshold else None

    if current:
        events.append({
            "type":   "possession",
            "player": str(current["id"]),
            "team":   current.get("team"),
            "x":      ball["center"][0],
            "y":      ball["center"][1]
        })
        state["possession_time"] += 1
        team_key = current.get("team")
        if team_key is not None:
            state["team_possession"][team_key] = \
                state["team_possession"].get(team_key, 0) + 1

    # ── PRESSURE ─────────────────────────
    if current:
        opponents = [p for p in players if p.get("team") != current.get("team")]
        close_opp = sum(
            1 for o in opponents
            if distance(o["center"], current["center"]) < frame_w * 0.05
        )
        if close_opp >= 2:
            events.append({"type": "under_pressure", "player": str(current["id"])})
            state["pressing"] = True

    # ── PROGRESSIVE RUN ───────────────────
    if state["last_ball_pos"] and current:
        if is_progressive(state["last_ball_pos"][0], ball["center"][0], frame_w):
            events.append({
                "type":   "progressive_run",
                "player": str(current["id"]),
                "team":   current.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })

    # ── PASS / INTERCEPTION ───────────────
    last = state["last_player"]
    if last and current and str(last["id"]) != str(current["id"]):
        same_team = last.get("team") == current.get("team")
        if same_team:
            pass_event = {
                "type":   "pass",
                "from":   str(last["id"]),
                "to":     str(current["id"]),
                "team":   current.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1],
                "xA":     0.0,
                "player": str(last["id"])
            }
            events.append(pass_event)
            state["events_buffer"].append(pass_event)
        else:
            events.append({
                "type":   "interception",
                "player": str(current["id"]),
                "team":   current.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })
            state["turnover_window"] = 15

    # ── FAST BREAK ───────────────────────
    if state["turnover_window"] > 0 and state["last_ball_pos"]:
        v = speed(state["last_ball_pos"], ball["center"])
        if v > frame_w * 0.08:
            events.append({"type": "fast_break"})
            state["turnover_window"] = 0
    state["turnover_window"] = max(0, state["turnover_window"] - 1)

    # ── DRIBBLE ──────────────────────────
    if current and state["last_ball_pos"]:
        if speed(state["last_ball_pos"], ball["center"]) > frame_w * 0.025:
            events.append({
                "type":   "dribble",
                "player": str(current["id"]),
                "team":   current.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })

    # ── BUILD UP ─────────────────────────
    state["sequence"].append(ball["center"])
    if detect_build_up(state["sequence"], frame_w):
        events.append({"type": "build_up"})
        state["sequence"].clear()

    # ── SHOTS / GOALS ─────────────────────
    # Récupération seuils dynamiques (depuis learner ou defaults state)
    ball_speed_min   = state.get("_ball_speed_min",   0.02)
    player_near_pct  = state.get("_player_near_goal", 0.15)
    goal_frames_min  = state.get("_goal_frames_min",  12)
    shot_cd_max      = state.get("_shot_cd_max",      75)
    goal_cd_max      = state.get("_goal_cd_max",      3750)

    if current:
        x, y = ball["center"]

        # FIX — vérification zone FP apprise
        if learner and learner.is_fp_zone(x, y, frame_w, frame_h):
            pass  # zone connue comme faux positif → on skip tir
        else:
            is_shot, is_goal_zone = is_shot_zone(
                x, y, sport, shot_zones, frame_w, frame_h
            )

            # ── SHOT ─────────────────────────
            ball_speed   = speed(state["last_ball_pos"], ball["center"]) \
                           if state["last_ball_pos"] else 0
            # FIX: seuil réduit 0.03→0.02 ET on accepte aussi les tirs lents
            # si le joueur est très près du but (zone goal)
            shot_speed_ok = (ball_speed > frame_w * ball_speed_min) or \
                            (is_goal_zone and ball_speed > frame_w * 0.01)

            if is_shot and state["shot_cd"] == 0 and shot_speed_ok:
                xg_val = compute_xg(x, y, frame_w, frame_h, learner)
                shot   = {
                    "type":   "shot",
                    "player": str(current["id"]),
                    "team":   current.get("team"),
                    "x":      x,
                    "y":      y,
                    "xg":     xg_val,
                    "danger": compute_danger({"type": "shot", "xg": xg_val})
                }
                events.append(shot)
                state["shot_cd"] = shot_cd_max

                if state["events_buffer"]:
                    last_pass       = state["events_buffer"][-1]
                    last_pass["xA"] = compute_xa(last_pass, shot)

            # ── GOAL ─────────────────────────
            ball_is_real     = not ball.get("interpolated", False)
            player_near_goal = dist < frame_w * player_near_pct

            if is_goal_zone:
                if ball_is_real and player_near_goal:
                    state["ball_in_goal_zone"] += 1
                elif state["ball_in_goal_zone"] > 0:
                    state["ball_in_goal_zone"] += 1
                else:
                    state["ball_in_goal_zone"] = 0
            else:
                state["ball_in_goal_zone"] = 0

            if state["ball_in_goal_zone"] >= goal_frames_min and state["goal_cd"] == 0:
                events.append({
                    "type":   "goal",
                    "player": str(current["id"]),
                    "team":   current.get("team"),
                    "x":      x,
                    "y":      y,
                    "danger": compute_danger({"type": "goal"})
                })
                state["goal_cd"]           = goal_cd_max
                state["ball_in_goal_zone"] = 0

    else:
        state["ball_in_goal_zone"] = 0

    # ── LONG PASS ────────────────────────
    if state["last_ball_pos"]:
        if distance(state["last_ball_pos"], ball["center"]) > frame_w * 0.2:
            events.append({
                "type":   "long_pass",
                "player": str(current["id"]) if current else None,
                "team":   current.get("team") if current else None
            })

    # ── UPDATE STATE ─────────────────────
    state["last_player"]   = current
    state["last_ball_pos"] = ball["center"]
    state["last_team"]     = current.get("team") if current else None

    return events, state


# ─────────────────────────────────────────
# MATCH PROCESSOR
# ─────────────────────────────────────────
def process_match(frames_data, sport="football", shot_zones=None, learner=None):
    state      = None
    all_events = []

    for frame in frames_data:
        events, state = detect_events(
            players    = frame.get("players"),
            ball       = frame.get("ball"),
            sport      = sport,
            state      = state,
            shot_zones = shot_zones,
            frame_w    = frame.get("frame_w", 1280),
            frame_h    = frame.get("frame_h", 720),
            learner    = learner,
        )
        for e in events:
            e["frame"] = frame.get("frame")
        all_events.extend(events)

    return all_events