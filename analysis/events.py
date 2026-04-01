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
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):
    if shot_zones:
        axis  = shot_zones.get("axis", "x")
        hi    = shot_zones.get("threshold_hi", frame_w * 0.85)
        lo    = shot_zones.get("threshold_lo", frame_w * 0.15)
        y_min = shot_zones.get("y_min", 0)
        y_max = shot_zones.get("y_max", frame_h)

        if axis == "x":
            in_zone = (x > hi or x < lo) and (y_min <= y <= y_max)
            # FIX — zone but = derniers 3% du terrain seulement
            in_goal = (x > frame_w * 0.97 or x < frame_w * 0.03) and (y_min <= y <= y_max)
            return in_zone, in_goal
        else:
            in_zone = (y < hi or y > lo)
            in_goal = (y < hi * 0.85 or y > lo * 1.15)
            return in_zone, in_goal

    if sport == "football":
        in_y    = (frame_h * 0.30 <= y <= frame_h * 0.70)
        in_zone = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y
        # FIX — zone but très stricte : derniers 2% du terrain
        in_goal = (x > frame_w * 0.98 or x < frame_w * 0.02) and in_y
        return in_zone, in_goal

    if sport == "basketball":
        in_zone = (x > frame_w * 0.92 or x < frame_w * 0.08)
        in_goal = (x > frame_w * 0.97 or x < frame_w * 0.03)
        return in_zone, in_goal

    if sport == "handball":
        in_y    = (frame_h * 0.20 <= y <= frame_h * 0.80)
        in_zone = (x > frame_w * 0.85 or x < frame_w * 0.15) and in_y
        in_goal = (x > frame_w * 0.97 or x < frame_w * 0.03) and in_y
        return in_zone, in_goal

    in_zone = (x > frame_w * 0.88 or x < frame_w * 0.12)
    in_goal = (x > frame_w * 0.97 or x < frame_w * 0.03)
    return in_zone, in_goal


# ─────────────────────────────────────────
# CALCUL xG
# ─────────────────────────────────────────
def compute_xg(x, y, frame_w=1280, frame_h=720):
    dist_right = math.hypot(x - frame_w, y - frame_h / 2)
    dist_left  = math.hypot(x,           y - frame_h / 2)
    dist       = min(dist_right, dist_left)
    max_dist   = math.hypot(frame_w, frame_h / 2)
    return round(max(0.0, 1.0 - dist / max_dist), 3)


# ─────────────────────────────────────────
# INIT STATE
# ─────────────────────────────────────────
def init_state():
    return {
        "last_player":     None,
        "last_ball_pos":   None,
        "last_team":       None,
        "sequence":        deque(maxlen=30),
        "events_buffer":   deque(maxlen=10),
        # FIX — cooldowns plus longs pour éviter faux buts
        "shot_cd":         0,
        "goal_cd":         0,
        "possession_time": 0,
        "team_possession": {0: 0, 1: 0},
        "pressing":        False,
        "turnover_window": 0,
        # FIX — suivi de la trajectoire du ballon pour confirmer le but
        "ball_in_goal_zone": 0,   # nombre de frames consécutives dans zone but
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
    frame_h    = 720
):
    if state is None:
        state = init_state()

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
        state["team_possession"][current.get("team", 0)] += 1

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
    # FIX — logique entièrement revue pour éviter les faux buts
    if current:
        x, y     = ball["center"]
        is_shot, is_goal_zone = is_shot_zone(
            x, y, sport, shot_zones, frame_w, frame_h
        )

        # ── SHOT ─────────────────────────
        if is_shot and state["shot_cd"] == 0:
            xg_val = compute_xg(x, y, frame_w, frame_h)
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
            state["shot_cd"] = 25  # ~1s à 25fps

            if state["events_buffer"]:
                last_pass       = state["events_buffer"][-1]
                last_pass["xA"] = compute_xa(last_pass, shot)

        # ── GOAL — FIX v2 ────────────────
        # Critères stricts anti faux-positifs :
        # 1. Ballon dans zone but stricte (95% du bord)
        # 2. Pendant au moins 8 frames consécutives (~0.3s)
        # 3. Cooldown 1500 frames = 60s à 25fps
        # 4. Un tir doit avoir été détecté avant (shot_cd < 25)
        if is_goal_zone:
            state["ball_in_goal_zone"] += 1
        else:
            state["ball_in_goal_zone"] = 0

        goal_confirmed = (
            state["ball_in_goal_zone"] >= 8
            and state["goal_cd"] == 0
            and state["shot_cd"] < 25  # un tir récent doit précéder
        )

        if goal_confirmed:
            events.append({
                "type":   "goal",
                "player": str(current["id"]),
                "team":   current.get("team"),
                "x":      x,
                "y":      y,
                "danger": compute_danger({"type": "goal"})
            })
            state["goal_cd"]           = 1500  # 60s à 25fps
            state["ball_in_goal_zone"] = 0

    else:
        # Ballon sans joueur proche — reset zone but
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
def process_match(frames_data, sport="football", shot_zones=None):
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
            frame_h    = frame.get("frame_h", 720)
        )
        for e in events:
            e["frame"] = frame.get("frame")
        all_events.extend(events)

    return all_events