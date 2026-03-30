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
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def speed(a, b):
    return distance(a, b)


def get_closest_player(players, ball):
    closest = None
    min_dist = float("inf")

    for p in players:
        d = distance(p["center"], ball["center"])
        if d < min_dist:
            min_dist = d
            closest = p

    return closest, min_dist


# ─────────────────────────────────────────
# SHOT ZONES MULTI SPORT
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):

    if shot_zones:
        axis = shot_zones.get("axis", "x")
        hi   = shot_zones.get("threshold_hi", frame_w * 0.85)
        lo   = shot_zones.get("threshold_lo", frame_w * 0.15)

        if axis == "x":
            return (x > hi or x < lo), (x > hi*1.02 or x < lo*0.98)
        else:
            return (y < hi or y > lo), (y < hi*0.9 or y > lo*1.1)

    if sport == "football":
        return (x > frame_w*0.85 or x < frame_w*0.15), (x > frame_w*0.92 or x < frame_w*0.08)

    if sport == "basketball":
        return (x > frame_w*0.92 or x < frame_w*0.08), (x > frame_w*0.95 or x < frame_w*0.05)

    if sport == "handball":
        return (x > frame_w*0.82 or x < frame_w*0.18), (x > frame_w*0.88 or x < frame_w*0.12)

    return (x > frame_w*0.85 or x < frame_w*0.15), (x > frame_w*0.92 or x < frame_w*0.08)


# ─────────────────────────────────────────
# INIT STATE GOD MODE
# ─────────────────────────────────────────
def init_state():
    return {
        "last_player": None,
        "last_ball_pos": None,
        "last_team": None,

        "sequence": deque(maxlen=30),
        "events_buffer": deque(maxlen=10),

        "shot_cd": 0,
        "goal_cd": 0,

        "possession_time": 0,
        "team_possession": {0: 0, 1: 0},

        "pressing": False,
        "turnover_window": 0
    }


# ─────────────────────────────────────────
# MAIN DETECTOR GOD MODE
# ─────────────────────────────────────────
def detect_events(
    players,
    ball,
    sport="football",
    state=None,
    shot_zones=None,
    frame_w=1280,
    frame_h=720
):
    if state is None:
        state = init_state()

    events = []

    # cooldowns
    state["shot_cd"] = max(0, state["shot_cd"] - 1)
    state["goal_cd"] = max(0, state["goal_cd"] - 1)

    if not players or not ball:
        return events, state

    # ─────────────────────────────
    # POSSESSION
    # ─────────────────────────────
    closest, dist = get_closest_player(players, ball)
    threshold = frame_w * 0.06

    current = closest if dist < threshold else None

    if current:
        events.append({
            "type": "possession",
            "player": str(current["id"]),
            "team": current.get("team"),
            "x": ball["center"][0],
            "y": ball["center"][1]
        })

        state["possession_time"] += 1
        state["team_possession"][current.get("team", 0)] += 1

    # ─────────────────────────────
    # PRESSURE DETECTION
    # ─────────────────────────────
    if current:
        opponents = [p for p in players if p.get("team") != current.get("team")]

        close_opponents = sum(
            1 for o in opponents
            if distance(o["center"], current["center"]) < frame_w * 0.05
        )

        if close_opponents >= 2:
            events.append({
                "type": "under_pressure",
                "player": str(current["id"])
            })
            state["pressing"] = True

    # ─────────────────────────────
    # PROGRESSIVE RUN
    # ─────────────────────────────
    if state["last_ball_pos"] and current:
        if is_progressive(state["last_ball_pos"][0], ball["center"][0], frame_w):
            events.append({
                "type": "progressive_run",
                "player": str(current["id"])
            })

    # ─────────────────────────────
    # PASS / INTERCEPTION / TURNOVER
    # ─────────────────────────────
    last = state["last_player"]

    if last and current and str(last["id"]) != str(current["id"]):

        same_team = last.get("team") == current.get("team")

        if same_team:
            pass_event = {
                "type": "pass",
                "from": str(last["id"]),
                "to": str(current["id"]),
                "team": current.get("team"),
                "x": ball["center"][0],
                "y": ball["center"][1],
                "xA": 0.0
            }
            events.append(pass_event)
            state["events_buffer"].append(pass_event)

        else:
            events.append({
                "type": "interception",
                "player": str(current["id"])
            })

            state["turnover_window"] = 15

    # ─────────────────────────────
    # FAST BREAK
    # ─────────────────────────────
    if state["turnover_window"] > 0:
        if state["last_ball_pos"]:
            v = speed(state["last_ball_pos"], ball["center"])
            if v > frame_w * 0.08:
                events.append({"type": "fast_break"})
                state["turnover_window"] = 0

    state["turnover_window"] = max(0, state["turnover_window"] - 1)

    # ─────────────────────────────
    # DRIBBLE
    # ─────────────────────────────
    if current and state["last_ball_pos"]:
        if speed(state["last_ball_pos"], ball["center"]) > frame_w * 0.025:
            events.append({
                "type": "dribble",
                "player": str(current["id"])
            })

    # ─────────────────────────────
    # BUILD UP
    # ─────────────────────────────
    state["sequence"].append(ball["center"])

    if detect_build_up(state["sequence"], frame_w):
        events.append({"type": "build_up"})
        state["sequence"].clear()

    # ─────────────────────────────
    # SHOTS / GOALS + xG + xA
    # ─────────────────────────────
    if current:
        x, y = ball["center"]
        is_shot, is_goal = is_shot_zone(x, y, sport, shot_zones, frame_w, frame_h)

        if is_shot and state["shot_cd"] == 0:
            shot = {
                "type": "shot",
                "player": str(current["id"]),
                "team": current.get("team"),
                "x": x,
                "y": y,
                "xG": round(1 - (abs(x - frame_w/2) / frame_w), 2),
                "danger": compute_danger({"type": "shot"})
            }

            events.append(shot)
            state["shot_cd"] = 20

            if state["events_buffer"]:
                last_pass = state["events_buffer"][-1]
                last_pass["xA"] = compute_xa(last_pass, shot)

        if is_goal and state["goal_cd"] == 0:
            events.append({
                "type": "goal",
                "player": str(current["id"]),
                "team": current.get("team"),
                "x": x,
                "y": y,
                "danger": compute_danger({"type": "goal"})
            })

            state["goal_cd"] = 120

    # ─────────────────────────────
    # LONG PASS
    # ─────────────────────────────
    if state["last_ball_pos"]:
        if distance(state["last_ball_pos"], ball["center"]) > frame_w * 0.2:
            events.append({"type": "long_pass"})

    # ─────────────────────────────
    # UPDATE STATE
    # ─────────────────────────────
    state["last_player"] = current
    state["last_ball_pos"] = ball["center"]
    state["last_team"] = current.get("team") if current else None

    return events, state


# ─────────────────────────────────────────
# MATCH PROCESSOR
# ─────────────────────────────────────────
def process_match(frames_data, sport="football", shot_zones=None):
    state = None
    all_events = []

    for frame in frames_data:
        events, state = detect_events(
            players=frame.get("players"),
            ball=frame.get("ball"),
            sport=sport,
            state=state,
            shot_zones=shot_zones,
            frame_w=frame.get("frame_w", 1280),
            frame_h=frame.get("frame_h", 720)
        )

        for e in events:
            e["frame"] = frame.get("frame")

        all_events.extend(events)

    return all_events