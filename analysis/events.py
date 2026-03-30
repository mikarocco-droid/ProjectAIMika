# analysis/events.py
# -*- coding: utf-8 -*-

from analysis.intelligence import (
    compute_xa,
    is_progressive,
    compute_danger,
    detect_build_up
)


# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


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
# ZONES DE TIR
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):

    if shot_zones:
        axis  = shot_zones.get("axis", "x")
        hi    = shot_zones.get("threshold_hi", frame_w * 0.85)
        lo    = shot_zones.get("threshold_lo", frame_w * 0.15)
        y_min = shot_zones.get("y_min", 0)
        y_max = shot_zones.get("y_max", frame_h)
        y_ok  = y_min <= y <= y_max
        if axis == "x":
            is_shot  = (x > hi or x < lo) and y_ok
            is_score = (x > hi * 1.02 or x < lo * 0.98) and y_ok
        else:
            is_shot  = (y < hi or y > lo)
            is_score = (y < hi * 0.85 or y > lo * 1.15)
        return is_shot, is_score

    if sport in ["football", "mini-foot"]:
        y_ok     = frame_h * 0.20 <= y <= frame_h * 0.80
        is_shot  = (x > frame_w * 0.85 or x < frame_w * 0.15) and y_ok
        is_score = (x > frame_w * 0.92 or x < frame_w * 0.08) and y_ok
        return is_shot, is_score

    elif sport == "basketball":
        y_ok     = frame_h * 0.20 <= y <= frame_h * 0.80
        is_shot  = (x > frame_w * 0.92 or x < frame_w * 0.08) and y_ok
        is_score = (x > frame_w * 0.95 or x < frame_w * 0.05) and y_ok
        return is_shot, is_score

    elif sport == "handball":
        y_ok     = frame_h * 0.20 <= y <= frame_h * 0.80
        is_shot  = (x > frame_w * 0.82 or x < frame_w * 0.18) and y_ok
        is_score = (x > frame_w * 0.88 or x < frame_w * 0.12) and y_ok
        return is_shot, is_score

    else:
        is_shot  = x > frame_w * 0.85 or x < frame_w * 0.15
        is_score = x > frame_w * 0.92 or x < frame_w * 0.08
        return is_shot, is_score


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE
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
        state = {
            "last_player":    None,
            "last_ball_pos":  None,
            "last_team":      None,
            "shot_cooldown":  0,
            "score_cooldown": 0,
            "sequence":       [],
            "events_buffer":  []
        }

    events = []
    state["shot_cooldown"]  = max(0, state["shot_cooldown"]  - 1)
    state["score_cooldown"] = max(0, state["score_cooldown"] - 1)

    if not players or not ball:
        return events, state

    # ─────────────────────────────────────────
    # POSSESSION
    # ─────────────────────────────────────────
    closest, dist        = get_closest_player(players, ball)
    possession_threshold = frame_w * 0.06
    current_player       = closest if dist < possession_threshold else None

    if current_player:
        events.append({
            "type":   "possession",
            "player": str(current_player["id"]),
            "team":   current_player.get("team"),
            "x":      ball["center"][0],
            "y":      ball["center"][1]
        })

    # ─────────────────────────────────────────
    # PROGRESSIVE RUN
    # ─────────────────────────────────────────
    if state["last_ball_pos"] and current_player:
        if is_progressive(
            state["last_ball_pos"][0],
            ball["center"][0],
            frame_w
        ):
            events.append({
                "type":   "progressive_run",
                "player": str(current_player["id"]),
                "team":   current_player.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })

    # ─────────────────────────────────────────
    # PASS / INTERCEPTION
    # ─────────────────────────────────────────
    last_player = state.get("last_player")

    if last_player and current_player:
        if str(last_player["id"]) != str(current_player["id"]):
            same_team = last_player.get("team") == current_player.get("team")

            if same_team:
                pass_event = {
                    "type": "pass",
                    "from": str(last_player["id"]),
                    "to":   str(current_player["id"]),
                    "team": current_player.get("team"),
                    "x":    ball["center"][0],
                    "y":    ball["center"][1],
                    "xA":   0.0
                }
                events.append(pass_event)
                state["events_buffer"].append(pass_event)
                if len(state["events_buffer"]) > 5:
                    state["events_buffer"].pop(0)
            else:
                events.append({
                    "type":      "interception",
                    "player":    str(current_player["id"]),
                    "from_team": last_player.get("team"),
                    "to_team":   current_player.get("team"),
                    "x":         ball["center"][0],
                    "y":         ball["center"][1]
                })

    # ─────────────────────────────────────────
    # DRIBBLE
    # ─────────────────────────────────────────
    if current_player and last_player:
        if str(current_player["id"]) == str(last_player["id"]):
            if state.get("last_ball_pos"):
                move = distance(state["last_ball_pos"], ball["center"])
                if move > frame_w * 0.025:
                    events.append({
                        "type":   "dribble",
                        "player": str(current_player["id"]),
                        "x":      ball["center"][0],
                        "y":      ball["center"][1]
                    })

    # ─────────────────────────────────────────
    # BUILD-UP
    # ─────────────────────────────────────────
    state["sequence"].append(ball["center"])
    if len(state["sequence"]) > 20:
        state["sequence"].pop(0)

    if detect_build_up(state["sequence"], frame_w):
        events.append({"type": "build_up_play"})
        state["sequence"] = []

    # ─────────────────────────────────────────
    # TIRS ET BUTS
    # ─────────────────────────────────────────
    if current_player:
        x, y = ball["center"]
        is_shot, is_score = is_shot_zone(
            x, y, sport, shot_zones, frame_w, frame_h
        )

        if is_shot and state["shot_cooldown"] == 0:
            shot_event = {
                "type":   "shot",
                "player": str(current_player["id"]),
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y,
                "danger": compute_danger({"type": "shot"})
            }
            events.append(shot_event)
            state["shot_cooldown"] = 20
            if state["events_buffer"]:
                last_pass        = state["events_buffer"][-1]
                last_pass["xA"] = compute_xa(last_pass, shot_event)

        if is_score and state["score_cooldown"] == 0:
            events.append({
                "type":   "score",
                "player": str(current_player["id"]),
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y,
                "danger": compute_danger({"type": "goal"})
            })
            state["score_cooldown"] = int(25 * 5)

    # ─────────────────────────────────────────
    # PASSE LONGUE
    # ─────────────────────────────────────────
    if state.get("last_ball_pos") and current_player:
        dist_ball = distance(state["last_ball_pos"], ball["center"])
        if dist_ball > frame_w * 0.20:
            events.append({
                "type":   "long_pass",
                "player": str(current_player["id"]),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })

    # ─────────────────────────────────────────
    # UPDATE STATE
    # ─────────────────────────────────────────
    state["last_player"]   = current_player
    state["last_ball_pos"] = ball["center"]
    state["last_team"]     = current_player.get("team") if current_player else None

    return events, state


# ─────────────────────────────────────────
# PROCESS MATCH COMPLET
# ─────────────────────────────────────────
def process_match(frames_data, sport="football", shot_zones=None):
    state      = None
    all_events = []

    for frame in frames_data:
        frame_w = frame.get("frame_w", 1280)
        frame_h = frame.get("frame_h", 720)

        events, state = detect_events(
            players    = frame.get("players", []),
            ball       = frame.get("ball"),
            sport      = sport,
            state      = state,
            shot_zones = shot_zones,
            frame_w    = frame_w,
            frame_h    = frame_h
        )

        frame_id = frame.get("frame", len(all_events))
        for e in events:
            e["frame"] = frame_id

        all_events.extend(events)

    return all_events