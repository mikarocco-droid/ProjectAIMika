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
# FIX 1 — IDENTIFIER LE GARDIEN
# ─────────────────────────────────────────
def get_goalkeeper(players, frame_w):
    threshold = frame_w * 0.15
    gks = [
        p for p in players
        if p["center"][0] < threshold or p["center"][0] > frame_w - threshold
    ]
    if not gks:
        return None
    return min(gks, key=lambda p: min(p["center"][0], frame_w - p["center"][0]))


# ─────────────────────────────────────────
# FIX 2 — DÉTECTION RELANCE À LA MAIN
# ─────────────────────────────────────────
def is_goalkeeper_throw(ball_pos, last_ball_pos, frame_w, frame_h, gk):
    if gk is None or last_ball_pos is None:
        return False

    bx, by = ball_pos
    lx, ly = last_ball_pos
    gx     = gk["center"][0]

    in_gk_zone = (lx < frame_w * 0.15 or lx > frame_w * 0.85)
    if not in_gk_zone:
        return False

    dx  = bx - lx
    dy  = by - ly
    spd = math.hypot(dx, dy)

    speed_ok = frame_w * 0.01 < spd < frame_w * 0.08

    moves_away_from_goal = (
        (gx < frame_w * 0.15 and dx > 0) or
        (gx > frame_w * 0.85 and dx < 0)
    )

    return speed_ok and moves_away_from_goal


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
            y_tol     = (y_max - y_min) * 0.50
            in_y_shot = (y_min - y_tol <= y <= y_max + y_tol)
            in_y_goal = (y_min - y_tol * 0.3 <= y <= y_max + y_tol * 0.3)
            in_zone   = (x > hi or x < lo) and in_y_shot
            in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
            return in_zone or in_goal, in_goal
        else:
            in_zone = (y < hi or y > lo)
            in_goal = (y < hi * 0.88 or y > lo * 1.12)
            return in_zone, in_goal

    if sport == "football":
        in_y_shot = (frame_h * 0.15 <= y <= frame_h * 0.90)
        in_y_goal = (frame_h * 0.20 <= y <= frame_h * 0.90)
        in_zone   = (x > frame_w * 0.80 or x < frame_w * 0.20) and in_y_shot
        in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
        return in_zone or in_goal, in_goal

    if sport == "basketball":
        in_zone = (x > frame_w * 0.88 or x < frame_w * 0.12)
        in_goal = (x > frame_w * 0.94 or x < frame_w * 0.06)
        return in_zone, in_goal

    if sport == "handball":
        in_y_shot = (frame_h * 0.15 <= y <= frame_h * 0.85)
        in_y_goal = (frame_h * 0.20 <= y <= frame_h * 0.85)
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
    thr = learner.get_thresholds() if learner else {}

    fps            = 25
    shot_cd_frames = int(thr.get("shot_cooldown",   3.0) * fps)
    goal_cd_frames = int(thr.get("goal_cooldown", 150.0) * fps)
    ball_speed_min = thr.get("ball_speed_min",    0.02)
    player_near    = thr.get("player_near_goal",  0.15)
    goal_frames    = int(thr.get("goal_frames_min", 12))

    return {
        "last_player":              None,
        "last_ball_pos":            None,
        "last_team":                None,
        "sequence":                 deque(maxlen=30),
        "events_buffer":            deque(maxlen=10),
        "shot_cd":                  0,
        "goal_cd":                  0,
        "possession_time":          0,
        "team_possession":          {0: 0, 1: 0},
        "pressing":                 False,
        "turnover_window":          0,
        "ball_in_goal_zone":        0,
        "_goal_zone_speeds":        [],
        "_shot_cd_max":             shot_cd_frames,
        "_goal_cd_max":             goal_cd_frames,
        "_ball_speed_min":          ball_speed_min,
        "_player_near_goal":        player_near,
        "_goal_frames_min":         goal_frames,
        "_last_ball_interpolated":  False,
        "_last_dribble_time":       -999.0,
        "_dribble_cooldown":        1.5,
        # ── FIX 1 : possession gardien ──
        "_gk_possession_frames":    0,
        "_gk_possession_min":       8,
        "_gk_holding_ball":         False,
        "_gk_release_cd":           0,
        "_gk_release_cd_max":       75,
        # ── FIX shot contré ──
        "_last_shot_x":             None,
        "_last_shot_y":             None,
        "_last_shot_time":          -999.0,   # FIX — timestamp du dernier tir
        "_shot_blocked_cd":         0,
        "_shot_blocked_cd_max":     40,
        # ── FIX touche devant but ──
        "_last_event_types":        deque(maxlen=10),  # historique types récents
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
    fps        = 25,
):
    if state is None:
        state = init_state(learner)

    events = []

    state["shot_cd"]          = max(0, state["shot_cd"] - 1)
    state["goal_cd"]          = max(0, state["goal_cd"] - 1)
    state["_gk_release_cd"]   = max(0, state["_gk_release_cd"] - 1)
    state["_shot_blocked_cd"] = max(0, state["_shot_blocked_cd"] - 1)

    if not players or not ball:
        state["ball_in_goal_zone"] = 0
        state["_goal_zone_speeds"] = []
        return events, state

    current_frame = ball.get("frame", 0) or 0
    current_time  = current_frame / fps

    # ── POSSESSION ───────────────────────
    closest, dist = get_closest_player(players, ball)
    threshold     = frame_w * 0.06
    current       = closest if dist < threshold else None

    if current:
        team_key = current.get("team")
        events.append({
            "type":   "possession",
            "player": str(current["id"]),
            "team":   team_key,
            "x":      ball["center"][0],
            "y":      ball["center"][1]
        })
        state["possession_time"] += 1
        if team_key is not None:
            state["team_possession"][team_key] = \
                state["team_possession"].get(team_key, 0) + 1
        else:
            if players:
                team_counts = {}
                for p in players:
                    t = p.get("team")
                    if t is not None:
                        team_counts[t] = team_counts.get(t, 0) + 1
                if team_counts:
                    dominant = max(team_counts, key=team_counts.get)
                    state["team_possession"][dominant] = \
                        state["team_possession"].get(dominant, 0) + 0.5

    # ── FIX 1 : DÉTECTION POSSESSION GARDIEN ─────────
    gk = get_goalkeeper(players, frame_w)

    if gk and state["last_ball_pos"]:
        ball_spd     = speed(state["last_ball_pos"], ball["center"])
        gk_near_ball = distance(gk["center"], ball["center"]) < frame_w * 0.06
        ball_slow    = ball_spd < frame_w * 0.025

        if gk_near_ball and ball_slow:
            state["_gk_possession_frames"] += 1
            if state["_gk_possession_frames"] >= state["_gk_possession_min"]:
                state["_gk_holding_ball"] = True
        else:
            if state["_gk_holding_ball"]:
                if is_goalkeeper_throw(
                    ball["center"], state["last_ball_pos"],
                    frame_w, frame_h, gk
                ):
                    state["_gk_release_cd"]       = state["_gk_release_cd_max"]
                    state["_gk_holding_ball"]      = False
                    state["_gk_possession_frames"] = 0
                    state["ball_in_goal_zone"]     = 0
                    state["_goal_zone_speeds"]     = []
                    print(f"  GK throw détecté à t={current_time:.1f}s — goal bloqué 3s")
                else:
                    state["_gk_holding_ball"]      = False
                    state["_gk_possession_frames"] = 0
            else:
                state["_gk_possession_frames"] = max(
                    0, state["_gk_possession_frames"] - 1
                )
    elif state["_gk_holding_ball"]:
        state["_gk_holding_ball"]      = False
        state["_gk_possession_frames"] = 0

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

    # ── DRIBBLE ───────────────────────────
    if current and state["last_ball_pos"]:
        ball_spd           = speed(state["last_ball_pos"], ball["center"])
        time_since_dribble = current_time - state.get("_last_dribble_time", -999)
        if (ball_spd > frame_w * 0.025
                and time_since_dribble >= state["_dribble_cooldown"]):
            events.append({
                "type":   "dribble",
                "player": str(current["id"]),
                "team":   current.get("team"),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })
            state["_last_dribble_time"] = current_time

    # ── BUILD UP ─────────────────────────
    state["sequence"].append(ball["center"])
    if detect_build_up(state["sequence"], frame_w):
        events.append({"type": "build_up"})
        state["sequence"].clear()

    # ── LONG PASS ────────────────────────
    if state["last_ball_pos"]:
        if distance(state["last_ball_pos"], ball["center"]) > frame_w * 0.2:
            events.append({
                "type":   "long_pass",
                "player": str(current["id"]) if current else None,
                "team":   current.get("team") if current else None
            })

    # ── SHOTS / GOALS ─────────────────────
    ball_speed_min  = state.get("_ball_speed_min",   0.02)
    player_near_pct = state.get("_player_near_goal", 0.15)
    goal_frames_min = state.get("_goal_frames_min",  12)
    shot_cd_max     = state.get("_shot_cd_max",      75)
    goal_cd_max     = state.get("_goal_cd_max",      3750)

    if current:
        x, y = ball["center"]

        if learner and learner.is_fp_zone(x, y, frame_w, frame_h):
            pass
        else:
            is_shot, is_goal_zone = is_shot_zone(
                x, y, sport, shot_zones, frame_w, frame_h
            )

            ball_interpolated = ball.get("interpolated", False)

            if (state["last_ball_pos"]
                    and not ball_interpolated
                    and not state.get("_last_ball_interpolated", False)):
                ball_speed = speed(state["last_ball_pos"], ball["center"])
            else:
                ball_speed = frame_w * 0.025

            shot_speed_ok = (
                ball_speed > frame_w * ball_speed_min
                or (is_goal_zone and ball_speed > frame_w * 0.01)
            )

            # ── FIX shot contré ──────────────────────────────────────────
            # Seuil relevé à 0.12 (était 0.08) + contrainte :
            # - _last_shot_x doit être défini (un vrai tir précédent existe)
            # - balle dans zone de tir
            # - cooldown tir encore actif
            # - pas de possession gardien
            time_since_last_shot = current_time - state.get("_last_shot_time", -999)

            if (state["shot_cd"] > 0
                    and is_shot
                    and ball_speed > frame_w * 0.12           # FIX seuil 0.08→0.12
                    and state["_last_shot_x"] is not None     # FIX contrainte tir précédent
                    and time_since_last_shot < 8.0            # FIX max 8s depuis dernier tir
                    and not state["_gk_holding_ball"]
                    and state["_gk_release_cd"] == 0):
                state["shot_cd"]          = min(state["shot_cd"], 15)
                state["_shot_blocked_cd"] = state["_shot_blocked_cd_max"]
                events.append({
                    "type":   "shot_blocked",
                    "player": str(current["id"]),
                    "team":   current.get("team"),
                    "x":      state["_last_shot_x"],
                    "y":      state["_last_shot_y"],
                    "danger": 5.0,
                })

            # ── Tir rapide en lucarne ────────────────────────────────────
            fast_shot_in_goal = (
                is_goal_zone
                and ball_speed > frame_w * 0.07
                and state["shot_cd"] == 0
                and not ball_interpolated
                and not state.get("_last_ball_interpolated", False)
                and not state["_gk_holding_ball"]
                and state["_gk_release_cd"] == 0
            )

            if is_shot and state["shot_cd"] == 0 and shot_speed_ok:
                if state["_gk_holding_ball"] or state["_gk_release_cd"] > 0:
                    pass
                else:
                    xg_val = compute_xg(x, y, frame_w, frame_h, learner)
                    shot   = {
                        "type":      "shot",
                        "player":    str(current["id"]),
                        "team":      current.get("team"),
                        "x":         x,
                        "y":         y,
                        "xg":        xg_val,
                        "danger":    compute_danger({"type": "shot", "xg": xg_val}),
                        "on_target": fast_shot_in_goal,
                    }
                    events.append(shot)
                    state["shot_cd"]         = shot_cd_max
                    state["_last_shot_x"]    = x
                    state["_last_shot_y"]    = y
                    state["_last_shot_time"] = current_time   # FIX timestamp

                    if state["events_buffer"]:
                        last_pass       = state["events_buffer"][-1]
                        last_pass["xA"] = compute_xa(last_pass, shot)

            elif fast_shot_in_goal and state["shot_cd"] > 0:
                if ball_speed > frame_w * 0.10:
                    xg_val = compute_xg(x, y, frame_w, frame_h, learner)
                    shot   = {
                        "type":      "shot",
                        "player":    str(current["id"]),
                        "team":      current.get("team"),
                        "x":         x,
                        "y":         y,
                        "xg":        xg_val,
                        "danger":    compute_danger({"type": "shot", "xg": xg_val}),
                        "on_target": True,
                        "fast_shot": True,
                    }
                    events.append(shot)
                    state["shot_cd"]         = shot_cd_max
                    state["_last_shot_x"]    = x
                    state["_last_shot_y"]    = y
                    state["_last_shot_time"] = current_time   # FIX timestamp

            # ── GOAL ─────────────────────────────────────────────────────
            ball_is_real     = not ball_interpolated
            player_near_goal = dist < frame_w * player_near_pct
            gk_blocking_goal = state["_gk_holding_ball"] or state["_gk_release_cd"] > 0

            if is_goal_zone and not gk_blocking_goal:
                if ball_is_real and player_near_goal:
                    state["ball_in_goal_zone"] += 1
                    state["_goal_zone_speeds"].append(ball_speed)
                elif state["ball_in_goal_zone"] > 0:
                    state["ball_in_goal_zone"] += 1
                    state["_goal_zone_speeds"].append(ball_speed)
                else:
                    state["ball_in_goal_zone"] = 0
                    state["_goal_zone_speeds"]  = []
            else:
                if state["ball_in_goal_zone"] > 0 and not gk_blocking_goal:
                    speeds = state["_goal_zone_speeds"]
                    if speeds:
                        avg_speed = sum(speeds) / len(speeds)
                        if avg_speed > frame_w * 0.04:
                            state["ball_in_goal_zone"] = 0
                            state["_goal_zone_speeds"]  = []
                state["ball_in_goal_zone"] = 0
                state["_goal_zone_speeds"]  = []

            goal_frames_threshold = goal_frames_min
            if state.get("_shot_blocked_cd", 0) > 0:
                goal_frames_threshold = max(4, goal_frames_min // 2)

            if (state["ball_in_goal_zone"] >= goal_frames_threshold
                    and state["goal_cd"] == 0
                    and not gk_blocking_goal):
                speeds    = state["_goal_zone_speeds"]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0

                if avg_speed < frame_w * 0.06:
                    events.append({
                        "type":   "goal",
                        "player": str(current["id"]),
                        "team":   current.get("team"),
                        "x":      x,
                        "y":      y,
                        "danger": compute_danger({"type": "goal"})
                    })
                    state["goal_cd"] = goal_cd_max
                else:
                    print(f"  goal rejeté vitesse_avg={avg_speed:.0f}px "
                          f"> seuil={frame_w * 0.06:.0f}px (dégagement)")

                state["ball_in_goal_zone"] = 0
                state["_goal_zone_speeds"]  = []

    else:
        state["ball_in_goal_zone"] = 0
        state["_goal_zone_speeds"]  = []

    # ── UPDATE STATE ─────────────────────
    state["last_player"]             = current
    state["last_ball_pos"]           = ball["center"]
    state["_last_ball_interpolated"] = ball.get("interpolated", False)
    state["last_team"]               = current.get("team") if current else None

    return events, state


# ─────────────────────────────────────────
# MATCH PROCESSOR
# ─────────────────────────────────────────
def process_match(frames_data, sport="football", shot_zones=None, learner=None):
    state      = None
    all_events = []
    fps        = frames_data[0].get("fps", 25) if frames_data else 25

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
            fps        = fps,
        )
        for e in events:
            e["frame"] = frame.get("frame")
        all_events.extend(events)

    return all_events