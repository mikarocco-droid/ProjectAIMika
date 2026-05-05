# -*- coding: utf-8 -*-
import math

# =========================================================
# UTILS
# =========================================================

def _get_ball_center(ball):
    if not ball:
        return None

    if ball.get("center"):
        return float(ball["center"][0]), float(ball["center"][1])

    if ball.get("bbox"):
        b = ball["bbox"]
        return (b[0]+b[2])/2, (b[1]+b[3])/2

    if ball.get("x") is not None:
        return float(ball["x"]), float(ball.get("y", 0))


def _compute_speeds(frames_data):
    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i-1].get("ball"))
        if c1 and c0:
            speeds.append(abs(c1[0]-c0[0]))
        else:
            speeds.append(0.0)
    return speeds


def _trajectory_ok(frames_data, i):
    dxs = []
    for k in range(1,5):
        c0 = _get_ball_center(frames_data[i-k].get("ball")) if i-k>=0 else None
        c1 = _get_ball_center(frames_data[i].get("ball"))
        if c0 and c1:
            dxs.append(c1[0]-c0[0])

    if len(dxs) < 2:
        return False

    pos = sum(d>0 for d in dxs)
    neg = sum(d<0 for d in dxs)
    return pos>=3 or neg>=3


def _post_event_signature(frames_data, i, GX_L, GX_R):
    stuck = 0
    disappear = 0

    for j in range(i, min(i+15, len(frames_data))):
        c = _get_ball_center(frames_data[j].get("ball"))
        if c is None:
            disappear += 1
            continue

        if c[0] < GX_L or c[0] > GX_R:
            stuck += 1
        else:
            break

    return stuck, disappear


def _net_rebound_signature(speeds, frames_data, i):
    if i < 5 or i+5 >= len(speeds):
        return False

    pre  = max(speeds[i-5:i])
    post = max(speeds[i:i+5])

    if pre < 1e-3:
        return False

    strong_drop = (post / pre) < 0.5

    c0 = _get_ball_center(frames_data[i-1].get("ball"))
    c1 = _get_ball_center(frames_data[i].get("ball"))
    c2 = _get_ball_center(frames_data[i+2].get("ball"))

    flip = False
    if c0 and c1 and c2:
        dx1 = c1[0]-c0[0]
        dx2 = c2[0]-c1[0]
        if dx1*dx2 < 0:
            flip = True

    return strong_drop or flip


# =========================================================
# MAIN V10
# =========================================================

def detect_fast_goals_from_ball(frames_data, events, fps=25):

    if not frames_data:
        return []

    goals = []

    frame_w = 1920
    frame_h = 1080

    GOAL_PCT = 0.06
    GX_L = frame_w * GOAL_PCT
    GX_R = frame_w * (1 - GOAL_PCT)

    GY_T = frame_h * 0.2
    GY_B = frame_h * 0.8

    LINE_MARGIN = frame_w * 0.002

    GOAL_COOLDOWN = 45  # 🔥 clé anti doublon / kickoff

    speeds = _compute_speeds(frames_data)

    speed_base = sorted(speeds)[len(speeds)//2]
    SPEED_THRESHOLD = speed_base * 2.0

    shots = sorted([e for e in events if e.get("type")=="shot"],
                   key=lambda e:e.get("time",0))

    existing = []

    i = 5

    while i < len(frames_data)-10:

        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            i += 1
            continue

        x,y = c

        # zone proche but
        if not (x < GX_L+frame_w*0.15 or x > GX_R-frame_w*0.15):
            i += 1
            continue

        if not (GY_T < y < GY_B):
            i += 1
            continue

        # crossing
        c_prev = _get_ball_center(frames_data[i-1].get("ball"))
        if not c_prev:
            i += 1
            continue

        xp,_ = c_prev

        cross_left  = xp > GX_L and x <= GX_L
        cross_right = xp < GX_R and x >= GX_R

        if not (cross_left or cross_right):
            i += 1
            continue

        if not _trajectory_ok(frames_data, i):
            i += 1
            continue

        peak = max(speeds[max(0,i-8):i+1])
        if peak < SPEED_THRESHOLD:
            i += 1
            continue

        stuck, disappear = _post_event_signature(frames_data, i, GX_L, GX_R)
        rebound = _net_rebound_signature(speeds, frames_data, i)

        if stuck < 2 or (stuck < 3 and disappear < 2 and not rebound):
            i += 1
            continue

        # temps réel
        frame_id = frames_data[i].get("frame", i)
        t = frame_id / fps

        # 🔥 cooldown anti faux but / kickoff
        if any(abs(t - t0) < GOAL_COOLDOWN for t0 in existing):
            i += 1
            continue

        # 🔥 validation tir
        valid_shot = False
        for s in reversed(shots):
            dt = t - s.get("time",0)
            if dt < 0:
                continue
            if dt > 6:
                break

            if s.get("speed",0) >= 2.0:
                valid_shot = True
                break

        if not valid_shot:
            i += 1
            continue

        # score
        score = 5.0

        if stuck >= 5:
            score += 2
        if disappear >= 2:
            score += 1
        if rebound:
            score += 2

        if score < 6:
            i += 1
            continue

        confidence = round(min(0.6 + score*0.07, 0.95),2)

        goals.append({
            "type":"goal",
            "time":round(t,2),
            "frame":frame_id,
            "x":x,
            "y":y,
            "confidence":confidence,
            "score":score
        })

        print(f"⚽ GOAL {t:.2f}s | score={score}")

        existing.append(t)

        i += max(5, stuck)

    return goals