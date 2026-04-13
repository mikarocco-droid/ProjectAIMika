# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-

import math


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def _get_ball_center(ball):
    if not ball:
        return None
    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return None


def _in_goal_zone(x, w):
    return x < w * 0.08 or x > w * 0.92


# ─────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────

def calibrate_match_context(frames_data):
    speeds = []
    visibility = []

    for i in range(1, min(500, len(frames_data))):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))

        if c1 and c0:
            speeds.append(abs(c1[0] - c0[0]))

        visibility.append(frames_data[i].get("ball") is not None)

    if not speeds:
        return {}

    speeds_sorted = sorted(speeds)

    return {
        "speed_base": speeds_sorted[int(len(speeds)*0.5)],
        "speed_peak": speeds_sorted[int(len(speeds)*0.95)],
        "tracking_quality": sum(visibility)/len(visibility)
    }


# ─────────────────────────────────────────────
# MAIN V6
# ─────────────────────────────────────────────

def detect_fast_goals_from_ball(frames_data, events, fps=25, frame_w=1920, frame_h=1080):

    goals = []

    if not frames_data:
        return goals

    ctx = calibrate_match_context(frames_data)

    speed_base = ctx.get("speed_base", 5)
    tracking_quality = ctx.get("tracking_quality", 1.0)

    SPEED_THRESHOLD = speed_base * 1.4
    STUCK_MIN = 3 if tracking_quality < 0.8 else 5
    SCORE_THRESHOLD = 3.0 if tracking_quality < 0.8 else 3.5

    print(f"goal_posthoc V6 | speed_base={speed_base:.2f} | tracking={tracking_quality:.2f}")

    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i-1].get("ball"))

        if c1 and c0:
            dx = (c1[0] - c0[0]) / frame_w
            speeds.append(abs(dx))
        else:
            speeds.append(0.0)

    shots = sorted(
        [e for e in events if e.get("type") == "shot"],
        key=lambda e: e.get("time", 0)
    )

    existing = [e.get("time", 0) for e in events if e.get("type") == "goal"]

    for i in range(5, len(frames_data)-10):

        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            continue

        x, y = c

        if not _in_goal_zone(x, frame_w):
            continue

        score = 2

        peak = max(speeds[i-5:i+1])
        if peak > SPEED_THRESHOLD / frame_w:
            score += 1

        # stuck
        stuck = 0
        for j in range(i, i+15):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj and _in_goal_zone(cj[0], frame_w):
                stuck += 1
            else:
                break

        if stuck >= STUCK_MIN:
            score += 2
        elif stuck >= 2:
            score += 1

        goal_time = frames_data[i].get("frame", i) / fps

        if any(abs(s.get("time", 0) - goal_time) < 5 for s in shots):
            score += 0.5

        if score < SCORE_THRESHOLD:
            continue

        if any(abs(goal_time - t) < 5 for t in existing):
            continue

        goals.append({
            "type": "goal",
            "time": round(goal_time, 2),
            "frame": frames_data[i].get("frame", i),
            "x": x,
            "y": y,
            "confidence": round(min(0.5 + score*0.1, 0.95), 2),
            "score": round(score, 2),
            "detected_from": "goal_posthoc_v6",
            "gemini_validated": False
        })

        print(f"🔥 V6 GOAL {goal_time:.2f}s score={score}")

        existing.append(goal_time)

    return goals