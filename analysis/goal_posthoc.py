# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-

import math


# =========================================================
# Utils
# =========================================================

def _get_ball_center(ball):
    if not ball:
        return None

    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])

    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    x = ball.get("x")
    y = ball.get("y")
    if x is not None:
        return float(x), float(y or 0)

    return None


def _resolve_resolution(frames_data, frame_w_hint, frame_h_hint):
    if frames_data:
        fw = frames_data[0].get("frame_w")
        fh = frames_data[0].get("frame_h")
        if fw and fh:
            return int(fw), int(fh)

    max_x = 0
    for f in frames_data[:300]:
        c = _get_ball_center(f.get("ball"))
        if c:
            max_x = max(max_x, c[0])

    if max_x > 1500:
        return 1920, 1080
    elif max_x > 900:
        return 1280, 720
    elif max_x > 0:
        return 960, 540

    return frame_w_hint, frame_h_hint


def calibrate_match_context(frames_data, frame_w):
    speeds = []
    visibility = []

    for i in range(1, min(500, len(frames_data))):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))

        if c1 and c0:
            speeds.append(abs(c1[0] - c0[0]))

        visibility.append(frames_data[i].get("ball") is not None)

    if not speeds:
        return {
            "speed_base": frame_w * 0.005,
            "speed_peak": frame_w * 0.02,
            "tracking_quality": 1.0
        }

    s = sorted(speeds)

    return {
        "speed_base": s[int(len(s) * 0.50)],
        "speed_peak": s[int(len(s) * 0.95)],
        "tracking_quality": sum(visibility) / len(visibility),
    }


# =========================================================
# MAIN
# =========================================================

def detect_fast_goals_from_ball(
    frames_data,
    events,
    fps=25,
    frame_w=1920,
    frame_h=1080,
    shot_window=5.0,
):

    goals = []
    if not frames_data:
        return goals

    # -----------------------------------------------------
    # Résolution
    # -----------------------------------------------------
    frame_w, frame_h = _resolve_resolution(frames_data, frame_w, frame_h)

    ctx = calibrate_match_context(frames_data, frame_w)
    speed_base = ctx["speed_base"]

    SPEED_THRESHOLD = speed_base * 2.0
    SCORE_THRESHOLD = 4.5

    # Zone but (5%)
    GOAL_PCT = 0.05
    GOAL_X_LEFT = frame_w * GOAL_PCT
    GOAL_X_RIGHT = frame_w * (1 - GOAL_PCT)

    GOAL_Y_TOP = frame_h * 0.2
    GOAL_Y_BOTTOM = frame_h * 0.8

    LINE_MARGIN = frame_w * 0.002

    print(f"[goal_posthoc_v8] res={frame_w}x{frame_h} speed_base={speed_base:.1f}")

    # -----------------------------------------------------
    # Vitesses
    # -----------------------------------------------------
    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))

        if c1 and c0:
            speeds.append(abs(c1[0] - c0[0]))
        else:
            speeds.append(0.0)

    # Shots
    shots = sorted(
        [e for e in events if e.get("type") == "shot"],
        key=lambda e: e.get("time", 0)
    )

    existing = [e.get("time", 0) for e in events if e.get("type") == "goal"]

    # -----------------------------------------------------
    # LOOP
    # -----------------------------------------------------
    i = 5

    while i < len(frames_data) - 10:

        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            i += 1
            continue

        x, y = c

        # -----------------------------
        # Zone verticale
        # -----------------------------
        if not (GOAL_Y_TOP < y < GOAL_Y_BOTTOM):
            i += 1
            continue

        # -----------------------------
        # Frame précédente
        # -----------------------------
        c_prev = _get_ball_center(frames_data[i - 1].get("ball"))
        if not c_prev:
            i += 1
            continue

        x_prev, _ = c_prev

        # -----------------------------
        # CROSSING (clé)
        # -----------------------------
        cross_left = (
            x_prev > GOAL_X_LEFT + LINE_MARGIN and
            x <= GOAL_X_LEFT - LINE_MARGIN
        )

        cross_right = (
            x_prev < GOAL_X_RIGHT - LINE_MARGIN and
            x >= GOAL_X_RIGHT + LINE_MARGIN
        )

        if not (cross_left or cross_right):
            i += 1
            continue

        # -----------------------------
        # Direction cohérente
        # -----------------------------
        dx = x - x_prev

        if cross_left and dx > 0:
            i += 1
            continue

        if cross_right and dx < 0:
            i += 1
            continue

        # -----------------------------
        # SCORE
        # -----------------------------
        score = 0.0

        # 1. Crossing validé
        score += 3.0

        # -----------------------------
        # Vitesse (obligatoire)
        # -----------------------------
        peak = max(speeds[max(0, i - 8):i + 1])
        if peak > SPEED_THRESHOLD:
            score += 1.5
        else:
            i += 1
            continue

        # -----------------------------
        # STUCK (balle reste dedans)
        # -----------------------------
        stuck = 0
        for j in range(i, min(i + 20, len(frames_data))):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj and (cj[0] < GOAL_X_LEFT or cj[0] > GOAL_X_RIGHT):
                stuck += 1
            else:
                break

        if stuck >= 6:
            score += 1.5
        elif stuck >= 4:
            score += 0.5

        # -----------------------------
        # Shot récent
        # -----------------------------
        goal_time = i / fps

        recent_shot = None
        for s in reversed(shots):
            if goal_time - s.get("time", 0) <= shot_window:
                recent_shot = s
                break

        if recent_shot:
            score += 0.5

        # -----------------------------
        # Validation
        # -----------------------------
        if score < SCORE_THRESHOLD:
            i += 1
            continue

        # Anti doublon
        if any(abs(goal_time - t) < 5 for t in existing):
            i += stuck + 1
            continue

        confidence = round(min(0.5 + score * 0.08, 0.95), 2)

        goals.append({
            "type": "goal",
            "time": round(goal_time, 2),
            "frame": i,
            "x": x,
            "y": y,
            "confidence": confidence,
            "score": round(score, 2),
            "detected_from": "goal_posthoc_v8",
            "shot_linked": recent_shot is not None,
        })

        print(f"⚽ BUT détecté {goal_time:.2f}s | score={score:.2f} | stuck={stuck}")

        existing.append(goal_time)
        i += max(stuck, 5)

    return goals