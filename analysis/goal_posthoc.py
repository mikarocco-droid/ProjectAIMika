# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-
#
# Détecteur de buts rapides — v3 PRODUCTION
#
# Corrections v3 :
#   - résolution auto-détectée (fix bug coords 960px vs frame_w=1920)
#   - detect_game_reset avec retour None/True/False (sentinelle)
#   - seuils assouplis pour terrain réel (caméra large)
#   - compatible pipeline existant (même signature detect_fast_goals_from_ball)

import math


def _infer_resolution(frames_data):
    max_x, max_y = 0, 0

    for f in frames_data[:200]:
        ball = f.get("ball")
        if ball:
            cx, cy = ball.get("center", [0, 0])
            max_x = max(max_x, cx)
            max_y = max(max_y, cy)

    if max_x < 1000:
        return 960, 540
    elif max_x < 1500:
        return 1280, 720
    else:
        return 1920, 1080


def detect_game_reset(frames_data, idx, window=25):
    speeds = []
    xs = []

    for j in range(max(0, idx - window), idx):
        players = frames_data[j].get("players", [])
        for p in players:
            if "speed" in p:
                speeds.append(p["speed"])
            if "center" in p:
                xs.append(p["center"][0])

    if not speeds or not xs:
        print(f"[goal_posthoc] reset inconnu idx={idx}")
        return None

    avg_speed = sum(speeds) / len(speeds)
    spread = max(xs) - min(xs)

    return avg_speed < 2.0 and spread < 200


def detect_fast_goals_from_ball(frames_data, events, fps=25):
    if not frames_data:
        return events

    frame_w = frames_data[0].get("frame_w")
    frame_h = frames_data[0].get("frame_h")

    if not frame_w:
        frame_w, frame_h = _infer_resolution(frames_data)

    print(f"[goal_posthoc] resolution utilisée : {frame_w}x{frame_h}")

    existing_goal_times = {
        round(e["time"], 1)
        for e in events
        if e.get("type") == "goal"
    }

    new_goals = []

    for i in range(3, len(frames_data) - 12):
        f_prev = frames_data[i - 3]
        f_now = frames_data[i]
        f_next = frames_data[i + 1]

        ball_prev = f_prev.get("ball")
        ball_now = f_now.get("ball")
        ball_next = f_next.get("ball")

        if not ball_prev or not ball_now:
            continue

        # disparition immédiate
        if ball_next is not None:
            continue

        # disparition prolongée
        reappears = False
        for j in range(i + 1, min(i + 12, len(frames_data))):
            if frames_data[j].get("ball") is not None:
                reappears = True
                break

        if reappears:
            continue

        bx_prev, by_prev = ball_prev.get("center", [0, 0])
        bx_now, by_now = ball_now.get("center", [0, 0])

        dx = bx_now - bx_prev
        dy = by_now - by_prev

        speed = math.sqrt(dx * dx + dy * dy) / frame_w

        if speed < 0.02:
            continue

        in_goal_zone = (
            bx_now < frame_w * 0.18 or
            bx_now > frame_w * 0.82
        )

        if not in_goal_zone:
            continue

        reset = detect_game_reset(frames_data, i)
        if reset is False:
            continue

        t = f_now.get("time", i / fps)

        if any(abs(t - tg) < 3 for tg in existing_goal_times):
            continue

        confidence = min(0.95, 0.6 + speed * 2)

        print(f"[goal_posthoc] BUT détecté à {t:.2f}s (speed={speed:.3f})")

        new_goals.append({
            "type": "goal",
            "time": t,
            "frame": i,
            "confidence": confidence,
            "source": "goal_posthoc"
        })

    return events + new_goals