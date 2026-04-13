# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-
#
# V7 — Fix critique résolution
#
# BUG V6 : frame_w=1920 hardcodé mais coords en 960px
#   → _in_goal_zone(x, 1920) : x < 0.08*1920 = 153
#   → milieu du terrain (x≈480 en 960px) passe SOUS 153
#   → toute la moitié gauche = "zone de but" → 126 faux buts
#
# FIX V7 :
#   - résolution auto-détectée (frames_data[0].frame_w en priorité)
#   - zone de but recalculée sur la vraie résolution
#   - score threshold = 3.5 (relevé pour réduire les faux positifs)
#   - cooldown interne 5s pour éviter doublons consécutifs

import math


def _resolve_resolution(frames_data, frame_w_hint, frame_h_hint):
    """Résolution réelle : frames_data[0] → inférence → hint."""
    if frames_data:
        fw = frames_data[0].get("frame_w")
        fh = frames_data[0].get("frame_h")
        if fw and fh:
            return int(fw), int(fh)

    max_x = 0
    for f in frames_data[:300]:
        ball = f.get("ball")
        if not ball:
            continue
        c = ball.get("center") or [ball.get("x", 0), 0]
        if c and c[0]:
            max_x = max(max_x, c[0])

    if max_x > 1500:
        return 1920, 1080
    elif max_x > 900:
        return 1280, 720
    elif max_x > 0:
        return 960, 540
    return frame_w_hint, frame_h_hint


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
        return {"speed_base": frame_w * 0.005, "speed_peak": frame_w * 0.02,
                "tracking_quality": 1.0}

    s = sorted(speeds)
    return {
        "speed_base":       s[int(len(s) * 0.50)],
        "speed_peak":       s[int(len(s) * 0.95)],
        "tracking_quality": sum(visibility) / len(visibility),
    }


def detect_fast_goals_from_ball(
    frames_data,
    events,
    fps                  = 25,
    frame_w              = 1920,
    frame_h              = 1080,
    disappearance_frames = 8,
    speed_threshold      = 0.008,
    shot_window          = 5.0,
    xg_min               = 0.10,
):
    goals = []
    if not frames_data:
        return goals

    # ── Résolution réelle ─────────────────────────────────────────────────────
    actual_w, actual_h = _resolve_resolution(frames_data, frame_w, frame_h)
    if actual_w != frame_w:
        print(f"  goal_posthoc : résolution corrigée {frame_w}x{frame_h} → {actual_w}x{actual_h}")
    frame_w = actual_w
    frame_h = actual_h

    ctx              = calibrate_match_context(frames_data, frame_w)
    speed_base       = ctx["speed_base"]
    tracking_quality = ctx["tracking_quality"]

    SPEED_THRESHOLD = speed_base * 1.8
    STUCK_MIN       = 4 if tracking_quality < 0.8 else 6
    SCORE_THRESHOLD = 3.5

    # Zone de but en pixels réels (8% de chaque côté)
    GOAL_X_LEFT  = frame_w * 0.08
    GOAL_X_RIGHT = frame_w * 0.92

    print(f"  goal_posthoc V7 | res={frame_w}x{frame_h} | "
          f"speed_base={speed_base:.1f}px | tracking={tracking_quality:.2f} | "
          f"goal_zone=[0,{GOAL_X_LEFT:.0f}]∪[{GOAL_X_RIGHT:.0f},{frame_w}]")

    # Pré-calcul vitesses
    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))
        speeds.append(abs(c1[0] - c0[0]) if c1 and c0 else 0.0)

    shots   = sorted([e for e in events if e.get("type") == "shot"],
                     key=lambda e: e.get("time", 0))
    existing = [e.get("time", 0) for e in events if e.get("type") == "goal"]

    i = 5
    while i < len(frames_data) - 10:
        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            i += 1
            continue

        x, y = c

        # ── Zone de but STRICTE avec vraie résolution ─────────────────────────
        if not (x < GOAL_X_LEFT or x > GOAL_X_RIGHT):
            i += 1
            continue

        score = 2.0

        # Vitesse pic avant
        peak = max(speeds[max(0, i - 5):i + 1])
        if peak > SPEED_THRESHOLD:
            score += 1.0

        # Ball stuck
        stuck = 0
        for j in range(i, min(i + 20, len(frames_data))):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj and (cj[0] < GOAL_X_LEFT or cj[0] > GOAL_X_RIGHT):
                stuck += 1
            else:
                break

        if stuck >= STUCK_MIN:
            score += 2.0
        elif stuck >= 3:
            score += 1.0

        goal_time = frames_data[i].get("frame", i) / fps

        # Tir récent
        recent_shot = None
        for s in reversed(shots):
            s_time = s.get("time", 0)
            if s_time > goal_time:
                continue
            if goal_time - s_time > shot_window:
                break
            recent_shot = s
            break

        if recent_shot is not None:
            score += 0.5

        # Ralentissement
        speed_after = max(speeds[i + 1:i + 6]) if i + 6 < len(speeds) else 0
        if peak > 0 and speed_after < peak * 0.5:
            score += 0.5

        if score < SCORE_THRESHOLD:
            i += 1
            continue

        if any(abs(goal_time - t) < 5.0 for t in existing):
            i += max(stuck, 3) + 1
            continue

        shot_xg    = recent_shot.get("xg", 0) if recent_shot else 0.0
        confidence = round(min(0.40 + score * 0.08, 0.95), 2)

        goals.append({
            "type":             "goal",
            "time":             round(goal_time, 2),
            "frame":            frames_data[i].get("frame", i),
            "x":                x,
            "y":                y,
            "xg":               shot_xg or 0.0,
            "player":           recent_shot.get("player") if recent_shot else None,
            "team":             recent_shot.get("team")   if recent_shot else None,
            "confidence":       confidence,
            "score":            round(score, 2),
            "detected_from":    "goal_posthoc_v7",
            "shot_linked":      recent_shot is not None,
            "gemini_validated": False,
        })

        print(f"  goal_posthoc BUT à {int(goal_time//60):02d}:{int(goal_time%60):02d} "
              f"(score={score:.1f} | conf={confidence} | x={x:.0f} | stuck={stuck}f)")

        existing.append(goal_time)
        i += max(stuck, 5) + 1

    return goals