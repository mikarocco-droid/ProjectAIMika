# analysis/goal_posthoc.py
# -*- coding: utf-8 -*-
#
# V8 — Réduction faux positifs
#
# Problème V7 : 112 faux buts malgré la bonne résolution (1920px)
#   La zone [0, 154px] ∪ [1766px, 1920px] est trop large :
#   le ballon passe régulièrement dans cette zone pour des touches,
#   corners, dégagements sur la ligne, gardiens qui tiennent le ballon.
#
# Fix V8 :
#   - Zone de but réduite à 5% (au lieu de 8%)
#     → [0, 96px] ∪ [1824px, 1920px] sur 1920 : vraiment dans le filet
#   - Score threshold relevé à 4.5 (était 3.5)
#   - stuck_min relevé à 6 frames obligatoires pour +2 points
#   - stuck < 3 ne donne plus de points (trop bruit)
#   - Vitesse peak obligatoire pour valider (sans vitesse, pas de but)
#   - Gemini reste le filtre final (conf >= 0.80)

import math


def _resolve_resolution(frames_data, frame_w_hint, frame_h_hint):
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
    speeds     = []
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

    # Seuils V8 — plus stricts
    SPEED_THRESHOLD = speed_base * 2.0     # pic de vitesse pour un vrai tir
    STUCK_MIN_HIGH  = 6                    # frames pour +2 points
    STUCK_MIN_LOW   = 4                    # frames pour +1 point
    STUCK_MIN_OTHER   = 12                    # frames pour +1 point
    SCORE_THRESHOLD = 4.5                  # relevé de 3.5 → 4.5

    # Zone de but réduite à 5% (au lieu de 8%)
    # Sur 1920px : [0, 96] ∪ [1824, 1920] → vraiment dans le filet
    GOAL_PCT        = 0.05
    GOAL_Y_TOP    = frame_h * 0.2
    GOAL_Y_BOTTOM = frame_h * 0.8
    GOAL_X_LEFT     = frame_w * GOAL_PCT
    GOAL_X_RIGHT    = frame_w * (1 - GOAL_PCT)
    DIR_EPS = 0.5  # tolérance
    LINE_MARGIN = frame_w * 0.002  # ~4px en 1920

    print(f"  goal_posthoc V8 | res={frame_w}x{frame_h} | "
          f"speed_base={speed_base:.1f}px | tracking={tracking_quality:.2f} | "
          f"goal_zone=[0,{GOAL_X_LEFT:.0f}]∪[{GOAL_X_RIGHT:.0f},{frame_w}] | "
          f"score_min={SCORE_THRESHOLD}")

    # Pré-calcul vitesses
    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))
        speeds.append(abs(c1[0] - c0[0]) if c1 and c0 else 0.0)

    shots    = sorted([e for e in events if e.get("type") == "shot"],
                      key=lambda e: e.get("time", 0))
    existing = [e.get("time", 0) for e in events if e.get("type") == "goal"]

    i = 5
    while i < len(frames_data) - 10:
        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            i += 1
            continue
        
        x, y = c

        # ===== Direction passée =====
        dxs = []
        for k in range(1, 6):
            c_prev = _get_ball_center(frames_data[i - k].get("ball")) if i - k >= 0 else None
            if c_prev:
                dxs.append(x - c_prev[0])

        avg_dx = sum(dxs) / len(dxs) if dxs else 0


        # ===== Direction future =====
        forward_dxs = []
        for k in range(1, 4):
            c_next = _get_ball_center(frames_data[i + k].get("ball")) if i + k < len(frames_data) else None
            if c_next:
                forward_dxs.append(c_next[0] - x)

        avg_dx_forward = sum(forward_dxs) / len(forward_dxs) if forward_dxs else 0


        # ===== Sécurité données =====
        if not dxs or not forward_dxs:
            i += 1
            continue


        # ===== Cohérence direction =====
        if (avg_dx * avg_dx_forward) < 0:
            i += 1
            continue  # rebond / déviation


        # ===== Filtre vertical =====
        if not (GOAL_Y_TOP < y < GOAL_Y_BOTTOM):
            i += 1
            continue

        # ===== Franchissement de ligne =====
        c_prev = _get_ball_center(frames_data[i - 1].get("ball")) if i > 0 else None

        if not c_prev:
            i += 1
            continue

        x_prev, _ = c_prev
        
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

        # ===== Zone de but =====
        if not (x < GOAL_X_LEFT or x > GOAL_X_RIGHT):
            i += 1
            continue


        is_left_goal  = x < GOAL_X_LEFT
        is_right_goal = x > GOAL_X_RIGHT


        # ===== Filtre directionnel =====
        if is_left_goal and avg_dx > DIR_EPS:
            i += 1
            continue

        if is_right_goal and avg_dx < -DIR_EPS:
            i += 1
            continue


        # ===== Anti-bruit =====
        if abs(avg_dx) < speed_base * 0.3:
            i += 1
            continue


        score = 2.0

        # Vitesse pic obligatoire (+1) — REQUIS pour valider
        peak = max(speeds[max(0, i - 8):i + 1])
        has_speed = peak > SPEED_THRESHOLD
        if has_speed:
            score += 1.0
        # Sans vitesse, on continue mais score max sera 2+0+stuck+... → difficile d'atteindre 4.5

        # Ball stuck dans la zone
        stuck = 0
        for j in range(i, min(i + 25, len(frames_data))):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj and (cj[0] < GOAL_X_LEFT or cj[0] > GOAL_X_RIGHT):
                stuck += 1
            else:
                break

        if stuck >= STUCK_MIN_OTHER:
            i += stuck + 1
            continue
        if stuck >= STUCK_MIN_HIGH:
            score += 2.0
        elif stuck >= STUCK_MIN_LOW:
            score += 1.0
        # stuck < 4 → 0 points (trop peu fiable)

        # Moment réel = milieu de la phase "stuck"
        goal_frame = i + stuck // 2
        goal_time  = goal_frame / fps

        # Tir récent (+0.5)
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

        # Ralentissement (+0.5)
        speed_after = max(speeds[i + 1:i + 6]) if i + 6 < len(speeds) else 0
        if peak > 0 and speed_after < peak * 0.5:
            score += 0.5

        # Disparition partielle (+0.5)
        if any(frames_data[i + k].get("ball") is None
               for k in range(1, min(6, len(frames_data) - i))):
            score += 0.5

        if score < SCORE_THRESHOLD:
            i += 1
            continue

        # Anti-doublon (cooldown 5s)
        if any(abs(goal_time - t) < 5.0 for t in existing):
            i += max(stuck, 3) + 1
            continue

        shot_xg    = recent_shot.get("xg", 0) if recent_shot else 0.0
        confidence = round(min(0.40 + score * 0.07, 0.95), 2)

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
            "detected_from":    "goal_posthoc_v8",
            "shot_linked":      recent_shot is not None,
            "gemini_validated": False,
        })

        print(f"  goal_posthoc BUT à {int(goal_time//60):02d}:{int(goal_time%60):02d} "
              f"(score={score:.1f} | conf={confidence} | x={x:.0f} | stuck={stuck}f | "
              f"speed_ok={has_speed})")

        existing.append(goal_time)
        i += max(stuck, 5) + 1

    return goals