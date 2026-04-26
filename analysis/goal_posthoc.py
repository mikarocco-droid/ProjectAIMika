# -*- coding: utf-8 -*-

import math


# =========================================================
# UTILS
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

    if ball.get("x") is not None:
        return float(ball["x"]), float(ball.get("y", 0))

    return None


def _resolve_resolution(frames_data, fw, fh):
    if frames_data and frames_data[0].get("frame_w"):
        return int(frames_data[0]["frame_w"]), int(frames_data[0]["frame_h"])

    max_x = 0
    for f in frames_data[:200]:
        c = _get_ball_center(f.get("ball"))
        if c:
            max_x = max(max_x, c[0])

    if max_x > 1500:
        return 1920, 1080
    if max_x > 900:
        return 1280, 720
    return fw, fh


def _compute_speeds(frames_data):
    speeds = [0.0]
    for i in range(1, len(frames_data)):
        c1 = _get_ball_center(frames_data[i].get("ball"))
        c0 = _get_ball_center(frames_data[i - 1].get("ball"))
        if c1 and c0:
            speeds.append(abs(c1[0] - c0[0]))
        else:
            speeds.append(0.0)
    return speeds


def _trajectory_ok(frames_data, i):
    dxs = []
    for k in range(1, 5):
        c_prev = _get_ball_center(frames_data[i - k].get("ball")) if i - k >= 0 else None
        c_curr = _get_ball_center(frames_data[i].get("ball"))
        if c_prev and c_curr:
            dxs.append(c_curr[0] - c_prev[0])

    if len(dxs) < 2:
        return False

    pos = sum(1 for d in dxs if d > 0)
    neg = sum(1 for d in dxs if d < 0)
    return pos >= 3 or neg >= 3


def _post_event_signature(frames_data, i, GOAL_X_LEFT, GOAL_X_RIGHT):
    stuck = 0
    disappear = 0

    for j in range(i, min(i + 15, len(frames_data))):
        c = _get_ball_center(frames_data[j].get("ball"))

        if c is None:
            disappear += 1
            continue

        if c[0] < GOAL_X_LEFT or c[0] > GOAL_X_RIGHT:
            stuck += 1
        else:
            break

    return stuck, disappear


# =========================================================
# 🔥 NOUVEAU : REBOND FILET
# =========================================================

def _net_rebound_signature(speeds, frames_data, i):
    """
    Détecte :
    - décélération brutale
    - petit rebond (changement de direction)
    """

    if i < 5 or i + 5 >= len(speeds):
        return False

    # vitesse avant impact
    pre = max(speeds[i - 5:i])

    # vitesse après impact
    post = max(speeds[i:i + 5])

    # 1. chute brutale
    if pre < 1e-3:
        return False

    drop_ratio = post / pre

    strong_drop = drop_ratio < 0.5

    # 2. mini inversion direction (optionnel mais fort)
    c_before = _get_ball_center(frames_data[i - 1].get("ball"))
    c_curr   = _get_ball_center(frames_data[i].get("ball"))
    c_after  = _get_ball_center(frames_data[i + 2].get("ball"))

    direction_flip = False
    if c_before and c_curr and c_after:
        dx1 = c_curr[0] - c_before[0]
        dx2 = c_after[0] - c_curr[0]
        if dx1 * dx2 < 0:
            direction_flip = True

    return strong_drop or direction_flip


# =========================================================
# MAIN V9.5
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

    frame_w, frame_h = _resolve_resolution(frames_data, frame_w, frame_h)

    GOAL_PCT = 0.06   # zone but standard
    GOAL_X_LEFT = frame_w * GOAL_PCT
    GOAL_X_RIGHT = frame_w * (1 - GOAL_PCT)

    GOAL_Y_TOP = frame_h * 0.2
    GOAL_Y_BOTTOM = frame_h * 0.8

    LINE_MARGIN = frame_w * 0.002

    speeds = _compute_speeds(frames_data)
    speed_base = sorted(speeds)[int(len(speeds) * 0.5)]
    SPEED_THRESHOLD = speed_base * 2.0   # seuil standard

    print(f"[goal_posthoc_v9.6] speed_base={speed_base:.2f}")

    # ── V9.6 — Seuils gardes-fous ────────────────────────────────────
    SHOT_LOOKBACK_STRICT = 6.0   # tir direct → accepté sans condition
    SHOT_LOOKBACK_LOOSE  = 20.0  # tir ancien → accepté uniquement si signal physique fort
    MIN_SHOT_SPEED     = 2.0   # vitesse min du tir (px/frame)
    MIN_PEAK_SPEED     = 4.5   # pic de vitesse requis avant impact
    MIN_DIRECTIONALITY = 0.55  # ratio dx/norme → orientation vers but
    MIN_RECENT_MOTION  = 1.0   # vitesse moyenne min → pas de ballon mort
    # ─────────────────────────────────────────────────────────────────

    shots = sorted([e for e in events if e.get("type") == "shot"],
                   key=lambda e: e.get("time", 0))

    existing = [e.get("time", 0) for e in events if e.get("type") == "goal"]

    i = 5

    while i < len(frames_data) - 10:

        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            i += 1
            continue

        x, y = c

        if not (GOAL_Y_TOP < y < GOAL_Y_BOTTOM):
            i += 1
            continue

        c_prev = _get_ball_center(frames_data[i - 1].get("ball"))
        if not c_prev:
            i += 1
            continue

        x_prev, _ = c_prev

        cross_left = (x_prev > GOAL_X_LEFT + LINE_MARGIN and x <= GOAL_X_LEFT)
        cross_right = (x_prev < GOAL_X_RIGHT - LINE_MARGIN and x >= GOAL_X_RIGHT)

        if not (cross_left or cross_right):
            i += 1
            continue

        if not _trajectory_ok(frames_data, i):
            i += 1
            continue

        peak = max(speeds[max(0, i - 8):i + 1])
        if peak < SPEED_THRESHOLD:
            i += 1
            continue

        stuck, disappear = _post_event_signature(
            frames_data, i, GOAL_X_LEFT, GOAL_X_RIGHT
        )

        rebound = _net_rebound_signature(speeds, frames_data, i)

        # 🔥 règle clé V9.5
        if stuck < 2 or (stuck < 3 and disappear < 2 and not rebound):
            i += 1
            continue

        # Calculer goal_time via frame_id absolu (pas l'index dans la liste filtrée)
        # i = index dans frames_data ≠ frame_id réel → décalage avec frame_skip > 1
        _frame_id_abs = frames_data[i].get("frame", i)
        goal_time = _frame_id_abs / fps
        recent_window = speeds[max(0, i - 15):i]
        recent_motion = (sum(recent_window) / len(recent_window)) if recent_window else 0

        # ── V9.6 — Garde-fou 1 : tir récent obligatoire (HARD FILTER) ──
        # Sans tir récent → pas un but, ballon mort ou phase morte
        # ── V9.6 — Hard filter hybride : tir strict ou loose+signal fort ──
        recent_shot_strict = False
        recent_shot_loose  = False

        for s in reversed(shots):
            dt = goal_time - s.get("time", 0)
            if dt < 0:
                continue
            if dt <= SHOT_LOOKBACK_STRICT:
                recent_shot_strict = True
                break
            if dt <= SHOT_LOOKBACK_LOOSE:
                recent_shot_loose = True
            else:
                break  # trié par temps → inutile de chercher plus loin

        recent_motion_ok = recent_motion >= MIN_RECENT_MOTION * 2.0

        if not recent_shot_strict:
            # Fallback strict (AND) : tir ancien + rebond + vitesse requis simultanément
            # OR était trop permissif → faux positifs sur centres/dégagements rapides
            valid_loose = (
                recent_shot_loose
                and rebound          # rebond filet obligatoire
                and recent_motion_ok # vitesse cohérente obligatoire
            )
            if not valid_loose:
                i += 1
                continue  # ❌ signal insuffisant

        # ── Garde-fou 2 : pic de vitesse avant impact ─────────────────
        peak_before = max(speeds[max(0, i - 20):i + 1]) if i > 0 else 0
        if peak_before < MIN_PEAK_SPEED:
            i += 1
            continue

        # ── Garde-fou 3 : direction vers le but ───────────────────────
        c_10 = _get_ball_center(frames_data[i - 10].get("ball")) if i >= 10 else None
        if c_10 and c:
            dx = c[0] - c_10[0]
            dy = c[1] - c_10[1]
            norm = math.hypot(dx, dy) + 1e-6
            directionality = abs(dx) / norm
            if directionality < MIN_DIRECTIONALITY:
                i += 1
                continue

        # ── Garde-fou 4 : pas de ballon mort ──────────────────────────
        if recent_motion < MIN_RECENT_MOTION:
            i += 1
            continue

        # ─────────────────────────────────────────────────────────────

        score = 3.0 + 1.5

        if stuck >= 6:
            score += 2.0
        elif stuck >= 3:
            score += 1.0

        if disappear >= 2:
            score += 1.0

        if rebound:
            score += 1.5  # 🔥 signal très fort

        if recent_shot_strict:
            score += 0.5  # boost si tir très récent

        # V9.7 — seuil relevé pour réduire les faux positifs
        # score < 7.0 → bruit (stuck faible sans rebound)
        if score < 6.5:
            i += 1
            continue

        if any(abs(goal_time - t) < 10 for t in existing):
            i += 5
            continue

        confidence = round(min(0.6 + score * 0.07, 0.97), 2)

        goals.append({
            "type": "goal",
            "time": round(goal_time, 2),
            "frame": _frame_id_abs,
            "x": x,
            "y": y,
            "confidence": confidence,
            "score": round(score, 2),
            "detected_from": "goal_posthoc_v9.6",
            "shot_linked": recent_shot_strict or recent_shot_loose,
            "rebound": rebound,
        })

        print(f"⚽ GOAL {goal_time:.2f}s | score={score:.2f} | stuck={stuck} | rebound={rebound}")

        existing.append(goal_time)
        i += max(stuck, 5)

    return goals