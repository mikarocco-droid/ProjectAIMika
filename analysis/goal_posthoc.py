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
    goal_box=None,       # résultat de detect_goal_box() — zones dynamiques
):

    goals = []
    if not frames_data:
        return goals

    # ── Initialisation de TOUTES les variables — doivent précéder toute utilisation
    GOAL_PCT          = 0.06
    GOAL_X_LEFT       = frame_w * GOAL_PCT
    GOAL_X_RIGHT      = frame_w * (1 - GOAL_PCT)
    GOAL_Y_TOP        = frame_h * 0.20
    GOAL_Y_BOTTOM     = frame_h * 0.80
    LINE_MARGIN       = frame_w * 0.002
    MIN_PEAK_SPEED    = 8.0
    SHOT_LOOKBACK_LOOSE = 20.0
    _DISAPPEAR_L_MIN  = 0
    _DISAPPEAR_L_MAX  = frame_w * 0.20
    _DISAPPEAR_R_MIN  = frame_w * 0.80
    _DISAPPEAR_R_MAX  = frame_w
    STOP_NEAR_GOAL_X_MIN  = frame_w * 0.45
    STOP_NEAR_GOAL_X_MAX  = frame_w * 0.92
    STOP_SPEED_MAX        = 5.0
    DISAPPEAR_WINDOW      = 8
    REAPPEAR_X_JUMP_MIN   = frame_w * 0.20



    frame_w, frame_h = _resolve_resolution(frames_data, frame_w, frame_h)

    # GOAL_X_LEFT/RIGHT : valeurs fixes stables — ne pas surcharger depuis goal_box
    # goal_box utilisé UNIQUEMENT pour les zones disappear (plus précises)
    GOAL_X_LEFT  = frame_w * GOAL_PCT
    GOAL_X_RIGHT = frame_w * (1 - GOAL_PCT)

    if goal_box and goal_box.get("method") == "vision":
        try:
            from vision.detect_goal_box import goal_box_to_posthoc_params
            _p = goal_box_to_posthoc_params(goal_box)
            _DISAPPEAR_L_MIN = _p.get("DISAPPEAR_X_LEFT_MIN",  0)
            _DISAPPEAR_L_MAX = _p.get("DISAPPEAR_X_LEFT_MAX",  frame_w * 0.12)
            _DISAPPEAR_R_MIN = _p.get("DISAPPEAR_X_RIGHT_MIN", frame_w * 0.88)
            _DISAPPEAR_R_MAX = _p.get("DISAPPEAR_X_RIGHT_MAX", frame_w)
            print(f"  [GOAL_BOX] Zones disappear : L<{_DISAPPEAR_L_MAX:.0f} R>{_DISAPPEAR_R_MIN:.0f}")
        except Exception as _e:
            print(f"  [GOAL_BOX] fallback zones fixes : {_e}")
            _DISAPPEAR_L_MIN = 0
            _DISAPPEAR_L_MAX = frame_w * 0.12
            _DISAPPEAR_R_MIN = frame_w * 0.88
            _DISAPPEAR_R_MAX = frame_w
    else:
        _DISAPPEAR_L_MIN = 0
        _DISAPPEAR_L_MAX = frame_w * 0.12
        _DISAPPEAR_R_MIN = frame_w * 0.88
        _DISAPPEAR_R_MAX = frame_w

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

    # ── LOG DEBUG zone temporelle cible ────────────────────────
    DEBUG_ZONES = [(100, 160), (570, 600)]  # autour des 2 vrais buts
    _debug_logged = set()

    i = 5

    while i < len(frames_data) - 10:

        c = _get_ball_center(frames_data[i].get("ball"))

        # Log détaillé dans les zones cibles
        _frame_id_debug = frames_data[i].get("frame", i)
        _t_debug = _frame_id_debug / fps
        for _t0, _t1 in DEBUG_ZONES:
            if _t0 <= _t_debug <= _t1 and int(_t_debug * 2) not in _debug_logged:
                _debug_logged.add(int(_t_debug * 2))
                _bx = round(c[0], 1) if c else None
                _by = round(c[1], 1) if c else None
                _in_y = (GOAL_Y_TOP < c[1] < GOAL_Y_BOTTOM) if c else False
                _in_xl = (c[0] <= GOAL_X_LEFT) if c else False
                _in_xr = (c[0] >= GOAL_X_RIGHT) if c else False
                _spd = round(speeds[i], 2) if i < len(speeds) else 0
                print(f"  [BALLTRACK] t={_t_debug:.1f}s ball=({_bx},{_by}) "
                      f"in_y={_in_y} in_xl={_in_xl} in_xr={_in_xr} speed={_spd}")

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

    # ── Détection complémentaire : ballon s'arrête near goal puis disparaît ──
    # Couvre le cas où le but est visible (caméra de face) mais le tracker
    # perd le ballon dans le filet avant qu'il ne franchisse la ligne x
    # Zones de disparition : utiliser les zones dynamiques si disponibles
    # Sinon fallback sur zones fixes couvrant les 2 côtés

    speeds = _compute_speeds(frames_data)  # déjà calculé mais recalcul propre

    for i in range(10, len(frames_data) - DISAPPEAR_WINDOW):
        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            continue

        x, y = c

        # 1. Ballon dans zone de tir
        # Utiliser zones dynamiques (goal_box) si disponibles
        near_right = _DISAPPEAR_R_MIN < x < _DISAPPEAR_R_MAX
        near_left  = _DISAPPEAR_L_MIN < x < _DISAPPEAR_L_MAX
        # Fallback : zones fixes si goal_box absent
        if not (near_right or near_left):
            near_right = STOP_NEAR_GOAL_X_MIN < x < STOP_NEAR_GOAL_X_MAX
            near_left  = (frame_w - STOP_NEAR_GOAL_X_MAX) < x < (frame_w - STOP_NEAR_GOAL_X_MIN)
        if not (near_right or near_left):
            continue

        # 2. Ballon dans la hauteur de but
        if not (GOAL_Y_TOP < y < GOAL_Y_BOTTOM):
            continue

        # 3. Ballon quasi-arrêté
        if speeds[i] > STOP_SPEED_MAX:
            continue

        # 4. Pic de vitesse récent (tir avant l'arrêt)
        peak_before = max(speeds[max(0, i-15):i+1]) if i > 0 else 0
        if peak_before < MIN_PEAK_SPEED:
            continue

        # 4b. Contrainte directionnelle : ballon doit se déplacer vers le but
        # Calculer la vélocité moyenne sur les 3 frames précédentes
        vx_sum = 0
        vx_count = 0
        for k in range(max(0, i-3), i):
            ck = _get_ball_center(frames_data[k].get("ball"))
            ck1 = _get_ball_center(frames_data[k+1].get("ball")) if k+1 < len(frames_data) else None
            if ck and ck1:
                vx_sum += ck1[0] - ck[0]
                vx_count += 1
        vx_avg = vx_sum / vx_count if vx_count > 0 else 0

        # Pour but droite : ballon doit aller vers la droite (vx > 0) ou être arrêté
        # Pour but gauche : ballon doit aller vers la gauche (vx < 0) ou être arrêté
        if near_right and vx_avg < -20:  # va fortement vers gauche = pas ce but
            i += 1
            continue
        if near_left and vx_avg > 20:   # va fortement vers droite = pas ce but
            i += 1
            continue

        # 5. Disparition + saut aberrant dans les frames suivantes
        disappeared = False
        for j in range(i+1, min(i+DISAPPEAR_WINDOW, len(frames_data))):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj is None:
                disappeared = True
                break
            xj = cj[0]
            if abs(xj - x) > REAPPEAR_X_JUMP_MIN:
                disappeared = True
                break

        if not disappeared:
            continue

        # 6. Tir récent obligatoire
        _frame_id_abs = frames_data[i].get("frame", i)
        goal_time = _frame_id_abs / fps

        recent_shot = any(
            0 < goal_time - s.get("time", 0) <= SHOT_LOOKBACK_LOOSE
            for s in shots
        )
        if not recent_shot:
            continue

        # 7. Pas doublon
        if any(abs(goal_time - t) < 10 for t in existing):
            continue

        score = 6.5  # score de base légèrement plus bas que cross_line
        confidence = round(min(0.6 + score * 0.07, 0.90), 2)

        goals.append({
            "type":          "goal",
            "time":          round(goal_time, 2),
            "frame":         _frame_id_abs,
            "x":             x,
            "y":             y,
            "confidence":    confidence,
            "score":         round(score, 2),
            "detected_from": "goal_posthoc_disappear",
            "shot_linked":   True,
            "rebound":       False,
        })

        print(f"⚽ GOAL_DISAPPEAR {goal_time:.2f}s | score={score:.2f} | x={x:.0f} near={'right' if near_right else 'left'}")
        existing.append(goal_time)

    # ── Détection complémentaire : ballon s'arrête near goal puis disparaît ──
    # Couvre le cas où le but est visible (caméra de face) mais le tracker
    # perd le ballon dans le filet avant qu'il ne franchisse la ligne x
    # Zones de disparition : utiliser les zones dynamiques si disponibles
    # Sinon fallback sur zones fixes couvrant les 2 côtés

    speeds = _compute_speeds(frames_data)  # déjà calculé mais recalcul propre

    for i in range(10, len(frames_data) - DISAPPEAR_WINDOW):
        c = _get_ball_center(frames_data[i].get("ball"))
        if not c:
            continue

        x, y = c

        # 1. Ballon dans zone de tir
        # Utiliser zones dynamiques (goal_box) si disponibles
        near_right = _DISAPPEAR_R_MIN < x < _DISAPPEAR_R_MAX
        near_left  = _DISAPPEAR_L_MIN < x < _DISAPPEAR_L_MAX
        # Fallback : zones fixes si goal_box absent
        if not (near_right or near_left):
            near_right = STOP_NEAR_GOAL_X_MIN < x < STOP_NEAR_GOAL_X_MAX
            near_left  = (frame_w - STOP_NEAR_GOAL_X_MAX) < x < (frame_w - STOP_NEAR_GOAL_X_MIN)
        if not (near_right or near_left):
            continue

        # 2. Ballon dans la hauteur de but
        if not (GOAL_Y_TOP < y < GOAL_Y_BOTTOM):
            continue

        # 3. Ballon quasi-arrêté
        if speeds[i] > STOP_SPEED_MAX:
            continue

        # 4. Pic de vitesse récent (tir avant l'arrêt)
        peak_before = max(speeds[max(0, i-15):i+1]) if i > 0 else 0
        if peak_before < MIN_PEAK_SPEED:
            continue

        # 4b. Contrainte directionnelle : ballon doit se déplacer vers le but
        # Calculer la vélocité moyenne sur les 3 frames précédentes
        vx_sum = 0
        vx_count = 0
        for k in range(max(0, i-3), i):
            ck = _get_ball_center(frames_data[k].get("ball"))
            ck1 = _get_ball_center(frames_data[k+1].get("ball")) if k+1 < len(frames_data) else None
            if ck and ck1:
                vx_sum += ck1[0] - ck[0]
                vx_count += 1
        vx_avg = vx_sum / vx_count if vx_count > 0 else 0

        # Pour but droite : ballon doit aller vers la droite (vx > 0) ou être arrêté
        # Pour but gauche : ballon doit aller vers la gauche (vx < 0) ou être arrêté
        if near_right and vx_avg < -20:  # va fortement vers gauche = pas ce but
            i += 1
            continue
        if near_left and vx_avg > 20:   # va fortement vers droite = pas ce but
            i += 1
            continue

        # 5. Disparition + saut aberrant dans les frames suivantes
        disappeared = False
        for j in range(i+1, min(i+DISAPPEAR_WINDOW, len(frames_data))):
            cj = _get_ball_center(frames_data[j].get("ball"))
            if cj is None:
                disappeared = True
                break
            xj = cj[0]
            if abs(xj - x) > REAPPEAR_X_JUMP_MIN:
                disappeared = True
                break

        if not disappeared:
            continue

        # 6. Tir récent obligatoire
        _frame_id_abs = frames_data[i].get("frame", i)
        goal_time = _frame_id_abs / fps

        recent_shot = any(
            0 < goal_time - s.get("time", 0) <= SHOT_LOOKBACK_LOOSE
            for s in shots
        )
        if not recent_shot:
            continue

        # 7. Pas doublon
        if any(abs(goal_time - t) < 10 for t in existing):
            continue

        score = 6.5  # score de base légèrement plus bas que cross_line
        confidence = round(min(0.6 + score * 0.07, 0.90), 2)

        goals.append({
            "type":          "goal",
            "time":          round(goal_time, 2),
            "frame":         _frame_id_abs,
            "x":             x,
            "y":             y,
            "confidence":    confidence,
            "score":         round(score, 2),
            "detected_from": "goal_posthoc_disappear",
            "shot_linked":   True,
            "rebound":       False,
        })

        print(f"⚽ GOAL_DISAPPEAR {goal_time:.2f}s | score={score:.2f} | x={x:.0f} near={'right' if near_right else 'left'}")
        existing.append(goal_time)


    return goals