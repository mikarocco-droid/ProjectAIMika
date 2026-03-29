# analysis/events.py
# -*- coding: utf-8 -*-


# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


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
# DÉTECTION ZONES DE TIR
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):
    """
    Vérifie si la balle est dans une zone de tir.
    Seuils relatifs à la résolution de la vidéo.
    """

    # Zones calibrées dynamiquement
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

    # Seuils relatifs à la résolution
    if sport in ["football", "mini-foot"]:
        # Zone but = 15% droite ou gauche de l'image
        shot_x_hi = frame_w * 0.85
        shot_x_lo = frame_w * 0.15
        shot_y_min = frame_h * 0.25
        shot_y_max = frame_h * 0.75

        y_ok     = shot_y_min <= y <= shot_y_max
        is_shot  = (x > shot_x_hi or x < shot_x_lo) and y_ok
        is_score = (x > frame_w * 0.92 or x < frame_w * 0.08) and y_ok
        return is_shot, is_score

    elif sport == "basketball":
        # Paniers à 8% gauche et 92% droite
        shot_x_hi  = frame_w * 0.92
        shot_x_lo  = frame_w * 0.08
        shot_y_min = frame_h * 0.20
        shot_y_max = frame_h * 0.80

        y_ok     = shot_y_min <= y <= shot_y_max
        is_shot  = (x > shot_x_hi or x < shot_x_lo) and y_ok
        is_score = (x > frame_w * 0.95 or x < frame_w * 0.05) and y_ok
        return is_shot, is_score

    elif sport == "handball":
        shot_x_hi  = frame_w * 0.82
        shot_x_lo  = frame_w * 0.18
        shot_y_min = frame_h * 0.20
        shot_y_max = frame_h * 0.80

        y_ok     = shot_y_min <= y <= shot_y_max
        is_shot  = (x > shot_x_hi or x < shot_x_lo) and y_ok
        is_score = (x > frame_w * 0.88 or x < frame_w * 0.12) and y_ok
        return is_shot, is_score

    elif sport == "rugby":
        is_shot  = x > frame_w * 0.90 or x < frame_w * 0.10
        is_score = x > frame_w * 0.95 or x < frame_w * 0.05
        return is_shot, is_score

    else:
        is_shot  = x > frame_w * 0.85 or x < frame_w * 0.15
        is_score = x > frame_w * 0.92 or x < frame_w * 0.08
        return is_shot, is_score


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE
# ─────────────────────────────────────────
def detect_events_v5(
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
        }

    events = []

    state["shot_cooldown"]  = max(0, state["shot_cooldown"]  - 1)
    state["score_cooldown"] = max(0, state["score_cooldown"] - 1)

    if not players or not ball:
        return events, state

    # ─────────────────────────────────────────
    # JOUEUR LE PLUS PROCHE
    # ─────────────────────────────────────────
    closest, dist = get_closest_player(players, ball)

    # Seuil possession relatif à la résolution
    possession_threshold = {
        "football":        frame_w * 0.06,
        "mini-foot":       frame_w * 0.05,
        "basketball":      frame_w * 0.07,
        "handball":        frame_w * 0.06,
        "rugby":           frame_w * 0.06,
        "tennis":          frame_w * 0.12,
        "padel":           frame_w * 0.12,
        "tennis de table": frame_w * 0.05,
    }.get(sport, frame_w * 0.06)

    current_player = None

    if closest and dist < possession_threshold:
        current_player = closest
        events.append({
            "type":   "possession",
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
                events.append({
                    "type": "pass",
                    "from": str(last_player["id"]),
                    "to":   str(current_player["id"]),
                    "team": current_player.get("team"),
                    "x":    ball["center"][0],
                    "y":    ball["center"][1]
                })
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
    dribble_threshold = {
        "football":        frame_w * 0.025,
        "mini-foot":       frame_w * 0.020,
        "basketball":      frame_w * 0.020,
        "handball":        frame_w * 0.025,
        "rugby":           frame_w * 0.025,
        "tennis":          frame_w * 0.040,
        "padel":           frame_w * 0.040,
        "tennis de table": frame_w * 0.015,
    }.get(sport, frame_w * 0.025)

    if current_player and last_player:
        if str(current_player["id"]) == str(last_player["id"]):
            if state.get("last_ball_pos"):
                move = distance(state["last_ball_pos"], ball["center"])
                if move > dribble_threshold:
                    events.append({
                        "type":   "dribble",
                        "player": str(current_player["id"]),
                        "x":      ball["center"][0],
                        "y":      ball["center"][1]
                    })

    # ─────────────────────────────────────────
    # TIRS ET BUTS
    # ─────────────────────────────────────────
    if current_player:
        x, y = ball["center"]

        is_shot, is_score = is_shot_zone(
            x, y, sport, shot_zones, frame_w, frame_h
        )

        if is_shot and state["shot_cooldown"] == 0:
            events.append({
                "type":   "shot",
                "player": str(current_player["id"]),
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y
            })
            state["shot_cooldown"] = 20  # ~0.8s à 25fps

        if is_score and state["score_cooldown"] == 0:
            events.append({
                "type":   "score",
                "player": str(current_player["id"]),
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y
            })
            state["score_cooldown"] = int(25 * 5)  # 5s à 25fps

    # ─────────────────────────────────────────
    # PASSE LONGUE
    # ─────────────────────────────────────────
    # Seuil relatif — 20% de la largeur
    long_pass_threshold = frame_w * 0.20

    if state.get("last_ball_pos") and current_player:
        dist_ball = distance(state["last_ball_pos"], ball["center"])
        if dist_ball > long_pass_threshold:
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
        # Récupérer dimensions depuis la frame si disponible
        frame_w = frame.get("frame_w", 1280)
        frame_h = frame.get("frame_h", 720)

        events, state = detect_events_v5(
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