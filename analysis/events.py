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
# DÉTECTION TIRS AVEC ZONES CALIBRÉES
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None):
    """
    Vérifie si la balle est dans une zone de tir.

    shot_zones : dict calibré par calibration.py
                 Si None, utilise les valeurs fixes par défaut.

    Retourne : (is_shot, is_score)
    """

    # ── Zones calibrées dynamiquement ────
    if shot_zones:
        axis    = shot_zones.get("axis", "x")
        hi      = shot_zones.get("threshold_hi", 900)
        lo      = shot_zones.get("threshold_lo", 0)
        y_min   = shot_zones.get("y_min", 0)
        y_max   = shot_zones.get("y_max", 9999)
        y_ok    = y_min <= y <= y_max

        if axis == "x":
            is_shot  = (x > hi or x < lo) and y_ok
            is_score = (x > hi * 1.02 or x < lo * 0.98) and y_ok
        else:
            is_shot  = (y < hi or y > lo)
            is_score = (y < hi * 0.85 or y > lo * 1.15)

        return is_shot, is_score

    # ── Valeurs fixes par défaut ──────────
    if sport in ["football", "mini-foot"]:
        is_shot  = x > 900
        is_score = x > 950
        return is_shot, is_score

    elif sport == "basketball":
        # Panier haut OU bas de l'image selon caméra
        is_shot  = y < 200 or y > 520
        is_score = y < 150 or y > 570
        return is_shot, is_score

    elif sport == "handball":
        is_shot  = x > 850
        is_score = x > 900
        return is_shot, is_score

    elif sport == "rugby":
        is_shot  = x > 900 or x < 100
        is_score = x > 950 or x < 50
        return is_shot, is_score

    elif sport in ["tennis", "padel", "tennis de table"]:
        # Pas de tir en tennis — on ne détecte pas ce type d'event
        return False, False

    else:
        is_shot  = x > 900
        is_score = x > 950
        return is_shot, is_score


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE V5
# ─────────────────────────────────────────
def detect_events_v5(
    players,
    ball,
    sport      = "football",
    state      = None,
    shot_zones = None
):
    """
    Détecte les events sur une frame.

    Paramètres :
        players    : [{"id": int, "center": [x,y], "team": int}]
        ball       : {"center": [x,y]} ou None
        sport      : sport en cours
        state      : mémoire entre frames (None = init)
        shot_zones : zones calibrées par calibration.py (optionnel)

    Retourne :
        events : liste des events de cette frame
        state  : état mis à jour pour la frame suivante
    """

    if state is None:
        state = {
            "last_player":   None,
            "last_ball_pos": None,
            "last_team":     None,
            "shot_cooldown": 0,    # évite les tirs en rafale
            "score_cooldown": 0,   # évite les buts en rafale
        }

    events = []

    # Décrémenter les cooldowns
    state["shot_cooldown"]  = max(0, state["shot_cooldown"]  - 1)
    state["score_cooldown"] = max(0, state["score_cooldown"] - 1)

    if not players or not ball:
        return events, state

    # ─────────────────────────────────────────
    # JOUEUR LE PLUS PROCHE DU BALLON
    # ─────────────────────────────────────────
    closest, dist = get_closest_player(players, ball)

    # Seuil de possession adapté au sport
    possession_threshold = {
        "football":        80,
        "mini-foot":       60,
        "basketball":      90,
        "handball":        80,
        "rugby":           80,
        "tennis":         150,
        "padel":          150,
        "tennis de table": 60,
        "hockey sur glace":70,
        "hockey sur gazon":70,
    }.get(sport, 80)

    current_player = None

    if closest and dist < possession_threshold:
        current_player = closest

        events.append({
            "type":   "possession",
            "player": current_player["id"],
            "team":   current_player.get("team"),
            "x":      ball["center"][0],
            "y":      ball["center"][1]
        })

    # ─────────────────────────────────────────
    # PASS / INTERCEPTION
    # ─────────────────────────────────────────
    last_player = state.get("last_player")

    if last_player and current_player:

        if last_player["id"] != current_player["id"]:

            same_team = last_player.get("team") == current_player.get("team")

            if same_team:
                events.append({
                    "type": "pass",
                    "from": last_player["id"],
                    "to":   current_player["id"],
                    "team": current_player.get("team"),
                    "x":    ball["center"][0],
                    "y":    ball["center"][1]
                })
            else:
                events.append({
                    "type":      "interception",
                    "player":    current_player["id"],
                    "from_team": last_player.get("team"),
                    "to_team":   current_player.get("team"),
                    "x":         ball["center"][0],
                    "y":         ball["center"][1]
                })

    # ─────────────────────────────────────────
    # DRIBBLE
    # ─────────────────────────────────────────
    dribble_threshold = {
        "football":        30,
        "mini-foot":       20,
        "basketball":      25,
        "handball":        25,
        "rugby":           30,
        "tennis":          50,
        "padel":           50,
        "tennis de table": 15,
    }.get(sport, 30)

    if current_player and last_player:
        if current_player["id"] == last_player["id"]:
            if state.get("last_ball_pos"):
                move = distance(state["last_ball_pos"], ball["center"])
                if move > dribble_threshold:
                    events.append({
                        "type":   "dribble",
                        "player": current_player["id"],
                        "x":      ball["center"][0],
                        "y":      ball["center"][1]
                    })

    # ─────────────────────────────────────────
    # TIRS ET BUTS
    # ─────────────────────────────────────────
    if current_player:
        x, y = ball["center"]

        is_shot, is_score = is_shot_zone(x, y, sport, shot_zones)

        if is_shot and state["shot_cooldown"] == 0:
            events.append({
                "type":   "shot",
                "player": current_player["id"],
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y
            })
            state["shot_cooldown"] = 15  # ~0.5s à 30fps

        if is_score and state["score_cooldown"] == 0:
            events.append({
                "type":   "score",
                "player": current_player["id"],
                "team":   current_player.get("team"),
                "x":      x,
                "y":      y
            })
            state["score_cooldown"] = 90  # ~3s à 30fps

    # ─────────────────────────────────────────
    # PASSE LONGUE
    # ─────────────────────────────────────────
    long_pass_threshold = {
        "football":   120,
        "mini-foot":   80,
        "basketball":  100,
        "handball":    100,
        "rugby":       150,
        "tennis":      200,
    }.get(sport, 120)

    if state.get("last_ball_pos") and current_player:
        dist_ball = distance(state["last_ball_pos"], ball["center"])
        if dist_ball > long_pass_threshold:
            events.append({
                "type":   "long_pass",
                "player": current_player["id"],
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
    """
    Traite toutes les frames et retourne la liste complète des events.

    Paramètres :
        frames_data : liste de dicts {players, ball, frame}
        sport       : sport en cours
        shot_zones  : zones calibrées (optionnel)
    """
    state      = None
    all_events = []

    for frame in frames_data:
        events, state = detect_events_v5(
            players    = frame.get("players", []),
            ball       = frame.get("ball"),
            sport      = sport,
            state      = state,
            shot_zones = shot_zones
        )

        frame_id = frame.get("frame", len(all_events))
        for e in events:
            e["frame"] = frame_id

        all_events.extend(events)

    return all_events


# ─────────────────────────────────────────
# TEST LOCAL
# ─────────────────────────────────────────
if __name__ == "__main__":

    frames = [
        {
            "players": [
                {"id": 1, "center": [100, 200], "team": 0},
                {"id": 2, "center": [300, 200], "team": 1}
            ],
            "ball":  {"center": [110, 210]},
            "frame": 0
        },
        {
            "players": [
                {"id": 1, "center": [150, 200], "team": 0},
                {"id": 2, "center": [300, 200], "team": 1}
            ],
            "ball":  {"center": [160, 210]},
            "frame": 1
        },
        {
            "players": [
                {"id": 1, "center": [150, 200], "team": 0},
                {"id": 2, "center": [320, 200], "team": 1}
            ],
            "ball":  {"center": [310, 210]},
            "frame": 2
        },
        {
            "players": [
                {"id": 1, "center": [150, 200], "team": 0},
                {"id": 2, "center": [320, 200], "team": 1}
            ],
            "ball":  {"center": [950, 350]},
            "frame": 3
        }
    ]

    print("\nTest football :")
    events = process_match(frames, sport="football")
    for e in events:
        print(f"  {e}")

    print("\nTest basketball avec zones calibrees :")
    shot_zones_basket = {
        "axis":         "x",
        "threshold_hi": 0.88,
        "threshold_lo": 0.12,
        "y_min":        0.30,
        "y_max":        0.70
    }
    events = process_match(frames, sport="basketball", shot_zones=shot_zones_basket)
    for e in events:
        print(f"  {e}")