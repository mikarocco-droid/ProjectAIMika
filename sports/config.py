# sports/config.py
# -*- coding: utf-8 -*-

import math

# ─────────────────────────────────────────
# CONFIG PAR SPORT
# ─────────────────────────────────────────
SPORT_CONFIG = {
    "football": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble",
                       "progressive_run", "interception", "fast_break"]
        },
        "goal_cooldown":   3750,   # 2.5 min à 25fps
        "shot_cooldown":   75,     # 3s à 25fps
        "context_before":  12,     # 12s avant — capture le tir + la phase d'attaque
        "context_after":   4,      # 4s après un tir
        "context_goal":    5,      # 5s après un but — juste pour voir rentrer dans le filet
    },
    "mini-foot": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble", "interception"]
        },
        "goal_cooldown":   2500,
        "shot_cooldown":   50,
        "context_before":  10,
        "context_after":   3,
        "context_goal":    4,
    },
    "basketball": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble",
                       "interception", "fast_break"]
        },
        "goal_cooldown":   125,
        "shot_cooldown":   30,
        "context_before":  6,
        "context_after":   3,
        "context_goal":    4,
    },
    "handball": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble", "fast_break"]
        },
        "goal_cooldown":   500,
        "shot_cooldown":   50,
        "context_before":  8,
        "context_after":   3,
        "context_goal":    5,
    },
    "rugby": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score"],
            "player": ["goal", "score", "interception", "fast_break"]
        },
        "goal_cooldown":   1250,
        "shot_cooldown":   75,
        "context_before":  10,
        "context_after":   5,
        "context_goal":    8,
    },
    "hockey sur glace": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "interception", "fast_break"]
        },
        "goal_cooldown":   750,
        "shot_cooldown":   50,
        "context_before":  8,
        "context_after":   4,
        "context_goal":    5,
    },
    "hockey sur gazon": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble"]
        },
        "goal_cooldown":   1250,
        "shot_cooldown":   75,
        "context_before":  10,
        "context_after":   4,
        "context_goal":    6,
    },
    "tennis": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score", "dribble"]
        },
        "goal_cooldown":   250,
        "shot_cooldown":   25,
        "context_before":  4,
        "context_after":   3,
        "context_goal":    4,
    },
    "tennis de table": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score"]
        },
        "goal_cooldown":   125,
        "shot_cooldown":   15,
        "context_before":  3,
        "context_after":   2,
        "context_goal":    3,
    },
    "padel": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score", "dribble"]
        },
        "goal_cooldown":   250,
        "shot_cooldown":   25,
        "context_before":  4,
        "context_after":   3,
        "context_goal":    4,
    },
}


# ─────────────────────────────────────────
# GETTER
# ─────────────────────────────────────────
def get_sport_config(sport):
    return SPORT_CONFIG.get(sport, SPORT_CONFIG["football"])


# ─────────────────────────────────────────
# HIGHLIGHT TYPES PAR SPORT + MODE
# ─────────────────────────────────────────
def get_highlight_types(sport, mode="match"):
    cfg = get_sport_config(sport)
    return cfg.get("highlight_types", {}).get(mode, ["goal", "shot"])


# ─────────────────────────────────────────
# XG PAR SPORT
# ─────────────────────────────────────────
def compute_xg_sport(x, y=None, sport="football"):
    cfg  = get_sport_config(sport)
    gx   = cfg.get("goal_x", 1.0)

    if gx <= 0:
        return 0.1
    dist = float(x) / gx

    exponent = max(-100.0, min(100.0, 4 * (dist - 0.5)))
    xg = 1 / (1 + math.exp(exponent))

    return round(max(0.01, min(0.95, xg)), 3)