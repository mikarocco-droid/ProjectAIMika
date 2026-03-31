# sports/config.py (V15)

import math

SPORT_CONFIG = {
    "football":   {"goal_x": 1200},
    "basketball": {"goal_x": 900},
    "handball":   {"goal_x": 950},
}


def get_sport_config(sport):
    return SPORT_CONFIG.get(sport, SPORT_CONFIG["football"])


def compute_xg_sport(x, y=None, sport="football"):
    cfg = get_sport_config(sport)

    # distance normalisée
    dist = x / cfg["goal_x"]

    # modèle logistique (plus réaliste)
    xg = 1 / (1 + math.exp(4 * (dist - 0.5)))

    return round(max(0.01, min(0.95, xg)), 3)