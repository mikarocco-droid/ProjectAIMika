# sports/config.py (V15)

import math

SPORTS_CONFIG = {
    "football": {
        "highlight_types": ["goal", "shot"],
        ...
    },
    "basketball": {
        "highlight_types": ["goal", "shot", "dribble"],  # panier + dunk + interception
        ...
    },
    "handball": {
        "highlight_types": ["goal", "shot"],
        ...
    },
    "rugby": {
        "highlight_types": ["goal", "score"],  # essai + transformation
        ...
    },
    "tennis": {
        "highlight_types": ["shot"],  # point gagnant
        ...
    }
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