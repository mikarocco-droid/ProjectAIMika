# sports/config.py

SPORT_CONFIG = {
    "football": {"goal_x": 1200},
    "basketball": {"goal_x": 900},
    "handball": {"goal_x": 950},
}


def get_sport_config(sport):
    return SPORT_CONFIG.get(sport, SPORT_CONFIG["football"])


def compute_xg_sport(x, sport):
    cfg = get_sport_config(sport)
    dist = 1 - (x / cfg["goal_x"])
    return max(0.05, min(0.95, 1 - dist * 1.3))