import math


def compute_xg_advanced(x, y, goal_x=1920, goal_y=540):
    dx = goal_x - x
    dy = goal_y - y

    distance = math.sqrt(dx*dx + dy*dy)

    # angle simplifié
    angle = abs(math.atan2(dy, dx))

    # modèle réaliste
    xg = (
        1 / (1 + math.exp((distance - 400) / 80))
    ) * (1 - angle / 3.14)

    return max(0.01, min(xg, 0.6))