import math


def compute_speed(p1, p2, dt):
    if not p1 or not p2 or dt <= 0:
        return 0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx + dy*dy) / dt


def detect_real_shot(event, prev_event, speed_thresh=120):
    if not prev_event:
        return False

    if "ball" not in event or "ball" not in prev_event:
        return False

    p1 = prev_event["ball"]
    p2 = event["ball"]

    dt = event.get("time", 0) - prev_event.get("time", 0)
    speed = compute_speed(p1, p2, dt)

    return speed > speed_thresh