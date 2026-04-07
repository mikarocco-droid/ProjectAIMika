def detect_pressing_intensity(events):
    high_press = 0

    for e in events:
        if e.get("type") == "interception":
            if e.get("x", 0) > 1200:  # terrain adverse
                high_press += 1

    return "high" if high_press > 10 else "medium"


def detect_play_style(events):
    long_passes = sum(1 for e in events if e.get("type") == "long_pass")
    short_passes = sum(1 for e in events if e.get("type") == "pass")

    if long_passes > short_passes:
        return "direct"
    return "possession"