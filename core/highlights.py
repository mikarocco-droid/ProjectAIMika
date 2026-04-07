def create_highlights(events, video_path=None, sport=None):
    highlights = []

    for e in events:
        t = e["time"]

        if e["type"] == "goal":
            highlights.append({
                "time_start": max(0, t - 5),
                "time_end": t + 5,
                "label": "goal"
            })

        elif e["type"] == "shot" and e.get("xg", 0) > 0.3:
            highlights.append({
                "time_start": max(0, t - 3),
                "time_end": t + 3,
                "label": "shot"
            })

    return highlights