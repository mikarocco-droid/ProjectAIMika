def deduplicate_goals(events, window=3.0):
    goals = sorted(
        [e for e in events if e.get("type") == "goal"],
        key=lambda x: x.get("time", 0)
    )

    kept = []

    for g in goals:
        if kept and abs(g["time"] - kept[-1]["time"]) < window:
            if g.get("confidence", 0) > kept[-1].get("confidence", 0):
                kept[-1] = g
        else:
            kept.append(g)

    others = [e for e in events if e.get("type") != "goal"]

    return sorted(others + kept, key=lambda x: x.get("time", 0))