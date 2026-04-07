def detect_passes(events, max_time=2.0):
    passes = []

    for i in range(1, len(events)):
        e1 = events[i-1]
        e2 = events[i]

        if e1.get("player") != e2.get("player"):
            dt = e2.get("time", 0) - e1.get("time", 0)

            if dt < max_time:
                passes.append({
                    "type": "pass",
                    "from": e1.get("player"),
                    "to": e2.get("player"),
                    "time": e2.get("time")
                })

    return passes