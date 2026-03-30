# analysis/tactical.py

def assign_teams(events):
    teams = {}
    for e in events:
        pid = e.get("player_id")
        if pid:
            team = "A" if e.get("x", 0) < 640 else "B"
            teams[pid] = team
            e["team"] = team
    return events, teams


def detect_formation(events):
    lines = {"def": 0, "mid": 0, "att": 0}

    for e in events:
        if "y" not in e:
            continue

        y = e["y"]

        if y < 250:
            lines["def"] += 1
        elif y < 500:
            lines["mid"] += 1
        else:
            lines["att"] += 1

    return f"{lines['def']}-{lines['mid']}-{lines['att']}"


def detect_pressing(events):
    interceptions = sum(1 for e in events if e.get("type") == "interception")
    return "high" if interceptions > 20 else "low"


def detect_phases(events):
    phases = []
    for e in events:
        phases.append({
            "team": e.get("team"),
            "time": e.get("timestamp", 0)
        })
    return phases