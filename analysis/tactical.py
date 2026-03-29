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
    return "4-3-3"


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