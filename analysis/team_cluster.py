from collections import defaultdict


def assign_teams_by_color(events):
    team_map = {}

    for e in events:
        color = e.get("color")
        if not color:
            continue

        # simple clustering (2 équipes)
        if color[0] > color[2]:
            team_map[e["player"]] = "team_A"
        else:
            team_map[e["player"]] = "team_B"

    for e in events:
        e["team"] = team_map.get(e.get("player"))

    return events