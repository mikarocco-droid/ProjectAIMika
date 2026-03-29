# analytics/advanced.py

def compute_action_score(e):
    if e.get("type") == "goal":
        return 10
    if e.get("type") == "shot":
        return 5 + e.get("xg", 0) * 5
    if e.get("type") == "pass":
        return 2
    return 1


def build_pass_network(events):
    net = {}
    for e in events:
        if e.get("type") == "pass":
            key = (e.get("player_id"), e.get("receiver_id"))
            net[key] = net.get(key, 0) + 1
    return net


def compute_xa(events):
    for i in range(len(events) - 1):
        if events[i].get("type") == "pass" and events[i+1].get("type") == "shot":
            events[i]["xA"] = events[i+1].get("xg", 0)
    return events


def detect_offside(events):
    offsides = []
    for e in events:
        if e.get("type") == "pass" and e.get("x", 0) > 1000:
            offsides.append(e)
    return offsides