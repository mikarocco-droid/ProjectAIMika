def filter_goals(goals, frame_w=1920, cooldown=30):

    if not goals:
        return []

    # 🔥 aligné avec V9
    position_threshold = 0.05

    left_max = frame_w * position_threshold
    right_min = frame_w * (1 - position_threshold)

    # ---------------------------------
    # Filtre position
    # ---------------------------------
    goals = [
        g for g in goals
        if g["x"] <= left_max or g["x"] >= right_min
    ]

    # ---------------------------------
    # Grouper par temps
    # ---------------------------------
    goals.sort(key=lambda x: x["time"])

    filtered = []
    group = [goals[0]]

    for g in goals[1:]:
        if g["time"] - group[-1]["time"] <= cooldown:
            group.append(g)
        else:
            best = max(group, key=lambda x: x["confidence"])
            filtered.append(best)
            group = [g]

    if group:
        filtered.append(max(group, key=lambda x: x["confidence"]))

    print(f"[filter_goals_v9] {len(goals)} → {len(filtered)}")

    return filtered