def compute_xa(events):
    for e in events:
        if e.get("type") == "pass":
            # simple : bonus si suivi d'un tir
            e["xa"] = 0.05
        else:
            e["xa"] = 0.0
    return events