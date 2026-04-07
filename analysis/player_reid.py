from analysis.smart_game_ai import (
    compute_possession,
    clean_events_smart,
    cluster_events
)
from analysis.ball_physics import detect_real_shot
from analysis.xg_model import compute_xg_advanced


# ─────────────────────────────────────────
# V18 SMART FILTERING
# ─────────────────────────────────────────

# 1. nettoyage intelligent
events = clean_events_smart(events)

# 2. validation tirs (ULTRA IMPORTANT)
validated = []
for i in range(1, len(events)):
    e = events[i]
    prev = events[i-1]

    if e.get("type") == "shot":
        if not detect_real_shot(e, prev):
            continue

    validated.append(e)

events = validated

# 3. clustering actions
clusters = cluster_events(events)

# garder événements centraux
events = [c[len(c)//2] for c in clusters]

print(f"  SMART events: {len(events)}")

# 4. xG avancé
for e in events:
    if e.get("type") == "shot":
        e["xg"] = compute_xg_advanced(e.get("x", 0), e.get("y", 0))

# 5. possession réelle
possession = compute_possession(events)
print(f"  Possession: {possession}")