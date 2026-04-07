# analysis/player_reid.py
# -*- coding: utf-8 -*-
import math


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def reidentify_players(events, max_dist=100):
    """
    Fusionne les IDs tracker proches pour réduire
    les doublons (76 joueurs → ~22).
    """
    memory  = {}   # player_id -> last position
    new_ids = {}
    next_id = 1

    for e in events:
        if "x" not in e or "y" not in e:
            continue

        pid = e.get("player")
        if pid is None:
            continue

        pos      = (e["x"], e["y"])
        assigned = None

        for stored_pid, last_pos in memory.items():
            if distance(pos, last_pos) < max_dist:
                assigned = stored_pid
                break

        if assigned is None:
            assigned = next_id
            next_id += 1

        memory[assigned] = pos
        new_ids[pid]     = assigned
        e["player"]      = assigned

    return events