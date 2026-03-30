# analysis/tactical.py
# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# DÉTECTION FORME D'ÉQUIPE
# ─────────────────────────────────────────
def detect_team_shape(players, frame_h=720):
    zones = defaultdict(int)
    for p in players:
        y = p.get("center", [0, 0])[1]
        if y < frame_h * 0.33:
            zones["defense"]  += 1
        elif y < frame_h * 0.66:
            zones["midfield"] += 1
        else:
            zones["attack"]   += 1
    return dict(zones)


# ─────────────────────────────────────────
# DÉTECTION FORMATION
# ─────────────────────────────────────────
def detect_formation(events):
    """
    Estime la formation depuis la distribution
    des joueurs dans les events de possession.
    """
    zones = defaultdict(int)

    for e in events:
        if e.get("type") == "possession":
            y = e.get("y", 0)
            if y < 240:
                zones["defense"]  += 1
            elif y < 480:
                zones["midfield"] += 1
            else:
                zones["attack"]   += 1

    total = sum(zones.values()) or 1
    d = round(zones["defense"]  / total * 10)
    m = round(zones["midfield"] / total * 10)
    a = round(zones["attack"]   / total * 10)

    # Formation la plus proche
    if m >= 4 and a >= 3:
        return "4-3-3"
    elif m >= 4 and d >= 4:
        return "4-4-2"
    elif d >= 4 and m >= 3:
        return "4-5-1"
    elif m >= 5:
        return "3-5-2"
    else:
        return "4-4-2"


# ─────────────────────────────────────────
# DÉTECTION PRESSING
# ─────────────────────────────────────────
def detect_pressing(events):
    pressure = sum(1 for e in events if e.get("type") == "interception")
    return pressure > 15


# ─────────────────────────────────────────
# ASSIGNATION ÉQUIPES
# ─────────────────────────────────────────
def assign_teams(events):
    """
    Retourne les events avec team assignée
    et un résumé des équipes détectées.
    """
    teams_found = set()

    for e in events:
        team = e.get("team")
        if team is not None:
            teams_found.add(team)

    teams = {
        str(t): {"id": t, "players": set()}
        for t in teams_found
    }

    for e in events:
        team = e.get("team")
        pid  = e.get("player")
        if team is not None and pid:
            teams[str(team)]["players"].add(str(pid))

    # Convertir sets en listes pour JSON
    for t in teams:
        teams[t]["players"] = list(teams[t]["players"])

    return events, teams


# ─────────────────────────────────────────
# DÉTECTION PHASES DE JEU
# ─────────────────────────────────────────
def detect_phases(events, fps=25):
    """
    Identifie les phases de jeu :
    - possession
    - transition
    - pressing
    - contre-attaque
    """
    phases = []
    window = []

    for e in events:
        window.append(e)

        if len(window) > 50:
            window.pop(0)

        types  = [x.get("type") for x in window]
        n_pass = types.count("pass")
        n_int  = types.count("interception")
        n_prog = types.count("progressive_run")

        if n_pass >= 5 and n_int == 0:
            phase = "possession"
        elif n_int >= 2 and n_prog >= 1:
            phase = "contre_attaque"
        elif n_int >= 3:
            phase = "pressing"
        else:
            phase = "transition"

        if not phases or phases[-1]["phase"] != phase:
            minute = int(e.get("frame", 0) / fps / 60)
            phases.append({
                "phase":  phase,
                "minute": minute,
                "frame":  e.get("frame", 0)
            })

    return phases


# ─────────────────────────────────────────
# STYLE DE JEU
# ─────────────────────────────────────────
def detect_play_style(events):
    passes = sum(1 for e in events if e.get("type") == "pass")
    shots  = sum(1 for e in events if e.get("type") == "shot")
    progs  = sum(1 for e in events if e.get("type") == "progressive_run")

    if shots > passes * 0.3:
        return "direct"
    if passes > shots * 2 and progs < passes * 0.2:
        return "possession"
    if progs > passes * 0.3:
        return "vertical"
    return "balanced"


# ─────────────────────────────────────────
# RAPPORT TACTIQUE COMPLET
# ─────────────────────────────────────────
def tactical_report(events, players_frames=None, frame_h=720):

    shapes = []
    if players_frames:
        for frame in players_frames[:100]:
            if frame:
                shapes.append(detect_team_shape(frame, frame_h))

    return {
        "formation":  detect_formation(events),
        "style":      detect_play_style(events),
        "pressing":   detect_pressing(events),
        "phases":     detect_phases(events)[:10],
        "avg_shape":  shapes[:5] if shapes else []
    }