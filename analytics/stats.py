# analytics/stats.py
# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# DÉTECTION GARDIEN
# Un joueur est gardien si sa zone d'action moyenne
# est dans les 12% extrêmes du terrain (x < 0.12 ou x > 0.88)
# ─────────────────────────────────────────
def is_goalkeeper(player_stats, frame_w=1920):
    zone_x = player_stats.get("_zone_x_sum", 0)
    n      = player_stats.get("_zone_x_n",   1)
    if n == 0:
        return False
    avg_x   = zone_x / n
    avg_x_n = avg_x / frame_w   # normalisé 0-1
    return avg_x_n < 0.12 or avg_x_n > 0.88


def compute_stats(events, jersey_map=None, frame_w=1920):
    players = defaultdict(lambda: {
        "touches":            0,
        "passes":             0,
        "passes_reussies":    0,
        "progressive_passes": 0,
        "key_passes":         0,
        "tirs":               0,
        "buts":               0,
        "arrets":             0,   # FIX — arrêts gardien
        "interceptions":      0,
        "dribbles":           0,
        "long_passes":        0,
        "progressive_runs":   0,
        "xg_total":           0.0,
        "xa_total":           0.0,
        "danger_total":       0.0,
        "is_goalkeeper":      False,
        "jersey":             None,
        "label":              None,
        "team":               None,
        "_team_votes":        defaultdict(int),
        "_zone_x_sum":        0.0,   # pour détection gardien
        "_zone_x_n":          0,
    })

    for e in events:
        t   = e.get("type")
        pid = str(e.get("player", e.get("from", "")))
        if not pid:
            continue

        # Label jersey
        if players[pid]["jersey"] is None and jersey_map:
            jersey = (
                jersey_map.get(pid)
                or jersey_map.get(int(pid) if pid.isdigit() else pid)
            )
            if jersey:
                players[pid]["jersey"] = jersey
                players[pid]["label"]  = f"#{jersey}"

        # Vote équipe
        team = e.get("team")
        if team is not None:
            players[pid]["_team_votes"][team] += 1

        # Accumule position pour détection gardien
        x = e.get("x")
        if x is not None and float(x) > 0:
            players[pid]["_zone_x_sum"] += float(x)
            players[pid]["_zone_x_n"]   += 1

        if t == "possession":
            players[pid]["touches"] += 1

        elif t == "pass":
            from_pid = str(e.get("from", ""))
            if from_pid:
                players[from_pid]["passes"]          += 1
                players[from_pid]["passes_reussies"] += 1
                players[from_pid]["xa_total"]        += e.get("xA", 0)
                if team is not None:
                    players[from_pid]["_team_votes"][team] += 1

        elif t == "progressive_pass":
            players[pid]["progressive_passes"] += 1

        elif t == "key_pass":
            players[pid]["key_passes"] += 1

        elif t == "shot":
            players[pid]["tirs"]         += 1
            players[pid]["xg_total"]     += e.get("xg", 0)
            players[pid]["danger_total"] += e.get("danger", 0)

        elif t in ["goal", "score"]:
            players[pid]["buts"] += 1

        elif t == "shot_blocked":
            # Le joueur qui bloque = potentiellement le gardien
            players[pid]["arrets"] += 1

        elif t == "interception":
            players[pid]["interceptions"] += 1

        elif t == "dribble":
            players[pid]["dribbles"] += 1

        elif t == "long_pass":
            players[pid]["long_passes"] += 1

        elif t == "progressive_run":
            players[pid]["progressive_runs"] += 1

    # ── Filtre touches minimum ──
    filtered = {
        pid: s
        for pid, s in players.items()
        if s["touches"] >= 15
    }

    # ── Post-processing : label, équipe, gardien ──
    for pid, s in filtered.items():

        if s["label"] is None:
            s["label"] = f"ID-{pid}"

        votes = s.pop("_team_votes", {})
        if votes:
            s["team"] = max(votes, key=votes.get)
        else:
            s["team"] = None

        # FIX — détecter si gardien
        s["is_goalkeeper"] = is_goalkeeper(s, frame_w)

        # FIX — si gardien, ses "buts" sont des arrêts
        if s["is_goalkeeper"] and s["buts"] > 0:
            s["arrets"] += s["buts"]
            s["buts"]    = 0

        # Nettoyer les champs internes
        s.pop("_zone_x_sum", None)
        s.pop("_zone_x_n",   None)

    return filtered


# ─────────────────────────────────────────
# POSSESSION CORRIGÉE
# ─────────────────────────────────────────
def compute_possession_from_stats(events, stats):
    pid_to_team = {
        pid: s["team"]
        for pid, s in stats.items()
        if s.get("team") is not None
    }

    possession = defaultdict(float)
    all_sorted = sorted(events, key=lambda x: x.get("time", 0))

    for i in range(1, len(all_sorted)):
        e1 = all_sorted[i - 1]
        e2 = all_sorted[i]
        dt = max(0, e2.get("time", 0) - e1.get("time", 0))

        pid  = str(e1.get("player", ""))
        team = e1.get("team")

        if team is None and pid in pid_to_team:
            team = pid_to_team[pid]

        if team is not None:
            possession[team] += dt

    total = sum(possession.values()) or 1

    result = {
        team: round(100 * t / total, 1)
        for team, t in possession.items()
    }

    if len(result) <= 1:
        result = {0: 50.0, 1: 50.0}

    return result