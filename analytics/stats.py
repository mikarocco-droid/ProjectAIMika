# analytics/stats.py
# -*- coding: utf-8 -*-

from collections import defaultdict


def compute_stats(events, jersey_map=None):
    """
    Calcule les stats par joueur.

    FIX — jersey_map optionnel : si fourni, chaque entrée contient
    aussi le numéro de maillot pour l'affichage dans le rapport.
    """

    players = defaultdict(lambda: {
        "touches":           0,
        "passes":            0,
        "passes_reussies":   0,
        "progressive_passes": 0,
        "key_passes":        0,
        "tirs":              0,
        "buts":              0,
        "interceptions":     0,
        "dribbles":          0,
        "long_passes":       0,
        "progressive_runs":  0,
        "xg_total":          0.0,
        "xa_total":          0.0,
        "danger_total":      0.0,
        "jersey":            None,   # FIX — numéro maillot
        "label":             None,   # FIX — label affiché (#9 ou ID-xxxx)
    })

    for e in events:
        t   = e.get("type")
        pid = str(e.get("player", e.get("from", "")))
        if not pid:
            continue

        # FIX — assigner jersey dès le premier event du joueur
        if jersey_map and players[pid]["jersey"] is None:
            jersey = jersey_map.get(pid) or jersey_map.get(int(pid) if pid.isdigit() else pid)
            if jersey:
                players[pid]["jersey"] = jersey
                players[pid]["label"]  = f"#{jersey}"
            else:
                players[pid]["label"]  = f"ID-{pid}"

        if t == "possession":
            players[pid]["touches"] += 1

        elif t == "pass":
            from_pid = str(e.get("from", ""))
            if from_pid:
                players[from_pid]["passes"]          += 1
                players[from_pid]["passes_reussies"] += 1
                players[from_pid]["xa_total"]        += e.get("xA", 0)

        elif t == "progressive_pass":
            players[pid]["progressive_passes"] += 1

        elif t == "key_pass":
            players[pid]["key_passes"] += 1

        elif t == "shot":
            players[pid]["tirs"]      += 1
            players[pid]["xg_total"]  += e.get("xg", 0)
            players[pid]["danger_total"] += e.get("danger", 0)

        elif t in ["goal", "score"]:
            players[pid]["buts"] += 1

        elif t == "interception":
            players[pid]["interceptions"] += 1

        elif t == "dribble":
            players[pid]["dribbles"] += 1

        elif t == "long_pass":
            players[pid]["long_passes"] += 1

        elif t == "progressive_run":
            players[pid]["progressive_runs"] += 1

    # Filtre : garder seulement les joueurs avec au moins 5 touches
    filtered = {
        pid: s
        for pid, s in players.items()
        if s["touches"] >= 5
    }

    # FIX — pour les joueurs sans jersey_map fourni, assigner le label par défaut
    for pid, s in filtered.items():
        if s["label"] is None:
            s["label"] = f"ID-{pid}"

    return filtered