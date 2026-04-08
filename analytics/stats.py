# analytics/stats.py
# -*- coding: utf-8 -*-

from collections import defaultdict


def compute_stats(events, jersey_map=None):
    """
    Calcule les stats par joueur.

    FIX label — priorité d'affichage :
      1. Numéro maillot OCR  → #9
      2. Numéro maillot Gemini → #9
      3. Fallback             → ID-{pid}

    FIX équipe — on déduit l'équipe majoritaire de chaque joueur
    depuis ses events pour corriger les cas où team=None ou 100% équipe 1.
    """

    players = defaultdict(lambda: {
        "touches":            0,
        "passes":             0,
        "passes_reussies":    0,
        "progressive_passes": 0,
        "key_passes":         0,
        "tirs":               0,
        "buts":               0,
        "interceptions":      0,
        "dribbles":           0,
        "long_passes":        0,
        "progressive_runs":   0,
        "xg_total":           0.0,
        "xa_total":           0.0,
        "danger_total":       0.0,
        "jersey":             None,
        "label":              None,
        "team":               None,
        "_team_votes":        defaultdict(int),  # FIX équipe
    })

    for e in events:
        t   = e.get("type")
        pid = str(e.get("player", e.get("from", "")))
        if not pid:
            continue

        # FIX label — assigner jersey dès le premier event
        if players[pid]["jersey"] is None and jersey_map:
            jersey = (
                jersey_map.get(pid)
                or jersey_map.get(int(pid) if pid.isdigit() else pid)
            )
            if jersey:
                players[pid]["jersey"] = jersey
                players[pid]["label"]  = f"#{jersey}"

        # FIX équipe — vote majoritaire sur tous les events du joueur
        team = e.get("team")
        if team is not None:
            players[pid]["_team_votes"][team] += 1

        if t == "possession":
            players[pid]["touches"] += 1

        elif t == "pass":
            from_pid = str(e.get("from", ""))
            if from_pid:
                players[from_pid]["passes"]          += 1
                players[from_pid]["passes_reussies"] += 1
                players[from_pid]["xa_total"]        += e.get("xA", 0)
                # vote équipe aussi pour from_pid
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

    # ── FIX label + équipe — post-processing ──
    for pid, s in filtered.items():

        # Label fallback si pas de jersey
        if s["label"] is None:
            s["label"] = f"ID-{pid}"

        # FIX équipe — assigner l'équipe majoritaire
        votes = s.pop("_team_votes", {})
        if votes:
            s["team"] = max(votes, key=votes.get)
        else:
            s["team"] = None

    return filtered


# ─────────────────────────────────────────
# POSSESSION CORRIGÉE
# FIX — utilise les stats joueurs (team majoritaire)
# pour corriger la possession si smart_game_ai donne 100%
# ─────────────────────────────────────────
def compute_possession_from_stats(events, stats):
    """
    Calcule la possession en s'appuyant sur les équipes
    déduites depuis les stats joueurs (vote majoritaire).
    Évite le 100% équipe 1 quand player_reid n'a pas bien assigné.
    """
    # Construire un mapping pid → team depuis les stats
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

        # FIX — si team=None dans l'event, chercher dans stats
        if team is None and pid in pid_to_team:
            team = pid_to_team[pid]

        if team is not None:
            possession[team] += dt

    total = sum(possession.values()) or 1

    result = {
        team: round(100 * t / total, 1)
        for team, t in possession.items()
    }

    # FIX — si toujours 100% équipe 1 (aucune équipe 0 détectée),
    # estimer 50/50 plutôt que d'afficher une donnée trompeuse
    if len(result) <= 1:
        result = {0: 50.0, 1: 50.0}

    return result