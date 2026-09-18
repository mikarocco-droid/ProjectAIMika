# analysis/highlight_ranker.py
# -*- coding: utf-8 -*-

from analysis.intelligence import compute_danger


def rank_highlights(events):

    ranked = []

    for e in events:

        score = compute_danger(e) * 10

        if e["type"] == "goal":
            score += 100

        elif e["type"] == "shot":
            score += 50 + e.get("xg", 0) * 50

        elif e["type"] == "pass":
            score += e.get("xA", 0) * 30

        e["highlight_score"] = round(score, 2)
        ranked.append(e)

    # V5.2 (17/09/2026) FIX : triait sur x.get("t", 0), mais tous les
    # events du pipeline utilisent la cle "time" (pas "t") - confirme
    # partout ailleurs ce soir (goal_posthoc.py, post_processing.py,
    # terminal_events.py, zone_analyzer.py, etc.). Ce tri etait donc un
    # no-op silencieux : "t" est toujours absent, x.get("t",0) retourne
    # toujours 0 pour tous les events, l'ordre d'insertion original
    # etait conserve au lieu d'un vrai tri chronologique.
    ranked.sort(key=lambda x: x.get("time", 0))

    return ranked