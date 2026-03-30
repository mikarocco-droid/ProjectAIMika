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

    ranked.sort(key=lambda x: x["highlight_score"], reverse=True)

    return ranked