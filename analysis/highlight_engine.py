# analysis/highlight_engine.py

from analysis.intelligence import compute_danger


def score_highlights(events):

    highlights = []

    for i, e in enumerate(events):

        score = compute_danger(e)

        # BOOST contexte (action décisive)
        if e.get("type") == "pass":
            if i + 1 < len(events) and events[i+1].get("type") == "shot":
                score += 4

        if e.get("type") == "shot":
            if i + 1 < len(events) and events[i+1].get("type") in ["goal", "score"]:
                score += 6

        if score > 3:
            highlights.append({
                "event": e,
                "score": round(score, 2)
            })

    return sorted(highlights, key=lambda x: -x["score"])[:30]