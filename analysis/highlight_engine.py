# analysis/highlight_engine.py

from analysis.intelligence import compute_danger

def score_highlights(events):
    highlights = []

    for e in events:
        score = compute_danger(e)

        if score > 3:
            highlights.append({
                "event": e,
                "score": score
            })

    return sorted(highlights, key=lambda x: -x["score"])[:20]