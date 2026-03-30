# analysis/highlights.py
# -*- coding: utf-8 -*-

from collections import defaultdict
from analysis.intelligence import compute_danger


def group_sequences(events, fps):
    sequences = []
    current = []

    for e in events:
        if not current:
            current = [e]
            continue

        if e["frame"] - current[-1]["frame"] < fps * 2:
            current.append(e)
        else:
            sequences.append(current)
            current = [e]

    if current:
        sequences.append(current)

    return sequences


def score_sequence(seq):
    score = sum(compute_danger(e) for e in seq)

    # boost si but
    if any(e["type"] in ["goal", "score"] for e in seq):
        score += 10

    return score


def extract_highlights(events, max_highlights=20, fps=30):

    if not events:
        return []

    events = sorted(events, key=lambda x: x["frame"])

    sequences = group_sequences(events, fps)

    scored = []

    for seq in sequences:
        s = score_sequence(seq)

        if s < 5:
            continue

        start = seq[0]["frame"]
        end   = seq[-1]["frame"]

        scored.append({
            "frame_start": start - int(fps * 4),
            "frame_end":   end + int(fps * 3),
            "score":       round(s, 2),
            "events":      seq,
            "main_type":   max(seq, key=lambda e: compute_danger(e))["type"]
        })

    scored.sort(key=lambda x: -x["score"])

    return scored[:max_highlights]