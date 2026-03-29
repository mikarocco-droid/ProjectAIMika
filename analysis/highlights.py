# analysis/highlights.py
# -*- coding: utf-8 -*-

from collections import defaultdict


# ─────────────────────────────────────────
# POIDS PAR TYPE D'EVENT
# ─────────────────────────────────────────
EVENT_WEIGHTS = {
    "goal":          10,
    "score":         10,
    "shot":           6,
    "interception":   5,
    "dribble":        4,
    "long_pass":      3,
    "pass":           2,
    "possession":     1
}


def score_event(e):
    base  = EVENT_WEIGHTS.get(e.get("type", ""), 1)
    bonus = e.get("xg", 0) * 5
    return base + bonus


# ─────────────────────────────────────────
# GROUPER EN SÉQUENCES
# ─────────────────────────────────────────
def group_into_sequences(events, window_frames=90):
    if not events:
        return []

    sequences = []
    current   = [events[0]]

    for e in events[1:]:
        last_frame    = current[-1].get("frame", 0)
        current_frame = e.get("frame", 0)

        if current_frame - last_frame <= window_frames:
            current.append(e)
        else:
            sequences.append(current)
            current = [e]

    sequences.append(current)
    return sequences


def score_sequence(seq):
    return sum(score_event(e) for e in seq)


# ─────────────────────────────────────────
# EXTRACTION HIGHLIGHTS
# ─────────────────────────────────────────
def extract_highlights(
    events,
    max_highlights = 15,
    min_score      = 4,
    fps            = 30
):
    if not events:
        return []

    # Filtrer events sans frame
    events = [e for e in events if "frame" in e]
    if not events:
        return []

    events = sorted(events, key=lambda e: e["frame"])

    # Filtrer les events peu intéressants pour les highlights
    key_events = [
        e for e in events
        if e.get("type") in [
            "goal", "score", "shot",
            "interception", "dribble", "long_pass"
        ]
    ]

    # Si pas assez d'events clés — prendre tous les events
    if len(key_events) < 3:
        key_events = events

    # Grouper en séquences
    sequences = group_into_sequences(key_events, window_frames=int(fps * 3))

    # Scorer
    scored = []
    for seq in sequences:
        s = score_sequence(seq)
        if s >= min_score:
            scored.append((s, seq))

    # Trier par score
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:max_highlights]

    highlights = []
    for score, seq in scored:
        first_frame = seq[0].get("frame", 0)
        last_frame  = seq[-1].get("frame", 0)

        # Timestamp en secondes
        time_start = max(0, (first_frame / fps) - 5)
        time_end   = (last_frame  / fps) + 4

        # Type dominant
        type_counts = defaultdict(int)
        for e in seq:
            type_counts[e.get("type", "action")] += EVENT_WEIGHTS.get(
                e.get("type", ""), 1
            )

        main_type = max(type_counts, key=type_counts.get)

        highlights.append({
            "frame_start": max(0, first_frame - int(fps * 5)),
            "frame_end":   last_frame + int(fps * 4),
            "time_start":  round(time_start, 2),
            "time_end":    round(time_end,   2),
            "score":       round(score, 2),
            "events":      seq,
            "main_type":   main_type
        })

    # Retrier par ordre chronologique
    highlights.sort(key=lambda h: h["frame_start"])

    return highlights