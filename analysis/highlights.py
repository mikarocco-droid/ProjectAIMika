# analysis/highlights.py

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


# ─────────────────────────────────────────
# SCORE D'UN EVENT
# ─────────────────────────────────────────
def score_event(e):
    base  = EVENT_WEIGHTS.get(e["type"], 1)
    bonus = e.get("xg", 0) * 5  # tir à haute xG = plus intéressant
    return base + bonus


# ─────────────────────────────────────────
# GROUPER LES EVENTS EN SÉQUENCES
# (évite 10 highlights sur la même action)
# ─────────────────────────────────────────
def group_into_sequences(events, window_frames=90):
    """
    Regroupe les events proches en séquences.
    window_frames : nb de frames max entre deux events
                    d'une même séquence (90 ≈ 3s à 30fps)
    """
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


# ─────────────────────────────────────────
# SCORE D'UNE SÉQUENCE
# ─────────────────────────────────────────
def score_sequence(seq):
    """
    Score total d'une séquence = somme des scores
    des events qui la composent.
    """
    return sum(score_event(e) for e in seq)


# ─────────────────────────────────────────
# EXTRACTION DES HIGHLIGHTS
# ─────────────────────────────────────────
def extract_highlights(events, max_highlights=15, min_score=4, fps=30):
    """
    À partir de la liste complète des events du match,
    retourne les moments les plus intéressants.

    Paramètres :
        events         : liste de tous les events (avec champ "frame")
        max_highlights : nombre max de highlights à retourner
        min_score      : score minimum pour qu'une séquence soit retenue
        fps            : framerate de la vidéo

    Retourne :
        liste de dicts {
            "frame_start", "frame_end",
            "time_start",  "time_end",
            "score",       "events",
            "main_type"
        }
    """

    if not events:
        return []

    # Filtrer les events sans frame
    events = [e for e in events if "frame" in e]
    events = sorted(events, key=lambda e: e["frame"])

    # Regrouper en séquences
    sequences = group_into_sequences(events, window_frames=90)

    # Scorer chaque séquence
    scored = []
    for seq in sequences:
        s = score_sequence(seq)
        if s >= min_score:
            scored.append((s, seq))

    # Trier par score décroissant
    scored.sort(key=lambda x: x[0], reverse=True)

    # Garder les meilleurs
    scored = scored[:max_highlights]

    # Construire les highlights
    highlights = []
    for score, seq in scored:

        first_frame = seq[0]["frame"]
        last_frame  = seq[-1]["frame"]

        # Contexte : 5s avant, 4s après
        frame_start = max(0, first_frame - 5 * fps)
        frame_end   =        last_frame  + 4 * fps

        # Type dominant de la séquence
        type_counts = defaultdict(int)
        for e in seq:
            type_counts[e["type"]] += EVENT_WEIGHTS.get(e["type"], 1)

        main_type = max(type_counts, key=type_counts.get)

        highlights.append({
            "frame_start": frame_start,
            "frame_end":   frame_end,
            "time_start":  round(frame_start / fps, 2),
            "time_end":    round(frame_end   / fps, 2),
            "score":       round(score, 2),
            "events":      seq,
            "main_type":   main_type
        })

    # Retrier par ordre chronologique pour le montage
    highlights.sort(key=lambda h: h["frame_start"])

    return highlights