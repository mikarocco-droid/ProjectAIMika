# analysis/match_story.py
# -*- coding: utf-8 -*-

from analysis.intelligence import compute_team_dominance


# ─────────────────────────────────────────
# SCORE MATCH
# ─────────────────────────────────────────
def compute_score(events):

    score = {0: 0, 1: 0}

    for e in events:
        if e.get("type") in ["goal", "score"]:
            team = e.get("team")
            if team in score:
                score[team] += 1

    return score


# ─────────────────────────────────────────
# PHASES CLÉS
# ─────────────────────────────────────────
def extract_key_moments(events, limit=5):

    moments = []

    for e in events:
        if e["type"] in ["goal", "shot", "interception", "dribble"]:
            moments.append(e)

    return moments[:limit]


# ─────────────────────────────────────────
# GÉNÉRATION TEXTE
# ─────────────────────────────────────────
def generate_match_story(events, stats=None):

    if not events:
        return "Match sans données exploitables."

    score = compute_score(events)
    dominance = compute_team_dominance(events)

    team0 = score.get(0, 0)
    team1 = score.get(1, 0)

    # ── INTRO
    story = f"Le match se termine sur un score de {team0}-{team1}. "

    # ── DOMINATION
    if dominance.get(0, 0) > dominance.get(1, 0):
        story += "L'équipe 1 a globalement dominé la rencontre. "
    elif dominance.get(1, 0) > dominance.get(0, 0):
        story += "L'équipe 2 a pris le dessus dans le jeu. "
    else:
        story += "Le match a été globalement équilibré. "

    # ── STYLE
    passes = sum(1 for e in events if e["type"] == "pass")
    shots  = sum(1 for e in events if e["type"] == "shot")

    if passes > shots * 2:
        story += "Le jeu s'est appuyé sur la possession et la construction. "
    elif shots > passes * 0.3:
        story += "Le match a été très direct avec beaucoup d'occasions. "
    else:
        story += "Le rythme a été intermédiaire entre construction et jeu direct. "

    # ── MOMENTS CLÉS
    moments = extract_key_moments(events)

    if moments:
        story += "Moments clés : "

        for m in moments:
            minute = int(m.get("frame", 0) / 25 / 60)

            if m["type"] in ["goal", "score"]:
                story += f"But à la {minute}e minute. "

            elif m["type"] == "shot":
                story += f"Tir dangereux à la {minute}e. "

            elif m["type"] == "interception":
                story += f"Interception clé à la {minute}e. "

            elif m["type"] == "dribble":
                story += f"Belle percée individuelle à la {minute}e. "

    return story


# ─────────────────────────────────────────
# VERSION STRUCTURÉE (API / PDF)
# ─────────────────────────────────────────
def generate_match_summary_structured(events):

    score = compute_score(events)
    dominance = compute_team_dominance(events)

    return {
        "score": score,
        "dominance": dominance,
        "key_moments": extract_key_moments(events)
    }