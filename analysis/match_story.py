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
# CONVERSION FRAME → MINUTE
# FIX — utilise le fps réel stocké dans l'event si disponible
# ─────────────────────────────────────────
def frame_to_minute(frame, fps=25):
    """Convertit un numéro de frame en minute de match."""
    if not frame or frame <= 0:
        return None
    return int(frame / fps / 60)


def frame_to_mmss(frame, fps=25):
    """Convertit un numéro de frame en MM:SS."""
    if not frame or frame <= 0:
        return None
    total_sec = int(frame / fps)
    return f"{total_sec // 60:02d}:{total_sec % 60:02d}"


# ─────────────────────────────────────────
# PHASES CLÉS
# ─────────────────────────────────────────
def extract_key_moments(events, limit=5, fps=25):
    moments = []
    for e in events:
        if e.get("type") in ["goal", "shot", "interception", "dribble"]:
            # FIX — ne garder que les events avec un frame valide
            if e.get("frame") and e.get("frame") > 0:
                moments.append(e)

    # Trier par frame pour avoir les moments dans l'ordre chronologique
    moments.sort(key=lambda e: e.get("frame", 0))
    return moments[:limit]


# ─────────────────────────────────────────
# GÉNÉRATION TEXTE
# ─────────────────────────────────────────
def generate_match_story(events, stats=None, fps=25):

    if not events:
        return "Match sans données exploitables."

    score     = compute_score(events)
    dominance = compute_team_dominance(events)

    team0 = score.get(0, 0)
    team1 = score.get(1, 0)

    # ── INTRO ────────────────────────────
    story = f"Le match se termine sur un score de {team0}-{team1}. "

    # ── DOMINATION ───────────────────────
    d0 = dominance.get(0, 0)
    d1 = dominance.get(1, 0)
    if d0 > d1 * 1.2:
        story += "L'équipe 1 a globalement dominé la rencontre. "
    elif d1 > d0 * 1.2:
        story += "L'équipe 2 a pris le dessus dans le jeu. "
    else:
        story += "Le match a été globalement équilibré. "

    # ── STYLE ────────────────────────────
    passes = sum(1 for e in events if e.get("type") == "pass")
    shots  = sum(1 for e in events if e.get("type") == "shot")

    if passes > shots * 3:
        story += "Le jeu s'est appuyé sur la possession et la construction. "
    elif shots > passes * 0.3:
        story += "Le match a été très direct avec beaucoup d'occasions. "
    else:
        story += "Le rythme a été intermédiaire entre construction et jeu direct. "

    # ── MOMENTS CLÉS ─────────────────────
    moments = extract_key_moments(events, fps=fps)

    if moments:
        story += "Moments clés : "
        for m in moments:
            mmss = frame_to_mmss(m.get("frame"), fps)
            if not mmss:
                continue

            if m["type"] in ["goal", "score"]:
                story += f"But à la {mmss}. "
            elif m["type"] == "shot":
                story += f"Tir dangereux à la {mmss}. "
            elif m["type"] == "interception":
                story += f"Interception clé à la {mmss}. "
            elif m["type"] == "dribble":
                story += f"Belle percée individuelle à la {mmss}. "

    return story


# ─────────────────────────────────────────
# VERSION STRUCTURÉE (API / PDF)
# ─────────────────────────────────────────
def generate_match_summary_structured(events, fps=25):
    score     = compute_score(events)
    dominance = compute_team_dominance(events)
    return {
        "score":        score,
        "dominance":    dominance,
        "key_moments":  extract_key_moments(events, fps=fps)
    }