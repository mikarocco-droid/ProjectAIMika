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
# CONVERSION TEMPS → MINUTE LISIBLE
# FIX — utilise "time" (secondes) en priorité,
#        fallback sur frame / fps
# ─────────────────────────────────────────
def event_to_minute(event, fps=25):
    """
    Retourne le temps de l'event en secondes.
    Priorité : champ "time" → champ "frame" / fps.
    """
    t = event.get("time")
    if t and float(t) > 0:
        return float(t)
    frame = event.get("frame")
    if frame and int(frame) > 0:
        return int(frame) / fps
    return None


def seconds_to_label(t_sec):
    """
    Convertit des secondes en label lisible.
    < 60s  → "à la 1ère minute"
    sinon  → "à la 25e minute"
    """
    if t_sec is None:
        return None
    minute = int(t_sec // 60) + 1   # +1 car la minute 0 = "1ère minute"
    if minute == 1:
        return "à la 1ère minute"
    return f"à la {minute}e minute"


def seconds_to_mmss(t_sec):
    """Retourne MM:SS pour usage technique."""
    if t_sec is None:
        return None
    t = int(t_sec)
    return f"{t // 60:02d}:{t % 60:02d}"


# ─────────────────────────────────────────
# PHASES CLÉS
# FIX — filtre les events sans temps valide
#        + priorité aux events importants
# ─────────────────────────────────────────
def extract_key_moments(events, limit=5, fps=25):
    # Priorité : buts > tirs > interceptions > dribbles
    priority = {"goal": 0, "score": 0, "shot": 1, "interception": 2, "dribble": 3}

    moments = []
    for e in events:
        if e.get("type") not in priority:
            continue
        t = event_to_minute(e, fps)
        if t is None or t <= 0:
            continue
        moments.append((priority[e["type"]], t, e))

    # Tri : d'abord par priorité type, puis chronologique
    moments.sort(key=lambda x: (x[0], x[1]))

    # Dédoublonner — pas deux events dans la même minute
    seen_minutes = set()
    result       = []
    for _, t, e in moments:
        minute = int(t // 60)
        if minute not in seen_minutes:
            seen_minutes.add(minute)
            result.append(e)
        if len(result) >= limit:
            break

    return result


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

    # ── CONTEXTE TIRS (si context_engine actif) ──────────────────────────
    shots_ctx = [e for e in events
                 if e.get("type") == "shot" and e.get("shot_context")]
    if shots_ctx:
        under_pressure = sum(1 for e in shots_ctx
                             if e.get("shot_context") == "under_pressure")
        counter        = sum(1 for e in shots_ctx
                             if e.get("shot_context") == "counter_attack")
        if counter > len(shots_ctx) * 0.4:
            story += "Les occasions ont souvent été créées en contre-attaque. "
        elif under_pressure > len(shots_ctx) * 0.5:
            story += "La majorité des tirs ont été effectués sous forte pression. "

    # ── MOMENTS CLÉS ─────────────────────
    moments = extract_key_moments(events, limit=5, fps=fps)

    if moments:
        story += "Moments clés : "
        for m in moments:
            t_sec = event_to_minute(m, fps)
            label = seconds_to_label(t_sec)
            if not label:
                continue

            if m["type"] in ["goal", "score"]:
                story += f"But {label}. "
            elif m["type"] == "shot":
                # Enrichi si shot_context disponible
                ctx = m.get("shot_context", "")
                if ctx == "counter_attack":
                    story += f"Tir en contre {label}. "
                elif ctx == "under_pressure":
                    story += f"Tir sous pression {label}. "
                else:
                    story += f"Tir dangereux {label}. "
            elif m["type"] == "interception":
                story += f"Interception clé {label}. "
            elif m["type"] == "dribble":
                story += f"Belle percée individuelle {label}. "

    return story.strip()


# ─────────────────────────────────────────
# VERSION STRUCTURÉE (API / PDF)
# ─────────────────────────────────────────
def generate_match_summary_structured(events, fps=25):
    score     = compute_score(events)
    dominance = compute_team_dominance(events)
    moments   = extract_key_moments(events, fps=fps)

    # Enrichit les moments avec le label minute lisible
    for m in moments:
        t = event_to_minute(m, fps)
        m["minute_label"] = seconds_to_label(t)
        m["time_mmss"]    = seconds_to_mmss(t)

    return {
        "score":       score,
        "dominance":   dominance,
        "key_moments": moments,
    }