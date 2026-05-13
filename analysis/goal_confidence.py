# analysis/goal_confidence.py
# -*- coding: utf-8 -*-
"""
Confiance fusionnée par but — ScoutIA V9.8

Agrège 4 sources de signal indépendantes en un score unique [0.0, 1.0] :
  - Score physique  (posthoc ball tracker)
  - Score Gemini    (vision LLM)
  - Qualité du tir  (xG + on_target + contexte)
  - Stabilité tracking  (confidence tracker + fusion ReID)

Usage :
    from analysis.goal_confidence import compute_goal_confidence, format_confidence_label
    conf = compute_goal_confidence(event)
    label = format_confidence_label(conf)  # "✅ Très probable (0.91)"
"""


# ─────────────────────────────────────────
# POIDS PAR SOURCE
# ─────────────────────────────────────────
_W_PHYSICAL  = 0.35
_W_GEMINI    = 0.35
_W_SHOT      = 0.20
_W_TRACKING  = 0.10


def _score_physical(event):
    """
    Score physique normalisé [0, 1] depuis les champs posthoc.
    Sources : physical_score, score, stuck, rebound, cross_line, ball_in_goal
    """
    raw   = float(event.get("physical_score", 0)
                  or event.get("score", 0) or 0)
    stuck = int(event.get("stuck", 0) or 0)
    reb   = bool(event.get("rebound", False))
    cross = bool(event.get("cross_line", False))
    b_gol = bool(event.get("ball_appears_in_goal", False))

    # Normaliser le score brut (typiquement 5-10 → 0.5-1.0)
    norm = min(raw / 10.0, 1.0)

    # Bonus stuck : ballon immobile dans le but = signal fort
    stuck_bonus = min(stuck / 15.0, 0.3)

    # Bonus visuels
    visual_bonus = 0.0
    if cross:  visual_bonus += 0.3
    if b_gol:  visual_bonus += 0.2
    if reb:    visual_bonus += 0.05

    return min(norm + stuck_bonus * 0.3 + visual_bonus * 0.3, 1.0)


def _score_gemini(event):
    """
    Score Gemini normalisé [0, 1].
    Sources : gemini_validated, gemini_conf, gemini_type, _goal_votes, goal_score
    """
    if not event.get("gemini_validated", False):
        # Auto-accept = signal physique fort sans Gemini
        if event.get("_auto") == "accept":
            return 0.85
        return 0.0

    conf  = float(event.get("gemini_conf", 0)
                  or event.get("confiance", 0) or 0)
    gtype = event.get("gemini_type", "")
    votes = int(event.get("_goal_votes", 0)
                or event.get("goal_votes", 0) or 0)

    if gtype == "goal":
        base = conf
    elif gtype == "shot":
        base = conf * 0.4   # tir ≠ but
    else:
        base = 0.0

    # Bonus multi-offset : plusieurs offsets confirment = plus fiable
    vote_bonus = min(votes * 0.1, 0.2)

    return min(base + vote_bonus, 1.0)


def _score_shot_quality(event):
    """
    Qualité du tir source [0, 1].
    Sources : xg, on_target, shot_context, fast_shot, gemini_shot_confirmed
    """
    xg         = float(event.get("xg", 0) or 0)
    on_target  = bool(event.get("on_target", False))
    fast       = bool(event.get("fast_shot", False))
    confirmed  = bool(event.get("gemini_shot_confirmed", False))
    ctx        = event.get("shot_context", "")

    # xG normalisé (max réaliste ~0.60)
    xg_score = min(xg / 0.60, 1.0)

    bonus = 0.0
    if on_target:  bonus += 0.15
    if fast:       bonus += 0.10
    if confirmed:  bonus += 0.15
    if ctx == "counter_attack":   bonus += 0.05
    if ctx == "open_play":        bonus += 0.03

    # Si c'est un but shot_to_goal_gemini sans tir source, xG=0.5 par défaut
    source = event.get("source", "") or event.get("detected_from", "")
    if "shot_to_goal" in str(source) and xg == 0:
        xg_score = 0.5

    return min(xg_score + bonus, 1.0)


def _score_tracking(event):
    """
    Stabilité du tracking [0, 1].
    Sources : confidence (tracker), gemini_valid, source fiabilité
    """
    tracker_conf = float(event.get("confidence", 0.5)
                         or event.get("tracker_conf", 0.5) or 0.5)
    source = str(event.get("source", "") or event.get("detected_from", ""))

    # Pénalité selon la source
    source_penalty = 0.0
    if "posthoc" in source:      source_penalty = 0.0   # posthoc = fiable
    elif "shot_to_goal" in source: source_penalty = 0.05  # shot→goal = très fiable
    elif "events" in source:     source_penalty = 0.10  # events bruts = moins fiable

    return max(tracker_conf - source_penalty, 0.0)


# ─────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────
def compute_goal_confidence(event):
    """
    Calcule la confiance fusionnée d'un but [0.0, 1.0].

    Formule :
        confidence = 0.35 * physical
                   + 0.35 * gemini
                   + 0.20 * shot_quality
                   + 0.10 * tracking_stability

    Retourne un dict avec le score global et les composantes détaillées.
    """
    p = _score_physical(event)
    g = _score_gemini(event)
    s = _score_shot_quality(event)
    t = _score_tracking(event)

    fused = (
        _W_PHYSICAL * p
        + _W_GEMINI  * g
        + _W_SHOT    * s
        + _W_TRACKING * t
    )
    fused = round(min(max(fused, 0.0), 1.0), 3)

    return {
        "confidence":  fused,
        "physical":    round(p, 3),
        "gemini":      round(g, 3),
        "shot":        round(s, 3),
        "tracking":    round(t, 3),
    }


def format_confidence_label(conf_dict):
    """
    Retourne un label lisible pour affichage/log.
    Ex : "✅ Très probable (0.91)"
    """
    v = conf_dict["confidence"] if isinstance(conf_dict, dict) else float(conf_dict)
    if v >= 0.85:
        return f"✅ Très probable ({v:.2f})"
    if v >= 0.65:
        return f"🟡 Probable ({v:.2f})"
    if v >= 0.45:
        return f"⚠️  Incertain ({v:.2f})"
    return f"❌ Improbable ({v:.2f})"


def enrich_goal_event(event):
    """
    Ajoute les champs de confiance fusionnée directement sur l'event.
    Modifie l'event in-place et le retourne.
    """
    conf = compute_goal_confidence(event)
    event["goal_confidence"]          = conf["confidence"]
    event["goal_confidence_physical"] = conf["physical"]
    event["goal_confidence_gemini"]   = conf["gemini"]
    event["goal_confidence_shot"]     = conf["shot"]
    event["goal_confidence_tracking"] = conf["tracking"]
    return event
