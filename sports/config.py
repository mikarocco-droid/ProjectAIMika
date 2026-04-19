# sports/config.py
# -*- coding: utf-8 -*-

import math

# ─────────────────────────────────────────
# CONFIG PAR SPORT
# ─────────────────────────────────────────
SPORT_CONFIG = {
    "football": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble",
                       "progressive_run", "interception", "fast_break"]
        },
        "goal_cooldown":   3750,
        "shot_cooldown":   75,
        "context_before":  12,
        "context_after":   4,
        "context_goal":    5,
        "goal_width_m":    7.32,
        "pitch_length_m":  105.0,
        "pitch_width_m":   68.0,
    },
    "mini-foot": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble", "interception"]
        },
        "goal_cooldown":   2500,
        "shot_cooldown":   50,
        "context_before":  10,
        "context_after":   3,
        "context_goal":    4,
        "goal_width_m":    3.0,
        "pitch_length_m":  40.0,
        "pitch_width_m":   20.0,
    },
    "basketball": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble",
                       "interception", "fast_break"]
        },
        "goal_cooldown":   125,
        "shot_cooldown":   30,
        "context_before":  6,
        "context_after":   3,
        "context_goal":    4,
        "goal_width_m":    0.45,
        "pitch_length_m":  28.0,
        "pitch_width_m":   15.0,
    },
    "handball": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble", "fast_break"]
        },
        "goal_cooldown":   500,
        "shot_cooldown":   50,
        "context_before":  8,
        "context_after":   3,
        "context_goal":    5,
        "goal_width_m":    3.0,
        "pitch_length_m":  40.0,
        "pitch_width_m":   20.0,
    },
    "rugby": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score"],
            "player": ["goal", "score", "interception", "fast_break"]
        },
        "goal_cooldown":   1250,
        "shot_cooldown":   75,
        "context_before":  10,
        "context_after":   5,
        "context_goal":    8,
        "goal_width_m":    5.6,
        "pitch_length_m":  100.0,
        "pitch_width_m":   70.0,
    },
    "hockey sur glace": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "interception", "fast_break"]
        },
        "goal_cooldown":   750,
        "shot_cooldown":   50,
        "context_before":  8,
        "context_after":   4,
        "context_goal":    5,
        "goal_width_m":    1.83,
        "pitch_length_m":  61.0,
        "pitch_width_m":   30.0,
    },
    "hockey sur gazon": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["goal", "score", "shot"],
            "player": ["goal", "score", "shot", "dribble"]
        },
        "goal_cooldown":   1250,
        "shot_cooldown":   75,
        "context_before":  10,
        "context_after":   4,
        "context_goal":    6,
        "goal_width_m":    3.66,
        "pitch_length_m":  91.4,
        "pitch_width_m":   55.0,
    },
    "tennis": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score", "dribble"]
        },
        "goal_cooldown":   250,
        "shot_cooldown":   25,
        "context_before":  4,
        "context_after":   3,
        "context_goal":    4,
        "goal_width_m":    8.23,
        "pitch_length_m":  23.77,
        "pitch_width_m":   8.23,
    },
    "tennis de table": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score"]
        },
        "goal_cooldown":   125,
        "shot_cooldown":   15,
        "context_before":  3,
        "context_after":   2,
        "context_goal":    3,
        "goal_width_m":    1.525,
        "pitch_length_m":  2.74,
        "pitch_width_m":   1.525,
    },
    "padel": {
        "goal_x":          1.0,
        "shot_zones":      None,
        "highlight_types": {
            "match":  ["shot", "score"],
            "player": ["shot", "score", "dribble"]
        },
        "goal_cooldown":   250,
        "shot_cooldown":   25,
        "context_before":  4,
        "context_after":   3,
        "context_goal":    4,
        "goal_width_m":    10.0,
        "pitch_length_m":  20.0,
        "pitch_width_m":   10.0,
    },
}


# ─────────────────────────────────────────
# OFFSETS CONTEXTE xG (V9.7)
# ─────────────────────────────────────────
# V9.7 — remplacé multiplicateurs × par offsets additifs
# Multiplier une probabilité est mathématiquement faux
# (0.30 × 1.30 = 0.39 ne veut rien dire statistiquement)
# À la place : on ajoute un offset au SCORE avant sigmoid
# Valeurs calibrées pour rester dans [0.03, 0.35]

PHASE_OFFSETS = {
    "set_play":       0.0,    # jeu placé, défense en place
    "open_play":      0.0,    # jeu ouvert normal
    "transition":     0.15,   # transition rapide
    "counter_attack": 0.30,   # contre-attaque → défense désorganisée
    "counter":        0.30,   # alias
    "fast_break":     0.20,   # montée rapide
    "press":         -0.15,   # tir sous pressing défensif
}

ACTION_BEFORE_OFFSETS = {
    "dribble":          0.15,  # dribble réussi → espace créé
    "progressive_run":  0.10,  # course progressive → bonne position
    "pass":             0.0,   # passe normale
    "key_pass":         0.05,  # passe clé
    "cross":           -0.25,  # centre → angle souvent fermé
    "long_pass":       -0.15,  # long ball → contrôle difficile
    "interception":     0.10,  # récupération → transition rapide
    "none":             0.0,
}

# Compatibilité arrière — anciens noms toujours accessibles
PHASE_MULTIPLIERS      = {k: 1.0 for k in PHASE_OFFSETS}
ACTION_BEFORE_MULTIPLIERS = {k: 1.0 for k in ACTION_BEFORE_OFFSETS}


# ─────────────────────────────────────────
# GETTER
# ─────────────────────────────────────────
def get_sport_config(sport):
    return SPORT_CONFIG.get(sport, SPORT_CONFIG["football"])


# ─────────────────────────────────────────
# HIGHLIGHT TYPES PAR SPORT + MODE
# ─────────────────────────────────────────
def get_highlight_types(sport, mode="match"):
    cfg = get_sport_config(sport)
    return cfg.get("highlight_types", {}).get(mode, ["goal", "shot"])


# ─────────────────────────────────────────
# XG PAR SPORT — Version avancée calibrée
#
# Références StatsBomb / Opta :
#   tir à 30m axe          → 0.01 – 0.03
#   tir à 20m axe          → 0.03 – 0.08
#   surface excentré       → 0.05 – 0.15
#   surface axe            → 0.20 – 0.40
#   face au but 10m        → 0.40 – 0.70
#   1v1 gardien            → 0.60 – 0.80
#   penalty                → 0.76
#
# Features utilisées :
#   1. distance + angle    (base géométrique)
#   2. pressure            (pression défensive — ContextEngine)
#   3. phase               (counter/transition/set_play)
#   4. action_before       (dribble/cross/passe)
#   5. sequence_length     (longueur de la séquence d'attaque)
# ─────────────────────────────────────────
def compute_xg_sport(
    x,
    y               = None,
    sport           = "football",
    frame_w         = 1920,
    frame_h         = 1080,
    pressure        = 0.0,
    phase           = "open_play",
    action_before   = "none",
    sequence_length = 1,
):
    cfg       = get_sport_config(sport)
    goal_w_m  = cfg.get("goal_width_m",   7.32)
    pitch_l_m = cfg.get("pitch_length_m", 105.0)

    # ── Sécurité ──────────────────────────
    if frame_w <= 0 or frame_h <= 0:
        return 0.01
    if y is None:
        y = frame_h / 2.0

    x = float(x)
    y = float(y)

    # ── Centre but le plus proche ─────────
    goal_y_px  = frame_h / 2.0
    goal_left  = (0.0,     goal_y_px)
    goal_right = (frame_w, goal_y_px)

    dist_left  = math.hypot(x - goal_left[0],  y - goal_left[1])
    dist_right = math.hypot(x - goal_right[0], y - goal_right[1])
    goal_cx, goal_cy = goal_left if dist_left <= dist_right else goal_right

    # ── Distance pixels → mètres ──────────
    dist_px   = math.hypot(x - goal_cx, y - goal_cy)
    px_per_m  = frame_w / max(pitch_l_m, 1e-6)
    dist_m    = dist_px / max(px_per_m, 1e-6)

    # ── Angle entre les deux poteaux ──────
    half_goal_px = (goal_w_m / pitch_l_m) * frame_w * 0.5
    post1 = (goal_cx, goal_cy - half_goal_px)
    post2 = (goal_cx, goal_cy + half_goal_px)

    v1x, v1y = post1[0] - x, post1[1] - y
    v2x, v2y = post2[0] - x, post2[1] - y

    dot   = v1x * v2x + v1y * v2y
    norm1 = math.hypot(v1x, v1y) + 1e-6
    norm2 = math.hypot(v2x, v2y) + 1e-6

    cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    angle = math.acos(cos_a)

    # ── Normalisation ─────────────────────
    distance_norm   = min(dist_m / pitch_l_m, 1.0)
    angle_norm      = angle / math.pi

    # ── Non-linéarités calibrées ──────────
    distance_effect = (1.0 - distance_norm) ** 1.3
    angle_effect    = angle_norm ** 1.7

    # ── Score de base calibré ─────────────
    # -2.8 → distribution réaliste (majorité des tirs < 0.10)
    score = -2.8 + (3.5 * distance_effect) + (1.2 * angle_effect)

    # ── Offsets contextuels (V9.7) ────────
    # Ajoutés au score AVANT sigmoid → mathématiquement correct
    phase_key    = str(phase).lower().replace("-", "_").replace(" ", "_")
    phase_off    = PHASE_OFFSETS.get(phase_key, 0.0)

    action_key   = str(action_before).lower().replace("-", "_")
    action_off   = ACTION_BEFORE_OFFSETS.get(action_key, 0.0)

    # ── Pression défensive ────────────────
    # Offset négatif si défenseur collé (max -0.5)
    pressure_off = -0.5 * max(0.0, min(1.0, float(pressure)))

    # ── Longueur de séquence ──────────────
    seq = max(1, int(sequence_length))
    if seq <= 2:
        seq_off = 0.05    # transition rapide
    elif seq <= 5:
        seq_off = 0.0     # séquence normale
    elif seq <= 10:
        seq_off = -0.05   # séquence longue
    else:
        seq_off = -0.10   # très longue séquence → défense replacée

    # ── Application des offsets au score ──
    score = score + phase_off + action_off + pressure_off + seq_off

    # ── Sigmoïde → xG ─────────────────────
    xg = 1.0 / (1.0 + math.exp(-score))

    # ── xG minimum adaptatif ──────────────
    min_xg = 0.005 if distance_norm > 0.7 else 0.01

    return round(max(min_xg, min(0.99, xg)), 3)


# ─────────────────────────────────────────
# XA — EXPECTED ASSIST
#
# xA = xG du tir suivant * qualité de la passe
# ─────────────────────────────────────────
def compute_xa(xg_of_shot, pass_quality=1.0):
    """
    xA = xG du tir × qualité de la passe
    pass_quality ∈ [0, 1]
    """
    pass_quality = max(0.0, min(1.0, float(pass_quality)))
    xa = xg_of_shot * pass_quality
    return round(max(0.0, min(0.99, xa)), 3)