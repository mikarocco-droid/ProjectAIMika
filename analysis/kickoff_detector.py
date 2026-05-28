# analysis/kickoff_detector.py
# -*- coding: utf-8 -*-
#
# Détection du coup d'envoi par système de score pondéré.
#
# Signaux (tous optionnels sauf symétrie) :
#   [3.0] Symétrie joueurs : ≥3 chaque côté de la ligne médiane
#   [2.0] Nb joueurs total : ≥10 détectés
#   [3.0] Ballon au centre : x∈[38-62%] y∈[33-67%]  — OPTIONNEL
#   [2.0] Ballon immobile  : speed < 8px/s            — OPTIONNEL
#   [2.0] Pas d'action récente : aucun shot/goal dans les 60s précédentes
#   [1.5] Première apparition des deux équipes
#   [1.0] Arbitre visible  : joueur couleur neutre seul
#
# Seuil : 7.0 / 14.5 → kickoff validé même sans ballon visible.
#
# API publique (inchangée) :
#   find_kickoff_offset(events, video_duration_s, frames_data=None, fps=25)
#   apply_kickoff_offset(events, offset, fps=25)
#   apply_kickoff_offset_frames(frames_data, offset, fps=25)
#   reset_pre_kickoff_state(jersey_map, offset, fps=25)
#   find_match_end(frames_data, fps, ...)
#   apply_match_end(events, frames_data, match_end_s, fps=25)

import math

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────
_SCORE_THRESHOLD   = 7.0    # score minimum pour valider un kickoff
_MIN_T_RATIO       = 0.03   # chercher après 3% de la durée vidéo (évite faux positifs début)
_MIN_T_ABS         = 30.0   # et au moins 30s dans la vidéo
_CONSECUTIVE_FRAMES = 3     # nombre de frames consécutives validant le score
_NO_ACTION_WINDOW  = 60.0   # secondes avant le kickoff sans shot/goal

# Poids des signaux
_W_SYMMETRY    = 3.0
_W_N_PLAYERS   = 2.0
_W_BALL_CENTER = 3.0
_W_BALL_STILL  = 2.0
_W_NO_ACTION   = 2.0
_W_FIRST_TEAMS = 1.5
_W_REFEREE     = 1.0


# ─────────────────────────────────────────
# SCORE D'UNE FRAME
# ─────────────────────────────────────────
def _score_frame(fd, fps, action_times, first_two_teams_seen, frame_w, frame_h):
    """
    Calcule le score kickoff d'une frame.
    Retourne (score, détails_dict).
    """
    score   = 0.0
    details = {}

    players  = fd.get("players") or []
    ball     = fd.get("ball") or {}
    t        = fd.get("frame", 0) / max(fps, 1)

    # ── 1. Symétrie joueurs ───────────────────────────────────────────────────
    mid_x    = frame_w * 0.50
    left_ps  = [p for p in players if _player_cx(p) < mid_x]
    right_ps = [p for p in players if _player_cx(p) >= mid_x]
    has_sym  = len(left_ps) >= 3 and len(right_ps) >= 3
    if has_sym:
        score += _W_SYMMETRY
    details["symmetry"] = (len(left_ps), len(right_ps))

    # ── 2. Nombre total de joueurs ────────────────────────────────────────────
    if len(players) >= 10:
        score += _W_N_PLAYERS
    details["n_players"] = len(players)

    # ── 3. Ballon au centre ───────────────────────────────────────────────────
    bx, by, bspeed = _ball_pos(ball, frame_w, frame_h)
    ball_at_center = False
    if bx is not None and by is not None:
        ball_at_center = (
            frame_w * 0.38 < bx < frame_w * 0.62 and
            frame_h * 0.33 < by < frame_h * 0.67
        )
        if ball_at_center:
            score += _W_BALL_CENTER
    details["ball_center"] = ball_at_center

    # ── 4. Ballon immobile ────────────────────────────────────────────────────
    ball_still = bspeed is not None and bspeed < 8.0
    if ball_still and ball_at_center:
        score += _W_BALL_STILL
    details["ball_still"] = ball_still

    # ── 5. Pas d'action récente (shot/goal) ───────────────────────────────────
    no_recent = not any(abs(t - at) < _NO_ACTION_WINDOW for at in action_times)
    if no_recent:
        score += _W_NO_ACTION
    details["no_action"] = no_recent

    # ── 6. Première apparition des deux équipes ───────────────────────────────
    if not first_two_teams_seen:
        teams_here = set(p.get("team") for p in players if p.get("team") is not None)
        if len(teams_here) >= 2:
            score += _W_FIRST_TEAMS
            first_two_teams_seen = True
    details["first_teams"] = first_two_teams_seen

    # ── 7. Arbitre visible (joueur couleur neutre seul) ───────────────────────
    referee_seen = any(
        str(p.get("team", "")).lower() in ("ref", "referee", "arbitre", "gk", "-1")
        or p.get("is_referee", False)
        for p in players
    )
    if referee_seen:
        score += _W_REFEREE
    details["referee"] = referee_seen

    return score, details, first_two_teams_seen


def _player_cx(p):
    bbox = p.get("bbox") or []
    if len(bbox) >= 3:
        return (bbox[0] + bbox[2]) / 2
    return p.get("x", 0) or 0


def _ball_pos(ball, frame_w, frame_h):
    """Retourne (bx, by, speed) depuis un dict ball."""
    if not ball:
        return None, None, None
    center = ball.get("center")
    if center and len(center) >= 2 and center[0] is not None:
        bx, by = center[0], center[1]
    else:
        bx = ball.get("x")
        by = ball.get("y")
    speed = ball.get("speed")
    return bx, by, speed


# ─────────────────────────────────────────
# FIND KICKOFF OFFSET — API publique
# ─────────────────────────────────────────
def find_kickoff_offset(events, video_duration_s, frames_data=None, fps=25):
    """
    Détecte le coup d'envoi et retourne (offset_s, confidence).

    Passe 1 — events type=kickoff (méthode legacy)
    Passe 2 — score pondéré sur frames_data (nouvelle méthode robuste)

    Le premier kickoff trouvé est retourné.
    offset_s = 0 si non détecté.
    """
    # ── Passe 1 : events type=kickoff ─────────────────────────────────────────
    kickoff_events = sorted(
        [e for e in (events or []) if e.get("type") == "kickoff"],
        key=lambda e: e.get("time", 0)
    )
    if kickoff_events:
        first = kickoff_events[0]
        t     = float(first.get("time", 0))
        conf  = float(first.get("confidence", first.get("conf", 0.80)))
        if t > 0:
            print(f"  [KICKOFF] Détecté via events à t={t:.1f}s (conf={conf:.2f})")
            return t, conf

    # ── Passe 2 : score pondéré sur frames_data ───────────────────────────────
    if not frames_data:
        return 0.0, 0.0

    min_t = max(_MIN_T_ABS, video_duration_s * _MIN_T_RATIO)

    # Timestamps de shots/goals pour le signal "pas d'action récente"
    action_times = [
        float(e.get("time", 0)) for e in (events or [])
        if e.get("type") in ("shot", "goal", "score")
    ]

    # Résolution réelle
    frame_w = int(frames_data[0].get("frame_w") or 1920)
    frame_h = int(frames_data[0].get("frame_h") or 1080)

    consecutive        = 0
    first_two_teams    = False
    best_t             = None
    best_score         = 0.0
    candidate_start_t  = None

    for fd in frames_data:
        t = fd.get("frame", 0) / max(fps, 1)
        if t < min_t:
            continue

        score, details, first_two_teams = _score_frame(
            fd, fps, action_times, first_two_teams, frame_w, frame_h
        )

        if score >= _SCORE_THRESHOLD:
            consecutive += 1
            if consecutive == 1:
                candidate_start_t = t
            if consecutive >= _CONSECUTIVE_FRAMES:
                # Kickoff validé — prendre le début de la séquence
                best_t     = candidate_start_t
                best_score = score
                conf       = min(0.95, 0.60 + (score - _SCORE_THRESHOLD) * 0.05)
                print(f"  [KICKOFF PHYS] Score={score:.1f}/{_SCORE_THRESHOLD} "
                      f"→ offset={best_t:.1f}s conf={conf:.2f} "
                      f"(sym={details['symmetry']} "
                      f"n={details['n_players']} "
                      f"ball={'✓' if details['ball_center'] else '✗'})")
                return best_t, conf
        else:
            consecutive       = 0
            candidate_start_t = None

    print(f"  [KICKOFF] Aucun coup d'envoi détecté (score max insuffisant)")
    return 0.0, 0.0


# ─────────────────────────────────────────
# APPLY KICKOFF OFFSET
# ─────────────────────────────────────────
def apply_kickoff_offset(events, offset, fps=25):
    """
    Soustrait l'offset de tous les timestamps events.
    Supprime les events antérieurs au coup d'envoi.
    Retourne (events_corrigés, n_supprimés).
    """
    if not offset or offset <= 0:
        return events, 0

    corrected = []
    n_removed = 0
    for e in events:
        t_orig = float(e.get("time", 0))
        t_new  = t_orig - offset
        if t_new < -2.0:          # marge de 2s pour les events juste avant le sifflet
            n_removed += 1
            continue
        e = dict(e)
        e["time"]  = max(0.0, round(t_new, 3))
        if "frame" in e and e["frame"] is not None:
            e["frame"] = max(0, int(e["frame"] - offset * fps))
        corrected.append(e)

    return corrected, n_removed


def apply_kickoff_offset_frames(frames_data, offset, fps=25):
    """
    Soustrait l'offset des frames_data et supprime les frames avant le KO.
    """
    if not offset or offset <= 0:
        return frames_data

    offset_frames = int(offset * fps)
    result = []
    for fd in frames_data:
        f = fd.get("frame", 0)
        if f < offset_frames - int(fps * 2):    # marge 2s
            continue
        fd = dict(fd)
        fd["frame"] = max(0, f - offset_frames)
        result.append(fd)
    return result


# ─────────────────────────────────────────
# RESET PRE-KICKOFF STATE
# ─────────────────────────────────────────
def reset_pre_kickoff_state(jersey_map, offset, fps=25):
    """
    Nettoie le jersey_map des joueurs vus uniquement à l'échauffement.
    (Actuellement conserve tout — le mapping est utile même s'il vient de l'échauffement)
    """
    return jersey_map


# ─────────────────────────────────────────
# FIND MATCH END
# ─────────────────────────────────────────
def find_match_end(
    frames_data,
    fps,
    team_colors          = None,
    silence_threshold_s  = 1500.0,   # 25 min sans vrai jeu = fin du match
    min_match_duration_s = 2700.0,   # chercher fin seulement après 45 min
):
    """
    Détecte la fin du match dans les frames_data.

    Critère "vrai jeu" : ≥4 joueurs avec couleur d'équipe assignée.
    Les gamins sans vareuse après le match = ignorés (team=None).

    Retourne le timestamp de fin (s) ou None si non détecté.
    """
    if not frames_data:
        return None

    last_real_game_t = None
    last_checked_t   = 0.0

    for fd in frames_data:
        t       = fd.get("frame", 0) / max(fps, 1)
        players = fd.get("players") or []

        # Joueurs avec équipe assignée (couleur connue)
        real_players = [
            p for p in players
            if p.get("team") is not None and p.get("team") != -1
        ]

        if len(real_players) >= 4:
            last_real_game_t = t

        last_checked_t = t

    if last_real_game_t is None:
        return None

    # Vérifier que le match a duré assez longtemps
    if last_real_game_t < min_match_duration_s:
        print(f"  [MATCH_END] last_real_game={last_real_game_t:.0f}s < "
              f"min={min_match_duration_s:.0f}s → ignoré")
        return None

    # Vérifier qu'il y a un silence significatif après
    silence = last_checked_t - last_real_game_t
    if silence < silence_threshold_s:
        print(f"  [MATCH_END] Pas de fin de match détectée → vidéo utilisée entièrement")
        return None

    print(f"  [MATCH_END] Fin détectée à t={last_real_game_t:.0f}s "
          f"(silence={silence:.0f}s > {silence_threshold_s:.0f}s)")
    return last_real_game_t


# ─────────────────────────────────────────
# APPLY MATCH END
# ─────────────────────────────────────────
def apply_match_end(events, frames_data, match_end_s, fps=25):
    """
    Coupe les events et frames_data après la fin du match.
    Garde +60s de marge (célébrations, coup de sifflet final).
    """
    cutoff = match_end_s + 60.0

    events_cut = [
        e for e in events
        if float(e.get("time", 0)) <= cutoff
    ]
    frames_cut = [
        fd for fd in frames_data
        if fd.get("frame", 0) / max(fps, 1) <= cutoff
    ]

    n_ev = len(events) - len(events_cut)
    n_fr = len(frames_data) - len(frames_cut)
    if n_ev or n_fr:
        print(f"  [MATCH_END] Coupe à t={cutoff:.0f}s : "
              f"{n_ev} events + {n_fr} frames supprimés")

    return events_cut, frames_cut