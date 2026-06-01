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
_MIN_T_ABS         = 60.0   # chercher après 60s minimum (le KO ne peut pas être dans la 1ère minute)
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
def _score_frame(fd, fps, action_times, first_two_teams_seen, frame_w, frame_h,
                 ball_speed_px=None):
    """
    Calcule le score kickoff d'une frame.

    Signal principal : SÉPARATION DES ÉQUIPES (contrainte réglementaire football).
    Score continu — pas de seuil binaire.

    Retourne (score, détails_dict, first_two_teams_seen).
    """
    score   = 0.0
    details = {}

    players = fd.get("players") or []
    ball    = fd.get("ball") or {}
    t       = fd.get("frame", 0) / max(fps, 1)
    mid_x   = frame_w * 0.50

    # ── 1. SÉPARATION DES ÉQUIPES — score continu [max 5.0] ─────────────────
    # La clé d'équipe dans frames_data peut être 'team', 'team_id', 'color',
    # 'team_color', ou un index couleur. On essaie toutes les variantes.
    def _get_team(p):
        """Retourne un identifiant d'équipe depuis un dict joueur."""
        for key in ("team", "team_id", "team_idx"):
            v = p.get(key)
            if v is not None:
                return v
        # Fallback : utiliser la couleur dominante comme proxy d'équipe
        # La couleur est stockée sous 'color', 'team_color', ou 'jersey_color'
        for key in ("color", "team_color", "jersey_color"):
            v = p.get(key)
            if v is not None:
                # Convertir en tuple hashable si c'est une liste
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    return (int(v[0]) // 30, int(v[1]) // 30, int(v[2]) // 30)
                return v
        return None

    team_players = {}
    n_with_team_count = 0
    for p in players:
        t_id = _get_team(p)
        if t_id is None:
            continue
        n_with_team_count += 1
        cx = _player_cx(p)
        team_players.setdefault(t_id, []).append(cx)

    separation = 0.0
    if len(team_players) >= 2:
        # Prendre les 2 équipes avec le plus de joueurs
        sorted_teams = sorted(team_players.keys(),
                              key=lambda k: len(team_players[k]), reverse=True)
        teams = sorted_teams[:2]
        n0 = len(team_players[teams[0]])
        n1 = len(team_players[teams[1]])
        if n0 >= 3 and n1 >= 3:
            t0_left  = sum(1 for cx in team_players[teams[0]] if cx < mid_x) / n0
            t0_right = 1.0 - t0_left
            t1_left  = sum(1 for cx in team_players[teams[1]] if cx < mid_x) / n1
            t1_right = 1.0 - t1_left
            sep_a = (t0_left + t1_right) / 2.0
            sep_b = (t0_right + t1_left) / 2.0
            separation = max(sep_a, sep_b)

    score += 5.0 * separation
    details["team_separation"] = round(separation, 3)

    # ── 2. Nombre de joueurs avec équipe assignée [max 2.0] ──────────────────
    n_with_team = n_with_team_count
    n_score = min(1.0, (n_with_team - 6) / 4.0) * 2.0 if n_with_team >= 6 else 0.0
    score += max(0.0, n_score)
    details["n_players"]   = len(players)
    details["n_with_team"] = n_with_team

    # ── 3. Ballon au centre [max 2.0] ─────────────────────────────────────────
    bx, by, bspeed = _ball_pos(ball, frame_w, frame_h)
    ball_at_center = False
    if bx is not None and by is not None:
        ball_at_center = (
            frame_w * 0.40 < bx < frame_w * 0.60 and
            frame_h * 0.35 < by < frame_h * 0.60
        )
        if ball_at_center:
            score += 2.0
    details["ball_center"] = ball_at_center

    # ── 4. 1-2 joueurs près du ballon [max 2.0] ───────────────────────────────
    # Au KO : 1 ou 2 tireurs près du ballon, personne d'autre
    # Échauffement / discussions : plusieurs joueurs groupés au centre
    if bx is not None and by is not None:
        near_ball_dist = frame_w * 0.08   # ~8% de la largeur
        players_near = [
            p for p in players
            if abs(_player_cx(p) - bx) < near_ball_dist
            and abs(_player_cy(p) - by) < near_ball_dist
        ]
        n_near = len(players_near)
        if 1 <= n_near <= 2:
            score += 2.0
        elif n_near == 3:
            score += 0.5
        details["players_near_ball"] = n_near
    else:
        details["players_near_ball"] = 0

    # ── 5. Peu de joueurs dans le rond central [max 1.0] ─────────────────────
    # Hors des 2 tireurs, personne ne doit être dans le rond
    cx_center = frame_w * 0.50
    cy_center = frame_h * 0.50
    r_x = frame_w * 0.09
    r_y = frame_h * 0.11
    players_in_circle = [
        p for p in players
        if abs(_player_cx(p) - cx_center) < r_x
        and abs(_player_cy(p) - cy_center) < r_y
    ]
    if 0 < len(players_in_circle) <= 3:
        score += 1.0
    details["players_in_circle"] = len(players_in_circle)

    # ── 6. Spread horizontal [bonus 0.5] ──────────────────────────────────────
    if players:
        xs = [_player_cx(p) for p in players]
        spread_x = (max(xs) - min(xs)) / max(frame_w, 1)
        if spread_x >= 0.60:
            score += 0.5
    else:
        spread_x = 0.0
    details["spread_x"] = spread_x

    # ── 7. Symétrie globale [bonus 0.5] ───────────────────────────────────────
    left_ps   = [p for p in players if _player_cx(p) < mid_x]
    right_ps  = [p for p in players if _player_cx(p) >= mid_x]
    n_left, n_right = len(left_ps), len(right_ps)
    sym_ratio = min(n_left, n_right) / max(n_left, n_right, 1)
    if sym_ratio >= 0.45 and n_left >= 3 and n_right >= 3:
        score += 0.5
    details["symmetry"]  = (n_left, n_right)
    details["sym_ratio"] = sym_ratio

    # ── 8. Première apparition des deux équipes [bonus 0.5] ──────────────────
    if not first_two_teams_seen:
        teams_here = set(p.get("team") for p in players if p.get("team") is not None)
        if len(teams_here) >= 2:
            score += 0.5
            first_two_teams_seen = True
    details["first_teams"] = first_two_teams_seen

    return score, details, first_two_teams_seen


def _player_cy(p):
    bbox = p.get("bbox") or []
    if len(bbox) >= 4:
        return (bbox[1] + bbox[3]) / 2
    return p.get("y", 0) or 0


def _player_cx(p):
    bbox = p.get("bbox") or []
    if len(bbox) >= 3:
        return (bbox[0] + bbox[2]) / 2
    return p.get("x", 0) or 0


def _ball_pos(ball, frame_w, frame_h):
    """Retourne (bx, by, speed) depuis un dict ball de frames_data.
    Note : frames_data ne stocke pas la vitesse — retourne None pour speed.
    La vitesse est calculée dans find_kickoff_offset via positions consécutives.
    """
    if not ball:
        return None, None, None
    center = ball.get("center")
    if center and len(center) >= 2 and center[0] is not None:
        bx, by = center[0], center[1]
    else:
        bx = ball.get("x")
        by = ball.get("y")
    # Pas de speed dans frames_data — sera injecté depuis l'extérieur
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

    # DEBUG : afficher la structure réelle du dict ball et player
    _debug_shown = False
    for _fd in frames_data[:50]:
        _b = _fd.get("ball")
        _ps = _fd.get("players") or []
        if _b and _ps and not _debug_shown:
            print(f"  [KICKOFF DEBUG] Structure ball   : {list(_b.keys())}")
            _p0 = _ps[0]
            print(f"  [KICKOFF DEBUG] Structure player : {list(_p0.keys())[:12]}")
            _team_keys = [k for k in _p0.keys()
                          if any(w in k.lower() for w in ('team','color','jersey','idx','class'))]
            print(f"  [KICKOFF DEBUG] Clés équipe      : {_team_keys}")
            if _team_keys:
                print(f"  [KICKOFF DEBUG] Valeurs équipe   : { {k: _p0.get(k) for k in _team_keys[:3]} }")
            _debug_shown = True
            break

    # Fenêtre de recherche pré-match
    # Borne haute : min(40% durée, 900s) ET premier event de jeu - 5s
    # Le KO ne peut pas être APRÈS le premier tir/goal détecté
    max_search_t = min(max(video_duration_s * 0.40, 360.0), 900.0)

    # Affiner avec le premier event de jeu FIABLE si disponible
    # On ignore events_standard (xg auto=0.5 même sur FP échauffement)
    # On garde seulement les sources physiques confirmées :
    #   - goal_posthoc : ballon détecté dans la cage physiquement
    #   - terminal_goal / terminal_goalkeeper_save : signal visuel fort
    #   - shot_to_goal_gemini : Gemini a confirmé visuellement
    RELIABLE_SOURCES = {"goal_posthoc", "terminal_goal",
                        "terminal_goalkeeper_save", "shot_to_goal_gemini",
                        "ball_appears_in_goal"}
    first_game_event_t = None
    for e in (events or []):
        if e.get("type") not in ("shot", "goal", "score"):
            continue
        src = e.get("detected_from", e.get("source", ""))
        # Ignorer events_standard et posthoc simples — sources trop bruyantes
        if src not in RELIABLE_SOURCES:
            continue
        t_e = float(e.get("time", 0))
        if t_e > min_t:
            if first_game_event_t is None or t_e < first_game_event_t:
                first_game_event_t = t_e
    if first_game_event_t is not None:
        event_bound = first_game_event_t - 10.0
        # N'appliquer la borne que si elle est significative :
        # > 50% de max_search → évite de réduire inutilement sur clips courts
        # où les events fiables sont eux-mêmes avant le vrai KO
        if event_bound > max_search_t * 0.5 and event_bound > min_t:
            max_search_t = min(max_search_t, event_bound)
            print(f"  [KICKOFF] Borne max ajustée : {max_search_t:.0f}s "
                  f"(premier event fiable à {first_game_event_t:.0f}s)")
        else:
            print(f"  [KICKOFF] Borne event ignorée ({event_bound:.0f}s < 50% de {max_search_t:.0f}s)"
                  f" → recherche jusqu'à {max_search_t:.0f}s")

    consecutive        = 0
    first_two_teams    = False
    best_t             = None
    best_score         = 0.0
    candidate_start_t  = None
    all_candidates     = []
    _prev_ball_center  = None  # pour calculer la vitesse inter-frames

    for fd in frames_data:
        t = fd.get("frame", 0) / max(fps, 1)
        if t < min_t:
            continue
        # Stopper la recherche une fois hors fenêtre pré-match
        if t > max_search_t:
            break

        # Calculer la vitesse du ballon depuis les positions consécutives
        _ball_speed = None
        _ball = fd.get("ball")
        if _ball:
            _c = _ball.get("center")
            if _c and len(_c) >= 2 and _c[0] is not None:
                _cx, _cy = float(_c[0]), float(_c[1])
                if _prev_ball_center is not None:
                    import math as _math
                    _dx = _cx - _prev_ball_center[0]
                    _dy = _cy - _prev_ball_center[1]
                    # Distance en pixels entre 2 frames analysées consécutives
                    # (pas de conversion temps — on compare directement les pixels)
                    _ball_speed = _math.hypot(_dx, _dy)
                _prev_ball_center = (_cx, _cy)
            else:
                _prev_ball_center = None
        else:
            _prev_ball_center = None

        score, details, first_two_teams = _score_frame(
            fd, fps, action_times, first_two_teams, frame_w, frame_h,
            ball_speed_px=_ball_speed
        )

        if score >= _SCORE_THRESHOLD:
            consecutive += 1
            if consecutive == 1:
                candidate_start_t = t
            if consecutive >= _CONSECUTIVE_FRAMES:
                all_candidates.append((candidate_start_t, score, dict(details)))
        else:
            consecutive       = 0
            candidate_start_t = None

    if not all_candidates:
        print(f"  [KICKOFF] Aucun coup d'envoi détecté dans les {max_search_t:.0f}s initiales")
        return 0.0, 0.0

    # ── Validation par activité dans les 30s suivant le candidat ─────────────
    # Le vrai KO déclenche une activité réelle : le ballon parcourt de longues
    # distances dans les 30s qui suivent.
    # Faux positifs (placement temporaire) → peu d'activité ensuite.
    def _post_activity(candidate_t, frames_data, fps, window=30.0, min_dist=200.0):
        """Distance totale parcourue par le ballon dans les 30s après candidate_t."""
        total_dist = 0.0
        prev_c = None
        import math as _mact
        for fd in frames_data:
            t = fd.get("frame", 0) / max(fps, 1)
            if t < candidate_t:
                continue
            if t > candidate_t + window:
                break
            ball = fd.get("ball")
            if not ball:
                prev_c = None
                continue
            center = ball.get("center")
            if not center or center[0] is None:
                prev_c = None
                continue
            cx, cy = float(center[0]), float(center[1])
            if prev_c is not None:
                total_dist += _mact.hypot(cx - prev_c[0], cy - prev_c[1])
            prev_c = (cx, cy)
        return total_dist

    # ── Bonus : events réels dans les 60s suivant le candidat ────────────────
    # Le vrai KO est immédiatement suivi d'actions de match.
    # L'échauffement → pas d'action dans la fenêtre suivante.
    _event_list = []
    for _e in (events or []):
        _t_e = float(_e.get("time", 0))
        if _t_e > min_t:
            _event_list.append((_t_e, _e.get("type", ""), _e.get("detected_from", _e.get("source", ""))))

    # Poids par type d'événement (signal de match réel)
    _EVENT_WEIGHTS = {
        "goal":                    4.0,
        "terminal_goal":           4.0,
        "goal_posthoc":            3.0,
        "shot_to_goal_gemini":     3.0,
        "ball_appears_in_goal":    3.0,
        "shot":                    2.0,
        "dangerous_attack":        1.5,
        "goalkeeper_save":         1.5,
        "clearance":               1.0,
        "interception":            1.0,
        "dribble":                 1.0,
        "pass":                    0.8,
        "progressive_run":         0.8,
        "possession":              0.5,
        "touche":                  1.0,
        "corner":                  1.5,
        "free_kick":               1.5,
    }

    def _future_event_bonus(candidate_t, event_list, window=60.0):
        """
        Bonus basé sur les events réels dans les 0-60s suivant le candidat.
        Fenêtre : 0s (le KO lui-même peut déclencher un penalty immédiatement)
        Retourne (bonus_total, premier_event_décrit).
        """
        best_bonus = 0.0
        best_desc  = None
        for t_e, etype, esrc in event_list:
            dt = t_e - candidate_t
            if dt < 0 or dt > window:
                continue
            # Pondération temporelle : bonus plein si < 20s, dégressif ensuite
            time_factor = 1.0 if dt <= 20 else (1.0 - (dt - 20) / 80.0)
            # Poids par type
            type_weight = _EVENT_WEIGHTS.get(etype, 0.5)
            # Chercher aussi dans la source (goal_posthoc vient souvent d'events_standard)
            src_weight  = _EVENT_WEIGHTS.get(esrc, 0.0)
            weight      = max(type_weight, src_weight)
            bonus       = weight * time_factor * 3.0 / max(type_weight, 0.5)  # normalise sur 3.0 max
            bonus       = min(bonus, 3.0)
            if bonus > best_bonus:
                best_bonus = bonus
                best_desc  = f"{etype}@{int(t_e//60)}:{int(t_e%60):02d}(+{dt:.0f}s)"
        return round(best_bonus, 2), best_desc

    # Appliquer le bonus sur tous les candidats
    bonused_candidates = []
    for cand_t, cand_score, cand_det in all_candidates:
        bonus, bonus_desc = _future_event_bonus(cand_t, _event_list)
        bonused_candidates.append((cand_t, cand_score + bonus, cand_det, bonus, bonus_desc))

    # Trier par score total décroissant
    bonused_candidates.sort(key=lambda x: x[1], reverse=True)

    # Debug : TOUS les candidats
    print(f"  [KICKOFF] {len(all_candidates)} candidat(s) dans {max_search_t:.0f}s :")
    for cand_t, cand_total, cand_det, cand_bonus, cand_desc in bonused_candidates:
        t_fmt = f"{int(cand_t//60)}:{int(cand_t%60):02d}"
        print(f"    t={t_fmt} score={cand_total:.1f} "
              f"(base={cand_total-cand_bonus:.1f} bonus={cand_bonus:+.1f} [{cand_desc or 'aucun'}]) "
              f"sep={cand_det.get('team_separation',0):.2f} "
              f"n={cand_det.get('n_with_team',0)} "
              f"ball={'✓' if cand_det.get('ball_center') else '✗'} "
              f"near={cand_det.get('players_near_ball',0)}")

    # scored_candidates pour la sélection finale
    scored_candidates = [(t, s, d) for t, s, d, _, _ in bonused_candidates]

    best_t     = None
    best_score = 0.0
    best_det   = None
    selection  = "meilleur score"

    for cand_t, cand_score, cand_details in scored_candidates[:5]:
        activity = _post_activity(cand_t, frames_data, fps)
        if activity >= 200.0:
            best_t     = cand_t
            best_score = cand_score
            best_det   = cand_details
            selection  = f"meilleur score validé (activité={activity:.0f}px)"
            break

    if best_t is None:
        best_t, best_score, best_det = scored_candidates[0]
        activity = _post_activity(best_t, frames_data, fps)
        selection = f"meilleur score sans validation (activité={activity:.0f}px)"

    conf = min(0.95, 0.60 + (best_score - _SCORE_THRESHOLD) * 0.05)
    print(f"  [KICKOFF PHYS] {len(all_candidates)} candidat(s) → {selection} : "
          f"Score={best_score:.1f}/{_SCORE_THRESHOLD} "
          f"→ offset={best_t:.1f}s conf={conf:.2f} "
          f"(sep={best_det.get('team_separation',0):.2f} "
          f"n_team={best_det.get('n_with_team',0)} "
          f"ball={'✓' if best_det.get('ball_center') else '✗'})")
    return best_t, conf


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