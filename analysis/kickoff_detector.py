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

    # ── 1. SÉPARATION GÉOMÉTRIQUE DES ÉQUIPES [max 5.0] ─────────────────────
    # Au coup d'envoi : une équipe à gauche, une à droite de la ligne médiane.
    # On mesure la pureté des deux moitiés de terrain sans dépendre des team_id
    # (qui peuvent être absents ou tous identiques dans frames_data).
    #
    # Méthode :
    #   - Compter n_left  = joueurs avec cx < mid_x
    #   - Compter n_right = joueurs avec cx >= mid_x
    #   - Si les deux moitiés sont bien peuplées (≥3 chacune),
    #     calculer purity = max(n_left, n_right) / total
    #     → 0.5 si parfaitement mélangés, 1.0 si tous d'un côté
    #   - separation = (purity - 0.5) * 2  → [0, 1]
    #
    # Bonus séparation par équipe si team_id disponible et discriminant.

    cx_list = [_player_cx(p) for p in players if _player_cx(p) is not None]
    separation = 0.0
    if len(cx_list) >= 6:
        n_left  = sum(1 for cx in cx_list if cx < mid_x)
        n_right = len(cx_list) - n_left
        if n_left >= 3 and n_right >= 3:
            purity = max(n_left, n_right) / len(cx_list)
            separation = (purity - 0.5) * 2.0   # [0.0 → 1.0]

    # Bonus si team_id disponible et distingue vraiment 2 équipes
    def _get_team(p):
        for key in ("team", "team_id", "team_idx"):
            v = p.get(key)
            if v is not None:
                return v
        for key in ("color", "team_color", "jersey_color"):
            v = p.get(key)
            if v is not None:
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

    if len(team_players) >= 2:
        sorted_teams = sorted(team_players.keys(),
                              key=lambda k: len(team_players[k]), reverse=True)
        teams = sorted_teams[:2]
        n0 = len(team_players[teams[0]])
        n1 = len(team_players[teams[1]])
        if n0 >= 3 and n1 >= 3:
            t0_left  = sum(1 for cx in team_players[teams[0]] if cx < mid_x) / n0
            t1_left  = sum(1 for cx in team_players[teams[1]] if cx < mid_x) / n1
            sep_a = (t0_left + (1 - t1_left)) / 2.0
            sep_b = ((1 - t0_left) + t1_left) / 2.0
            team_sep = max(sep_a, sep_b)
            # Utiliser le meilleur des deux signaux
            separation = max(separation, team_sep)

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
def find_kickoff_offset(events, video_duration_s, frames_data=None, fps=25,
                        video_path=None, gemini_verify_fn=None):
    """
    Détecte le coup d'envoi et retourne (offset_s, confidence).

    Passe 1 — events type=kickoff (méthode legacy)
    Passe 2 — score pondéré sur frames_data (nouvelle méthode robuste)

    Le premier kickoff trouvé est retourné.
    offset_s = 0 si non détecté.
    """
    # ── Passe 1 : events type=kickoff ─────────────────────────────────────────
    # Vérification early motion : si la vidéo commence déjà en jeu
    # (p_motion élevée dès les 60 premières secondes), ignorer les kickoff events
    # qui seraient de faux positifs du terminal_events.
    _p1_early_high = False
    if frames_data:
        _p1_early_frames = [f for f in frames_data if float(f.get("time", 0)) <= 60.0]
        if _p1_early_frames:
            # Calculer la p_motion par distances inter-frames (comme la Passe 2)
            # car les players dans frames_data n'ont pas de champ "speed"
            _p1_prev_pos = {}
            _p1_frame_avgs = []
            for f in _p1_early_frames:
                players = f.get("players") or []
                moves = []
                for p in players:
                    if not isinstance(p, dict):
                        continue
                    pid = p.get("id") or p.get("player_id") or p.get("tracker_id")
                    if pid is None:
                        continue
                    cx = _player_cx(p)
                    cy = _player_cy(p)
                    if cx is None or cy is None:
                        continue
                    if pid in _p1_prev_pos:
                        ox, oy = _p1_prev_pos[pid]
                        moves.append(math.hypot(cx - ox, cy - oy))
                    _p1_prev_pos[pid] = (cx, cy)
                if moves:
                    _p1_frame_avgs.append(sum(moves) / len(moves))
            if _p1_frame_avgs:
                _p1_avg = sum(_p1_frame_avgs) / len(_p1_frame_avgs)
                print(f"  [KICKOFF P1] p_motion moy 0-60s = {_p1_avg:.1f}")
                if _p1_avg >= 8.0:
                    _p1_early_high = True
                    print(f"  [KICKOFF P1] p_motion moy 0-60s = {_p1_avg:.1f} → vidéo commence en jeu, passe 1 ignorée")

    kickoff_events = sorted(
        [e for e in (events or []) if e.get("type") == "kickoff"],
        key=lambda e: e.get("time", 0)
    )
    if kickoff_events and not _p1_early_high:
        first = kickoff_events[0]
        t     = float(first.get("time", 0))
        conf  = float(first.get("confidence", first.get("conf", 0.80)))
        if t > 0:
            print(f"  [KICKOFF] Détecté via events à t={t:.1f}s (conf={conf:.2f})")
            return t, conf

    # ── Passe 2 : score pondéré sur frames_data ───────────────────────────────
    # Si la Passe 1 a détecté p_motion élevée dès 0-60s (_p1_early_high),
    # deux cas sont possibles :
    #
    #   CAS A — vidéo commence en cours de match (pas d'avant-match) :
    #     p_motion élevée dès 0s, reste élevée → offset=0 correct
    #
    #   CAS B — vidéo commence à l'échauffement :
    #     p_motion élevée dès 0s (joueurs qui s'échauffent), puis transition
    #     vers le vrai coup d'envoi → offset > 0 nécessaire
    #     Exemple : andrimont_0, kickoff réel à 308s
    #
    # Ancien comportement : court-circuit systématique → offset=0 dans les deux cas.
    # Problème : CAS B non géré → tous les events d'avant-match entraient dans
    # le pipeline comme s'ils appartenaient au match (FP structurels).
    #
    # Correctif : si video_path est disponible, laisser la Passe 2 chercher
    # une transition faible→fort. Sans video_path → conservateur, offset=0.
    if _p1_early_high:
        _p1_avg_val = sum(_p1_frame_avgs) / len(_p1_frame_avgs) if _p1_frame_avgs else 0.0
        if not video_path:
            print(f"  [KICKOFF P2] p_motion={_p1_avg_val:.1f} >= 8.0, pas de video_path"
                  f" → offset=0 (match supposé en cours)")
            return 0.0, 0.0
        print(f"  [KICKOFF P2] p_motion={_p1_avg_val:.1f} >= 8.0 + video_path disponible"
              f" → recherche transition échauffement→match (CAS B)")

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
    _kp_sep_by_t  = {}  # t → (sep, n) — accumulé par la boucle principale pour le probe joueurs

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

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL BALLON — BALLON STABLE AU CENTRE PUIS PREMIER MOUVEMENT
    #
    # CAS B (échauffement) : le ballon est posé au rond central pour la
    # cérémonie (tirage au sort, serrage de mains). Il y reste immobile
    # pendant plusieurs minutes, puis part au coup d'envoi réel.
    #
    # Problème connu : le BallTracker génère des faux positifs dès t=0
    # (ballon "détecté" sur des formes circulaires hors terrain), donc
    # on NE peut PAS se fier à l'absence initiale du ballon.
    #
    # Solution : chercher la PREMIÈRE période prolongée (_BALL_STABLE_MIN_S)
    # où le ballon est détecté de façon continue et stable au centre.
    # Cette période correspond à la cérémonie pré-match. Le premier
    # mouvement significatif qui part du centre est le kickoff.
    #
    # Filtre clé : les mouvements hors centre (gardien qui envoie le ballon
    # dans sa moitié, joueur qui s'échauffe) sont ignorés — seul un mouvement
    # DEPUIS le rond central compte.
    # ─────────────────────────────────────────────────────────────────────────
    _BALL_STABLE_MIN_S  = 20.0  # Ballon doit rester au centre pendant >= 20s (cérémonie)
    _BALL_STABLE_TOL_PX = 30.0  # Tolérance de mouvement pour "immobile" (pixels)
    _BALL_MOVE_PX       = 20.0  # Mouvement minimal pour valider le coup d'envoi
    _BALL_STILL_MAX_S   = 600.0 # Le ballon peut rester immobile jusqu'à 10min après cérémonie
    _CENTER_TOL_X       = 0.22  # Tolérance horizontale autour du centre (bx ∈ [0.28, 0.72])
    _CENTER_TOL_Y       = 0.25  # Tolérance verticale autour du centre (by ∈ [0.25, 0.75])


    _ball_kickoff_t = None
    if _p1_early_high:
        # Phase 1 : trouver la fenêtre de cérémonie (ballon stable au centre ≥ 20s)
        # NOTE : sur low_side_zoom, le BallTracker ne détecte pas le ballon au rond
        # central (streak_max=1.9s prouvé). Cette phase reste en code pour les autres
        # types de caméra, mais ne produira pas de signal sur low_side_zoom.
        _ceremony_start_t  = None
        _ceremony_ref_pos  = None
        _center_streak_s   = 0.0
        _prev_t_ball       = None
        _prev_bx_stable    = None
        _prev_by_stable    = None

        for _fd in frames_data:
            _t = _fd.get("frame", 0) / max(fps, 1)
            if _t > max_search_t + 60:
                break

            _dt = (_t - _prev_t_ball) if _prev_t_ball is not None else (1.0 / max(fps, 1))
            _prev_t_ball = _t

            _bfd = _fd.get("ball")
            _bc  = _bfd.get("center") if _bfd else None
            _has_ball = (_bc is not None and len(_bc) >= 2 and _bc[0] is not None)

            if not _has_ball:
                _center_streak_s = 0.0
                _ceremony_start_t = None
                _prev_bx_stable = None
                _prev_by_stable = None
                continue

            _bx = float(_bc[0]) / max(frame_w, 1)
            _by = float(_bc[1]) / max(frame_h, 1)

            # Au centre ?
            _at_center = (
                abs(_bx - 0.50) <= _CENTER_TOL_X
                and 0.5 - _CENTER_TOL_Y <= _by <= 0.5 + _CENTER_TOL_Y
            )

            # Stable ? (bruit BallTracker ≤ _BALL_STABLE_TOL_PX)
            _is_stable = False
            if _at_center and _prev_bx_stable is not None:
                _dx_stab = (_bx - _prev_bx_stable) * frame_w
                _dy_stab = (_by - _prev_by_stable) * frame_h
                _is_stable = math.hypot(_dx_stab, _dy_stab) <= _BALL_STABLE_TOL_PX
            elif _at_center and _prev_bx_stable is None:
                _is_stable = True  # Première frame au centre → on commence le streak

            if _at_center and _is_stable:
                if _ceremony_start_t is None:
                    _ceremony_start_t = _t
                    _ceremony_ref_pos = (_bx, _by)
                _center_streak_s += _dt
                _prev_bx_stable = _bx
                _prev_by_stable = _by

                if _center_streak_s >= _BALL_STABLE_MIN_S:
                    # Cérémonie détectée !

                    break
            else:
                # Hors centre ou mouvement brusque → reset
                _center_streak_s  = 0.0
                _ceremony_start_t = None
                _prev_bx_stable   = None
                _prev_by_stable   = None

        # Phase 2 : si cérémonie trouvée, chercher le premier mouvement depuis le centre
        if _ceremony_start_t is not None and _center_streak_s >= _BALL_STABLE_MIN_S:
            _prev_t_ball2       = None
            _prev_ball_pos_ba   = None
            _ball_at_center_prev = False

            for _fd in frames_data:
                _t = _fd.get("frame", 0) / max(fps, 1)
                if _t < _ceremony_start_t:
                    continue
                if _t > _ceremony_start_t + _BALL_STILL_MAX_S:
                    break

                _dt2 = (_t - _prev_t_ball2) if _prev_t_ball2 is not None else (1.0 / max(fps, 1))
                _prev_t_ball2 = _t

                _bfd = _fd.get("ball")
                _bc  = _bfd.get("center") if _bfd else None
                _has_ball = (_bc is not None and len(_bc) >= 2 and _bc[0] is not None)

                if not _has_ball:
                    _prev_ball_pos_ba    = None
                    _ball_at_center_prev = False
                    continue

                _bx = float(_bc[0]) / max(frame_w, 1)
                _by = float(_bc[1]) / max(frame_h, 1)

                _at_center = (
                    abs(_bx - 0.50) <= _CENTER_TOL_X
                    and 0.5 - _CENTER_TOL_Y <= _by <= 0.5 + _CENTER_TOL_Y
                )

                if _prev_ball_pos_ba is not None and _ball_at_center_prev:
                    _dx = (_bx - _prev_ball_pos_ba[0]) * frame_w
                    _dy = (_by - _prev_ball_pos_ba[1]) * frame_h
                    _move = math.hypot(_dx, _dy)
                    if _move >= _BALL_MOVE_PX:
                        _ball_kickoff_t = _t

                        break
                    # else: immobile → cérémonie continue

                _prev_ball_pos_ba    = (_bx, _by)
                _ball_at_center_prev = _at_center
        # (else: pas de cérémonie détectée — normal sur low_side_zoom)

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL PRINCIPAL : transition d'activité joueurs + ballon
    #
    # Le pré-match (échauffement) = joueurs qui marchent / statiques.
    # Le match = joueurs qui courent, ballon qui se déplace en continu.
    #
    # On calcule une activité glissante (fenêtre 10s) sur :
    #   - player_motion : moyenne des distances inter-frames par joueur tracké
    #   - ball_motion   : distance inter-frames du ballon
    #
    # Le KO est la première transition durable faible→fort.
    # ─────────────────────────────────────────────────────────────────────────
    _motion_trace = []  # (t, player_motion_px, ball_motion_px)
    _motion_by_t  = {}  # t → p_motion  (pour log des groupes)
    _prev_player_pos = {}   # pid → (cx, cy)
    _prev_ball_pos_m = None

    for fd in frames_data:
        t = fd.get("frame", 0) / max(fps, 1)
        if t > max_search_t + 60:
            break

        # Player motion
        players_fd = fd.get("players", [])
        moves = []
        for p in players_fd:
            pid = p.get("id")
            if pid is None:
                continue
            cx = _player_cx(p)
            cy = _player_cy(p)
            if cx is None or cy is None:
                continue
            if pid in _prev_player_pos:
                ox, oy = _prev_player_pos[pid]
                moves.append(math.hypot(cx - ox, cy - oy))
            _prev_player_pos[pid] = (cx, cy)
        p_motion = sum(moves) / len(moves) if moves else 0.0
        _motion_by_t[t] = p_motion

        # Ball motion
        b_motion = 0.0
        ball_fd = fd.get("ball")
        if ball_fd:
            bc = ball_fd.get("center")
            if bc and len(bc) >= 2 and bc[0] is not None:
                bx, by = float(bc[0]), float(bc[1])
                if _prev_ball_pos_m is not None:
                    b_motion = math.hypot(bx - _prev_ball_pos_m[0],
                                          by - _prev_ball_pos_m[1])
                _prev_ball_pos_m = (bx, by)
            else:
                _prev_ball_pos_m = None
        else:
            _prev_ball_pos_m = None

        _motion_trace.append((t, p_motion, b_motion))

    # Activité glissante (fenêtre 5s)
    _WIN = 5.0
    _activity_curve = []
    for i, (t, pm, bm) in enumerate(_motion_trace):
        window = [x for x in _motion_trace if t - _WIN <= x[0] <= t]
        avg_pm = sum(x[1] for x in window) / max(len(window), 1)
        avg_bm = sum(x[2] for x in window) / max(len(window), 1)
        combined = avg_pm + avg_bm * 0.5
        _activity_curve.append((t, combined))

    # Log ASCII de la courbe (résolution 10s, jusqu'à max_search_t + 30s)
    if _activity_curve:
        max_act = max(a for _, a in _activity_curve) or 1.0
        print("  [MOTION] Profil activité (résolution 10s, █=haut, ░=bas) :")
        bucket_size = 10.0
        t_max_log = min(max_search_t + 30, _activity_curve[-1][0])
        t_cur = 0.0
        while t_cur <= t_max_log:
            bucket = [a for t, a in _activity_curve
                      if t_cur <= t < t_cur + bucket_size]
            if bucket:
                val = sum(bucket) / len(bucket)
                bar_len = int(val / max_act * 20)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                print(f"    {int(t_cur):4d}s [{bar}] {val:.1f}")
            t_cur += bucket_size

    # Détection de transition faible→fort
    # On cherche la première fenêtre de 10s où l'activité dépasse
    # le seuil = moyenne_globale * 1.5, précédée d'une fenêtre basse
    _motion_offset = None
    if len(_activity_curve) >= 10:
        vals = [a for _, a in _activity_curve]
        global_mean = sum(vals) / len(vals)
        threshold_hi = global_mean * 1.5
        threshold_lo = global_mean * 0.7

        # Calculer activité glissante 10s
        _win10 = 10.0
        _smoothed = []
        for t, _ in _activity_curve:
            w = [a for tt, a in _activity_curve if t - _win10 <= tt <= t]
            _smoothed.append((t, sum(w) / max(len(w), 1)))

        # Chercher première transition lo→hi
        _was_low = False
        for i, (t, act) in enumerate(_smoothed):
            if t < min_t:
                continue
            if act <= threshold_lo:
                _was_low = True
            if _was_low and act >= threshold_hi:
                _motion_offset = t
                print(f"  [MOTION] Transition détectée à t={t:.1f}s "
                      f"(act={act:.1f} > seuil={threshold_hi:.1f}, "
                      f"mean={global_mean:.1f})")
                break

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
        details["p_motion"] = _motion_by_t.get(t, 0.0)
        # Accumuler sep+n pour le probe joueurs (team_separation est dans details, pas dans fd)
        _kp_sep_by_t[t] = (details.get("team_separation", 0.0), len(fd.get("players", [])))

        # ── PROBE 250-320s : 4 champs seulement ─────────────────────────────
        if 250.0 <= t <= 320.0:
            sep  = details.get("team_separation", 0.0)
            n    = len(fd.get("players", []))
            ball_fd = fd.get("ball")
            if ball_fd:
                bc = ball_fd.get("center")
                if bc and len(bc) >= 2 and bc[0] is not None and frame_w and frame_h:
                    bx_n = round(float(bc[0]) / frame_w, 2)
                    by_n = round(float(bc[1]) / frame_h, 2)
                    spd  = f"{_ball_speed:.0f}" if _ball_speed is not None else "?"
                    ball_str = f"{bx_n},{by_n} spd={spd}"
                else:
                    ball_str = "none"
            else:
                ball_str = "none"
            print(f"  [KO_PROBE] t={t:6.1f}s sep={sep:.2f} n={n:2d} ball={ball_str} pm={_motion_by_t.get(t,0.0):.1f}")

        if score >= _SCORE_THRESHOLD:
            consecutive += 1
            if consecutive == 1:
                candidate_start_t = t
            if consecutive >= _CONSECUTIVE_FRAMES:
                all_candidates.append((candidate_start_t, score, dict(details)))
        else:
            consecutive       = 0
            candidate_start_t = None

    # ─────────────────────────────────────────────────────────────────────────
    # PROBE JOUEURS V1 : détection kickoff basée sur sep + n uniquement
    #
    # Sur low_side_zoom le BallTracker est inutilisable (streak_max=1.9s prouvé).
    # On cherche dans frames_data le premier groupe long (≥20s) satisfaisant :
    #   - sep >= _KP_SEP_MIN  (équipes bien séparées)
    #   - n   >= _KP_N_MIN    (assez de joueurs trackés)
    # Le début de ce groupe = candidat kickoff.
    # Ce probe est INSTRUMENTÉ UNIQUEMENT — il loggue mais ne modifie pas
    # _ball_kickoff_t tant que la validation multi-vidéos n'est pas faite.
    # ─────────────────────────────────────────────────────────────────────────
    _KP_SEP_MIN    = 0.30   # séparation inter-équipes minimale
    _KP_N_MIN      = 12     # nombre de joueurs trackés minimal
    _KP_MIN_DUR_S  = 5.0    # durée minimale du groupe (secondes) — abaissé de 20s: groupes réels ~8s
    _KP_WIN_S      = 2.0    # tolérance de gap dans un groupe (secondes) — élargi: gaps n=0 jusqu'à 1.4s observés

    # Diagnostic : taille et contenu de l'accumulateur
    kp_keys = sorted(_kp_sep_by_t.keys())
    kp_285_320 = [(t, sep, n) for t, (sep, n) in _kp_sep_by_t.items() if 285.0 <= t <= 320.0]
    kp_qual = [(t, sep, n) for t, sep, n in kp_285_320 if sep >= 0.30 and n >= 12]
    print(f"  [KICKOFF PLAYERS] probe start : len={len(_kp_sep_by_t)}"
          f" t_range=[{kp_keys[0]:.1f}s,{kp_keys[-1]:.1f}s]" if kp_keys else
          f"  [KICKOFF PLAYERS] probe start : len={len(_kp_sep_by_t)} (vide)")
    print(f"  [KICKOFF PLAYERS] dans 285-320s: {len(kp_285_320)} frames"
          f" | sep>=0.30 n>=12: {len(kp_qual)}")
    if kp_285_320:
        sep_vals = [sep for _, sep, _ in kp_285_320]
        n_vals   = [n   for _, _, n   in kp_285_320]
        print(f"  [KICKOFF PLAYERS]   sep max={max(sep_vals):.3f} mean={sum(sep_vals)/len(sep_vals):.3f}"
              f" | n max={max(n_vals)} mean={sum(n_vals)/len(n_vals):.1f}")
    if kp_qual:
        print(f"  [KICKOFF PLAYERS]   première qualif: t={kp_qual[0][0]:.1f}s sep={kp_qual[0][1]:.2f} n={kp_qual[0][2]}")
    # Collecter les frames satisfaisant les critères, depuis l'accumulateur de la boucle principale
    # (team_separation est dans details/_kp_sep_by_t, PAS dans frames_data directement)
    _kp_frames = []   # liste de (t, sep, n)
    for _t_kp in sorted(_kp_sep_by_t):
        if _t_kp > max_search_t:
            break
        _sep_kp, _n_kp = _kp_sep_by_t[_t_kp]
        if _sep_kp >= _KP_SEP_MIN and _n_kp >= _KP_N_MIN:
            _kp_frames.append((_t_kp, _sep_kp, _n_kp))

    # Grouper les frames consécutives (gap toléré = _KP_WIN_S)
    _kp_groups = []
    if _kp_frames:
        _cur_grp = [_kp_frames[0]]
        for _i in range(1, len(_kp_frames)):
            if _kp_frames[_i][0] - _kp_frames[_i-1][0] <= _KP_WIN_S:
                _cur_grp.append(_kp_frames[_i])
            else:
                _kp_groups.append(_cur_grp)
                _cur_grp = [_kp_frames[_i]]
        _kp_groups.append(_cur_grp)

    # Filtrer les groupes par durée minimale
    _kp_long_groups = []
    for _grp in _kp_groups:
        _dur = _grp[-1][0] - _grp[0][0]
        if _dur >= _KP_MIN_DUR_S:
            _sep_avg = sum(f[1] for f in _grp) / len(_grp)
            _n_avg   = sum(f[2] for f in _grp) / len(_grp)
            _kp_long_groups.append({
                "t_start": _grp[0][0],
                "t_end":   _grp[-1][0],
                "dur":     _dur,
                "sep_avg": _sep_avg,
                "n_avg":   _n_avg,
                "len":     len(_grp),
            })

    # Log du résultat
    if _kp_long_groups:
        print(f"  [KICKOFF PLAYERS] {len(_kp_long_groups)} groupe(s) "
              f"sep≥{_KP_SEP_MIN} n≥{_KP_N_MIN} dur≥{_KP_MIN_DUR_S:.0f}s :")
        for _gi, _g in enumerate(_kp_long_groups[:6]):
            _t0_fmt = f"{int(_g['t_start']//60)}:{int(_g['t_start']%60):02d}"
            _t1_fmt = f"{int(_g['t_end']//60)}:{int(_g['t_end']%60):02d}"
            print(f"    grp[{_gi}] {_t0_fmt}→{_t1_fmt} "
                  f"dur={_g['dur']:.0f}s  sep_avg={_g['sep_avg']:.2f}  "
                  f"n_avg={_g['n_avg']:.1f}  frames={_g['len']}")
        # Candidat = début du DERNIER groupe (le plus proche du vrai coup d'envoi)
        # Raisonnement : groupe 1 = cérémonie/positionnement (précurseur)
        #                groupe N = reprise réelle du jeu (événement)
        # Pour filtrer les tirs d'échauffement, on veut l'offset du jeu réel.
        _kp_long_groups.sort(key=lambda g: g["t_start"])
        _kp_candidate_t = _kp_long_groups[-1]["t_start"]
        _t_fmt = f"{int(_kp_candidate_t//60)}:{int(_kp_candidate_t%60):02d}"
        _n_groups = len(_kp_long_groups)
        if _n_groups > 1:
            _first_t = f"{int(_kp_long_groups[0]['t_start']//60)}:{int(_kp_long_groups[0]['t_start']%60):02d}"
            print(f"  [KICKOFF PLAYERS] → {_n_groups} groupe(s) : précurseur={_first_t} "
                  f"→ candidat kickoff t={_t_fmt} (dernier groupe, instrumenté, non appliqué)")
        else:
            print(f"  [KICKOFF PLAYERS] → candidat kickoff t={_t_fmt} "
                  f"(groupe unique, instrumenté, non appliqué)")
    else:
        print(f"  [KICKOFF PLAYERS] aucun groupe satisfaisant "
              f"sep≥{_KP_SEP_MIN} n≥{_KP_N_MIN} dur≥{_KP_MIN_DUR_S:.0f}s "
              f"→ signal joueurs absent")

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

    # Trier les candidats par score décroissant
    scored_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

    # Debug : top-20 candidats
    print(f"  [KICKOFF] {len(all_candidates)} candidat(s) dans {max_search_t:.0f}s :")
    for cand_t, cand_score, cand_det in scored_candidates[:20]:
        t_fmt = f"{int(cand_t//60)}:{int(cand_t%60):02d}"
        print(f"    t={t_fmt} score={cand_score:.1f} "
              f"sep={cand_det.get('team_separation',0):.2f} "
              f"n={cand_det.get('n_with_team',0)} "
              f"ball={'✓' if cand_det.get('ball_center') else '✗'} "
              f"near={cand_det.get('players_near_ball',0)}")

    best_t     = None
    best_score = 0.0
    best_det   = None
    selection  = "meilleur score"

    # Stratégie 1 : parmi les candidats avec sep >= 0.45, prendre celui dont
    # la séquence consécutive de frames avec sep ≥ 0.45 est la plus longue.
    # Le vrai KO a les équipes statiques sur leurs moitiés pendant ~5-15 secondes.
    # L'échauffement a des joueurs qui bougent partout (séquences courtes).
    high_sep = [(t, s, d) for t, s, d in scored_candidates
                if d.get("team_separation", 0) >= 0.45]

    if high_sep:
        # Grouper les candidats consécutifs (delta < 2s) et mesurer la longueur
        # de chaque séquence continue
        groups = []
        current_group = [high_sep[0]]
        for i in range(1, len(high_sep)):
            t_prev = high_sep[i-1][0]
            t_curr = high_sep[i][0]
            if abs(t_curr - t_prev) <= 2.0:
                current_group.append(high_sep[i])
            else:
                groups.append(current_group)
                current_group = [high_sep[i]]
        groups.append(current_group)

        # Trier les groupes par longueur décroissante (séquence la plus stable)
        groups.sort(key=lambda g: len(g), reverse=True)

        # Log des groupes avec p_motion moyen — pour décider ensuite si ce signal
        # discrimine effectivement les vrais KO des placements pré-match
        print(f"  [KICKOFF GROUPS] {len(groups)} groupe(s) sep≥0.45 :")
        for gi, grp in enumerate(groups[:8]):
            g_best = max(grp, key=lambda x: x[1])
            t_fmt = f"{int(g_best[0]//60)}:{int(g_best[0]%60):02d}"
            pm_vals = [d.get("p_motion", 0.0) for _, _, d in grp]
            pm_avg = sum(pm_vals) / max(len(pm_vals), 1)
            pm_min = min(pm_vals)
            pm_max = max(pm_vals)
            print(f"    grp[{gi}] t={t_fmt} len={len(grp)} "
                  f"sep={g_best[2].get('team_separation',0):.2f} "
                  f"p_motion avg={pm_avg:.1f} min={pm_min:.1f} max={pm_max:.1f}")

        # Parmi les 3 groupes les plus longs, prendre le premier validé par activité
        for grp in groups[:3]:
            # Représenter le groupe par son meilleur score
            best_in_grp = max(grp, key=lambda x: x[1])
            cand_t, cand_score, cand_details = best_in_grp
            activity = _post_activity(cand_t, frames_data, fps)
            if activity >= 200.0:
                best_t     = cand_t
                best_score = cand_score
                best_det   = cand_details
                selection  = (f"sep≥0.45 séquence={len(grp)}f "
                              f"validé (activité={activity:.0f}px)")
                break

        # ── Vérification Gemini des top candidats ────────────────────────────
        # Si un callback gemini_verify_fn est fourni ET video_path disponible,
        # on vérifie visuellement les top-5 groupes par score.
        # Le premier confirmé par Gemini remplace la sélection physique.
        if gemini_verify_fn and video_path and groups:
            print(f"  [KICKOFF GEMINI] Vérification visuelle des top groupes...")
            # Trier par timestamp croissant : le premier KO dans le temps est le bon.
            # On vérifie TOUS les groupes — pas seulement les plus longs.
            # Un match a toujours un KO : le premier groupe confirmé par Gemini est le vrai.
            # On vérifie TOUS les groupes avec p_motion >= 8.0.
            # On collecte tous ceux confirmés par Gemini, puis on retient
            # celui avec la p_motion la plus élevée (vrai KO = mouvement maximal).
            # ── Vérification : la vidéo commence-t-elle déjà en jeu ? ──────────
            # Même logique que Passe 1 : si video_path disponible, on laisse
            # la vérification Gemini chercher un KO même si p_motion élevée dès 0s
            # (CAS B : échauffement). Sans video_path → offset=0 conservateur.
            _early_motions = [v for t, v in _motion_by_t.items() if t <= 60.0]
            _early_pm_avg  = sum(_early_motions) / max(len(_early_motions), 1) if _early_motions else 0.0
            print(f"  [KICKOFF EARLY] p_motion moy 0-60s = {_early_pm_avg:.1f}")
            if _early_pm_avg >= 8.0 and not video_path:
                print(f"  [KICKOFF EARLY] ⚠️  p_motion élevée, pas de video_path → offset=0")
                best_t     = 0.0
                best_score = 0.0
                best_det   = {}
                selection  = "no_kickoff_video_starts_in_play"
                return 0.0, 0.0
            if _early_pm_avg >= 8.0:
                print(f"  [KICKOFF EARLY] p_motion élevée + video_path → vérification Gemini continue (CAS B échauffement)")

            _top_grps = sorted(groups, key=lambda g: min(x[0] for x in g))
            _gemini_confirmed_candidates = []  # (cand_t, score, details, conf, pm_avg)
            for grp in _top_grps:
                best_in_grp = max(grp, key=lambda x: x[1])
                cand_t = best_in_grp[0]
                pm_vals = [_motion_by_t.get(e[0], 0.0) for e in grp]
                pm_avg = sum(pm_vals) / max(len(pm_vals), 1)
                print(f"  [KICKOFF GEMINI] t={int(cand_t//60)}:{int(cand_t%60):02d} → p_motion={pm_avg:.1f} (seuil=8.0)")
                if pm_avg < 8.0:
                    print(f"  [KICKOFF GEMINI] t={int(cand_t//60)}:{int(cand_t%60):02d} → ⏭️  ignoré (p_motion={pm_avg:.1f} trop bas)")
                    continue
                activity = _post_activity(cand_t, frames_data, fps)
                if activity < 200.0:
                    continue
                offsets = [-2.0, 0.0, 2.0, 5.0]
                confirmed, conf_gemini = gemini_verify_fn(
                    video_path, cand_t, offsets=offsets
                )
                print(f"  [KICKOFF GEMINI] t={int(cand_t//60)}:{int(cand_t%60):02d} "
                      f"→ {'✅ KO confirmé' if confirmed else '❌ rejeté'} "
                      f"(conf={conf_gemini:.2f})")
                if confirmed:
                    _gemini_confirmed_candidates.append(
                        (cand_t, best_in_grp[1], best_in_grp[2], conf_gemini, pm_avg)
                    )
            # Sélection : parmi tous confirmés, score combiné p_motion + timestamp tardif.
            # Le vrai KO a toujours la p_motion la plus haute ET est le plus tardif.
            # Score = p_motion * 0.7 + position_relative * 0.3
            # (position_relative = rang chronologique / total)
            if _gemini_confirmed_candidates:
                _n = len(_gemini_confirmed_candidates)
                for _rank, _cand in enumerate(_gemini_confirmed_candidates):
                    _pos_score = _rank / max(_n - 1, 1)  # 0=premier, 1=dernier
                    _combined  = _cand[4] * 0.7 + _pos_score * 30 * 0.3  # normalise p_motion ~0-30
                    _cand = _cand + (_combined,)
                    _gemini_confirmed_candidates[_rank] = _cand
                _best = max(_gemini_confirmed_candidates, key=lambda x: x[5])
                best_t, best_score, best_det = _best[0], _best[1], _best[2]
                conf_gemini = _best[3]
                selection = f"gemini_confirmed (conf={conf_gemini:.2f})"
                print(f"  [KICKOFF GEMINI] ✅ Sélection finale : t={int(best_t//60)}:{int(best_t%60):02d} "
                      f"p_motion={_best[4]:.1f} score_combiné={_best[5]:.1f} "
                      f"parmi {_n} confirmé(s)")


    # Stratégie 2 : fallback sur les 5 meilleurs scores
    if best_t is None:
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

    # ── AFFINAGE PAR DÉTECTION VISUELLE DU BALLON AU ROND CENTRAL ────────────
    # Pour les top-5 candidats par sep, on analyse les frames originales
    # dans une fenêtre ±15s pour trouver un objet clair et petit (ballon)
    # qui est PRÉSENT puis DISPARAÎT dans la zone centrale.
    # C'est le vrai signal du coup d'envoi : ballon posé → joué.
    #
    # On ne modifie best_t que si on trouve un signal clair.
    # Sinon on garde le résultat actuel.

    def _detect_ball_played_in_center(video_path, candidate_t, frame_w, frame_h,
                                      fps, window=15.0):
        """
        Ouvre la vidéo originale autour de candidate_t.
        Cherche dans la zone centrale (40-60% x, 35-65% y) :
          - un pixel cluster clair/blanc/rond présent N frames consécutives
          - suivi d'une absence (le ballon vient d'être joué)
        Retourne le timestamp de la disparition, ou None.
        """
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError:
            return None

        if not video_path:
            return None

        try:
            cap = _cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            real_fps = cap.get(_cv2.CAP_PROP_FPS) or fps
            start_frame = max(0, int((candidate_t - window) * real_fps))
            end_frame   = int((candidate_t + window) * real_fps)

            cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

            cx1 = int(frame_w * 0.38)
            cx2 = int(frame_w * 0.62)
            cy1 = int(frame_h * 0.33)
            cy2 = int(frame_h * 0.67)

            present_streak  = 0
            absent_streak   = 0
            last_present_t  = None
            PRESENT_THRESH  = 4    # frames consécutives avec ballon
            ABSENT_THRESH   = 3    # frames consécutives sans ballon

            frame_idx = start_frame
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                t_cur = frame_idx / real_fps

                # Extraire la zone centrale et réduire
                roi = frame[cy1:cy2, cx1:cx2]
                small = _cv2.resize(roi, (64, 48))

                # Chercher un objet clair (ballon blanc/clair)
                gray = _cv2.cvtColor(small, _cv2.COLOR_BGR2GRAY)
                # Seuil adaptatif : pixels très clairs dans une scène verte
                _, thresh = _cv2.threshold(gray, 200, 255, _cv2.THRESH_BINARY)
                # Chercher des blobs petits et ronds
                contours, _ = _cv2.findContours(
                    thresh, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
                )
                ball_found = False
                for cnt in contours:
                    area = _cv2.contourArea(cnt)
                    if 8 < area < 200:  # taille compatible ballon
                        x, y, w, h = _cv2.boundingRect(cnt)
                        aspect = w / max(h, 1)
                        if 0.5 < aspect < 2.0:  # forme ronde/ovale
                            ball_found = True
                            break

                if ball_found:
                    present_streak += 1
                    absent_streak   = 0
                    if present_streak >= PRESENT_THRESH:
                        last_present_t = t_cur
                else:
                    absent_streak  += 1
                    present_streak  = 0
                    if absent_streak >= ABSENT_THRESH and last_present_t is not None:
                        cap.release()
                        return last_present_t  # instant de la disparition = KO

                frame_idx += 1

            cap.release()
        except Exception:
            pass
        return None

    # Récupérer le chemin vidéo — priorité au paramètre direct, sinon scan frames_data
    _video_path = video_path
    if not _video_path:
        for _fd in (frames_data or []):
            _vp = _fd.get("video_path") or _fd.get("source_video")
            if _vp:
                _video_path = _vp
                break

    # Tenter l'affinage sur les top-3 candidats sep
    _top_sep = sorted(
        [(t, s, d) for t, s, d in scored_candidates
         if d.get("team_separation", 0) >= 0.35],
        key=lambda x: x[1], reverse=True
    )[:3]

    _refined = False
    if _video_path and _top_sep:
        for _cand_t, _cand_s, _cand_d in _top_sep:
            _ball_t = _detect_ball_played_in_center(
                _video_path, _cand_t, frame_w, frame_h, fps
            )
            if _ball_t is not None:
                print(f"  [KICKOFF BALL] ballon joué détecté visuellement "
                      f"à t={_ball_t:.1f}s (candidat={_cand_t:.1f}s)")
                best_t    = _ball_t
                best_score = _cand_s
                best_det   = _cand_d
                selection  = f"ball_played visuel à {_ball_t:.1f}s"
                conf       = min(0.95, conf + 0.10)
                _refined   = True
                break
    if not _refined:
        print(f"  [KICKOFF BALL] affinage visuel non disponible "
              f"({'pas de chemin vidéo' if not _video_path else 'aucun signal trouvé'})")

        # ── Signal BALL_APPEAR prioritaire (CAS B échauffement) ──────────────
        # Si on a détecté le premier mouvement du ballon après une longue
        # absence, c'est le kickoff le plus fiable pour les vidéos d'échauffement.
        if _ball_kickoff_t is not None:
            print(f"  [KICKOFF] Signal ball_appear → offset={_ball_kickoff_t:.1f}s conf=0.85")
            return _ball_kickoff_t, 0.85

        # Si la Passe 1 avait détecté p_motion élevée (vidéo commence en jeu)
        # et que la détection visuelle ne confirme aucun kick off → offset=0
        # (la vidéo commence vraiment en cours de jeu, pas de pré-match à supprimer)
        if _p1_early_high:
            print(f"  [KICKOFF] p_motion élevée + aucun signal visuel → vidéo commence en jeu, offset=0")
            return 0.0, 0.0

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
    silence_threshold_s  = 1500.0,   # 25 min sans vrai jeu = fin du match (matchs complets)
    min_match_duration_s = 2700.0,   # chercher fin seulement après 45 min (matchs complets)
    video_duration_s     = None,     # durée totale de la vidéo (si connue)
):
    """
    Détecte la fin du match dans les frames_data.

    Critère "vrai jeu" : ≥4 joueurs avec couleur d'équipe assignée.
    Les gamins sans vareuse après le match = ignorés (team=None).

    Seuils adaptatifs : pour les vidéos courtes (extraits, fins de match),
    les seuils sont réduits proportionnellement à la durée vidéo.

    Retourne le timestamp de fin (s) ou None si non détecté.
    """
    if not frames_data:
        return None

    # ── Seuils adaptatifs selon durée vidéo ──────────────────────────────────
    # Si la vidéo est plus courte qu'un match complet, adapter les seuils :
    # - min_match_duration : au moins 30% de la durée vidéo (min 60s)
    # - silence_threshold  : au moins 10% de la durée vidéo (min 30s)
    if video_duration_s is not None and video_duration_s > 0:
        adaptive_min = max(60.0, video_duration_s * 0.30)
        adaptive_silence = max(30.0, video_duration_s * 0.10)
        # Prendre le minimum entre les valeurs fixes et adaptatives
        _min_match = min(min_match_duration_s, adaptive_min)
        _silence   = min(silence_threshold_s, adaptive_silence)
        if _min_match < min_match_duration_s or _silence < silence_threshold_s:
            print(f"  [MATCH_END] Seuils adaptatifs : "
                  f"min_match={_min_match:.0f}s (fixe={min_match_duration_s:.0f}s) | "
                  f"silence={_silence:.0f}s (fixe={silence_threshold_s:.0f}s)")
    else:
        _min_match = min_match_duration_s
        _silence   = silence_threshold_s

    last_real_game_t = None
    last_checked_t   = 0.0

    # Seuil de vitesse ballon — fenêtre glissante 15s
    # Observation andrimont_2 :
    #   match      → vitesses 100–400 px/frame, fréquentes et soutenues
    #   post-match → vitesses 3–96 px/frame, sporadiques, médiane ~5
    # ball_speed > 3 frame par frame = activité ballon (pas match en cours)
    # On utilise une fenêtre glissante : game_active ssi
    #   median(ball_speed, 15s) >= SLIDING_MEDIAN_MIN  (ex: 20)
    #   OU high_speed_ratio(>30 px/frame, 15s) >= HIGH_SPEED_RATIO_MIN (ex: 0.15)
    # Le seuil instantané 3 est gardé comme pré-filtre uniquement.
    _BALL_SPEED_MIN        = 3.0    # pré-filtre instantané (inchangé)
    _SLIDING_WINDOW_S      = 15.0   # fenêtre glissante en secondes
    _SLIDING_MEDIAN_MIN    = 20.0   # médiane min pour "match en cours"
    # USE_RATIO30 : activez pour A/B test median seul vs median+ratio
    # Observation andrimont_2 : post-match génère ratio30=0.40–0.52 (rafales réelles
    # mais sporadiques), médiane reste 2–7 (< 20). Le ratio seul est insuffisant.
    # Gardé dans le code pour comparaison future sur d'autres vidéos.
    _USE_RATIO30           = False  # désactivé — median seul suffit sur cette vidéo
    _HIGH_SPEED_RATIO_MIN  = 0.15   # seuil ratio30 (ignoré si _USE_RATIO30=False)
    _HIGH_SPEED_THRESH     = 30.0   # seuil "vitesse élevée" (px/frame)
    _speed_window          = []     # liste (t, ball_speed) sur la fenêtre

    # Déterminer si le ballon est bien tracké dans cette vidéo (>5% des frames)
    _n_ball_frames = sum(
        1 for f in frames_data
        if (f.get("ball") or {}).get("x") is not None
        or (f.get("ball") or {}).get("center") is not None
    )
    _use_ball_criterion = _n_ball_frames > len(frames_data) * 0.05
    _tracked_ratio = _n_ball_frames / max(len(frames_data), 1)

    # ── DEBUG résumé avant boucle ─────────────────────────────────────────────
    print(f"  [MATCH_END DEBUG] tracked_ratio={_tracked_ratio:.2f} "
          f"({_n_ball_frames}/{len(frames_data)} frames) "
          f"use_ball={_use_ball_criterion} "
          f"speed_min={_BALL_SPEED_MIN}")

    _prev_ball_end = None   # (bx, by) frame précédente — pour calculer speed

    for fd in frames_data:
        t       = fd.get("frame", 0) / max(fps, 1)
        ball    = fd.get("ball") or {}
        players = fd.get("players") or []

        if _use_ball_criterion:
            # Extraire la position du ballon
            _center = ball.get("center")
            if _center and len(_center) >= 2 and _center[0] is not None:
                ball_x = float(_center[0])
                ball_y = float(_center[1])
            else:
                ball_x = ball.get("x")
                ball_y = ball.get("y")

            # Calculer la vitesse inter-frames (px/frame analysée)
            if ball_x is not None and ball_y is not None:
                if _prev_ball_end is not None:
                    import math as _m
                    ball_speed = _m.hypot(ball_x - _prev_ball_end[0],
                                          ball_y - _prev_ball_end[1])
                else:
                    ball_speed = 0.0
                _prev_ball_end = (ball_x, ball_y)
                game_active = ball_speed >= _BALL_SPEED_MIN
            else:
                ball_x = ball_y = ball_speed = None
                _prev_ball_end = None
                game_active = False
        else:
            ball_x = ball_y = ball_speed = None
            # Fallback : ≥4 joueurs avec équipe assignée (ballon non tracké)
            real_players = [
                p for p in players
                if p.get("team") is not None and p.get("team") != -1
            ]
            game_active = len(real_players) >= 4

        # ── Fenêtre glissante 15s ─────────────────────────────────────────────
        if _use_ball_criterion and ball_speed is not None:
            _speed_window.append((t, ball_speed))
            # Purger les frames hors fenêtre
            _speed_window = [(ts, sp) for ts, sp in _speed_window
                             if t - ts <= _SLIDING_WINDOW_S]
            # Calculer métriques agrégées
            if len(_speed_window) >= 3:
                import statistics as _stats
                _speeds = [sp for _, sp in _speed_window]
                _sliding_median     = _stats.median(_speeds)
                _high_speed_ratio   = sum(1 for sp in _speeds if sp >= _HIGH_SPEED_THRESH) / len(_speeds)
                # Décision finale : game_active ssi activité soutenue
                game_active = (
                    _sliding_median >= _SLIDING_MEDIAN_MIN
                    or (_USE_RATIO30 and _high_speed_ratio >= _HIGH_SPEED_RATIO_MIN)
                )
            # else : pas assez de données → garder game_active instantané
        # (si _use_ball_criterion=False, game_active vient du fallback joueurs)

        # ── DEBUG zone 420s–520s — toutes les 5s ─────────────────────────────
        if 420.0 <= t <= 520.0 and int(t) % 5 == 0:
            _spd_str = f"{ball_speed:.1f}" if ball_speed is not None else "None"
            if _use_ball_criterion and len(_speed_window) >= 3:
                _speeds_now = [sp for _, sp in _speed_window]
                import statistics as _stats2
                _med  = _stats2.median(_speeds_now)
                _ratio = sum(1 for sp in _speeds_now if sp >= _HIGH_SPEED_THRESH) / len(_speeds_now)
                _ball_extra = f"median15s={_med:.1f} ratio30={_ratio:.2f}"
            else:
                _ball_extra = "window<3"
            # Métriques joueurs — déjà disponibles dans frames_data
            _nb_total   = len(players)
            _nb_team0   = sum(1 for p in players if p.get("team") == 0)
            _nb_team1   = sum(1 for p in players if p.get("team") == 1)
            _nb_unknown = sum(1 for p in players if p.get("team") not in (0, 1))
            _unk_ratio  = _nb_unknown / max(_nb_total, 1)
            _lr = f"{last_real_game_t:.0f}s" if last_real_game_t else "None"
            print(f"  [MATCH_END DEBUG] t={t:.1f}s "
                  f"game_active={game_active} last_real={_lr} | "
                  f"{_ball_extra} | "
                  f"players={_nb_total} t0={_nb_team0} t1={_nb_team1} "
                  f"unk={_nb_unknown} unk_ratio={_unk_ratio:.2f}")

        if game_active:
            last_real_game_t = t

        last_checked_t = t

    # ── DEBUG résumé après boucle ─────────────────────────────────────────────
    print(f"  [MATCH_END DEBUG] last_real_game_t={last_real_game_t} "
          f"last_checked_t={last_checked_t:.1f}s "
          f"silence={last_checked_t - last_real_game_t:.1f}s"
          if last_real_game_t is not None else
          f"  [MATCH_END DEBUG] last_real_game_t=None "
          f"last_checked_t={last_checked_t:.1f}s")

    if last_real_game_t is None:
        return None

    # Vérifier que le match a duré assez longtemps
    if last_real_game_t < _min_match:
        print(f"  [MATCH_END] last_real_game={last_real_game_t:.0f}s < "
              f"min={_min_match:.0f}s → ignoré")
        return None

    # Vérifier qu'il y a un silence significatif après
    silence = last_checked_t - last_real_game_t
    if silence < _silence:
        print(f"  [MATCH_END] Pas de fin de match détectée → vidéo utilisée entièrement")
        return None

    print(f"  [MATCH_END] Fin détectée à t={last_real_game_t:.0f}s "
          f"(silence={silence:.0f}s > {_silence:.0f}s)")
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