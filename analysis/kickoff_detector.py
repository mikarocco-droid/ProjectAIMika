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
def find_kickoff_offset(events, video_duration_s, frames_data=None, fps=25, video_path=None):
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
            print(f"  [KO_PROBE] t={t:6.1f}s sep={sep:.2f} n={n:2d} ball={ball_str}")

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