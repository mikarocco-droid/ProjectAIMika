# analysis/events.py
# -*- coding: utf-8 -*-

from collections import deque
import math

from analysis.intelligence import (
    compute_xa,
    compute_danger,
    is_progressive,
    detect_build_up
)

# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def speed(a, b):
    return distance(a, b)

def _locked_team(player, team_map):
    """
    V5.2 : retourne l'equipe "verrouillee" (vote majoritaire sur toute la
    trajectoire du track_id, via team_map) plutot que la classification
    instantanee de CETTE frame (player.get("team"), potentiellement
    ambigue meme pour un joueur dont l'equipe est connue avec confiance
    par ailleurs). Repli sur la classification instantanee si team_map
    n'a pas d'entree pour ce track_id (ou si team_map absent, pour
    compatibilite retro).

    C'est le principe Veo : un numero/une equipe, une fois identifie,
    reste attache au joueur pendant tout le match - pas redemande a
    chaque evenement.
    """
    if player is None:
        return None
    if team_map:
        tid = str(player.get("id", ""))
        locked = team_map.get(tid)
        if locked is not None:
            return locked
    return player.get("team")

def get_closest_player(players, ball):
    closest  = None
    min_dist = float("inf")
    for p in players:
        d = distance(p["center"], ball["center"])
        if d < min_dist:
            min_dist = d
            closest  = p
    return closest, min_dist


# ─────────────────────────────────────────
# IDENTIFIER LE GARDIEN
# ─────────────────────────────────────────
def get_goalkeeper(players, frame_w):
    threshold = frame_w * 0.15
    gks = [
        p for p in players
        if p["center"][0] < threshold or p["center"][0] > frame_w - threshold
    ]
    if not gks:
        return None
    return min(gks, key=lambda p: min(p["center"][0], frame_w - p["center"][0]))


# ─────────────────────────────────────────
# DÉTECTION RELANCE À LA MAIN
# ─────────────────────────────────────────
def is_goalkeeper_throw(ball_pos, last_ball_pos, frame_w, frame_h, gk):
    if gk is None or last_ball_pos is None:
        return False
    bx, by = ball_pos
    lx, ly = last_ball_pos
    gx     = gk["center"][0]
    in_gk_zone = (lx < frame_w * 0.15 or lx > frame_w * 0.85)
    if not in_gk_zone:
        return False
    dx  = bx - lx
    dy  = by - ly
    spd = math.hypot(dx, dy)
    speed_ok = frame_w * 0.01 < spd < frame_w * 0.08
    moves_away_from_goal = (
        (gx < frame_w * 0.15 and dx > 0) or
        (gx > frame_w * 0.85 and dx < 0)
    )
    return speed_ok and moves_away_from_goal


# ─────────────────────────────────────────
# SHOT ZONES MULTI SPORT
# ─────────────────────────────────────────
def is_shot_zone(x, y, sport, shot_zones=None, frame_w=1280, frame_h=720):
    if shot_zones:
        axis  = shot_zones.get("axis", "x")
        hi    = shot_zones.get("threshold_hi", frame_w * 0.85)
        lo    = shot_zones.get("threshold_lo", frame_w * 0.15)
        y_min = shot_zones.get("y_min", 0)
        y_max = shot_zones.get("y_max", frame_h)
        if axis == "x":
            y_tol     = (y_max - y_min) * 0.50
            in_y_shot = (y_min - y_tol <= y <= y_max + y_tol)
            in_y_goal = (y_min - y_tol * 0.3 <= y <= y_max + y_tol * 0.3)
            in_zone   = (x > hi or x < lo) and in_y_shot
            in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
            return in_zone or in_goal, in_goal
        else:
            in_zone = (y < hi or y > lo)
            in_goal = (y < hi * 0.88 or y > lo * 1.12)
            return in_zone, in_goal

    if sport == "football":
        in_y_shot = (frame_h * 0.15 <= y <= frame_h * 0.90)
        in_y_goal = (frame_h * 0.20 <= y <= frame_h * 0.90)
        # Élargi 0.80 → 0.65 pour détecter tirs de l'extérieur de la surface
        in_zone   = (x > frame_w * 0.65 or x < frame_w * 0.35) and in_y_shot
        in_goal   = (x > frame_w * 0.88 or x < frame_w * 0.12) and in_y_goal
        return in_zone or in_goal, in_goal

    if sport == "basketball":
        in_zone = (x > frame_w * 0.88 or x < frame_w * 0.12)
        in_goal = (x > frame_w * 0.94 or x < frame_w * 0.06)
        return in_zone, in_goal

    if sport == "handball":
        in_y_shot = (frame_h * 0.15 <= y <= frame_h * 0.85)
        in_y_goal = (frame_h * 0.20 <= y <= frame_h * 0.85)
        in_zone   = (x > frame_w * 0.82 or x < frame_w * 0.18) and in_y_shot
        in_goal   = (x > frame_w * 0.90 or x < frame_w * 0.10) and in_y_goal
        return in_zone or in_goal, in_goal

    in_zone = (x > frame_w * 0.85 or x < frame_w * 0.15)
    in_goal = (x > frame_w * 0.90 or x < frame_w * 0.10)
    return in_zone or in_goal, in_goal


# ─────────────────────────────────────────
# FILTRE TIR PHYSIQUE
# Élimine : touches, passes, balles arrêtées
# Garde   : vrais tirs (vitesse + trajectoire + direction but)
# ─────────────────────────────────────────
def is_valid_shot(_bt, frame_w, frame_h):
    """
    Valide un tir par critères physiques du BallTracker.
    Version robuste : garde les tirs lents mais propres (penalties, reprises).

    Critères :
      - toward  : direction vers but (obligatoire)
      - speed   : > 25% frame_w/s  (souple pour tirs lents)
      - stability > 0.5 si rapide
      - stability > 0.75 fallback tirs lents mais droits

    Élimine : touches, centres, rebonds, balles arrêtées.
    Garde   : tirs puissants + tirs lents propres + penalties.
    """
    if _bt is None:
        return True  # pas de tracker → on laisse passer (compatibilité)

    speed     = _bt.get_speed_per_second()
    toward, _ = _bt.ball_buffer.toward_goal(frame_w, frame_h)
    stability = _bt.get_direction_stability(3)

    # direction vers but = condition obligatoire
    if not toward:
        return False

    speed_ok = speed > frame_w * 0.25

    # cas principal : tir rapide + trajectoire stable
    if speed_ok and stability > 0.5:
        return True

    # fallback : tir lent mais trajectoire très propre (penalty, reprise)
    if stability > 0.75:
        return True

    return False


# ─────────────────────────────────────────
# ON TARGET — détection tir cadré
#
# Un tir est "on_target" (cadré) si :
#   1. fast_shot_in_goal : tir rapide dans la zone de but
#   2. in_goal_zone : ballon dans la zone de but au moment du tir
#   3. trajectoire vers le centre du but (angle < seuil)
#   4. proximité des poteaux (poteau = tir cadré dévié)
# ─────────────────────────────────────────
def compute_on_target(
    x, y,
    ball_speed,
    ball_tracker=None,
    frame_w=1280,
    frame_h=720,
    in_goal_zone=False,
    fast_shot=False,
):
    """
    Retourne True si le tir est cadré ou proche du cadre.

    Critères :
      - fast_shot_in_goal → cadré ✅
      - in_goal_zone → ballon dans la zone de but → cadré ✅
      - trajectoire du ballon vers le centre du but (via BallTracker) ✅
      - ballon très proche des poteaux (< 8% frame_w) → poteau probable ✅
    """
    # Critère 1 — tir rapide dans la zone de but
    if fast_shot or in_goal_zone:
        return True

    # Critère 2 — trajectoire via BallTracker
    if ball_tracker is not None and hasattr(ball_tracker, "toward_goal"):
        toward, _ = ball_tracker.toward_goal(
            frame_w, frame_h, threshold=0.50
        )
        if toward:
            stability = ball_tracker.get_direction_stability(3)
            if stability > 0.55:
                return True

    # Critère 3 — proximité des poteaux
    # But gauche : x ≈ 0, but droit : x ≈ frame_w
    # Poteaux verticaux : y ≈ frame_h * 0.45 et y ≈ frame_h * 0.85
    goal_y_top    = frame_h * 0.45
    goal_y_bottom = frame_h * 0.85
    near_left_goal  = x < frame_w * 0.08
    near_right_goal = x > frame_w * 0.92
    near_post_y     = (goal_y_top - frame_h * 0.05 <= y <= goal_y_bottom + frame_h * 0.05)

    if (near_left_goal or near_right_goal) and near_post_y:
        return True

    return False


# ─────────────────────────────────────────
# CALCUL xG
# ─────────────────────────────────────────
def compute_xg(x, y, frame_w=1280, frame_h=720, learner=None):
    if learner and learner.xg_model.get("n_samples", 0) >= 20:
        return learner.predict_xg(x, y, frame_w, frame_h)
    dist_right = math.hypot(x - frame_w, y - frame_h / 2)
    dist_left  = math.hypot(x,           y - frame_h / 2)
    dist       = min(dist_right, dist_left)
    max_dist   = math.hypot(frame_w, frame_h / 2)
    xg         = round(max(0.0, 1.0 - dist / max_dist), 3)
    return min(xg, 0.5)


# ─────────────────────────────────────────
# INIT STATE
# ─────────────────────────────────────────
def init_state(learner=None, fps=25):
    thr = learner.get_thresholds() if learner else {}
    # V5.2 (14/09/2026) : fps reçu en paramètre (rythme EFFECTIF d'appel,
    # pas le fps natif codé en dur auparavant) - voir le commentaire au
    # point d'appel pour le détail du bug corrigé.
    shot_cd_frames = int(thr.get("shot_cooldown",   8.0) * fps)
    goal_cd_frames = int(thr.get("goal_cooldown", 150.0) * fps)
    ball_speed_min = thr.get("ball_speed_min",    0.02)
    player_near    = thr.get("player_near_goal",  0.15)
    goal_frames    = int(thr.get("goal_frames_min", 8))

    return {
        "last_player":              None,
        "last_ball_pos":            None,
        "last_team":                None,
        "sequence":                 deque(maxlen=30),
        "events_buffer":            deque(maxlen=10),
        "shot_cd":                  0,
        "goal_cd":                  0,
        "possession_time":          0,
        "team_possession":          {0: 0, 1: 0},
        "pressing":                 False,
        "turnover_window":          0,
        "ball_in_goal_zone":        0,
        "_goal_zone_speeds":        [],
        "_goal_zone_speeds_gap":    0,
        "_shot_cd_max":             shot_cd_frames,
        "_goal_cd_max":             goal_cd_frames,
        "_ball_speed_min":          ball_speed_min,
        "_player_near_goal":        player_near,
        "_goal_frames_min":         goal_frames,
        "_last_ball_interpolated":  False,
        "_last_dribble_time":       -999.0,
        "_dribble_cooldown":        1.5,
        "_gk_possession_frames":    0,
        "_gk_possession_min":       8,
        "_gk_holding_ball":         False,
        "_gk_release_cd":           0,
        "_gk_release_cd_max":       75,
        "_last_shot_x":             None,
        "_last_shot_y":             None,
        "_last_shot_time":          -999.0,
        "_shot_blocked_cd":         0,
        "_shot_blocked_cd_max":     40,
        "_last_event_types":        deque(maxlen=10),
        "_shot_candidate_xg":       None,
        "_shot_candidate_player":   None,
        "_shot_candidate_team":     None,
        # ── AJOUT v2 : buffer tirs récents pour filtre xG=0 ──
        "_recent_shots_buffer":     deque(maxlen=20),
        # V5.2 (14/09/2026) : buffer de vitesses BRUTES du ballon,
        # independant du filtre strict is_shot_candidate() (4 criteres :
        # vitesse+alignement+stabilite+acceleration). Necessaire pour le
        # fallback GOALZONE+vitesse (voir plus bas) - un penalty peut
        # echouer le filtre strict (stability~0 sur une fenetre trop
        # courte, ~0,3-0,5s de vol reel) tout en ayant ete genuinement
        # rapide juste avant. Ne remplace PAS is_shot_candidate(), sert
        # de preuve complementaire independante.
        "_recent_ball_speeds":      deque(maxlen=30),
        # V5.2 (17/09/2026) : buffer de POSITIONS brutes du ballon (x,y),
        # pour un 4e signal corroborant NON-BLOQUANT (stabilisation de
        # position = plusieurs lectures quasi identiques consecutives,
        # cohérent avec un ballon qui s'arrête dans les filets apres un
        # but). Trouve par comparaison de donnees reelles sur 3 cas
        # labellises avec certitude (2 faux positifs, 1 vrai penalty) :
        # les faux positifs (coup d'envoi, passes au milieu) ne montrent
        # JAMAIS de position repetee identique ; le vrai but, si. MAIS ce
        # n'est PAS une condition absolue (rebond sur barre qui ressort
        # aussitot = vrai but sans stabilisation) - sert uniquement a
        # AUGMENTER la confiance quand present, jamais a bloquer.
        "_recent_ball_positions":   deque(maxlen=15),
    }


# ─────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────
def _diag_kickoff_geometrique(players, frame_w, frame_h, current_time, state):
    """
    V5.2 (14/09/2026) - PROTOTYPE EXPERIMENTAL, DIAGNOSTIQUE UNIQUEMENT.
    N'intervient PAS dans la decision CONFIRME/REJETE existante - log
    seulement un signal, pour validation avant integration reelle.

    CORRECTIF IMPORTANT (suite a une remarque juste de l'utilisateur) :
    ce signal ne doit PAS se declencher en continu - un regroupement de
    joueurs pres du centre arrive aussi en jeu normal (melee au milieu,
    touche pres du centre), sans rapport avec un but. Le signal n'a de
    valeur que comme CORROBORATION d'un candidat deja suspect : ne
    s'active que dans une fenetre de surveillance ouverte par un rejet
    "pas de tir recent" (state["_kickoff_watch_until"], pose au moment
    du rejet, voir plus bas dans ce fichier), pas en continu.

    Detecte une formation "coup d'envoi" approximative : au moins 2
    joueurs regroupes pres du centre estime du terrain.

    LIMITE HONNETE ET IMPORTANTE : centre_x/y estimes comme le simple
    centre de l'image (frame_w/2, frame_h/2) - PAS une vraie calibration
    geometrique du terrain. Casse silencieusement si la camera n'est
    pas fixe/centree sur le terrain (zoom, pan, angle lateral prononce).
    """
    try:
        from config import DEBUG as _DBG_KO
    except ImportError:
        _DBG_KO = False
    if not _DBG_KO or not players:
        return

    # Ne rien faire hors fenêtre de surveillance active
    _watch_until = state.get("_kickoff_watch_until", 0) if state else 0
    if current_time > _watch_until:
        return

    centre_x = frame_w / 2
    centre_y = frame_h / 2
    rayon    = frame_w * 0.08  # ~ rayon du cercle central, approximatif

    proches = []
    for p in players:
        cx, cy = p.get("center", [None, None])
        if cx is None:
            continue
        d = math.hypot(cx - centre_x, cy - centre_y)
        if d < rayon:
            proches.append(p)

    if len(proches) >= 2:
        _origine = state.get("_kickoff_watch_origin", "?") if state else "?"
        print(f"  [KICKOFF_GEO] t={current_time:.1f}s ⚠️ CORROBORATION : "
              f"formation possible coup d'envoi ({len(proches)} joueurs "
              f"près du centre estimé), dans la fenêtre de surveillance "
              f"ouverte par le candidat rejeté à t={_origine}s — "
              f"signal EXPERIMENTAL, centre non calibré géométriquement")


def detect_events(
    players,
    ball,
    sport      = "football",
    state      = None,
    shot_zones = None,
    frame_w    = 1280,
    frame_h    = 720,
    learner    = None,
    fps        = 25,
    team_map   = None,
):
    if state is None:
        # V5.2 (14/09/2026) FIX CRITIQUE : init_state() utilisait fps=25
        # codé en dur pour convertir goal_cooldown/shot_cooldown (en
        # secondes) en nombre d'appels - mais le compteur decremente une
        # fois par appel ANALYSE (state["goal_cd"] -= 1 par appel a
        # detect_events()), pas une fois par frame native. Avec
        # frame_skip, le rythme reel d'appel est bien plus bas que 25fps
        # - meme bug que celui deja corrige dans player_reid.py (TTL) et
        # main.py (fps_effectif pour PlayerReID). Constate concretement :
        # cooldown de but voulu = 150s, reel mesure = 625s (x4.17 trop
        # long) - bloquait un vrai penalty a 382s a cause d'un faux
        # positif a 324s (voir ANALYSE_NOUVELLE_ARCHITECTURE_DETECTION.md
        # section 3.8). fps ici est le fps NATIF (necessaire pour
        # current_time = frame_id/fps, correct) - PAS le rythme reel
        # d'appel, qu'on calcule ici separement via FRAME_SKIP_EVERY.
        try:
            from config import FRAME_SKIP_EVERY as _FSE
        except ImportError:
            _FSE = 1
        _fps_effectif_pour_cooldowns = max(1.0, fps / max(1, _FSE))
        state = init_state(learner, fps=_fps_effectif_pour_cooldowns)

    events = []

    state["shot_cd"]          = max(0, state["shot_cd"] - 1)
    state["goal_cd"]          = max(0, state["goal_cd"] - 1)
    state["_gk_release_cd"]   = max(0, state["_gk_release_cd"] - 1)
    state["_shot_blocked_cd"] = max(0, state["_shot_blocked_cd"] - 1)

    if not players or not ball:
        state["ball_in_goal_zone"]     = 0
        state["_goal_zone_speeds"]     = []
        state["_goal_zone_speeds_gap"] = 0
        return events, state

    current_frame = ball.get("frame", 0) or 0
    current_time  = current_frame / fps

    # V5.2 (14/09/2026) : prototype expérimental, purement diagnostique
    # (voir _diag_kickoff_geometrique ci-dessus) - n'affecte aucune
    # decision existante.
    _diag_kickoff_geometrique(players, frame_w, frame_h, current_time, state)

    # ── POSSESSION ───────────────────────
    closest, dist = get_closest_player(players, ball)
    threshold     = frame_w * 0.06
    current       = closest if dist < threshold else None

    # V5.2 (14/09/2026) : diagnostic complet ballon/proximité joueur, pour
    # comprendre les cas ou tout le bloc tirs/buts est saute silencieusement
    # (current=None) - notamment penalties, ou aucun joueur n'est proche du
    # ballon pendant son vol vers le but. Aide aussi a distinguer un ballon
    # reellement au sol/en jeu d'un ballon porte dans les mains d'un joueur
    # (ex. installation du ballon sur le point de penalty), qui peut
    # produire des positions erratiques ressemblant a tort a un tir rapide.
    try:
        from config import DEBUG as _DBG_BALL
    except ImportError:
        _DBG_BALL = False
    if _DBG_BALL:
        _bx, _by = ball.get("center", [None, None])
        _interp = ball.get("interpolated", False)
        _conf   = ball.get("conf", None)
        print(f"  [BALL] t={current_time:.1f}s pos=({_bx},{_by}) "
              f"interp={_interp} conf={_conf} "
              f"dist_joueur_proche={dist:.0f}px seuil={threshold:.0f}px "
              f"current={'None' if current is None else current.get('id')} "
              f"n_joueurs={len(players)}")

    if current:
        team_key = _locked_team(current, team_map)
        events.append({
            "type":   "possession",
            "player": str(current["id"]),
            "team":   team_key,
            "x":      ball["center"][0],
            "y":      ball["center"][1]
        })
        state["possession_time"] += 1
        if team_key is not None:
            state["team_possession"][team_key] = \
                state["team_possession"].get(team_key, 0) + 1
        else:
            if players:
                team_counts = {}
                for p in players:
                    t = _locked_team(p, team_map)
                    if t is not None:
                        team_counts[t] = team_counts.get(t, 0) + 1
                if team_counts:
                    dominant = max(team_counts, key=team_counts.get)
                    state["team_possession"][dominant] = \
                        state["team_possession"].get(dominant, 0) + 0.5

    # ── POSSESSION GARDIEN ───────────────
    gk = get_goalkeeper(players, frame_w)

    if gk and state["last_ball_pos"]:
        ball_spd     = speed(state["last_ball_pos"], ball["center"])
        gk_near_ball = distance(gk["center"], ball["center"]) < frame_w * 0.06
        ball_slow    = ball_spd < frame_w * 0.025

        if gk_near_ball and ball_slow:
            state["_gk_possession_frames"] += 1
            if state["_gk_possession_frames"] >= state["_gk_possession_min"]:
                state["_gk_holding_ball"] = True
        else:
            if state["_gk_holding_ball"]:
                if is_goalkeeper_throw(
                    ball["center"], state["last_ball_pos"],
                    frame_w, frame_h, gk
                ):
                    state["_gk_release_cd"]       = state["_gk_release_cd_max"]
                    state["_gk_holding_ball"]      = False
                    state["_gk_possession_frames"] = 0
                    state["ball_in_goal_zone"]     = 0
                    state["_goal_zone_speeds"]     = []
                    state["_goal_zone_speeds_gap"] = 0
                    print(f"  GK throw détecté à t={current_time:.1f}s — goal bloqué 3s")
                else:
                    state["_gk_holding_ball"]      = False
                    state["_gk_possession_frames"] = 0
            else:
                state["_gk_possession_frames"] = max(
                    0, state["_gk_possession_frames"] - 1
                )
    elif state["_gk_holding_ball"]:
        state["_gk_holding_ball"]      = False
        state["_gk_possession_frames"] = 0

    # ── PRESSURE ─────────────────────────
    if current:
        opponents = [p for p in players if _locked_team(p, team_map) != _locked_team(current, team_map)]
        close_opp = sum(
            1 for o in opponents
            if distance(o["center"], current["center"]) < frame_w * 0.05
        )
        if close_opp >= 2:
            events.append({"type": "under_pressure", "player": str(current["id"]),
                            "team": _locked_team(current, team_map)})
            state["pressing"] = True

    # ── PROGRESSIVE RUN ──────────────────
    if state["last_ball_pos"] and current:
        if is_progressive(state["last_ball_pos"][0], ball["center"][0], frame_w):
            events.append({
                "type":   "progressive_run",
                "player": str(current["id"]),
                "team":   _locked_team(current, team_map),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })

    # ── PASS / INTERCEPTION ──────────────
    last = state["last_player"]
    if last and current and str(last["id"]) != str(current["id"]):
        same_team = _locked_team(last, team_map) == _locked_team(current, team_map)

        _pass_dist     = distance(last["center"], current["center"])
        _min_pass_dist = frame_w * 0.05

        if same_team and _pass_dist > _min_pass_dist:
            pass_event = {
                "type":   "pass",
                "from":   str(last["id"]),
                "to":     str(current["id"]),
                "team":   _locked_team(current, team_map),
                "x":      ball["center"][0],
                "y":      ball["center"][1],
                "xA":     0.0,
                "player": str(last["id"])
            }
            events.append(pass_event)
            state["events_buffer"].append(pass_event)
        elif not same_team:
            events.append({
                "type":   "interception",
                "player": str(current["id"]),
                "team":   _locked_team(current, team_map),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })
            state["turnover_window"] = 15

    # ── FAST BREAK ───────────────────────
    if state["turnover_window"] > 0 and state["last_ball_pos"]:
        v = speed(state["last_ball_pos"], ball["center"])
        if v > frame_w * 0.08:
            events.append({"type": "fast_break", "team": state.get("last_team")})
            state["turnover_window"] = 0
    state["turnover_window"] = max(0, state["turnover_window"] - 1)

    # ── DRIBBLE ──────────────────────────
    if current and state["last_ball_pos"]:
        ball_spd           = speed(state["last_ball_pos"], ball["center"])
        time_since_dribble = current_time - state.get("_last_dribble_time", -999)
        _dribble_dist      = distance(state["last_ball_pos"], ball["center"])

        if (_dribble_dist > 5
                and ball_spd > frame_w * 0.025
                and time_since_dribble >= state["_dribble_cooldown"]):
            events.append({
                "type":   "dribble",
                "player": str(current["id"]),
                "team":   _locked_team(current, team_map),
                "x":      ball["center"][0],
                "y":      ball["center"][1]
            })
            state["_last_dribble_time"] = current_time

    # ── BUILD UP ─────────────────────────
    state["sequence"].append(ball["center"])
    if detect_build_up(state["sequence"], frame_w):
        events.append({"type": "build_up", "team": state.get("last_team")})
        state["sequence"].clear()

    # ── LONG PASS ────────────────────────
    if state["last_ball_pos"]:
        if distance(state["last_ball_pos"], ball["center"]) > frame_w * 0.2:
            events.append({
                "type":   "long_pass",
                "player": str(current["id"]) if current else None,
                "team":   _locked_team(current, team_map) if current else None
            })

    # ── SHOTS / GOALS ────────────────────
    ball_speed_min  = state.get("_ball_speed_min",   0.02)
    player_near_pct = state.get("_player_near_goal", 0.15)
    goal_frames_min = state.get("_goal_frames_min",  8)
    shot_cd_max     = state.get("_shot_cd_max",      75)
    goal_cd_max     = state.get("_goal_cd_max",      3750)

    # V5.2 (14/09/2026) FIX MAJEUR : "if current:" retiré ici. Ce bloc
    # entier (détection tirs/buts) était conditionné à la présence d'un
    # joueur proche du ballon (current, seuil 6% largeur image) — brisant
    # la détection pendant tout le vol du ballon lors d'un penalty
    # (aucun joueur proche pendant 7-9s, confirmé empiriquement via les
    # logs [BALL] : current=None en continu de t=380.4s à t=387.9s,
    # exactement la fenêtre du tir + but réels). La détection (position/
    # vitesse du ballon) ne doit dépendre que du ballon lui-même ;
    # seule l'ATTRIBUTION au joueur (current["id"]) a besoin de current,
    # et gère maintenant explicitement le cas current=None (buteur
    # inconnu) au lieu de sauter toute la détection.
    if True:
        x, y = ball["center"]

        if learner and learner.is_fp_zone(x, y, frame_w, frame_h):
            pass
        else:
            is_shot_z, is_goal_zone = is_shot_zone(
                x, y, sport, shot_zones, frame_w, frame_h
            )

            ball_interpolated = ball.get("interpolated", False)

            _bt = ball.get("_tracker_ref")
            if _bt is not None and hasattr(_bt, "get_speed_per_second"):
                ball_speed = _bt.get_speed_per_second() / max(fps, 1)
            elif (state["last_ball_pos"]
                    and not ball_interpolated
                    and not state.get("_last_ball_interpolated", False)):
                ball_speed = speed(state["last_ball_pos"], ball["center"])
            else:
                ball_speed = frame_w * 0.025

            # V5.2 (14/09/2026) FIX : ball_speed (ci-dessus) est divisé par
            # fps pour les usages existants (shot_speed_ok, avg_speed<seuil,
            # etc.) - ne pas y toucher, deja calibre pour cette echelle
            # ailleurs dans cette fonction. Pour le buffer du fallback,
            # calcule une valeur SEPAREE, a la MEME echelle que celle
            # utilisee par is_shot_candidate() en interne
            # (self.ball_buffer.speed_px_per_sec(), sans division
            # supplementaire) - sinon le seuil frame_w*0.10 ne peut
            # jamais etre depasse (bug constate : fallback vitesse=False
            # systematiquement, alors que la vraie vitesse etait elevee).
            if _bt is not None and hasattr(_bt, "get_speed_per_second"):
                _vitesse_brute_fallback = _bt.get_speed_per_second()
            else:
                _vitesse_brute_fallback = ball_speed
            state["_recent_ball_speeds"].append({"time": current_time, "speed": _vitesse_brute_fallback})
            state["_recent_ball_positions"].append({"time": current_time, "x": x, "y": y})

            shot_speed_ok = (
                ball_speed > frame_w * ball_speed_min
                or (is_goal_zone and ball_speed > frame_w * 0.01)
            )

            # ── SHOT CONTRÉ ──────────────────────────────────────────────
            time_since_last_shot = current_time - state.get("_last_shot_time", -999)

            if (state["shot_cd"] > 0
                    and is_shot_z
                    and ball_speed > frame_w * 0.12
                    and state["_last_shot_x"] is not None
                    and time_since_last_shot < 8.0
                    and not state["_gk_holding_ball"]
                    and state["_gk_release_cd"] == 0):
                state["shot_cd"]          = min(state["shot_cd"], 15)
                state["_shot_blocked_cd"] = state["_shot_blocked_cd_max"]
                events.append({
                    "type":   "shot_blocked",
                    "player": str(current["id"]) if current else None,
                    "team":   _locked_team(current, team_map),
                    "x":      state["_last_shot_x"],
                    "y":      state["_last_shot_y"],
                    "danger": 5.0,
                })

            # ── SHOT RAPIDE EN LUCARNE ───────────────────────────────────
            fast_shot_in_goal = (
                is_goal_zone
                and ball_speed > frame_w * 0.07
                and state["shot_cd"] == 0
                and not ball_interpolated
                and not state.get("_last_ball_interpolated", False)
                and not state["_gk_holding_ball"]
                and state["_gk_release_cd"] == 0
            )

            # ── HELPER _register_shot ────────────────────────────────────
            def _register_shot(xg_val, source, on_target=False, fast=False):
                # Calcul on_target enrichi
                _on_target = on_target or compute_on_target(
                    x, y,
                    ball_speed  = ball_speed,
                    ball_tracker = _bt,
                    frame_w     = frame_w,
                    frame_h     = frame_h,
                    in_goal_zone = is_goal_zone,
                    fast_shot   = fast,
                )
                shot = {
                    "type":      "shot",
                    "player":    str(current["id"]) if current else None,
                    "team":      _locked_team(current, team_map),
                    "x":         x,
                    "y":         y,
                    "xg":        xg_val,
                    "time":      current_time,
                    "danger":    compute_danger({"type": "shot", "xg": xg_val}),
                    "on_target": _on_target,
                    "source":    source,
                }
                if fast:
                    shot["fast_shot"] = True
                events.append(shot)
                state["shot_cd"]         = shot_cd_max
                state["_last_shot_x"]    = x
                state["_last_shot_y"]    = y
                state["_last_shot_time"] = current_time
                # ── AJOUT v2 : alimenter le buffer tirs récents ──
                state["_recent_shots_buffer"].append({
                    "time": current_time,
                    "xg":   xg_val,
                })
                if _bt is not None and hasattr(_bt, "register_shot_candidate"):
                    _bt.register_shot_candidate(
                        x=x, y=y, t=current_time,
                        xg=xg_val,
                        player=str(current["id"]) if current else None,
                        team=_locked_team(current, team_map)
                    )
                if state["events_buffer"]:
                    state["events_buffer"][-1]["xA"] = compute_xa(
                        state["events_buffer"][-1], shot
                    )

            if is_shot_z and state["shot_cd"] == 0:
                if not (state["_gk_holding_ball"] or state["_gk_release_cd"] > 0):
                    # V9.7+ — is_shot_candidate() est le filtre principal
                    # vitesse > frame_w*1.5 px/s + accélération x1.4 + zone 25%
                    # Élimine les passes, centres, dégagements (~33 → ~8-12 tirs)
                    if (_bt is not None
                            and hasattr(_bt, "is_shot_candidate")
                            and not ball_interpolated):
                        if _bt.is_shot_candidate(frame_w, frame_h, current_time=current_time):
                            _register_shot(
                                compute_xg(x, y, frame_w, frame_h, learner),
                                source    = "events_standard",
                                on_target = fast_shot_in_goal
                            )
                    elif _bt is None and is_valid_shot(_bt, frame_w, frame_h):
                        # Fallback si pas de tracker
                        _register_shot(
                            compute_xg(x, y, frame_w, frame_h, learner),
                            source    = "events_standard_fallback",
                            on_target = fast_shot_in_goal
                        )

            elif fast_shot_in_goal and state["shot_cd"] > 0:
                if ball_speed > frame_w * 0.10:
                    _register_shot(
                        compute_xg(x, y, frame_w, frame_h, learner),
                        source    = "events_fast",
                        on_target = True,
                        fast      = True
                    )

            # ── UPGRADE #5 — tick shot candidate → lien tir→but ─────────
            _in_goal_zone_now = is_goal_zone
            if _bt is not None and hasattr(_bt, "tick_shot_candidate"):
                _goal_confirmed = _bt.tick_shot_candidate(
                    in_goal_zone = _in_goal_zone_now,
                    current_t    = current_time,
                    speed        = ball_speed,
                    frame_w      = frame_w
                )
                if _goal_confirmed and state["goal_cd"] == 0:
                    sc = _bt.get_shot_candidate()
                    if sc:
                        shot_dist    = sc.distance_to_goal(frame_w, frame_h)
                        max_dist     = math.hypot(frame_w, frame_h / 2)
                        xg_from_dist = round(max(0.01, min(0.5,
                            1.0 - shot_dist / max_dist)), 3)
                        final_xg = max(sc.xg, xg_from_dist)

                        # V5.2 (14/09/2026) : ce chemin (_bt.tick_shot_candidate)
                        # enregistrait un but SANS AUCUN PRINT - silencieux,
                        # avec un cooldown de goal_cd_max (3750 appels, ~10min+
                        # a notre rythme effectif) qui bloque ensuite tout le
                        # reste, y compris le chemin standard REJETÉ/CONFIRMÉ.
                        # Decouvert en observant un compteur ball_in_goal_zone
                        # depassant le seuil de x5+ sans jamais imprimer -
                        # ce but silencieux (potentiellement un faux positif
                        # anterieur) explique le blocage complet observe.
                        _joueur_str_bt = sc.player or (str(current["id"]) if current else "inconnu")
                        print(f"  ✅ goal CONFIRMÉ (chemin _bt/tick_shot_candidate) "
                              f"à t={current_time:.1f}s (xg={final_xg:.3f}, "
                              f"joueur={_joueur_str_bt}, cooldown={goal_cd_max} appels)")

                        events.append({
                            "type":        "goal",
                            "player":      sc.player or (str(current["id"]) if current else None),
                            "team":        sc.team   or _locked_team(current, team_map),
                            "x":           x,
                            "y":           y,
                            "xg":          final_xg,
                            "time":        current_time,
                            "shot_x":      sc.x,
                            "shot_y":      sc.y,
                            "shot_dist":   round(shot_dist, 1),
                            "danger":      compute_danger({"type": "goal"}),
                            "shot_linked": True,
                            "on_target":   True,
                            "source":      "events_bt",
                            "confidence":  0.85,
                        })
                        state["goal_cd"] = goal_cd_max
                        state["ball_in_goal_zone"]     = 0
                        state["_goal_zone_speeds"]     = []
                        state["_goal_zone_speeds_gap"] = 0
                        _bt.clear_shot_candidate()

            # ── GOAL — logique standard (fallback) ───────────────────────
            ball_is_real     = not ball_interpolated
            player_near_goal = dist < frame_w * player_near_pct
            gk_blocking_goal = state["_gk_holding_ball"] or state["_gk_release_cd"] > 0

            _goal_already_added = any(e.get("type") == "goal" for e in events)

            if is_goal_zone and not gk_blocking_goal and not _goal_already_added:
                if ball_is_real and (player_near_goal or state["ball_in_goal_zone"] >= 3):
                    state["ball_in_goal_zone"] += 1
                    state["_goal_zone_speeds"].append(ball_speed)
                    state["_goal_zone_speeds_gap"] = 0
                elif state["ball_in_goal_zone"] > 0:
                    state["_goal_zone_speeds_gap"] = \
                        state.get("_goal_zone_speeds_gap", 0) + 1
                    if state["_goal_zone_speeds_gap"] <= 4:
                        state["ball_in_goal_zone"] += 1
                        state["_goal_zone_speeds"].append(ball_speed)
                    else:
                        state["ball_in_goal_zone"]     = 0
                        state["_goal_zone_speeds"]     = []
                        state["_goal_zone_speeds_gap"] = 0
                else:
                    state["ball_in_goal_zone"]     = 0
                    state["_goal_zone_speeds"]     = []
                    state["_goal_zone_speeds_gap"] = 0
            else:
                # V5.2 (14/09/2026) FIX : les 3 lignes de reset
                # inconditionnel qui suivaient ici (a la meme
                # indentation que le "if" ci-dessous) ecrasaient
                # systematiquement la logique conditionnelle de
                # tolerance (qui ne remet a 0 que si avg_speed etait
                # trop eleve) - rendant le compteur ball_in_goal_zone
                # fragile a la moindre sortie, meme d'un seul appel,
                # de la zone de but. Constate concretement : 0 but
                # confirme sur 5 vrais buts connus dans une fenetre de
                # test (~61 min), aucun rejet local meme a proximite
                # de ces timestamps - le compteur ne semble jamais
                # atteindre le seuil requis. Repli desormais SEULEMENT
                # conditionnel, comme le code semblait l'avoir prevu.
                if state["ball_in_goal_zone"] > 0 and not gk_blocking_goal:
                    speeds = state["_goal_zone_speeds"]
                    if speeds:
                        avg_speed = sum(speeds) / len(speeds)
                        if avg_speed > frame_w * 0.04:
                            state["ball_in_goal_zone"]     = 0
                            state["_goal_zone_speeds"]     = []
                            state["_goal_zone_speeds_gap"] = 0

            goal_frames_threshold = goal_frames_min
            if state.get("_shot_blocked_cd", 0) > 0:
                goal_frames_threshold = max(4, goal_frames_min // 2)

            # V5.2 (14/09/2026) : log de progression du compteur
            # ball_in_goal_zone, à CHAQUE frame - pour voir son évolution
            # complète (montée, plateau, reset) plutôt que seulement le
            # résultat final (REJETÉ/CONFIRMÉ). Utile pour distinguer :
            # ballon jamais vu dans la zone (compteur toujours 0),
            # compteur qui progresse mais reset avant le seuil, ou
            # gardien qui bloque (gk_blocking_goal=True) au mauvais moment.
            try:
                from config import DEBUG as _DBG_GOALZONE
            except ImportError:
                _DBG_GOALZONE = False
            if _DBG_GOALZONE and (is_goal_zone or state["ball_in_goal_zone"] > 0):
                print(f"  [GOALZONE] t={current_time:.1f}s "
                      f"is_goal_zone={is_goal_zone} "
                      f"gk_blocking={gk_blocking_goal} "
                      f"compteur={state['ball_in_goal_zone']}/{goal_frames_threshold} "
                      f"goal_cd={state['goal_cd']} "
                      f"ball_is_real={ball_is_real} "
                      f"speed={ball_speed:.0f}")
                # V5.2 (14/09/2026) : log explicite quand le seuil est
                # atteint/dépassé mais bloqué par goal_cd>0 (cooldown
                # partagé entre le chemin standard et _bt/tick_shot_candidate,
                # cf. lignes 688 et 807 - meme cle state["goal_cd"]). Sans ce
                # log, ce blocage est invisible : le compteur peut depasser le
                # seuil x5+ sans jamais rien imprimer, comme observe a t=387s
                # (compteur=47/8) suite a un but confirme (peut-etre a tort)
                # a t=324.3s par le chemin _bt, verrouillant tout pour ~625s.
                if (state["ball_in_goal_zone"] >= goal_frames_threshold
                        and state["goal_cd"] > 0):
                    print(f"  [GOALZONE] ⚠️ seuil atteint mais BLOQUÉ par "
                          f"goal_cd={state['goal_cd']} (cooldown actif, "
                          f"encore ~{state['goal_cd']/6:.0f}s avant déblocage "
                          f"à ~6fps effectif)")

            # V5.2 (14/09/2026) : diagnostic explicite des 4 conditions,
            # au lieu de deviner laquelle bloque - decouvert que goal_cd
            # n'est PAS toujours la cause (goal_cd=0 confirme dans les
            # logs alors qu'aucune confirmation n'apparait). Affiche les
            # 4 valeurs exactes des qu'un seuil est atteint, pour savoir
            # PRECISEMENT laquelle empeche la confirmation.
            if _DBG_GOALZONE and state["ball_in_goal_zone"] >= goal_frames_threshold:
                _speeds_debug = state["_goal_zone_speeds"]
                _avg_debug = sum(_speeds_debug) / len(_speeds_debug) if _speeds_debug else 0
                print(f"  [GOALZONE_CHECK] t={current_time:.1f}s "
                      f"compteur_ok={state['ball_in_goal_zone'] >= goal_frames_threshold} "
                      f"goal_cd_ok={state['goal_cd'] == 0} (goal_cd={state['goal_cd']}) "
                      f"gk_ok={not gk_blocking_goal} "
                      f"goal_already_added_ok={not _goal_already_added} "
                      f"avg_speed={_avg_debug:.0f} (seuil={frame_w*0.09:.0f}) "
                      f"n_speeds={len(_speeds_debug)}")

            if (state["ball_in_goal_zone"] >= goal_frames_threshold
                    and state["goal_cd"] == 0
                    and not gk_blocking_goal
                    and not _goal_already_added):
                speeds    = state["_goal_zone_speeds"]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0

                if avg_speed < frame_w * 0.09:
                    # ── FIX v2 : filtre xG=0 — chercher tir récent ───────
                    # Un but sans tir associé dans les 5s = tracking artifact
                    _recent_shot_xg = 0.0
                    _recent_shot_player = None
                    _recent_shot_team   = None
                    for _s in reversed(list(state.get("_recent_shots_buffer", []))):
                        if 0 < current_time - _s["time"] <= 5.0:
                            _recent_shot_xg     = _s.get("xg", 0) or 0
                            _recent_shot_player = _s.get("player")
                            _recent_shot_team   = _s.get("team")
                            break

                    if _recent_shot_xg <= 0.01:
                        # V5.2 (14/09/2026) — FALLBACK GOALZONE+VITESSE
                        # (architecture proposee par l'utilisateur, suite
                        # au debug du penalty a t=382s) : is_shot_candidate()
                        # a un pouvoir de veto total sur la confirmation
                        # d'un but, alors que son role est de filtrer les
                        # TIRS, pas de decider des BUTS. Un penalty (vol
                        # ~0,3-0,5s) peut echouer son critere de stabilite
                        # (fenetre trop courte pour notre echantillonnage)
                        # tout en etant un vrai but. Ce fallback verifie
                        # 3 conditions INDEPENDANTES de is_shot_candidate,
                        # comme preuve alternative :
                        #   1. vitesse elevee recente (buffer brut, pas le
                        #      filtre strict)
                        #   2. trajectoire compatible (ballon reellement
                        #      suivi, pas interpole/perdu, pendant l'approche)
                        #   3. aucun evenement contradictoire (pas de
                        #      blocage gardien recent, pas de tir contre
                        #      juste avant - suggererait un arret, pas un but)
                        # N'affecte PAS is_shot_candidate() lui-meme - reste
                        # inchange, continue a filtrer les tirs normalement.
                        _vitesse_recente_elevee = any(
                            s["speed"] > frame_w * 0.10
                            for s in state["_recent_ball_speeds"]
                            if 0 < current_time - s["time"] <= 3.0
                        )
                        _lost_frames_ok = (_bt.lost_frames <= 2) if (_bt is not None and hasattr(_bt, "lost_frames")) else True
                        _trajectoire_compatible = ball_is_real and _lost_frames_ok
                        _pas_evenement_contradictoire = (
                            not gk_blocking_goal
                            and state.get("_shot_blocked_cd", 0) == 0
                        )
                        _fallback_ok = (_vitesse_recente_elevee
                                        and _trajectoire_compatible
                                        and _pas_evenement_contradictoire)

                        # V5.2 (17/09/2026) : 4e signal, NON-BLOQUANT —
                        # stabilisation de position (plusieurs lectures
                        # quasi identiques consecutives), cohérent avec un
                        # ballon qui s'arrete dans les filets. Trouve par
                        # comparaison de donnees reelles sur 3 cas
                        # labellises avec certitude (2 faux positifs -
                        # coup d'envoi, passes au milieu -, 1 vrai penalty)
                        # : les faux positifs ne montrent JAMAIS de
                        # position repetee identique ; le vrai but, si.
                        # PAS une condition absolue (un tir sur la barre
                        # qui ressort aussitot est un vrai but sans
                        # stabilisation) - sert UNIQUEMENT a augmenter la
                        # confiance quand present, jamais a bloquer le
                        # fallback si absent.
                        _positions_recentes = [
                            p for p in state["_recent_ball_positions"]
                            if 0 <= current_time - p["time"] <= 1.0
                        ][-4:]
                        _position_stabilisee = False
                        if len(_positions_recentes) >= 3:
                            _px = [p["x"] for p in _positions_recentes]
                            _py = [p["y"] for p in _positions_recentes]
                            _position_stabilisee = (max(_px) - min(_px) <= 5
                                                     and max(_py) - min(_py) <= 5)

                        if _fallback_ok:
                            _joueur_fb = str(current["id"]) if current else None
                            _confidence_fb = 0.7 if _position_stabilisee else 0.5
                            print(f"  ✅ goal CONFIRMÉ (GOALZONE_SPEED_FALLBACK) "
                                  f"à t={current_time:.1f}s — pas de tir lié "
                                  f"(is_shot_candidate a echoue, probable fenetre "
                                  f"trop courte), mais vitesse recente elevee + "
                                  f"trajectoire compatible + aucun evenement "
                                  f"contradictoire. position_stabilisée={_position_stabilisee} "
                                  f"joueur={_joueur_fb or 'inconnu'}")
                            events.append({
                                "type":        "goal",
                                "player":      _joueur_fb,
                                "team":        _locked_team(current, team_map),
                                "x":           x,
                                "y":           y,
                                "xg":          0.3,  # valeur conservatrice, non calibrée
                                "time":        current_time,
                                "danger":      compute_danger({"type": "goal"}),
                                "shot_linked": False,
                                "on_target":   True,
                                "source":      "goalzone_speed_fallback",
                                "position_stabilisee": _position_stabilisee,
                                "confidence":  _confidence_fb,  # 0.7 si stabilisée, 0.5 sinon - à valider
                            })
                            state["goal_cd"] = goal_cd_max
                            state["_kickoff_watch_until"]  = current_time + 30.0
                            state["_kickoff_watch_origin"] = f"{current_time:.1f}"
                        else:
                            # Pas de tir récent avec xG > 0, ET fallback
                            # non satisfait → faux positif
                            # V5.2 (14/09/2026) : détail du buffer de tirs
                            # récents, pour distinguer précisément POURQUOI
                            # aucun tir n'a qualifié - buffer vide (aucun tir
                            # jamais enregistré), tirs trop vieux (>5s), ou
                            # tirs présents mais xG trop faible (<=0.01).
                            _buffer = list(state.get("_recent_shots_buffer", []))
                            if not _buffer:
                                _detail_buffer = "buffer vide (aucun tir jamais enregistré)"
                            else:
                                _ages = [f"{current_time - s['time']:.1f}s(xg={s.get('xg',0):.2f})"
                                         for s in reversed(_buffer[-5:])]
                                _detail_buffer = f"{len(_buffer)} tir(s) en mémoire, plus récents : {', '.join(_ages)}"
                            print(f"  goal REJETÉ à t={current_time:.1f}s "
                                  f"(xG=0.000 — pas de tir récent → faux positif, "
                                  f"fallback vitesse={_vitesse_recente_elevee} "
                                  f"trajectoire={_trajectoire_compatible} "
                                  f"pas_contradictoire={_pas_evenement_contradictoire} "
                                  f"position_stabilisée={_position_stabilisee}) "
                                  f"[{_detail_buffer}]")
                            # V5.2 (14/09/2026) : ouvre la fenêtre de
                            # surveillance pour le signal experimental
                            # _diag_kickoff_geometrique - 30s, temps
                            # plausible pour celebration + recuperation du
                            # ballon + replacement + reprise au centre.
                            state["_kickoff_watch_until"]  = current_time + 30.0
                            state["_kickoff_watch_origin"] = f"{current_time:.1f}"

                        state["ball_in_goal_zone"]     = 0
                        state["_goal_zone_speeds"]     = []
                        state["_goal_zone_speeds_gap"] = 0
                    else:
                        # Tir récent confirmé → but valide
                        _joueur_str = str(current["id"]) if current else "inconnu (aucun joueur proche)"
                        print(f"  ✅ goal CONFIRMÉ à t={current_time:.1f}s "
                              f"(xG_tir_lié={_recent_shot_xg:.3f}, "
                              f"joueur={_joueur_str}, team={_locked_team(current, team_map)})")
                        events.append({
                            "type":        "goal",
                            "player":      str(current["id"]) if current else None,
                            "team":        _locked_team(current, team_map),
                            "x":           x,
                            "y":           y,
                            "xg":          _recent_shot_xg,
                            "time":        current_time,
                            "danger":      compute_danger({"type": "goal"}),
                            "shot_linked": True,
                            "on_target":   True,
                            "source":      "events_standard",
                            "confidence":  min(0.5 + _recent_shot_xg, 0.90),
                        })
                        state["goal_cd"] = goal_cd_max
                else:
                    print(f"  goal rejeté vitesse_avg={avg_speed:.0f}px "
                          f"> seuil={frame_w * 0.09:.0f}px (dégagement)")

                state["ball_in_goal_zone"]     = 0
                state["_goal_zone_speeds"]     = []
                state["_goal_zone_speeds_gap"] = 0

    else:
        state["ball_in_goal_zone"]     = 0
        state["_goal_zone_speeds"]     = []
        state["_goal_zone_speeds_gap"] = 0

    # ── UPDATE STATE ─────────────────────
    state["last_player"]             = current
    state["last_ball_pos"]           = ball["center"]
    state["_last_ball_interpolated"] = ball.get("interpolated", False)
    state["last_team"]               = _locked_team(current, team_map) if current else None

    return events, state


# ─────────────────────────────────────────
# MATCH PROCESSOR
# ─────────────────────────────────────────
def process_match(frames_data, sport="football", shot_zones=None, learner=None, team_map=None):
    state      = None
    all_events = []
    fps        = frames_data[0].get("fps", 25) if frames_data else 25

    for frame in frames_data:
        events, state = detect_events(
            players    = frame.get("players"),
            ball       = frame.get("ball"),
            sport      = sport,
            state      = state,
            shot_zones = shot_zones,
            frame_w    = frame.get("frame_w", 1280),
            frame_h    = frame.get("frame_h", 720),
            learner    = learner,
            fps        = fps,
            team_map   = team_map,
        )
        for e in events:
            e["frame"] = frame.get("frame")
        all_events.extend(events)

    return all_events
    
def process_events(raw_events):
    clean_events = []

    for event in raw_events:

        if event.get("type") == "goal":

            # 🔥 FIX CRITIQUE
            if event.get("xg", 0) == 0.0 and event.get("source") != "goal_posthoc":
                print(f"[events] rejet but sans xG à {event.get('time')}")
                continue

        clean_events.append(event)

    return clean_events