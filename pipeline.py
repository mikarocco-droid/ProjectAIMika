# pipeline.py
# -*- coding: utf-8 -*-
#
# Chef d'orchestre du pipeline.
#
# Règles de responsabilité :
#   - deduplicate_goals   : défini et appelé ICI uniquement
#   - goal_cooldown       : calculé ICI, passé à post_process_events
#   - position_threshold  : calculé ICI (GOAL_PCT = 0.05), passé à post_process_events
#   - post_processing.py  : module de transformation pure, sans décision métier

import os
import sys
import json

# Garantir que le repo est dans sys.path — nécessaire sur Kaggle
_repo = os.path.dirname(os.path.abspath(__file__))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import config
from main import process_video
from analytics.stats import compute_stats, compute_possession_from_stats
from analytics.heatmap import generate_all_heatmaps
from analytics.advanced import (
    build_pass_network,
    detect_offside,
    compute_team_dominance,
    compute_action_score,
    extract_pass_sequences
)
from video_utils import create_highlights, create_highlight_reel
from analysis.events import detect_events
from analysis.tactical import (
    tactical_report,
    assign_teams,
    detect_formation,
    detect_pressing,
    detect_phases
)
from analysis.highlight_ranker import rank_highlights
from analysis.player_rating import compute_player_ratings, get_mvp, tag_key_passes
from analysis.match_story import generate_match_story
from ai.commentary import generate_commentary
from ai.learning import cluster_actions, learn_action_importance, detect_key_moments
from sports.config import get_sport_config, compute_xg_sport, get_owner_confidence_min
from analysis.event_validator import detect_real_shots
from analysis.context_engine import ContextEngine
from analysis.player_identity import resolve_player_identities, get_player_label
from analysis.goal_posthoc    import detect_fast_goals_from_ball
from analysis.terminal_events import detect_terminal_events, build_candidate_windows
from analysis.kickoff_detector import find_kickoff_offset, apply_kickoff_offset, apply_kickoff_offset_frames, reset_pre_kickoff_state, find_match_end, apply_match_end

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTE ZONE DE BUT — source unique
# Alignée avec goal_posthoc.py (GOAL_PCT = 0.05)
# ─────────────────────────────────────────────────────────────────────────────
GOAL_POSITION_THRESHOLD = 0.05   # 5% de chaque côté = zone filet réelle


# ─────────────────────────────────────────
# NORMALIZE HIGHLIGHTS
# ─────────────────────────────────────────
def normalize_highlights(highlights, mode="match"):
    fixed = []
    for h in highlights:
        ts = h.get("time_start") or h.get("timestamp_debut") or 0
        h["time_start"] = float(ts)

        te = h.get("time_end") or h.get("timestamp_fin") or 0
        h["time_end"] = float(te)

        if h["time_end"] <= h["time_start"]:
            h["time_end"] = h["time_start"] + 3.0

        if "main_type" not in h:
            h["main_type"] = h.get("type", "action")

        TYPE_MAP = {
            "arrêt_gardien":  "shot",
            "arret_gardien":  "shot",
            "shot_missed":    "shot",
            "shot_on_target": "shot",
            "phase de jeu":   "action",
            "phase_de_jeu":   "action",
            "tir":            "shot",
            "but":            "goal",
            "none":           "action",
            "corner":         "action",
            "touche":         "action"
        }
        t = h["main_type"].lower().strip()
        h["main_type"] = TYPE_MAP.get(t, t)

        if "score" not in h:
            h["score"] = 1.0

        fixed.append(h)

    if mode == "player":
        allowed = ["goal", "shot", "score", "dribble",
                   "progressive_run", "interception", "fast_break", "big_chance", "assist"]
    else:
        allowed = ["goal", "shot", "score", "big_chance", "assist"]

    fixed = [h for h in fixed if h["main_type"] in allowed]
    return fixed


# ─────────────────────────────────────────
# SANITIZE JSON
# ─────────────────────────────────────────
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): sanitize_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    else:
        return obj


# ─────────────────────────────────────────
# MATCH SUMMARY
# ─────────────────────────────────────────
def compute_match_summary(events, stats, total_frames=0, fps=25):
    total_xg     = sum(e.get("xg", 0) for e in events if e.get("type") == "shot")
    duration_sec = int(total_frames / fps) if fps > 0 else 0
    minutes      = duration_sec // 60
    seconds      = duration_sec % 60

    return {
        "total_events":     len(events),
        "goals":            sum(1 for e in events if e.get("type") in ["goal", "score"]),
        "shots":            sum(1 for e in events if e.get("type") == "shot"),
        "passes":           sum(1 for e in events if e.get("type") == "pass"),
        "interceptions":    sum(1 for e in events if e.get("type") == "interception"),
        "dribbles":         sum(1 for e in events if e.get("type") == "dribble"),
        "progressive_runs": sum(1 for e in events if e.get("type") == "progressive_run"),
        "build_up_plays":   sum(1 for e in events if e.get("type") == "build_up_play"),
        "total_xg":         round(total_xg, 2),
        "players":          len(stats),
        "duration":         f"{minutes:02d}:{seconds:02d}",
        "total_frames":     total_frames,
        "fps":              fps
    }


# ─────────────────────────────────────────
# CALLBACK PROGRESSION
# ─────────────────────────────────────────
def make_progress_callback(analysis_id=None):
    if analysis_id is None:
        return lambda pct: print(f"  {pct}%", end="\r")

    def callback(pct):
        try:
            from app import app, db, Analysis
            with app.app_context():
                a = db.session.get(Analysis, analysis_id)
                if a:
                    a.progress = pct
                    db.session.commit()
        except Exception as e:
            print(f"Progression DB error: {e}")

    return callback


# ─────────────────────────────────────────
# FIX MVP
# ─────────────────────────────────────────
def resolve_mvp_label(mvp, stats, jersey_map):
    if mvp is None:
        return None
    pid = str(mvp[0]) if isinstance(mvp, (list, tuple)) else str(mvp)
    if pid in stats and stats[pid].get("label"):
        return stats[pid]["label"]
    return get_player_label(pid, jersey_map)


# ─────────────────────────────────────────────────────────────────────────────
# DÉDUPLICATION BUTS — SOURCE UNIQUE (définie ici, nulle part ailleurs)
# ─────────────────────────────────────────────────────────────────────────────
def deduplicate_goals(events, window=3.0):
    """
    Dans une fenêtre de `window` secondes, garde le but avec la meilleure
    confidence. Appelé plusieurs fois dans le pipeline à des moments précis.

    Priorité : gemini_validated=True > confidence > score
    """
    goals  = sorted(
        [e for e in events if e.get("type") in ("goal", "score")],
        key=lambda x: x.get("time", 0)
    )
    others = [e for e in events if e.get("type") not in ("goal", "score")]

    kept = []
    for g in goals:
        if kept and abs(g["time"] - kept[-1]["time"]) < window:
            # Priorité : gemini_validated > confidence > score
            cur_key  = (1 if kept[-1].get("gemini_validated") else 0,
                        kept[-1].get("confidence", 0),
                        kept[-1].get("score", 0))
            new_key  = (1 if g.get("gemini_validated") else 0,
                        g.get("confidence", 0),
                        g.get("score", 0))
            if new_key > cur_key:
                kept[-1] = g
        else:
            kept.append(g)

    return sorted(others + kept, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────
# MODE DEBUG — valeur centralisée dans config.py
# Rechargé depuis os.environ pour fonctionner sur Kaggle
# ─────────────────────────────────────────
import os as _os_debug
try:
    from config import DEBUG as _DEBUG_CONFIG
    DEBUG = _os_debug.getenv("DEBUG", str(_DEBUG_CONFIG)).lower() == "true"
except ImportError:
    DEBUG = _os_debug.getenv("DEBUG", "false").lower() == "true"

# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_pipeline(
    video_path,
    sport             = "football",
    output_dir        = "outputs",
    analysis_id       = None,
    save_annotated    = False,
    plan              = "free",
    mode              = "match",
    player_id         = None,
    goals_real        = None,
    use_coarse_scan   = False,   # V2 : scan léger avant deep analysis
    team_names        = None,    # {"team_0": "RSC Stavelot B", "team_1": "FC Liège"}
    user_team_id      = None,    # 0 ou 1 — équipe de l'utilisateur
    player_position   = None,    # "Gardien", "Attaquant", etc. — pour mode player
):
    os.makedirs(output_dir, exist_ok=True)
    progress = make_progress_callback(analysis_id)

    # ── Auto frame_skip selon FPS réel de la vidéo ───────────────────────────
    # Garantit ~6.25 fps analysés quelle que soit la source (25/30/50/60fps)
    # Override LOCAL uniquement — n'affecte pas les runs parallèles
    import cv2 as _cv2 ; import re as _re_fps
    try:
        _cap     = _cv2.VideoCapture(video_path)
        _src_fps = _cap.get(_cv2.CAP_PROP_FPS)
        _cap.release()
        if not _src_fps or _src_fps < 10:
            _src_fps = 25.0
        _auto_skip = max(1, round(_src_fps / 6.25))
        _cfg_skip  = int(os.environ.get("FRAME_SKIP_EVERY",
                         getattr(config, "FRAME_SKIP_EVERY", 4)))
        if _auto_skip != _cfg_skip:
            print(f"  [AUTO SKIP] FPS={_src_fps:.1f} → "
                  f"FRAME_SKIP_EVERY={_auto_skip} "
                  f"(config={_cfg_skip}, "
                  f"{_src_fps/_auto_skip:.1f} fps analysés)")
            # Injecter dans l'environnement pour que config.py le relise
            os.environ["FRAME_SKIP_EVERY"] = str(_auto_skip)
            config.FRAME_SKIP_EVERY = _auto_skip
        else:
            print(f"  [AUTO SKIP] FPS={_src_fps:.1f} → "
                  f"FRAME_SKIP_EVERY={_auto_skip} (inchangé)")
    except Exception as _eskip:
        print(f"  [AUTO SKIP] Ignoré : {_eskip}")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\nPIPELINE START - {sport.upper()} | mode={mode}")

    # Initialisations globales
    _team_bootstrapper = None

    # ─────────────────────────────────────────
    # INIT LEARNING MODEL
    # ─────────────────────────────────────────
    learner = None
    try:
        from ai.learning_model import MatchLearner
        learner = MatchLearner(
            sport    = sport,
            base_dir = os.path.join(output_dir, "..", "learning")
        )
        ls = learner.stats()
        xg_adv_status = "ok" if ls.get("xg_advanced_ready") else str(ls.get("xg_advanced_samples", 0)) + "/50"
        print(f"  Learning : {ls['n_matches']} matchs | "
              f"{ls['n_events']} events | "
              f"xG n={ls['xg_samples']} | "
              f"xG avancé={xg_adv_status} | "
              f"FP zones={ls['fp_zones']} | "
              f"spatial={ls['spatial_max_dist']:.0f}px")
    except Exception as e:
        print(f"  Learning model ignoré : {e}")

    # ─────────────────────────────────────────
    # 0. DÉTECTION SPORT + CALIBRATION
    # ─────────────────────────────────────────
    print("Step 0 : Détection sport + Calibration...")
    calib      = None
    shot_zones = None

    try:
        from ai.sport_detector import detect_sport
        sport_detected = detect_sport(video_path, fallback=sport)
        if sport_detected != sport:
            print(f"  Sport détecté : {sport_detected} (demandé : {sport})")
            sport = sport_detected
    except Exception as e:
        print(f"  Sport detector ignoré : {e}")

    try:
        from vision.calibration import calibrate
        calib = calibrate(video_path, sport)
        if calib and calib.get("play_zone"):
            from vision.detector import PLAY_ZONES
            PLAY_ZONES[sport] = calib["play_zone"]
        if calib and calib.get("shot_zones"):
            shot_zones = calib["shot_zones"]
        print(f"  OK calibration : {calib.get('camera_angle','?')} | {calib.get('pitch_color','?')}")
    except Exception as e:
        print(f"  Calibration ignoree : {e}")

    # Détection automatique des buts (poteaux blancs)
    _goal_box = None
    try:
        from vision.detect_goal_box import detect_goal_box
        _goal_box = detect_goal_box(video_path, n_frames=60, fps=25)  # fps fixe — video pas encore analysée
        if _goal_box and _goal_box.get("method") == "vision":
            print(f"  [GOAL_BOX] Poteaux détectés via vision")
        else:
            print(f"  [GOAL_BOX] Fallback zones fixes")
    except Exception as _e:
        print(f"  [GOAL_BOX] Non disponible : {_e}")

    # ─────────────────────────────────────────
    # 1. TRACKING + EVENTS
    # ─────────────────────────────────────────
    # ─────────────────────────────────────────
    # STEP 0b — COARSE SCAN (optionnel)
    # ─────────────────────────────────────────
    candidate_segments = None
    coarse_stats       = {}

    if use_coarse_scan and mode == "match":
        try:
            from coarse_scan import run_coarse_scan
            candidate_segments, coarse_stats = run_coarse_scan(
                video_path = video_path,
                sport      = sport,
            )
            if candidate_segments:
                coverage = coarse_stats.get("coverage_pct", 100)
                print(f"  Coarse scan : {len(candidate_segments)} segments "
                      f"({coverage:.1f}% du match)")
        except Exception as e:
            print(f"  Coarse scan ignoré : {e}")
            candidate_segments = None

    print("Step 1 : Tracking + Events...")

    annotated_path = os.path.join(output_dir, "annotated.mp4") \
        if save_annotated else None

    if candidate_segments:
        # ── VRAI 2-PASS : extraction clips + analyse fine ────────────────
        print("Step 1 : Deep analysis sur segments chauds...")
        try:
            from segment_extractor import extract_segments, analyze_segments, cleanup_segments

            _seg_dir    = os.path.join(output_dir, "segments_tmp")
            _deep_skip  = coarse_stats.get("deep_frame_skip", 2)
            _deep_batch = coarse_stats.get("deep_batch_size", 8)
            _deep_imgsz = coarse_stats.get("deep_imgsz", 960)

            print(f"  Deep pass : imgsz={_deep_imgsz} | "
                  f"frame_skip={_deep_skip} | "
                  f"{len(candidate_segments)} segments")

            # 1. Extraire les clips via ffmpeg
            segment_clips = extract_segments(
                video_path  = video_path,
                segments    = candidate_segments,
                output_dir  = _seg_dir,
            )

            # 2. Analyser uniquement les clips
            events, _jersey, fps, total_frames, frames_data = analyze_segments(
                segment_clips = segment_clips,
                sport         = sport,
                shot_zones    = shot_zones,
                frame_skip    = _deep_skip,
                batch_size    = _deep_batch,
                imgsz         = _deep_imgsz,
            )

            # jersey_map depuis le coarse (déjà disponible)
            jersey_map = _jersey if _jersey else {}

            # 3. Cleanup clips temporaires
            cleanup_segments(segment_clips)

            coarse_stats["deep_frames_analyzed"] = len(frames_data)
            coarse_stats["deep_events_found"]    = len(events)

        except Exception as _e:
            print(f"  ⚠️  Segment extractor échoué ({_e}) — fallback full pass")
            candidate_segments = None  # fallback ci-dessous

    if not candidate_segments:
        # ── FALLBACK : process_video() complet ───────────────────────────
        print("Step 1 : Tracking + Events (full pass)...")
        events, jersey_map, fps, total_frames, frames_data = process_video(
            video_path        = video_path,
            sport             = sport,
            progress_callback = progress,
            save_annotated    = save_annotated,
            annotated_path    = annotated_path,
            shot_zones        = shot_zones,
            return_frames     = True,
        )
    print(f"  RAW {len(events)} events | {len(jersey_map)} maillots")

    # ── DÉTECTION COUP D'ENVOI ────────────────────────────────────────────────
    # Le coup d'envoi est détecté par terminal_events pendant le tracking
    # (event type="kickoff" : ballon au rond central + joueurs symétriques).
    # On cherche le premier kickoff pour corriger tous les timestamps.
    # ── DÉTECTION COUP D'ENVOI ────────────────────────────────────────────────
    # Le coup d'envoi est détecté par terminal_events pendant le tracking
    # (event type="kickoff" : ballon au rond central + joueurs symétriques).
    # On cherche le premier kickoff pour corriger tous les timestamps.
    _video_duration_s = total_frames / max(fps, 1)

    # Callback Gemini pour vérification visuelle des candidats kickoff
    def _gemini_verify_kickoff(vpath, candidate_t, offsets=(-2.0, 0.0, 2.0, 5.0)):
        """
        Extrait 4 frames autour de candidate_t et demande à Gemini si c'est
        un coup d'envoi. La frame à +5s est clé : après un vrai KO le jeu
        est lancé (joueurs en course). Après un simple placement les joueurs
        marchent encore vers leurs positions.
        Retourne (is_kickoff: bool, confidence: float).
        """
        try:
            import cv2 as _cv2
            import base64 as _b64
            import json as _json
            import re as _re
            import google.generativeai as _genai
            cap = _cv2.VideoCapture(vpath)
            if not cap.isOpened():
                return False, 0.0
            real_fps = cap.get(_cv2.CAP_PROP_FPS) or fps
            frames_b64 = []
            for off in offsets:
                t_target = max(0.0, candidate_t + off)
                cap.set(_cv2.CAP_PROP_POS_FRAMES, int(t_target * real_fps))
                ret, frame = cap.read()
                if not ret:
                    continue
                _, buf = _cv2.imencode(".jpg", frame,
                                       [_cv2.IMWRITE_JPEG_QUALITY, 75])
                frames_b64.append(_b64.b64encode(buf).decode())
            cap.release()
            if not frames_b64:
                return False, 0.0

            model = _genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "You are analyzing a football (soccer) match video. "
                "I will show you 4 frames taken at -2s, 0s, +2s and +5s around a specific moment. "
                "Your task: determine if this moment is a kickoff — "
                "the moment the match starts or restarts after a goal. "
                "A kickoff sequence looks like: "
                "at 0s the ball is placed at the center circle with 1 or 2 players near it, "
                "referee nearby, all other players on their halves — "
                "AND at +5s the game is clearly in open play (players running, ball moving away from center). "
                "Answer NO if: at +5s players are still walking to their positions (pre-kickoff placement), "
                "or if it is a team huddle/cheer, or normal open play without a restart. "
                "The key signal is the +5s frame: real kickoff = active open play; "
                "false positive = players still settling into position. "
                "Respond ONLY with valid JSON: "
                "{\"is_kickoff\": true/false, \"confidence\": 0.0-1.0, "
                "\"reason\": \"one short sentence\"}"
            )
            parts = [prompt]
            for b64 in frames_b64:
                img_data = _b64.b64decode(b64)
                parts.append({"mime_type": "image/jpeg", "data": img_data})
            response = model.generate_content(parts)

            # Parser le JSON de la réponse
            raw = response.text.strip()
            # Extraire le JSON même s'il est entouré de backticks
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            result = None
            if m:
                try:
                    result = _json.loads(m.group())
                except Exception:
                    pass
            if result is None:
                return False, 0.0
            is_ko  = bool(result.get("is_kickoff", False))
            conf   = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")
            print(f"    [KICKOFF GEMINI DETAIL] is_kickoff={is_ko} conf={conf:.2f} reason={reason}")
            return is_ko, conf
        except Exception as e:
            print(f"  [KICKOFF GEMINI ERROR] {e}")
            return False, 0.0

    _kickoff_offset, _kickoff_conf = find_kickoff_offset(
        events, _video_duration_s,
        frames_data      = frames_data,
        fps              = fps,
        video_path       = video_path,
        gemini_verify_fn = _gemini_verify_kickoff,
    )

    if _kickoff_offset > 0:
        print(f"  [KICKOFF] Application offset={_kickoff_offset:.1f}s "
              f"(conf={_kickoff_conf:.2f}) — suppression pré-match")

        # 1. Corriger timestamps + supprimer events avant le coup d'envoi
        events, _n_removed = apply_kickoff_offset(events, _kickoff_offset, fps=fps)
        print(f"  [KICKOFF] {len(events)} events après correction "
              f"({_n_removed} events pré-match supprimés)")

        # 2. Supprimer les frames_data avant le coup d'envoi
        frames_data = apply_kickoff_offset_frames(frames_data, _kickoff_offset, fps=fps)

        # 3. Nettoyer le jersey_map (joueurs vus seulement à l'échauffement)
        jersey_map = reset_pre_kickoff_state(jersey_map, _kickoff_offset, fps=fps)
    else:
        print(f"  [KICKOFF] Pas de coup d'envoi détecté → timestamps inchangés")

    # ── DÉTECTION FIN DE MATCH ───────────────────────────────────────────────
    # Cherche la fin du match dans les frames_data pour couper l'après-match.
    # Prolongation et tirs au but sont couverts (seuil silence = 25 min).
    # Les gamins sans vareuse après le match ne seront pas comptés comme joueurs.
    _team_colors = None
    try:
        _team_colors = {
            int(k): v.get("color_bgr") or v.get("color")
            for k, v in teams.items()
            if v.get("color_bgr") or v.get("color")
        } if teams else None
    except Exception:
        pass

    _match_end_s = find_match_end(
        frames_data          = frames_data,
        fps                  = fps,
        team_colors          = _team_colors,
        silence_threshold_s  = 1500.0,   # 25 min sans jeu = fin du match
        min_match_duration_s = 2700.0,   # chercher fin seulement après 45 min de jeu
    )

    if _match_end_s is not None:
        events, frames_data = apply_match_end(events, frames_data, _match_end_s, fps=fps)

    # ── Alimenter le bootstrapper depuis frames_data ────────────────────────
    if _team_bootstrapper is not None and frames_data:
        try:
            _bootstrap_seconds = 120.0
            for _fd in frames_data:
                _t = _fd.get("frame", 0) / max(fps, 1)
                if _t > _bootstrap_seconds:
                    break
                _frame_orig = _fd.get("_frame_orig")
                # frames_data n'a plus _frame_orig (supprimé après rendu)
                # Utiliser les couleurs déjà extraites par player_reid
                for _p in _fd.get("players", []):
                    _pid   = _p.get("id") or _p.get("player_id")
                    _color = _p.get("color")  # couleur extraite par PlayerReID
                    _bbox  = _p.get("bbox")
                    if _pid is None or _color is None or _bbox is None:
                        continue
                    # Injecter directement la couleur sans re-extraire
                    _color_arr = np.array(_color, dtype=np.float32)
                    if np.any(_color_arr > 10):
                        pid_str = str(_pid)
                        _team_bootstrapper._pid_colors[pid_str].append(_color_arr)
                        if pid_str not in _team_bootstrapper._pid_first_seen:
                            _team_bootstrapper._pid_first_seen[pid_str] = _t
                        _team_bootstrapper._pid_last_seen[pid_str] = _t
                        _team_bootstrapper._n_frames += 1
        except Exception as _efb:
            print(f"  [BOOTSTRAP] Alimentation échouée : {_efb}")

    # ── Finaliser le bootstrapper d'équipes ──────────────────────────────────
    if _team_bootstrapper is not None:
        try:
            _bs_summary = _team_bootstrapper.summary()
            print(f"  [BOOTSTRAP] {_bs_summary['n_pids_total']} pids vus | "
                  f"{_bs_summary['n_stable']} stables")
            if _team_bootstrapper.is_ready() or _bs_summary['n_stable'] >= 4:
                _bs_result = _team_bootstrapper.finalize()
                if _bs_result:
                    # Injecter dans teams (priorité sur KMeans du Step 4)
                    for _tid, _tdata in _bs_result.items():
                        if _tid not in teams:
                            teams[_tid] = {}
                        teams[_tid]["color_bgr"]    = _tdata["color_bgr"]
                        teams[_tid]["color_name"]   = _tdata["color_name"]
                        teams[_tid]["_bootstrap_conf"] = _tdata["confidence"]
                    print(f"  [BOOTSTRAP] Couleurs injectées dans teams ✅")
        except Exception as _ebs:
            print(f"  [BOOTSTRAP] Finalisation échouée : {_ebs}")

    # Récupérer les couleurs équipes depuis le module tracker (source la plus fiable)
    try:
        from vision import tracker as _tracker_mod
        if hasattr(_tracker_mod, "_LAST_TEAM_COLORS") and _tracker_mod._LAST_TEAM_COLORS:
            _captured_team_colors = dict(_tracker_mod._LAST_TEAM_COLORS)
            print(f"  Couleurs equipes depuis tracker : {_captured_team_colors}")
    except Exception:
        pass

    # Capturer les couleurs équipes depuis le tracker dès le Step 1
    # Elles sont dans frames_data[*].players[*].color ou team_color
    _captured_team_colors = {}
    try:
        _color_votes = {}  # {team_id: [bgr_tuples]}
        for _fd in frames_data[:300]:
            for _pp in (_fd.get("players") or []):
                _t  = _pp.get("team")
                _c  = _pp.get("color") or _pp.get("team_color") or _pp.get("jersey_color")
                if _t is not None and _c is not None:
                    if isinstance(_c, (list, tuple)) and len(_c) == 3:
                        _color_votes.setdefault(_t, []).append(_c)
        for _t, _cols in _color_votes.items():
            # Moyenne des couleurs pour l'équipe
            _b = int(sum(c[0] for c in _cols) / len(_cols))
            _g = int(sum(c[1] for c in _cols) / len(_cols))
            _r = int(sum(c[2] for c in _cols) / len(_cols))
            _captured_team_colors[_t] = (_b, _g, _r)
        if _captured_team_colors:
            print(f"  Couleurs équipes capturées : {_captured_team_colors}")
    except Exception as _ect:
        pass

    for e in events:
        if not e.get("time"):
            frame     = e.get("frame", 0) or 0
            e["time"] = round(frame / fps, 2) if fps > 0 else 0

    # ─────────────────────────────────────────
    # 1a. POST PROCESSING
    #
    # Ordre et responsabilités :
    #   pipeline.py décide : résolution, goal_cooldown, position_threshold
    #   Étape 1 : filtre xG=0 sur buts bruts (events.py artifacts)
    #   Étape 2 : goal_posthoc (détection physique)
    #   Étape 3 : deduplicate_goals fenêtre courte 3s (fusion doublons immédiats)
    #   Étape 4 : post_process_events (merge, temporal, infer, filter_goals)
    #   Étape 5 : deduplicate_goals fenêtre 2s (sécurité post-traitement)
    # ─────────────────────────────────────────
    is_summary = False

    # ── Résolution réelle ─────────────────────────────────────────────────────
    if frames_data:
        _frame_w = int(frames_data[0].get("frame_w") or 1920)
        _frame_h = int(frames_data[0].get("frame_h") or 1080)
    else:
        _frame_w, _frame_h = 1920, 1080

    # Vérification cohérence résolution vs coordonnées réelles
    _max_ball_x = 0
    for f in frames_data[:200]:
        ball = f.get("ball")
        if ball:
            c = ball.get("center") or [ball.get("x", 0), 0]
            if c[0]:
                _max_ball_x = max(_max_ball_x, c[0])

    if _max_ball_x > 0 and _max_ball_x < _frame_w * 0.6:
        ratio = _max_ball_x / _frame_w
        if ratio < 0.5:
            _frame_w = int(_max_ball_x * 2)
        print(f"  WARNING résolution corrigée → {_frame_w}x{_frame_h} "
              f"(max_ball_x={_max_ball_x:.0f})")

    print(f"  Résolution pipeline : {_frame_w}x{_frame_h}")

    try:
        from analysis.post_processing import post_process_events

        video_duration = total_frames / fps if fps > 0 else 600
        is_summary     = video_duration < 480

        # ── PARAMÈTRES DÉCIDÉS ICI — source unique ────────────────────────────
        if is_summary:
            goal_cooldown = 10.0
            print(f"  Résumé détecté ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={GOAL_POSITION_THRESHOLD*100:.0f}%")
        else:
            if learner and learner.stats().get("n_matches", 0) >= 25:
                goal_cooldown = learner.get_thresholds().get("goal_cooldown", 45.0)
                goal_cooldown = min(goal_cooldown, 45.0)
            else:
                goal_cooldown = 45.0
            print(f"  Match complet ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={GOAL_POSITION_THRESHOLD*100:.0f}%")

        # position_threshold = GOAL_POSITION_THRESHOLD (0.05) dans tous les cas
        # Légèrement élargi à 0.06 pour absorber le bruit de tracking (±1px)
        _position_threshold = 0.06

        # ── Étape 1 : filtre xG=0 sur buts bruts ─────────────────────────────
        n_goals_raw = sum(1 for e in events if e.get("type") in ("goal", "score"))
        events = [
            e for e in events
            if not (
                e.get("type") in ("goal", "score")
                and (e.get("xg", 0) or 0) <= 0.01
                and e.get("confidence", 0) < 0.5
                and e.get("source") not in ("goal_posthoc", "ball_physics_v3",
                                            "ball_physics_v2", "events_v2",
                                            "goal_posthoc_v9", "goal_posthoc_v9.5")
            )
        ]
        n_goals_filtered = sum(1 for e in events if e.get("type") in ("goal", "score"))
        if n_goals_raw != n_goals_filtered:
            print(f"  Filtre xG=0 : {n_goals_raw - n_goals_filtered} faux but(s) supprimés")

        # ── Étape 2 : goal_posthoc ────────────────────────────────────────────
        _raw_posthoc_times = []  # init avant try — préservé pour SHOT→GOAL
        try:
            fast_goals = detect_fast_goals_from_ball(
                frames_data = frames_data,
                events      = events,
                fps         = fps,
                frame_w     = _frame_w,
                frame_h     = _frame_h,
                goal_box    = _goal_box,
            )
            if fast_goals:
                events.extend(fast_goals)
                events.sort(key=lambda e: e.get("time", 0))
                print(f"  goal_posthoc : {len(fast_goals)} but(s) détecté(s)")
            # Sauvegarder les temps posthoc bruts AVANT tout filtrage
            # Utilisé plus tard par SHOT→GOAL pour valider la présence d'un signal physique
            _raw_posthoc_times = [
                e.get("time", 0) for e in (fast_goals or [])
                if isinstance(e, dict)
            ]
        except Exception as eg:
            print(f"  goal_posthoc ignoré : {eg}")

        # ── Étape 2b : terminal_events — fenêtres candidates football-native ─────
        _terminal_goals = []
        try:
            _terminal_evts = detect_terminal_events(
                frames_data = frames_data,
                fps         = fps,
                frame_w     = _frame_w,
                frame_h     = _frame_h,
                goal_box    = _goal_box,
                sport       = sport,
            )
            if _terminal_evts:
                # Convertir en events "goal" candidats pour Gemini
                # Chaque événement terminal génère une fenêtre [-15s, +2s]
                _existing = [e for e in events if e.get("type") in ("goal", "score")]
                _windows  = build_candidate_windows(
                    _terminal_evts,
                    frames_data = frames_data,
                    fps         = fps,
                    existing_candidates = [
                        {"time": e.get("time", 0), "confidence": e.get("conf", 0.5),
                         "source": e.get("source", "posthoc")}
                        for e in _existing
                    ],
                )
                # Ajouter les fenêtres non couvertes par posthoc comme nouveaux candidats
                _posthoc_times = {e.get("time", 0) for e in events if e.get("type") in ("goal","score")}
                for w in _windows:
                    if w["source"].startswith("terminal_"):
                        # Vérifier qu'aucun candidat posthoc n'est déjà dans cette fenêtre
                        covered = any(
                            w["window_start"] <= t <= w["window_end"]
                            for t in _posthoc_times
                        )
                        if not covered:
                            # Calculer frame_id depuis time et fps
                            _t_ev = w["time"]
                            _fid  = int(_t_ev * fps) if fps > 0 else 0
                            _terminal_goals.append({
                                "type":       "goal",
                                "time":       _t_ev,
                                "frame":      _fid,
                                "source":     w["source"],
                                "conf":       w["confidence"],
                                "xg":         0.3,
                                "_terminal":  True,
                                "_window_start": w["window_start"],
                            })
                if _terminal_goals:
                    # Déduplication contre posthoc : évite les doublons ±10s
                    _posthoc_times = [
                        e["time"] for e in events
                        if e.get("type") == "goal"
                        and "posthoc" in e.get("source", "")
                    ]
                    _filtered_terminal = []
                    for _tg in _terminal_goals:
                        _too_close = any(
                            abs(_tg["time"] - _pt) < 10.0
                            for _pt in _posthoc_times
                        )
                        if not _too_close:
                            _filtered_terminal.append(_tg)
                    events.extend(_filtered_terminal)
                    events.sort(key=lambda e: e.get("time", 0))
                    print(f"  terminal_events : {len(_filtered_terminal)} nouveau(x) candidat(s) ({len(_terminal_goals)-len(_filtered_terminal)} dédupliqués vs posthoc)")
        except Exception as et:
            print(f"  terminal_events ignoré : {et}")

        # ── Étape 2c : zone_analyzer — analyse dense sur zones terminal_events ──
        try:
            from analysis.zone_analyzer import analyze_dense_zones
            if _terminal_evts:
                _dense_candidates = analyze_dense_zones(
                    video_path      = video_path,
                    terminal_events = _terminal_evts,
                    fps             = fps,
                    frames_data     = frames_data,   # pass 1 → fenêtres dynamiques
                    sport           = sport,
                )
                if _dense_candidates:
                    _existing_times = [
                        e["time"] for e in events if e.get("type") in ("goal", "score")
                    ]
                    for _dc in _dense_candidates:
                        _too_close = any(
                            abs(_dc["time"] - _et) < 10.0
                            for _et in _existing_times
                        )
                        if not _too_close:
                            events.append(_dc)
                            _existing_times.append(_dc["time"])
                    events.sort(key=lambda e: e.get("time", 0))
                    print(f"  zone_analyzer : {len(_dense_candidates)} candidat(s) dense(s) ajouté(s)")
        except Exception as _ez:
            print(f"  zone_analyzer ignoré : {_ez}")

        # ── Étape 3 : dedup fenêtre courte 3s (doublons immédiats posthoc/events)
        n_before = sum(1 for e in events if e.get("type") in ("goal", "score"))
        events   = deduplicate_goals(events, window=3.0)
        n_after  = sum(1 for e in events if e.get("type") in ("goal", "score"))
        if n_before != n_after:
            print(f"  Dedup 3s : {n_before} → {n_after} buts")

        # ── Étape 4 : post_process_events ─────────────────────────────────────
        # On lui passe TOUS les paramètres décidés ici
        events = post_process_events(
            events             = events,
            goal_cooldown      = goal_cooldown,
            position_threshold = _position_threshold,
            frame_w            = _frame_w,
            frame_h            = _frame_h,
            frames_data        = frames_data,
            fps                = fps,
        )

        # ── Étape 5 : dedup fenêtre 2s (sécurité après post-traitement) ───────
        events = deduplicate_goals(events, window=2.0)

        n_goals = sum(1 for e in events if e.get("type") in ["goal", "score"])
        print(f"  CLEAN {len(events)} events | {n_goals} but(s) | "
              f"goal_cooldown={goal_cooldown:.0f}s")

    except Exception as e:
        print(f"  Post processing ignoré : {e}")

    # ─────────────────────────────────────────
    # 1b. VALIDATION GEMINI
    # ─────────────────────────────────────────

    # ── Mode éco — détection phases creuses ──────────────────────────────────
    # Si aucun tir/but détecté depuis > 15s → on considère la phase creuse
    # Réduit les appels Gemini sur les events sans intérêt
    def _has_danger_near(t, window=15.0):
        """Retourne True si un tir/but existe dans les [t-window, t+window]."""
        for e in events:
            if e.get("type") in ("shot", "goal", "score"):
                if abs(e.get("time", 0) - t) <= window:
                    return True
        return False

    # Marquer les events "en phase creuse" pour filtrage Gemini
    for e in events:
        e["_eco"] = not _has_danger_near(e.get("time", 0))

    n_eco = sum(1 for e in events if e.get("_eco"))
    print(f"  Mode éco : {n_eco}/{len(events)} events en phase creuse (skip Gemini)")
    # ─────────────────────────────────────────────────────────────────────────

    print("Step 1b : Validation Gemini...")
    try:
        from ai.gemini_validator import validate_events_with_gemini, read_jersey_numbers, read_goal_scorers

        # Filtrer les events éco AVANT Gemini — réduit ~30% des appels
        events_for_gemini = [e for e in events if not e.get("_eco")]
        events_eco        = [e for e in events if e.get("_eco")]

        # Log PRE-GEMINI — état de tous les buts avant validation
        goals_pre = [e for e in events_for_gemini if e.get("type") == "goal"]
        print(f"  [PRE-GEMINI PIPELINE] {len(goals_pre)} but(s) candidats")
        if DEBUG:
            for e in goals_pre:
                t    = e.get("time", 0)
                src  = e.get("detected_from", e.get("source", "?"))
                # Offsets réels depuis gemini_validator (pas hardcodés)
                try:
                    import ai.gemini_validator as _gv
                    _offsets = _gv.OFFSETS_POSTHOC if "posthoc" in str(src) else _gv.OFFSETS_EVENTS
                    offs = str(_offsets) + "s"
                except Exception:
                    offs = "[-18,-14,-10,-7,-4,-2]s" if "posthoc" in str(src) else "[0,+2,+4]s"
                print(f"    t={int(t//60):02d}:{int(t%60):02d} "
                      f"source={src} "
                      f"conf={e.get('confidence',0):.2f} "
                      f"xg={e.get('xg',0):.3f} "
                      f"frame={e.get('frame',0)} "
                      f"offsets={offs}")

        # ── DEBUG clips candidats buts ──────────────────────────────
        if DEBUG:
            import subprocess as _sp
            _goals_pre = [e for e in events_for_gemini if e.get("type") == "goal"]
            if _goals_pre:
                _debug_dir = os.path.join(output_dir, "debug_goals")
                os.makedirs(_debug_dir, exist_ok=True)
                for _e in _goals_pre:
                    _t   = _e.get("time", 0)
                    _t0  = max(0, _t - 10)
                    _out = os.path.join(_debug_dir,
                        f"candidate_{int(_t//60):02d}m{int(_t%60):02d}s.mp4")
                    _sp.run([
                        "ffmpeg", "-y", "-ss", str(_t0),
                        "-i", video_path, "-t", "25",
                        "-c:v", "libx264", "-crf", "28",
                        "-c:a", "aac", "-loglevel", "error", _out
                    ], capture_output=True)
                    if os.path.exists(_out):
                        print(f"  [DEBUG CLIP] {os.path.basename(_out)}")
        # ────────────────────────────────────────────────────────────

        # ── Filtre posthoc : déclencheurs offensifs seulement ────────────────
        # Un posthoc n'est conservé que si proche d'un terminal OFFENSIF.
        # goalkeeper_save / clearance / possession_recovery = contextes défensifs
        # → ils déclenchent des posthoc non-buts qui polluent Gemini
        #
        # Déclencheurs OFFENSIFS retenus : goal, shot_on_target, rebound_attack
        # Déclencheurs DÉFENSIFS rejetés : goalkeeper_save, clearance, corner_or_goalkick
        #
        # Exception : clearance/save à ±5s d'un posthoc fort (score ≥ 8)
        # → ballon dans filet visible juste après un dégagement = rebond rentré

        _OFFENSIVE_TYPES = {"goal", "shot_on_target", "rebound_attack",
                             "shot", "blocked_shot", "loose_ball"}
        _DEFENSIVE_TYPES = {"goalkeeper_save", "clearance",
                             "corner_or_goalkick", "possession_recovery"}

        _terminal_times_filter = [
            float(ev.get("time", 0)) for ev in (_terminal_evts or [])
        ]
        _terminal_goals_times = [
            float(ev.get("time", 0)) for ev in (_terminal_evts or [])
            if ev.get("terminal_type") in ("goal",)
        ]
        # Terminaux offensifs (hors goal pur — déjà couvert)
        _terminal_offensive_times = [
            float(ev.get("time", 0)) for ev in (_terminal_evts or [])
            if ev.get("terminal_type") in _OFFENSIVE_TYPES
            and ev.get("terminal_type") != "goal"
        ]
        # Terminaux défensifs — fenêtre très réduite (±5s uniquement)
        _terminal_defensive_times = [
            float(ev.get("time", 0)) for ev in (_terminal_evts or [])
            if ev.get("terminal_type") in _DEFENSIVE_TYPES
        ]

        def _is_posthoc(e):
            src = str(e.get("detected_from", e.get("source", "")))
            return "posthoc" in src

        def _near_terminal(e):
            if not _terminal_times_filter:
                return True  # pas de terminal → garder tout
            t = float(e.get("time", 0))
            # ±20s pour terminal_goal
            if any(abs(t - tt) <= 20.0 for tt in _terminal_goals_times):
                return True
            # ±15s pour terminal offensif (shot, rebound...)
            if any(abs(t - tt) <= 15.0 for tt in _terminal_offensive_times):
                return True
            # ±5s pour terminal défensif UNIQUEMENT si score posthoc très fort
            # (rebond rentré juste après un dégagement)
            posthoc_score = float(e.get("score", e.get("danger", 0)))
            if posthoc_score >= 8.0:
                if any(abs(t - tt) <= 5.0 for tt in _terminal_defensive_times):
                    return True
            return False

        _n_before = sum(1 for e in events_for_gemini if e.get("type") == "goal")
        events_for_gemini = [
            e for e in events_for_gemini
            if e.get("type") != "goal"           # non-buts : inchangés
            or not _is_posthoc(e)                # terminal : toujours gardés
            or _near_terminal(e)                 # posthoc proche terminal : gardés
        ]
        _n_after = sum(1 for e in events_for_gemini if e.get("type") == "goal")
        if _n_before != _n_after:
            print(f"  [FILTRE POSTHOC] {_n_before} → {_n_after} candidats "
                  f"({_n_before - _n_after} posthoc éloignés de tout terminal supprimés)")

        # Buts cross_line → confiance physique élevée → bypass Gemini
        cross_line_goals = [
            e for e in events_for_gemini
            if e.get("type") == "goal" and e.get("cross_line")
        ]
        events_for_gemini_filtered = [
            e for e in events_for_gemini
            if not (e.get("type") == "goal" and e.get("cross_line"))
        ]
        if cross_line_goals:
            print(f"  [CROSS_LINE] {len(cross_line_goals)} but(s) franchissement ligne → bypass Gemini")
            for g in cross_line_goals:
                g["gemini_validated"] = True
                g["gemini_type"]      = "goal"
                g["gemini_conf"]      = g.get("confidence", 0.90)

        events_validated = validate_events_with_gemini(
            events        = events_for_gemini_filtered,
            video_path    = video_path,
            fps           = fps,
            sport         = sport,
            MIN_CONF_GOAL = 0.85 if not is_summary else 0.75,
            MIN_CONF_SHOT = 0.70,
        )
        # Réintégrer les buts cross_line confirmés
        events_validated = sorted(
            events_validated + cross_line_goals,
            key=lambda e: e.get("time", 0)
        )

        # Log POST-GEMINI — résultat après validation
        goals_post = [e for e in events_validated if e.get("type") == "goal"]
        print(f"  [POST-GEMINI PIPELINE] {len(goals_post)} but(s) conservés")
        if DEBUG:
            for e in goals_post:
                t = e.get("time", 0)
                print(f"    t={int(t//60):02d}:{int(t%60):02d} "
                      f"gemini_type={e.get('gemini_type','?')} "
                      f"gemini_conf={e.get('gemini_conf',0):.2f} "
                      f"tracker_conf={e.get('confidence',0):.2f} "
                      f"validated={e.get('gemini_validated',False)}")

        # ── Cooldown post-Gemini — éliminer les doublons après validation ────────
        # V9.7+ : le cooldown long s'applique UNIQUEMENT sur les buts validés
        # Les candidats rejetés par Gemini ne bloquent plus les suivants
        _GOAL_COOLDOWN_POST = goal_cooldown  # 30s ou 45s selon le learning
        _confirmed_times = []
        _goals_deduped = []
        for _e in sorted(events_validated, key=lambda x: x.get("time", 0)):
            if _e.get("type") not in ("goal", "score"):
                _goals_deduped.append(_e)
                continue
            _t = _e.get("time", 0)
            _validated = _e.get("gemini_validated", False)
            if _validated:
                # But validé : vérifier cooldown contre autres buts validés
                if any(abs(_t - _ct) < _GOAL_COOLDOWN_POST for _ct in _confirmed_times):
                    print(f"  [COOLDOWN POST] t={int(_t//60):02d}:{int(_t%60):02d} "
                          f"trop proche d'un but confirmé → supprimé")
                    continue
                _confirmed_times.append(_t)
                print(f"  [GOAL CONFIRMED] t={int(_t//60):02d}:{int(_t%60):02d} "
                      f"→ cooldown actif ({_GOAL_COOLDOWN_POST:.0f}s)")
            _goals_deduped.append(_e)
        events_validated = _goals_deduped

        # ── SHOT→GOAL conditionnel ────────────────────────────────────────────────
        # Activé uniquement si un tir xG > 0.35 n'a pas de but dans les 30s suivantes.
        # Évite les faux positifs (gardien, dégagements, centres) sur les vidéos standard
        # où goal_posthoc + ball_appears_in_goal suffisent.
        try:
            from ai.gemini_validator import find_goal_after_shot

            _confirmed_goal_times = [
                e.get("time", 0) for e in events_validated
                if isinstance(e, dict) and e.get("type") == "goal"
            ]

            # Tirs candidats : xG > 0.35, on_target, pas de but dans les 30s suivantes
            _stg_candidates = [
                e for e in events_validated
                if isinstance(e, dict)
                and e.get("type") == "shot"
                and e.get("on_target", False)
                and float(e.get("xg", 0) or 0) > 0.35
                and not any(
                    0 <= gt - e.get("time", 0) <= 30
                    for gt in _confirmed_goal_times
                )
            ]

            if not _stg_candidates:
                print("  [SHOT→GOAL] Aucun tir éligible (xG>0.35 sans but dans 30s) → ignoré")
            else:
                print(f"  [SHOT→GOAL] {len(_stg_candidates)} tir(s) éligible(s) → analyse Gemini stricte")

                shots_on_target_sorted = sorted(_stg_candidates, key=lambda e: e.get("time", 0))
                shot_times_all = [e.get("time", 0) for e in shots_on_target_sorted]
                existing_goal_times = list(_confirmed_goal_times)
                shots_to_analyze = []
                detected_goal_times = []

                for i, shot in enumerate(shots_on_target_sorted):
                    st = shot.get("time", 0)
                    already_covered = any(abs(gt - st) < 45 for gt in existing_goal_times)
                    if already_covered:
                        continue
                    next_shot_t = shot_times_all[i + 1] if i + 1 < len(shot_times_all) else st + 999
                    window = max(35, min(60, next_shot_t - st - 5))
                    shots_to_analyze.append((shot, st, window))

                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _analyze_shot(args):
                    shot, st, window = args
                    # st est en temps RELATIF post-KO.
                    # find_goal_after_shot lit des frames dans la vidéo brute →
                    # il faut ajouter l'offset KO pour pointer sur la bonne frame.
                    _ko = _kickoff_offset if _kickoff_offset > 0 else 0
                    _abs_shot_time = st + _ko
                    # Un but ne peut pas arriver dans les 15 premières secondes
                    # après le coup d'envoi — filtre les faux positifs très proches du KO
                    if st < 15.0:
                        return shot, st, {"is_goal": False, "timestamp": None,
                                          "confidence": 0.0, "desc": "trop proche KO",
                                          "goal_votes": 0, "goal_score": 0}
                    # confirmed_goal_times est en relatif → convertir en absolu
                    # pour que les comparaisons internes de find_goal_after_shot soient cohérentes
                    _abs_confirmed = [gt + _ko for gt in existing_goal_times]
                    return shot, st, find_goal_after_shot(
                        video_path           = video_path,
                        shot_time            = _abs_shot_time,
                        window               = window,
                        fps                  = fps,
                        frame_w              = _frame_w,
                        frame_h              = _frame_h,
                        confirmed_goal_times = _abs_confirmed,
                    )

                shot_goal_candidates = []
                results_map = {}

                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(_analyze_shot, args): args for args in shots_to_analyze}
                    for future in as_completed(futures):
                        try:
                            shot, st, result = future.result()
                            results_map[st] = (shot, result)
                        except Exception as _e:
                            print(f"  [SHOT→GOAL] Erreur analyse : {_e}")

                for shot, st, window in shots_to_analyze:
                    result = results_map.get(st, (shot, None))[1]
                    # FIX : already_covered élargi à 200s — un kickoff peut rester visible
                    # longtemps après un but (remise en jeu lente, caméra large)
                    already_covered = any(abs(gt - st) < 200 for gt in detected_goal_times)
                    if already_covered:
                        continue
                    _gv_stg = result.get("goal_votes", 1) if result else 0
                    if (result and result.get("is_goal")
                            and result.get("confidence", 0) >= 0.85
                            and _gv_stg >= 2):
                        # result["timestamp"] est en temps ABSOLU vidéo (find_goal_after_shot
                        # travaille sur la vidéo brute). On le reconvertit en temps RELATIF post-KO.
                        _goal_t_abs = result["timestamp"]
                        goal_t = _goal_t_abs - (_kickoff_offset if _kickoff_offset > 0 else 0)
                        too_close = any(abs(gt - goal_t) < 20 for gt in existing_goal_times)
                        _kickoff_fp = False  # filtre retiré : rejetait de vraies actions
                        # Exiger un signal physique posthoc dans la fenêtre tir→but
                        # Évite de valider un kickoff initial confondu avec un kickoff après but
                        # Utiliser les posthoc BRUTS (avant filtrage Gemini)
                        # pour ne pas rater les buts dont le posthoc a été supprimé
                        # Posthoc BRUTS (avant filtrage Gemini) — évite de rejeter
                        # des buts dont le candidat posthoc a été supprimé en amont
                        _posthoc_times = _raw_posthoc_times if _raw_posthoc_times else [
                            e.get("time", 0) for e in events
                            if isinstance(e, dict)
                            and e.get("type") in ("goal", "score")
                            and e.get("source", "").startswith("goal_posthoc")
                        ]
                        # Fenêtre réduite à ±15s (était ±30s) — évite que le kickoff
                        # de la 2e mi-temps (~+60s) valide un FP à la fin de la 1ère
                        # Exclure les posthoc estampillés "kickoff" ou trop proches
                        # d'un début de période (t < 5s ou t > durée-5s)
                        _video_duration = total_frames / fps if fps > 0 else 9999
                        _posthoc_times_filtered = [
                            pt for pt in _posthoc_times
                            if pt > 10.0                        # pas un kickoff initial
                            and pt < _video_duration - 5.0     # pas en fin de vidéo
                        ]
                        _has_physical = any(
                            abs(pt - goal_t) <= 15 for pt in _posthoc_times_filtered
                        )
                        if not _has_physical:
                            print(f"  [SHOT→GOAL] ❌ Rejeté {int(goal_t//60):02d}:{int(goal_t%60):02d} — aucun signal posthoc dans ±15s")
                        if not too_close and _has_physical and not _kickoff_fp:
                            new_goal = {
                                "type":             "goal",
                                "time":             goal_t,
                                "source":           "shot_to_goal_gemini",
                                "detected_from":    "shot_to_goal_gemini",
                                "confidence":       result["confidence"],
                                "gemini_validated": True,
                                "gemini_type":      "goal",
                                "gemini_conf":      result["confidence"],
                                "xg":               shot.get("xg", 0.5),
                                "desc":             result.get("desc", ""),
                                "player":           shot.get("player"),
                                "team":             shot.get("team"),
                                "x":                shot.get("x", _frame_w * 0.85),
                                "y":                shot.get("y", _frame_h * 0.5),
                                "frame":            int(goal_t * fps),
                                "shot_linked":      True,
                            }
                            shot_goal_candidates.append(new_goal)
                            existing_goal_times.append(goal_t)
                            detected_goal_times.append(goal_t)
                            print(f"  [SHOT→GOAL] ✅ BUT détecté à {int(goal_t//60):02d}:{int(goal_t%60):02d} conf={result['confidence']:.2f}")

                if shot_goal_candidates:
                    # Parmi les buts détectés, garder le meilleur score par fenêtre de 45s
                    # Évite qu'un faux positif à score modéré bloque un vrai but à score élevé
                    shot_goal_candidates.sort(key=lambda e: float(e.get("gemini_conf", 0)), reverse=True)
                    _final_goals = []
                    for _cand in shot_goal_candidates:
                        _ct = _cand.get("time", 0)
                        _too_close = any(abs(_ct - _fg.get("time", 0)) < 45 for _fg in _final_goals)
                        if not _too_close:
                            _final_goals.append(_cand)
                    events_validated = events_validated + _final_goals
                    print(f"  [SHOT→GOAL] {len(_final_goals)} but(s) ajouté(s) via analyse tirs ({len(shot_goal_candidates)} candidats)")
                else:
                    print(f"  [SHOT→GOAL] Aucun but confirmé")

        except Exception as _e:
            print(f"  [SHOT→GOAL] Ignoré : {_e}")

        # Réassembler dans l'ordre chronologique
        events = sorted(
            events_validated + events_eco,
            key=lambda x: x.get("time", 0)
        )
        # Nettoyer le flag interne
        for e in events:
            e.pop("_eco", None)

        # FP zones — actif seulement après 25 matchs propres
        if learner and learner.stats().get("n_matches", 0) >= 25:
            n_before = len(events)
            events   = [
                e for e in events
                if not (e.get("type") == "shot"
                        and learner.is_fp_zone(e.get("x", 0), e.get("y", 0)))
            ]
            if len(events) < n_before:
                print(f"  FP zones : {n_before - len(events)} shots filtrés")

        # Lecture maillots prioritaire
        seen_pids    = set()
        prio_players = []

        for e in events:
            if e.get("type") not in ["goal", "shot"]:
                continue
            pid = str(e.get("player", ""))
            if not pid or pid in seen_pids:
                continue
            if str(pid) in jersey_map:
                seen_pids.add(pid)
                continue
            frame_id = e.get("frame", 0) or 0
            for offset in [-10, 0, 10]:
                prio_players.append({
                    "id":       pid,
                    "frame_id": max(0, frame_id + offset),
                    "bbox":     e.get("bbox", [100, 100, 200, 300])
                })
            seen_pids.add(pid)

        general_players = [
            str(e.get("player")) for e in events
            if e.get("player") and str(e.get("player")) not in jersey_map
        ]
        seen_general = set()
        for pid in general_players:
            if pid not in seen_pids and pid not in seen_general:
                prio_players.append({
                    "id":       pid,
                    "frame_id": int(fps * 30),
                    "bbox":     [100, 100, 200, 300]
                })
                seen_general.add(pid)
                if len(seen_general) >= 20:
                    break

        if prio_players:
            gemini_jerseys = read_jersey_numbers(
                video_path          = video_path,
                players_with_frames = prio_players[:40],
                fps                 = fps
            )
            jersey_map.update(gemini_jerseys)
            print(f"  Gemini jerseys : {len(gemini_jerseys)} numéros lus "
                  f"(buts+tirs 3 frames + {len(seen_general)} généraux)")

        # Lecture ciblée buteurs + passeurs sur frames de buts confirmés
        # Bien plus fiable que les crops génériques
        try:
            confirmed_goals = [
                e for e in events_validated
                if e.get("type") in ("goal", "score")
                and e.get("gemini_validated", False)
            ]
            if confirmed_goals:
                from ai.gemini_validator import read_goal_scorers, read_highlight_visuals

                # Construire le dict {goal_time: clip_path} depuis les highlights
                _hl_clips = {}
                for _hl in []:
                    _hl_t = float(_hl.get("time") or _hl.get("event_time") or 0)
                    _hl_f = _hl.get("file", "")
                    if _hl_t > 0 and _hl_f:
                        _hl_clips[_hl_t] = _hl_f

                # ── Phase 1 élargie : description visuelle sur tous les highlights ──
                # Tirs + actions dangereuses → pool de référence visuelle local au match
                _highlight_events = [
                    e for e in events
                    if e.get("type") in ("shot", "goal", "score", "big_chance", "key_pass")
                    and e.get("gemini_validated", False) or e.get("type") in ("shot",)
                ]
                _visual_pool = {}
                try:
                    _visual_pool = read_highlight_visuals(
                        video_path       = video_path,
                        highlight_events = _highlight_events,
                        fps              = fps,
                        highlight_clips  = _hl_clips or None,
                    )
                    print(f"  Highlight visuals : {len(_visual_pool)} joueurs dans le pool")
                except Exception as _ev:
                    print(f"  Highlight visuals ignoré : {_ev}")

                # ── Scorers des buts avec fallback visuel enrichi ──
                # Récupérer les couleurs équipes dès maintenant pour validation croisée
                _colors_for_scorer = {}
                try:
                    from analysis.player_reid import get_team_colors as _gtc
                    _raw_colors = _gtc()
                    # Convertir BGR → nom couleur approximatif pour comparaison
                    _BGR_TO_NAME = {
                        "bleu": lambda b,g,r: b > 100 and b > r and b > g,
                        "rouge": lambda b,g,r: r > 100 and r > b and r > g,
                        "vert": lambda b,g,r: g > 100 and g > b and g > r,
                        "blanc": lambda b,g,r: b > 180 and g > 180 and r > 180,
                        "noir": lambda b,g,r: b < 60 and g < 60 and r < 60,
                        "jaune": lambda b,g,r: g > 150 and r > 150 and b < 100,
                    }
                    for _tid, _bgr in _raw_colors.items():
                        if isinstance(_bgr, (list, tuple)) and len(_bgr) == 3:
                            _b, _g, _r = int(_bgr[0]), int(_bgr[1]), int(_bgr[2])
                            for _name, _fn in _BGR_TO_NAME.items():
                                if _fn(_b, _g, _r):
                                    _colors_for_scorer[_tid] = _name
                                    break
                            else:
                                _colors_for_scorer[_tid] = "inconnu"
                except Exception:
                    pass

                # Injecter _team_colors dans chaque but pour validation croisée
                for _eg in confirmed_goals:
                    if _colors_for_scorer:
                        _eg["_team_colors"] = _colors_for_scorer

                goal_jerseys = read_goal_scorers(
                    video_path      = video_path,
                    goal_events     = confirmed_goals,
                    fps             = fps,
                    highlight_clips = _hl_clips or None,
                    visual_pool     = _visual_pool,
                )
                jersey_map.update(goal_jerseys)
                print(f"  Gemini goal scorers : {len(goal_jerseys)} buteur(s)/passeur(s) identifié(s)")
        except Exception as _ej:
            print(f"  Goal scorers ignoré : {_ej}")

    except Exception as e:
        print(f"  Gemini validation ignoree : {e}")

    # ─────────────────────────────────────────
    # 1b-bis. RÉSOLUTION IDENTITÉS JOUEURS
    # ─────────────────────────────────────────
    try:
        events, jersey_map = resolve_player_identities(events, jersey_map)
        print(f"  Jersey map : {len(jersey_map)} joueurs identifiés")
    except Exception as e:
        print(f"  Identity resolver ignoré : {e}")

    # ─────────────────────────────────────────
    # 1b-ter. VALIDATION STRUCTURELLE
    # ─────────────────────────────────────────
    try:
        from analysis.event_validator import filter_events
        events, val_stats = filter_events(
            events  = events,
            sport   = sport,
            frame_w = _frame_w,
            frame_h = _frame_h,
        )
        print(f"  Validation structurelle : {val_stats}")
    except Exception as e:
        print(f"  filter_events ignoré : {e}")

    # ─────────────────────────────────────────
    # 1c. SMART FILTERING (tirs)
    # ─────────────────────────────────────────
    # Smart filtering — désactivé (detect_real_shots nécessite ball_history)
    # Les tirs sont déjà filtrés par is_valid_shot dans events.py

    # ─────────────────────────────────────────
    # 1d. RE-ID + TEAMS
    # ─────────────────────────────────────────
    try:
        from analysis.player_reid import reidentify_players
        from analysis.team_cluster import assign_teams_by_color
        from analysis.pass_detector import detect_passes
        from analysis.xa_model import compute_xa
        from analysis.tactical_v2 import detect_pressing_intensity, detect_play_style

        events         = reidentify_players(events)
        events         = assign_teams_by_color(events)
        real_pass      = detect_passes(events)
        events.extend(real_pass)
        events         = compute_xa(events)

        pressing_level = detect_pressing_intensity(events)
        play_style     = detect_play_style(events)

        # Extraire les couleurs équipes depuis player_reid ou events ou frames_data
        _team_colors = {}
        try:
            from analysis.player_reid import get_team_colors
            _team_colors = get_team_colors()
        except Exception:
            pass

        # Fallback 1 : depuis les events
        if not _team_colors:
            for _e in events:
                _t = _e.get("team")
                _c = _e.get("team_color") or _e.get("color")
                if _t is not None and _c is not None and _t not in _team_colors:
                    _team_colors[_t] = _c

        # Fallback 2 : depuis la capture Step 1
        if not _team_colors and _captured_team_colors:
            _team_colors = _captured_team_colors

        # Fallback 3 : depuis frames_data joueurs
        if not _team_colors:
            for _f in frames_data[:500]:
                for _p in (_f.get("players") or []):
                    _t = _p.get("team")
                    _c = _p.get("color") or _p.get("team_color")
                    if _t is not None and _c is not None and _t not in _team_colors:
                        _team_colors[_t] = _c
                if len(_team_colors) >= 2:
                    break

        print(f"  Re-ID OK | Pressing: {pressing_level} | Style: {play_style}")
    except Exception as e:
        print(f"  Re-ID ignoré : {e}")

    # ─────────────────────────────────────────
    # 1e. CONTEXT ENGINE
    # ─────────────────────────────────────────
    ctx_stats = {}
    try:
        ctx    = ContextEngine(fps=fps, frame_w=_frame_w, frame_h=_frame_h)
        events = ctx.process_events(events, frames_data)
        ctx_stats = ctx.get_match_stats(events)
        print(f"  Context : shots_ctx={ctx_stats['shots_by_context']} | "
              f"avg_pressure={ctx_stats['avg_pressure_on_shot']} | "
              f"avg_seq={ctx_stats['avg_sequence_length']}")
    except Exception as e:
        print(f"  Context engine ignoré : {e}")

    # ─────────────────────────────────────────
    # 2. ENRICH xG
    # ─────────────────────────────────────────
    print("Step 2 : xG avancé...")
    frame_w     = _frame_w
    frame_h     = _frame_h
    n_advanced  = 0
    n_geometric = 0

    for e in events:
        if e.get("type") != "shot":
            continue

        x               = e.get("x", 0)
        y               = e.get("y", frame_h / 2)
        pressure        = float(e.get("pressure", 0.0) or 0.0)
        phase           = e.get("shot_context", e.get("phase", "open_play")) or "open_play"
        action_before   = e.get("action_before", "none") or "none"
        sequence_length = int(e.get("sequence_length", 1) or 1)

        xg = None

        if learner and learner.has_advanced_xg():
            xg = learner.predict_advanced_xg(e, frame_w=frame_w, frame_h=frame_h)
            if xg is not None:
                n_advanced += 1

        if xg is None:
            xg = compute_xg_sport(
                x               = x,
                y               = y,
                sport           = sport,
                frame_w         = frame_w,
                frame_h         = frame_h,
                pressure        = pressure,
                phase           = phase,
                action_before   = action_before,
                sequence_length = sequence_length,
            )
            n_geometric += 1

        e["xg"] = round(min(float(xg), 0.60), 3)   # clamp xG max à 0.60

    n_shots = n_advanced + n_geometric
    if n_shots > 0:
        if n_advanced > 0:
            n_adv_samples = len(learner.xg_advanced) if learner else 0
            print(f"  xG sklearn actif : {n_advanced}/{n_shots} tirs "
                  f"(modèle appris sur {n_adv_samples} samples)")
        else:
            print(f"  xG géométrique : {n_geometric} tirs "
                  f"(features=distance+angle+phase+action+pression+séquence)")

    # ─────────────────────────────────────────
    # 3. STATS
    # ─────────────────────────────────────────
    print("Step 3 : Stats...")
    stats = compute_stats(events, jersey_map=jersey_map)
    print(f"  OK {len(stats)} joueurs")

    possession = {}
    try:
        possession = compute_possession_from_stats(events, stats)
        print(f"  Possession corrigée : {possession}")
    except Exception as e:
        print(f"  Possession correction ignorée : {e}")

    # ─────────────────────────────────────────
    # 4. TACTICAL
    # ─────────────────────────────────────────
    print("Step 4 : Tactical...")
    teams = {}; formation = "?"; pressing = False; phases = []; tactical = {}
    try:
        events, teams = assign_teams(events)
        formation     = detect_formation(events)
        pressing      = detect_pressing(events)
        phases        = detect_phases(events)
        tactical      = tactical_report(
            events,
            players_frames = [f.get("players", []) for f in frames_data[:100]],
            frame_h        = _frame_h
        )

        # analyze_tactics Gemini supprimé V9.6 — économie ~1 appel/run

        # Injecter les couleurs équipes dans teams
        # Source prioritaire : _team_colors (player_reid)
        # Fallback : _captured_team_colors (Step 1 frames_data)
        _colors_src = _team_colors if _team_colors else _captured_team_colors
        for _tid, _tdata in teams.items():
            _key = int(_tid) if str(_tid).isdigit() else _tid
            if _key in _colors_src:
                _tdata["color_bgr"] = _colors_src[_key]
            elif _tid in _colors_src:
                _tdata["color_bgr"] = _colors_src[_tid]
            # Stocker aussi le nom couleur pour le PDF
            if _tdata.get("color_bgr"):
                try:
                    from export.pdf import bgr_to_name as _btn
                    _cn = _btn(_tdata["color_bgr"])
                    if _cn:
                        _tdata["color_name"] = _cn
                except Exception:
                    pass

        # Dernier fallback : calculer couleur dominante depuis events trackés
        if not any("color_bgr" in td for td in teams.values()):
            _team_color_votes = {}
            for _fd in frames_data[:200]:
                for _pp in (_fd.get("players") or []):
                    _t = _pp.get("team")
                    _c = _pp.get("color") or _pp.get("team_color") or _pp.get("jersey_color")
                    if _t is not None and isinstance(_c, (list, tuple)) and len(_c) == 3:
                        _team_color_votes.setdefault(str(_t), []).append(_c)
            for _t, _cols in _team_color_votes.items():
                if _t in teams and _cols:
                    _b = int(sum(c[0] for c in _cols) / len(_cols))
                    _g = int(sum(c[1] for c in _cols) / len(_cols))
                    _r = int(sum(c[2] for c in _cols) / len(_cols))
                    teams[_t]["color_bgr"] = (_b, _g, _r)

        _colors_final = {k: v.get("color_bgr") for k, v in teams.items() if v.get("color_bgr")}

        # Injecter les noms équipes fournis par l'utilisateur
        if team_names:
            for _tid, _tname in team_names.items():
                _tk = str(_tid)
                if _tk in teams:
                    teams[_tk]["name"] = str(_tname)
                elif _tid in teams:
                    teams[_tid]["name"] = str(_tname)

        # Marquer l'équipe de l'utilisateur
        if user_team_id is not None:
            _utk = str(user_team_id)
            if _utk in teams:
                teams[_utk]["is_user_team"] = True

        print(f"  OK formation={formation} | style={tactical.get('style','?')} | colors={_colors_final}")
    except Exception as e:
        print(f"  Tactical error : {e}")

    # ─────────────────────────────────────────
    # 5. IA LEARNING
    # ─────────────────────────────────────────
    print("Step 5 : IA Learning...")
    key_moments = []
    try:
        events      = cluster_actions(events)
        events      = learn_action_importance(events)
        key_moments = detect_key_moments(events)
        print(f"  OK {len(key_moments)} key moments")
    except Exception as e:
        print(f"  Learning error : {e}")

    # ─────────────────────────────────────────
    # 6. ADVANCED ANALYTICS
    # ─────────────────────────────────────────
    print("Step 6 : Advanced analytics...")
    pass_network = {}; offsides = []; dominance = {}
    try:
        from analytics.advanced import compute_xa as compute_xa_list
        events       = compute_xa_list(events)
        pass_network = build_pass_network(events)
        offsides     = detect_offside(events)
        dominance    = compute_team_dominance(events)

        pass_network = {
            f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): v
            for k, v in pass_network.items()
        }
        print(f"  OK pass_network={len(pass_network)} | offsides={len(offsides)}")
    except Exception as e:
        print(f"  Advanced error : {e}")

    # ─────────────────────────────────────────
    # 7. HEATMAPS
    # ─────────────────────────────────────────
    print("Step 7 : Heatmaps...")
    heatmaps = {}; heatmap_path = None; heatmap_paths = {}
    highlights = []  # initialisé ici, rempli en Step 8, utilisé en Step 7b
    # Note : heatmaps tirs générées après Step 8 (highlights filtrés)
    # → voir Step 7b ci-dessous

    # ─────────────────────────────────────────
    # 8. HIGHLIGHTS
    # ─────────────────────────────────────────
    # ─────────────────────────────────────────
    # STEP 7b : Heatmaps avec events filtrés
    # ─────────────────────────────────────────
    # Step 7b déplacé après Step 8 — heatmaps générées sur events_clean
    # (highlights doivent être prêts pour filtrer les tirs)

    print("Step 8 : Highlights...")
    reel_path = None
    try:
        # Limite dynamique : ~1 highlight/min, min 10, max 30
        _duration_min = total_frames / fps / 60 if fps > 0 else 15
        _max_hl = max(10, min(int(_duration_min * 1.2), 30))
        print(f"  Highlights max : {_max_hl} (durée={_duration_min:.0f} min)")

        highlights = create_highlights(
            video_path = video_path,
            events     = events,
            output_dir = os.path.join(output_dir, "highlights"),
            fps        = fps,
            max_clips  = _max_hl,
            mode       = mode,
            player_id  = player_id,
            sport      = sport
        )
        highlights = normalize_highlights(highlights, mode=mode)

        # Propager team depuis les events vers les highlights
        # Chaque highlight a un time_start / time → on cherche l'event le plus proche
        _events_with_team = [e for e in events if e.get("team") is not None]
        for _h in highlights:
            if _h.get("team") is not None:
                continue  # déjà renseigné
            _h_time = float(_h.get("time_start") or _h.get("time") or 0)
            _pid    = str(_h.get("player") or "")
            # Chercher l'event du même joueur au même moment (±10s)
            _best, _best_dist = None, float("inf")
            for _e in _events_with_team:
                _et = float(_e.get("time") or _e.get("frame", 0) / fps)
                _dist = abs(_et - _h_time)
                if _dist < _best_dist:
                    # Bonus si même joueur
                    if _pid and str(_e.get("player", "")) == _pid:
                        _dist *= 0.1
                    _best_dist = _dist
                    _best = _e
            if _best and _best_dist < 15:
                _h["team"] = _best["team"]

        try:
            from ai.highlight_scorer import score_all_highlights
            highlights = score_all_highlights(
                highlights     = highlights,
                video_path     = video_path,
                sport          = sport,
                max_highlights = _max_hl,
                frame_w        = frame_w,
            )
            highlights = normalize_highlights(highlights, mode=mode)
            print(f"  Gemini scoring : {len(highlights)} highlights scorés")
        except Exception as eg:
            print(f"  Highlight scorer ignoré : {eg}")

        highlights.sort(key=lambda h: h.get("time_start", 0))

        # ── Overlay scoreboard si noms équipes disponibles ────────
        _overlay_dir = os.path.join(output_dir, "highlights_overlay")
        _use_overlay = bool(team_names)  # overlay si noms fournis
        if _use_overlay:
            try:
                from rendering.overlay import render_highlights_with_overlay
                highlights = render_highlights_with_overlay(
                    highlights  = highlights,
                    events      = events,
                    teams_data  = teams,
                    output_dir  = _overlay_dir,
                    sport       = sport,
                )
                print(f"  OK overlay scoreboard sur {len(highlights)} clips")
            except Exception as _eov:
                print(f"  Overlay ignoré : {_eov}")
                _use_overlay = False

        reel_path = create_highlight_reel(
            highlights  = [
                {**h, "file": h.get("file_overlay") or h.get("file")}
                for h in highlights
            ] if _use_overlay else highlights,
            output_path = os.path.join(output_dir, "reel.mp4")
        )
        print(f"  OK {len(highlights)} highlights")
    except Exception as e:
        print(f"  Highlights error : {e}")

    # ─────────────────────────────────────────
    # 9. MONTAGE
    # ─────────────────────────────────────────
    print("Step 9 : Montage ignoré (désactivé)")
    montage_path = None

    # ── Mode player : reel joueur spécifique ─────────────────────────────────
    # Activé si mode="player" ET player_id fourni (ex: "11" ou "11_bleu")
    _player_reel_path = None
    if mode == "player" and player_id is not None and result_pe_mgr is not None:
        try:
            _jersey     = None
            _team_color = None
            _pid_parts  = str(player_id).split("_")
            if _pid_parts[0].isdigit():
                _jersey = int(_pid_parts[0])
            if len(_pid_parts) > 1:
                _team_color = "_".join(_pid_parts[1:])

            # Mode gardien : identifier par position et équipe
            _is_gk = (player_position or "").lower() in ("gardien", "goalkeeper", "gk")
            if _is_gk:
                # Déterminer l'équipe du gardien depuis player_id ou user_team_id
                _gk_team_id = user_team_id if user_team_id is not None else 0
                _player_highlights = result_pe_mgr.get_goalkeeper_highlights(
                    team_id        = _gk_team_id,
                    min_confidence = 0.3,  # seuil plus bas pour gardien
                )
                print(f"  [MODE PLAYER] Gardien équipe {_gk_team_id} : "
                      f"{len(_player_highlights)} clips")
            else:
                _player_highlights = result_pe_mgr.get_player_highlights(
                    jersey         = _jersey,
                    team_color     = _team_color,
                    min_confidence = get_owner_confidence_min(sport),
                )

            if _player_highlights:
                _player_reel_path = os.path.join(
                    output_dir,
                    f"reel_player_{player_id}.mp4"
                )
                create_highlight_reel(
                    highlights  = _player_highlights,
                    output_path = _player_reel_path,
                )
                print(f"  [MODE PLAYER] Reel #{_jersey} ({_team_color}) : "
                      f"{len(_player_highlights)} clips → {_player_reel_path}")
            else:
                print(f"  [MODE PLAYER] Aucun highlight pour #{_jersey} "
                      f"(min_conf={get_owner_confidence_min(sport):.2f})")

        except Exception as _epr:
            print(f"  [MODE PLAYER] Reel joueur ignoré : {_epr}")

    # ─────────────────────────────────────────
    # 10. RANKINGS + RATINGS + COMMENTARY
    # ─────────────────────────────────────────
    print("Step 10 : Ratings + Commentary...")
    ratings = {}; mvp = None; mvp_label = None; commentary = []; story = ""
    try:
        for e in events:
            if not e.get("time"):
                frame     = e.get("frame", 0) or 0
                e["time"] = round(frame / fps, 2) if fps > 0 else 0

        events = tag_key_passes(events)

        ranked_highlights = rank_highlights(events)
        ratings           = compute_player_ratings(events, jersey_map=jersey_map, fps=fps)
        mvp               = get_mvp(ratings)
        mvp_label         = resolve_mvp_label(mvp, stats, jersey_map)

        commentary = generate_commentary(
            ranked_highlights[:10],
            jersey_map = jersey_map,
            sport      = sport,
            formation  = formation,
            style      = tactical.get("style"),
        )
        story = None  # calculé après events_clean (Step 11)
        print(f"  OK MVP={mvp_label} | commentary={len(commentary)} lines")
    except Exception as e:
        print(f"  Ratings error : {e}")

    # ─────────────────────────────────────────
    # events_clean — version filtrée pour summary + story + heatmaps + PDF
    # Garde : buts validés + tirs dans highlights + passes + tous les autres events
    # Retire : tirs bruts non confirmés (faux tirs / passes mal classées)
    # ─────────────────────────────────────────
    highlight_shot_times = set()
    for h in highlights:
        if h.get("main_type") == "shot":
            t = h.get("time_start", h.get("time", 0))
            highlight_shot_times.add(round(float(t), 1))

    events_clean = [
        e for e in events
        if e.get("type") != "shot"
        or round(float(e.get("time", 0)), 1) in highlight_shot_times
    ]

    n_shots_clean = sum(1 for e in events_clean if e.get("type") == "shot")
    n_shots_raw   = sum(1 for e in events if e.get("type") == "shot")
    print(f"  events_clean : {len(events_clean)} events | "
          f"{n_shots_clean} tirs validés / {n_shots_raw} bruts")

    # Recalculer story sur events_clean
    try:
        story = generate_match_story(events_clean, fps=fps)
    except Exception as _e:
        print(f"  Story error : {_e}")

    # ─────────────────────────────────────────
    # Step 7b : Heatmaps (sur events_clean + tirs highlights)
    # ─────────────────────────────────────────
    print("Step 7b : Heatmaps (events filtrés)...")
    try:
        # Injecter tirs highlights dans events_clean (events_clean a 0 tirs)
        _shot_events_from_hl = []
        for _h in highlights:
            if _h.get("main_type") == "shot":
                _t_hl = float(_h.get("time_start") or _h.get("time") or 0)
                _best = next(
                    (_e for _e in events
                     if _e.get("type") == "shot" and abs(_e.get("time", 0) - _t_hl) < 2.0),
                    None
                )
                if _best:
                    _shot_events_from_hl.append(_best)
                else:
                    _shot_events_from_hl.append({
                        "type": "shot", "time": _t_hl,
                        "x": _h.get("x", _frame_w * 0.8),
                        "y": _h.get("y", _frame_h * 0.5),
                        "xg": _h.get("xg", 0),
                        "team": _h.get("team"), "player": _h.get("player"),
                    })
        events_for_heatmap = events_clean + _shot_events_from_hl
        heatmaps = generate_all_heatmaps(
            events     = events_for_heatmap,
            output_dir = os.path.join(output_dir, "heatmaps"),
            width      = config.FRAME_WIDTH,
            height     = config.FRAME_HEIGHT,
            sport      = sport
        )
        heatmap_path  = heatmaps.get("global")
        heatmap_paths = heatmaps
        print(f"  OK {len(heatmaps)} heatmaps "
              f"({len(_shot_events_from_hl)} tirs highlights / {n_shots_raw} bruts)")
    except Exception as e:
        print(f"  Heatmaps error : {e}")

    # ─────────────────────────────────────────
    # 10b. PLAYER ENTITIES — identité joueur persistante
    # ─────────────────────────────────────────
    player_entities = None
    try:
        from analysis.player_entity import PlayerEntityManager
        _pe_mgr = PlayerEntityManager(sport=sport)
        _pe_mgr._debug = DEBUG  # logs [ENTITY MATCH] seulement en mode debug
        _pe_result = {
            # Utiliser events_clean (filtrés) plutôt que events bruts
            # pour éviter 814 entités avec des pids instables
            "events":     events_clean if events_clean else events,
            "highlights": highlights,
            "jersey_map": jersey_map,
            "teams":      teams,
        }
        _pe_mgr.build_from_pipeline_result(_pe_result)
        player_entities = _pe_mgr.summary()

        # Log résumé des entités high-conf
        for _ent in _pe_mgr.get_all_entities(min_confidence=0.60, has_jersey=True):
            print(f"  [ENTITY] #{_ent.jersey} {_ent.team_color or 'team'+str(_ent.team)} "
                  f"conf={_ent.identity_confidence:.2f} "
                  f"events={len(_ent.events)} highlights={len(_ent.highlights)}")

        # Stocker le manager pour usage en mode player
        result_pe_mgr = _pe_mgr
    except Exception as _epe:
        print(f"  Player entities ignoré : {_epe}")
        result_pe_mgr = None

    # ─────────────────────────────────────────
    # 11. SUMMARY — sur events_clean (buts + tirs validés uniquement)
    # ─────────────────────────────────────────
    print("Step 11 : Summary...")
    summary = compute_match_summary(events_clean, stats, total_frames, fps)
    summary["possession"]    = possession
    summary["is_summary"]    = is_summary
    summary["context_stats"] = ctx_stats

    # V9.7+ — shots/xG/goals depuis events_clean (tirs validés par highlights)
    n_highlight_shots = sum(1 for h in highlights if h.get("main_type") == "shot")
    n_highlight_goals = sum(1 for h in highlights if h.get("main_type") in ("goal", "score"))
    # Buts = depuis events_validated (source de vérité) — pas depuis highlights
    # Les highlights peuvent inclure des SHOT→GOAL faux positifs
    n_validated_goals = sum(1 for e in events_validated if e.get("type") in ("goal", "score"))
    summary["goals"] = n_validated_goals

    # V9.9 — shots = tirs dans les highlights (ce que l'utilisateur voit)
    # n_highlight_shots = tirs, n_highlight_goals = buts
    # Total = highlights sélectionnés = ce qui est dans le reel
    summary["shots"]    = n_highlight_shots   # tirs cadres dans les highlights
    summary["total_xg"] = round(
        sum(float(h.get("xg", 0) or 0) for h in highlights), 2
    )

    # V9.7 — recalculer stats joueurs (tirs + xG) depuis highlights filtrés
    # Les stats brutes de compute_stats incluent tous les faux tirs
    highlight_times = {round(h.get("time_start", 0)): h for h in highlights}
    for pid in stats:
        stats[pid]["tirs"]     = 0
        stats[pid]["xg_total"] = 0.0
    for h in highlights:
        if h.get("main_type") != "shot":
            continue
        pid = str(h.get("player", ""))
        if pid and pid in stats:
            stats[pid]["tirs"]     += 1
            stats[pid]["xg_total"] += float(h.get("xg", 0) or 0)
        elif pid:
            # joueur pas encore dans stats (edge case) → ignorer
            pass
    # Recalculer aussi les buts depuis highlights
    for h in highlights:
        if h.get("main_type") not in ("goal", "score"):
            continue
        pid = str(h.get("player", ""))
        if pid and pid in stats:
            stats[pid]["buts"] = stats[pid].get("buts", 0) + 1

    print(f"  buts={summary['goals']} | tirs={summary['shots']} | "
          f"xG={summary['total_xg']} | joueurs={summary['players']}")
    print(f"  Possession : {possession}")

    # ─────────────────────────────────────────
    # 11b. ENREGISTREMENT APPRENTISSAGE
    # ─────────────────────────────────────────
    learning_result = {}
    if learner:
        try:
            learning_result = learner.record_match(
                events     = events,
                summary    = summary,
                fps        = fps,
                jersey_map = jersey_map,
                highlights = highlights,
                goals_real = goals_real,
                frame_w    = _frame_w,
                frame_h    = _frame_h,
            )
            print(f"  Learning : {learner.stats()}")
        except Exception as e:
            print(f"  Learning record error : {e}")

    # ─────────────────────────────────────────
    # 12. AI SUMMARY (Claude)
    # ─────────────────────────────────────────
    print("Step 12 : AI summary...")
    ai_summary = None
    if config.CLAUDE_API_KEY:
        try:
            from ai.claude import summarize
            ai_summary = summarize(
                highlights = highlights,
                summary    = summary,
                stats      = stats,
                sport      = sport
            )
            print("  OK Resume IA")
        except Exception as e:
            print(f"  AI error : {e}")
    else:
        print("  CLAUDE_API_KEY manquante")

    # ─────────────────────────────────────────
    # 13. PDF
    # ─────────────────────────────────────────
    print("Step 13 : PDF...")
    pdf_path = None
    if config.PLANS.get(plan, {}).get("pdf", False):
        try:
            from export.pdf import generate_pdf
            pdf_path = generate_pdf(
                result = {
                    "summary":        summary,
                    "stats":          stats,
                    "highlights":     highlights,
                    "jersey_map":     jersey_map,
                    "heatmaps":       heatmap_paths,
                    "player_ratings": ratings,
                    "match_story":    story,
                    "mvp":            mvp_label,
                    "formation":      formation,
                    "tactical":       tactical,
                    "commentary":     commentary,
                    "possession":     possession,
                    "context_stats":  ctx_stats,
                    "teams":          teams,
                },
                output_path = os.path.join(output_dir, "rapport.pdf"),
                sport       = sport
            )
            print(f"  OK PDF -> {pdf_path}")
        except Exception as e:
            print(f"  PDF error : {e}")

    # ─────────────────────────────────────────
    # 14. SAVE JSON
    # ─────────────────────────────────────────
    result = {
        "summary":       summary,
        "coarse_stats":  coarse_stats,
        "events":        events_clean,
        "stats":         stats,
        "highlights":    highlights,
        "jersey_map":    jersey_map,
        "heatmaps":      heatmaps,
        "heatmap":       heatmap_path,
        "reel":          reel_path,
        "montage":       montage_path,
        "annotated":     annotated_path,
        "ai_summary":    ai_summary,
        "pdf":           pdf_path,
        "sport":         sport,
        "mode":          mode,
        "player_id":     player_id,
        "calib":         calib,
        "fps":           fps,
        "total_frames":  total_frames,
        "teams":         teams,
        "formation":     formation,
        "pressing":      pressing,
        "phases":        phases,
        "tactical":      tactical,
        "pass_network":  pass_network,
        "offsides":      offsides,
        "dominance":     dominance,
        "key_moments":   key_moments,
        "ratings":       ratings,
        "mvp":           mvp_label,
        "commentary":    commentary,
        "story":         story,
        "learning":      learning_result,
        "possession":    possession,
        "context_stats": ctx_stats,
        "player_entities": player_entities,
        "player_reel":     _player_reel_path,
    }

    result = sanitize_for_json(result)

    with open(os.path.join(output_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────
    # RÉSUMÉ FINAL + DEBUG BUTS
    # ─────────────────────────────────────────
    print(f"\nPIPELINE DONE")
    print(f"  {summary['goals']} buts | {summary['shots']} tirs | "
          f"xG: {summary['total_xg']} | {summary['players']} joueurs")

    # Résumé player entities
    if player_entities:
        _ne = player_entities.get("n_entities", 0)
        _nj = player_entities.get("n_with_jersey", 0)
        _nh = player_entities.get("n_high_conf", 0)
        print(f"  Entities: {_ne} total | {_nj} avec jersey | {_nh} high-conf")
    print(f"  Formation: {formation} | Style: {tactical.get('style','?')}")
    print(f"  MVP: {mvp_label}")
    print(f"  Possession: {possession}")
    if is_summary:
        print(f"  ℹ️  Mode résumé détecté — paramètres adaptés")

    goal_events       = [e for e in events if e.get("type") in ("goal", "score")]
    shot_events_final = [e for e in events if e.get("type") == "shot"]
    print(f"\n{'='*50}")
    print(f"DEBUG — Analyse buts + joueurs")
    print(f"{'='*50}")
    print(f"\n  Buts          : {len(goal_events)}")
    print(f"  Tirs          : {len(shot_events_final)}")
    print(f"  Jersey map    : {len(jersey_map)} joueurs identifiés")
    for g in goal_events:
        t  = g.get("time", 0)
        mm = int(t // 60)
        ss = int(t % 60)
        print(f"\n  ── But à {mm:02d}:{ss:02d} ──")
        pid   = g.get("player")
        label = get_player_label(str(pid), jersey_map) if pid else "?"
        print(f"     Buteur    : {label} (ID={pid})")
        print(f"     Source    : {g.get('source', g.get('detected_from', '?'))}")
        print(f"     gemini    : {g.get('gemini_validated', False)}")
        print(f"     conf      : {g.get('confidence', '?')}")
        print(f"     xG        : {g.get('xg', 0):.3f}")

    return result