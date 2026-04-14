# pipeline.py
# -*- coding: utf-8 -*-

import os
import json

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
from sports.config import get_sport_config, compute_xg_sport
from analysis.event_validator import detect_real_shots
from analysis.context_engine import ContextEngine
from analysis.player_identity import resolve_player_identities, get_player_label
from analysis.goal_posthoc import detect_fast_goals_from_ball


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


# ─────────────────────────────────────────
# DÉDUPLICATION BUTS (niveau pipeline)
# ─────────────────────────────────────────
def deduplicate_goals(events, window=3.0):
    """
    Fusionne les buts détectés en doublon (events + posthoc + Gemini).
    Conserve le but avec la meilleure confidence dans chaque fenêtre.
    """
    goals  = sorted(
        [e for e in events if e.get("type") in ("goal", "score")],
        key=lambda x: x.get("time", 0)
    )
    others = [e for e in events if e.get("type") not in ("goal", "score")]

    kept = []
    for g in goals:
        if kept and abs(g["time"] - kept[-1]["time"]) < window:
            if g.get("confidence", 0) > kept[-1].get("confidence", 0):
                kept[-1] = g
        else:
            kept.append(g)

    return sorted(others + kept, key=lambda x: x.get("time", 0))


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_pipeline(
    video_path,
    sport          = "football",
    output_dir     = "outputs",
    analysis_id    = None,
    save_annotated = False,
    plan           = "free",
    mode           = "match",
    player_id      = None,
    goals_real     = None,
):
    os.makedirs(output_dir, exist_ok=True)
    progress = make_progress_callback(analysis_id)

    print(f"\nPIPELINE START - {sport.upper()} | mode={mode}")

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

    # ─────────────────────────────────────────
    # 1. TRACKING + EVENTS
    # ─────────────────────────────────────────
    print("Step 1 : Tracking + Events...")

    annotated_path = os.path.join(output_dir, "annotated.mp4") \
        if save_annotated else None

    events, jersey_map, fps, total_frames, frames_data = process_video(
        video_path        = video_path,
        sport             = sport,
        progress_callback = progress,
        save_annotated    = save_annotated,
        annotated_path    = annotated_path,
        shot_zones        = shot_zones,
        return_frames     = True
    )
    print(f"  RAW {len(events)} events | {len(jersey_map)} maillots")

    for e in events:
        if not e.get("time"):
            frame     = e.get("frame", 0) or 0
            e["time"] = round(frame / fps, 2) if fps > 0 else 0

    # ─────────────────────────────────────────
    # 1a. POST PROCESSING
    #
    # Ordre critique :
    #   1. Résolution réelle (une fois, hors boucle)
    #   2. detect_goals_robust → remplace ancienne logique events.py
    #   3. goal_posthoc → rattrapage tirs rapides
    #   4. deduplicate_goals → fusion par confidence
    #   5. merge_players + temporal_filter
    #   6. infer_shots_from_goals
    #   7. filter_goals (cooldown)
    # ─────────────────────────────────────────
    is_summary = False

    # ── Résolution réelle (UNE SEULE FOIS, hors boucle) ──────────────────────
    # FIX CRITIQUE : ne jamais utiliser 1920 hardcodé si les coords sont en 960px
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
        # Coordonnées ballon bien inférieures à frame_w → résolution incohérente
        ratio = _max_ball_x / _frame_w

        if ratio < 0.5:
            _frame_w = int(_max_ball_x * 2)

        print(f"  WARNING résolution corrigée → {_frame_w}x{_frame_h} "
              f"(max_ball_x={_max_ball_x:.0f} incohérent avec frame_w original)")

    print(f"  Résolution pipeline : {_frame_w}x{_frame_h}")

    try:
        from analysis.post_processing import post_process_events

        video_duration = total_frames / fps if fps > 0 else 600
        is_summary     = video_duration < 480

        if is_summary:
            goal_cooldown      = 10.0
            position_threshold = 0.05
            print(f"  Résumé détecté ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={int(position_threshold*100)}%")
        else:
            goal_cooldown = learner.get_thresholds().get("goal_cooldown", 150.0) \
                            if learner else 150.0
            position_threshold = 0.05
            print(f"  Match complet ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={int(position_threshold*100)}%")

        # ── FILTRE XG == 0 SUR BUTS BRUTS (avant tout le reste) ──────────────
        # Supprime les faux buts générés par events.py sans contexte tir
        n_goals_raw = sum(1 for e in events if e.get("type") in ("goal", "score"))
        events = [
            e for e in events
            if not (
                e.get("type") in ("goal", "score")
                and (e.get("xg", 0) or 0) <= 0.01
                and e.get("confidence", 0) < 0.5
                and e.get("source") not in ("goal_posthoc", "ball_physics_v3",
                                            "ball_physics_v2", "events_v2")
            )
        ]
        n_goals_filtered = sum(1 for e in events if e.get("type") in ("goal", "score"))
        if n_goals_raw != n_goals_filtered:
            print(f"  Filtre xG=0 : {n_goals_raw - n_goals_filtered} faux but(s) "
                  f"supprimés à la source")

        # ── 1. Détecteur physique buts rapides (posthoc) ──────────────────────
        fast_goals = []
        try:
            fast_goals = detect_fast_goals_from_ball(
                frames_data = frames_data,
                events      = events,
                fps         = fps,
                frame_w     = _frame_w,   # résolution corrigée
                frame_h     = _frame_h,
            )
            if fast_goals:
                events.extend(fast_goals)
                events.sort(key=lambda e: e.get("time", 0))
                print(f"  goal_posthoc : {len(fast_goals)} but(s) détecté(s)")
        except Exception as eg:
            print(f"  goal_posthoc ignoré : {eg}")

        # ── 2. Déduplication par confidence (avant merge) ─────────────────────
        n_before = sum(1 for e in events if e.get("type") in ("goal", "score"))
        events   = deduplicate_goals(events, window=3.0)
        n_after  = sum(1 for e in events if e.get("type") in ("goal", "score"))
        if n_before != n_after:
            print(f"  Dedup buts : {n_before} → {n_after} (doublons fusionnés par confidence)")

        for e in events:
            if e.get("detected_from") == "goal_posthoc_v8":
                e["_keep"] = True

        # ── 3. Fusion joueurs proches ──────────────────────────────────────────
        events = post_process_events(
            events=events,
            frames_data=frames_data,
            fps=fps,
            frame_w=_frame_w,
            frame_h=_frame_h
        )
        print("  DEBUG buts après post_process:")
        for e in events:
            if e.get("type") in ("goal", "score"):
                print(f"    t={e.get('time'):.2f} | conf={e.get('confidence')} | src={e.get('detected_from')}")

        n_goals = sum(1 for e in events if e.get("type") in ["goal", "score"])
        print(f"  CLEAN {len(events)} events | {n_goals} but(s) | "
              f"goal_cooldown={goal_cooldown:.0f}s")

        # Réinjecter buts critiques si supprimés
        post_times = {round(e.get("time", 0), 1) for e in events}

        for e in fast_goals:
            t = round(e.get("time", 0), 1)
            if t not in post_times:
                print(f"  ⚠️ BUT RÉINJECTÉ {t}s (perdu dans post_process)")
                events.append(e)

        events.sort(key=lambda x: x.get("time", 0))

    except Exception as e:
        print(f"  Post processing ignoré : {e}")

    # ─────────────────────────────────────────
    # 1b. VALIDATION GEMINI
    # ─────────────────────────────────────────
    print("Step 1b : Validation Gemini...")
    try:
        # protéger les buts physiques forts
        protected_goals = []

        for e in events:
            if (
                e.get("type") == "goal"
                and e.get("detected_from") == "goal_posthoc_v8"
                and e.get("confidence", 0) >= 0.7
            ):
                e["_protected"] = True

        from ai.gemini_validator import validate_events_with_gemini, read_jersey_numbers

        events = validate_events_with_gemini(
            events        = events,
            video_path    = video_path,
            fps           = fps,
            sport         = sport,
            MIN_CONF_GOAL = 0.85 if not is_summary else 0.75,   # FIX : était 0.70 → trop permissif (ex: 0.65 accepté)
            MIN_CONF_SHOT = 0.70,
            frame_w       = _frame_w,
            frame_h       = _frame_h,
        )

        if learner:
            n_before = len(events)
            events   = [
                e for e in events
                if not (e.get("type") == "shot"
                        and learner.is_fp_zone(e.get("x", 0), e.get("y", 0)))
            ]
            if len(events) < n_before:
                print(f"  FP zones : {n_before - len(events)} shots filtrés")

        # ── Lecture maillots prioritaire sur 3 frames ──
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
    # 1c. SMART FILTERING
    # ─────────────────────────────────────────
    try:
        
        validated = []
        for i, e in enumerate(events):
            prev = events[i-1] if i > 0 else None
            if e.get("type") == "shot":
                if not e.get("detected_from") and not e.get("synthetic") and not detect_real_shot(e, prev):
                    continue
            validated.append(e)
        events = validated

        print(f"  SMART {len(events)} events | Possession: (calculée après stats)")
    except Exception as e:
        print(f"  Smart filtering ignoré : {e}")

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
        print(f"  Re-ID OK | Pressing: {pressing_level} | Style: {play_style}")
    except Exception as e:
        print(f"  Re-ID ignoré : {e}")

    # ─────────────────────────────────────────
    # 1e. CONTEXT ENGINE
    # ─────────────────────────────────────────
    try:
        ctx    = ContextEngine(fps=fps, frame_w=_frame_w, frame_h=_frame_h)
        events = ctx.process_events(events, frames_data)
        ctx_stats = ctx.get_match_stats(events)
        print(f"  Context : shots_ctx={ctx_stats['shots_by_context']} | "
              f"avg_pressure={ctx_stats['avg_pressure_on_shot']} | "
              f"avg_seq={ctx_stats['avg_sequence_length']}")
    except Exception as e:
        print(f"  Context engine ignoré : {e}")
        ctx_stats = {}

    # ─────────────────────────────────────────
    # 2. ENRICH xG
    # ─────────────────────────────────────────
    print("Step 2 : xG avancé...")
    frame_w    = _frame_w
    frame_h    = _frame_h
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

        e["xg"] = round(min(float(xg), 0.99), 3)

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

    try:
        possession = compute_possession_from_stats(events, stats)
        print(f"  Possession corrigée : {possession}")
    except Exception as e:
        print(f"  Possession correction ignorée : {e}")

    # ─────────────────────────────────────────
    # 4. TACTICAL
    # ─────────────────────────────────────────
    print("Step 4 : Tactical...")
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

        try:
            from ai.gemini_analyzer import analyze_tactics
            tactical_gemini = analyze_tactics(
                video_path = video_path,
                sport      = sport,
                fps        = fps,
                events     = events
            )
            if tactical_gemini.get("gemini_analysed"):
                tactical.update({k: v for k, v in tactical_gemini.items()
                                 if k != "gemini_analysed"})
                formation = tactical_gemini.get("formation", formation)
        except Exception as eg:
            print(f"  Gemini tactical ignoré : {eg}")

        print(f"  OK formation={formation} | style={tactical.get('style','?')}")
    except Exception as e:
        print(f"  Tactical error : {e}")
        teams = {}; formation = "?"; pressing = False
        phases = []; tactical = {}

    # ─────────────────────────────────────────
    # 5. IA LEARNING
    # ─────────────────────────────────────────
    print("Step 5 : IA Learning...")
    try:
        events      = cluster_actions(events)
        events      = learn_action_importance(events)
        key_moments = detect_key_moments(events)
        print(f"  OK {len(key_moments)} key moments")
    except Exception as e:
        print(f"  Learning error : {e}")
        key_moments = []

    # ─────────────────────────────────────────
    # 6. ADVANCED ANALYTICS
    # ─────────────────────────────────────────
    print("Step 6 : Advanced analytics...")
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
        pass_network = {}; offsides = []; dominance = {}

    # ─────────────────────────────────────────
    # 7. HEATMAPS
    # ─────────────────────────────────────────
    print("Step 7 : Heatmaps...")
    heatmaps      = {}
    heatmap_path  = None
    heatmap_paths = {}
    try:
        heatmaps = generate_all_heatmaps(
            events     = events,
            output_dir = os.path.join(output_dir, "heatmaps"),
            width      = config.FRAME_WIDTH,
            height     = config.FRAME_HEIGHT,
            sport      = sport
        )
        heatmap_path  = heatmaps.get("global")
        heatmap_paths = heatmaps
        print(f"  OK {len(heatmaps)} heatmaps")
    except Exception as e:
        print(f"  Heatmaps error : {e}")

    # ─────────────────────────────────────────
    # 8. HIGHLIGHTS
    # ─────────────────────────────────────────
    print("Step 8 : Highlights...")
    highlights = []
    reel_path  = None

    try:
        highlights = create_highlights(
            video_path = video_path,
            events     = events,
            output_dir = os.path.join(output_dir, "highlights"),
            fps        = fps,
            max_clips  = config.HIGHLIGHT_MAX,
            mode       = mode,
            player_id  = player_id,
            sport      = sport
        )
        highlights = normalize_highlights(highlights, mode=mode)

        try:
            from ai.highlight_scorer import score_all_highlights
            highlights = score_all_highlights(
                highlights     = highlights,
                video_path     = video_path,
                sport          = sport,
                max_highlights = config.HIGHLIGHT_MAX,
                frame_w        = frame_w,
            )
            highlights = normalize_highlights(highlights, mode=mode)
            print(f"  Gemini scoring : {len(highlights)} highlights scorés")
        except Exception as eg:
            print(f"  Highlight scorer ignoré : {eg}")

        highlights.sort(key=lambda h: h.get("time_start", 0))

        reel_path = create_highlight_reel(
            highlights  = highlights,
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

    # ─────────────────────────────────────────
    # 10. RANKINGS + RATINGS + COMMENTARY
    # ─────────────────────────────────────────
    print("Step 10 : Ratings + Commentary...")
    ratings    = {}
    mvp        = None
    mvp_label  = None
    commentary = []
    story      = ""
    possession = {}
    
    try:
        for e in events:
            if not e.get("time"):
                frame     = e.get("frame", 0) or 0
                e["time"] = round(frame / fps, 2) if fps > 0 else 0

        events = tag_key_passes(events)

        ranked_highlights = rank_highlights(events)
        ratings           = compute_player_ratings(
            events,
            jersey_map = jersey_map,
            fps        = fps,
        )
        mvp       = get_mvp(ratings)
        mvp_label = resolve_mvp_label(mvp, stats, jersey_map)

        commentary = generate_commentary(
            ranked_highlights[:10],
            jersey_map = jersey_map,
            sport      = sport,
            formation  = formation,
            style      = tactical.get("style"),
        )
        story = generate_match_story(events, fps=fps)
        print(f"  OK MVP={mvp_label} | commentary={len(commentary)} lines")
    except Exception as e:
        print(f"  Ratings error : {e}")

    # ─────────────────────────────────────────
    # 11. SUMMARY
    # ─────────────────────────────────────────
    print("Step 11 : Summary...")
    summary = compute_match_summary(events, stats, total_frames, fps)
    summary["possession"]    = possession
    summary["is_summary"]    = is_summary
    summary["context_stats"] = ctx_stats
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
                frame_w    = _frame_w,   # FIX : résolution réelle (960px possible)
                frame_h    = _frame_h,   # sans ça les features xG sont calculées en 1920 → faux
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
        "events":        events,
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
    }

    result = sanitize_for_json(result)

    with open(os.path.join(output_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ─────────────────────────────────────────
    print(f"\nPIPELINE DONE")
    print(f"  {summary['goals']} buts | {summary['shots']} tirs | "
          f"xG: {summary['total_xg']} | {summary['players']} joueurs")
    print(f"  Formation: {formation} | Style: {tactical.get('style','?')}")
    print(f"  MVP: {mvp_label}")
    print(f"  Possession: {possession}")
    if is_summary:
        print(f"  ℹ️  Mode résumé détecté — paramètres adaptés")

    # ─────────────────────────────────────────
    # DEBUG BUTS (toujours affiché)
    # ─────────────────────────────────────────
    goal_events = [e for e in events if e.get("type") in ("goal", "score")]
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
        pid = g.get("player")
        label = get_player_label(str(pid), jersey_map) if pid else "?"
        print(f"     Buteur    : {label} (ID={pid})")
        print(f"     Source    : {g.get('source', g.get('detected_from', '?'))}")
        print(f"     gemini    : {g.get('gemini_validated', False)}")
        print(f"     conf      : {g.get('confidence', '?')}")
        print(f"     xG        : {g.get('xg', 0):.3f}")

    return result