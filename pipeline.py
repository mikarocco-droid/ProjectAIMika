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
from analysis.event_validator import filter_events, detect_real_shots
from analysis.context_engine import ContextEngine


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
                   "progressive_run", "interception", "fast_break"]
    else:
        allowed = ["goal", "shot", "score"]

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

    if jersey_map:
        jersey = (
            jersey_map.get(pid)
            or jersey_map.get(int(pid) if pid.isdigit() else pid)
        )
        if jersey:
            return f"#{jersey}"

    return f"ID-{pid}"


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
        print(f"  Learning : {ls['n_matches']} matchs | "
              f"{ls['n_events']} events | "
              f"xG n={ls['xg_samples']} | "
              f"xG avancé={'✅' if ls.get('xg_advanced_ready') else f\"({ls.get('xg_advanced_samples',0)}/50)\"} | "
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
            frame = e.get("frame", 0) or 0
            e["time"] = round(frame / fps, 2) if fps > 0 else 0

    # ─────────────────────────────────────────
    # 1a. POST PROCESSING
    # ─────────────────────────────────────────
    is_summary = False
    _frame_w   = 1920
    _frame_h   = 1080

    try:
        from analysis.post_processing import (
            temporal_filter,
            filter_goals,
            merge_players,
        )

        video_duration = total_frames / fps if fps > 0 else 600
        is_summary     = video_duration < 480

        _frame_w = int(frames_data[0].get("frame_w", 1920)) if frames_data else 1920
        _frame_h = int(frames_data[0].get("frame_h", 1080)) if frames_data else 1080

        if is_summary:
            goal_cooldown      = 10.0
            position_threshold = 0.35
            print(f"  Résumé détecté ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={int(position_threshold*100)}%")
        else:
            goal_cooldown = learner.get_thresholds().get("goal_cooldown", 150.0) \
                            if learner else 150.0
            position_threshold = 0.20
            print(f"  Match complet ({video_duration:.0f}s) — "
                  f"goal_cooldown={goal_cooldown:.0f}s | "
                  f"position_threshold={int(position_threshold*100)}%")

        events = merge_players(events)
        events = temporal_filter(events, min_delta=2.0)
        events = filter_goals(
            events,
            window             = goal_cooldown,
            frame_w            = _frame_w,
            position_threshold = position_threshold,
        )
        print(f"  CLEAN {len(events)} events | goal_cooldown={goal_cooldown:.0f}s")
    except Exception as e:
        print(f"  Post processing ignoré : {e}")

    # ─────────────────────────────────────────
    # 1b. VALIDATION GEMINI
    # ─────────────────────────────────────────
    print("Step 1b : Validation Gemini...")
    try:
        from ai.gemini_validator import validate_events_with_gemini, read_jersey_numbers

        events = validate_events_with_gemini(
            events        = events,
            video_path    = video_path,
            fps           = fps,
            sport         = sport,
            MIN_CONF_GOAL = 0.80,
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

        top_players = [
            {"id": pid, "frame_id": int(fps * 30), "bbox": [100, 100, 200, 300]}
            for pid in list(set(
                str(e.get("player")) for e in events
                if e.get("player") and str(e.get("player")) not in jersey_map
            ))[:20]
        ]

        if top_players:
            gemini_jerseys = read_jersey_numbers(
                video_path          = video_path,
                players_with_frames = top_players,
                fps                 = fps
            )
            jersey_map.update(gemini_jerseys)
            print(f"  Gemini jerseys : {len(gemini_jerseys)} nouveaux numéros lus")

    except Exception as e:
        print(f"  Gemini validation ignoree : {e}")

    # ─────────────────────────────────────────
    # 1b-bis. VALIDATION STRUCTURELLE
    # ─────────────────────────────────────────
    try:
        events, val_stats = filter_events(
            events  = events,
            sport   = sport,
            frame_w = _frame_w,
            frame_h = _frame_h,
            verbose = False,
        )
        if val_stats["invalid"] > 0:
            print(f"  Validator : {val_stats['valid']} valides | "
                  f"{val_stats['invalid']} rejetés | "
                  f"{val_stats['by_reason']}")
    except Exception as e:
        print(f"  Event validator ignoré : {e}")

    # ─────────────────────────────────────────
    # 1b-ter. DÉTECTION RÉELLE DE TIRS
    # ─────────────────────────────────────────
    try:
        real_shots = detect_real_shots(
            events  = events,
            frame_w = _frame_w,
            frame_h = _frame_h,
            sport   = sport,
            fps     = fps,
        )
        if real_shots:
            events.extend(real_shots)
            events.sort(key=lambda e: e.get("time", 0))
    except Exception as e:
        print(f"  Shot detector ignoré : {e}")

    # ─────────────────────────────────────────
    # 1c. SMART FILTERING
    # ─────────────────────────────────────────
    possession = {}
    try:
        from analysis.smart_game_ai import clean_events_smart
        from analysis.ball_physics import detect_real_shot

        events = clean_events_smart(events)

        validated = []
        for i in range(len(events)):
            e    = events[i]
            prev = events[i-1] if i > 0 else None
            if e.get("type") == "shot":
                if not e.get("detected_from") and not detect_real_shot(e, prev):
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
    # 2. ENRICH xG — version avancée
    #
    # Priorité :
    #   1. learner.predict_advanced_xg() — sklearn avec features complètes
    #      (actif dès 50 tirs collectés)
    #   2. compute_xg_sport() — modèle géométrique calibré
    #      distance + angle + phase + action + pression + séquence
    # ─────────────────────────────────────────
    print("Step 2 : xG avancé...")
    frame_w = _frame_w
    frame_h = _frame_h
    n_advanced = 0
    n_geometric = 0

    for e in events:
        if e.get("type") != "shot":
            continue

        x        = e.get("x", 0)
        y        = e.get("y", frame_h / 2)
        pressure = float(e.get("pressure", 0.0) or 0.0)
        phase    = e.get("shot_context", e.get("phase", "open_play")) or "open_play"
        action_before   = e.get("action_before", "none") or "none"
        sequence_length = int(e.get("sequence_length", 1) or 1)

        xg = None

        # Priorité 1 — modèle sklearn avancé (apprend de tes données réelles)
        if learner and learner.has_advanced_xg():
            xg = learner.predict_advanced_xg(e, frame_w=frame_w, frame_h=frame_h)
            if xg is not None:
                n_advanced += 1

        # Priorité 2 — modèle géométrique calibré (fallback)
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

    # Log du mode xG utilisé
    n_shots = n_advanced + n_geometric
    if n_shots > 0:
        if n_advanced > 0:
            print(f"  xG sklearn actif : {n_advanced}/{n_shots} tirs "
                  f"(modèle appris sur {len(learner.xg_advanced) if learner else 0} samples)")
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

    print(f"\nPIPELINE DONE")
    print(f"  {summary['goals']} buts | {summary['shots']} tirs | "
          f"xG: {summary['total_xg']} | {summary['players']} joueurs")
    print(f"  Formation: {formation} | Style: {tactical.get('style','?')}")
    print(f"  MVP: {mvp_label}")
    print(f"  Possession: {possession}")
    if is_summary:
        print(f"  ℹ️  Mode résumé détecté — paramètres adaptés")

    return result