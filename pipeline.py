# pipeline.py
# -*- coding: utf-8 -*-

import os
import json

import config
from main import process_video
from analytics.stats import compute_stats
from analytics.heatmap import generate_all_heatmaps
from analytics.advanced import (
    build_pass_network,
    detect_offside,
    compute_team_dominance,
    compute_action_score,
    extract_pass_sequences
)
from video_utils import create_highlights, create_highlight_reel
from video.montage import create_montage
from analysis.events import detect_events
from analysis.tactical import (
    tactical_report,
    assign_teams,
    detect_formation,
    detect_pressing,
    detect_phases
)
from analysis.highlight_ranker import rank_highlights
from analysis.player_rating import compute_player_ratings, get_mvp
from analysis.match_story import generate_match_story
from ai.commentary import generate_commentary
from ai.learning import cluster_actions, learn_action_importance, detect_key_moments
from sports.config import get_sport_config, compute_xg_sport


# ─────────────────────────────────────────
# NORMALIZE HIGHLIGHTS
# ─────────────────────────────────────────
def normalize_highlights(highlights):
    fixed = []
    for h in highlights:
        if "time_start" not in h:
            h["time_start"] = h.get("timestamp_debut", 0)
        if "time_end" not in h:
            h["time_end"] = h.get("timestamp_fin", h["time_start"] + 3)
        if "main_type" not in h:
            h["main_type"] = h.get("type", "action")
        if "score" not in h:
            h["score"] = 1.0
        fixed.append(h)
    return fixed


# ─────────────────────────────────────────
# SANITIZE JSON — corrige les clés tuples
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
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_pipeline(
    video_path,
    sport          = "football",
    output_dir     = "outputs",
    analysis_id    = None,
    save_annotated = False,
    plan           = "free"
):
    os.makedirs(output_dir, exist_ok=True)
    progress = make_progress_callback(analysis_id)

    print(f"\nPIPELINE START - {sport.upper()}")

    # ─────────────────────────────────────────
    # 0. CALIBRATION
    # ─────────────────────────────────────────
    print("Step 0 : Calibration...")
    calib      = None
    shot_zones = None

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
    print(f"  OK {len(events)} events | {len(jersey_map)} maillots")

    # ─────────────────────────────────────────
    # 2. ENRICH xG
    # ─────────────────────────────────────────
    print("Step 2 : xG...")
    cfg = get_sport_config(sport)
    for e in events:
        if e.get("type") == "shot":
            xg = compute_xg_sport(e.get("x", 0), sport)
            e["xg"] = min(xg, 0.5)  # FIX — plafonné à 0.5

    # ─────────────────────────────────────────
    # 3. STATS
    # FIX — jersey_map passé pour numéros maillot
    # ─────────────────────────────────────────
    print("Step 3 : Stats...")
    stats = compute_stats(events, jersey_map=jersey_map)
    print(f"  OK {len(stats)} joueurs")

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
            frame_h        = int(frames_data[0].get("frame_h", 720)) if frames_data else 720
        )
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

        # FIX — convertir clés tuples en string
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
            max_clips  = config.HIGHLIGHT_MAX
        )
        highlights = normalize_highlights(highlights)

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
    print("Step 9 : Montage...")
    montage_path = None
    try:
        montage_path = create_montage(
            highlights = highlights,
            video_path = video_path,
            output     = os.path.join(output_dir, "montage.mp4"),
            title      = f"Analyse {sport.capitalize()}",
        )
        print(f"  OK montage -> {montage_path}")
    except Exception as e:
        print(f"  Montage error : {e}")

    # ─────────────────────────────────────────
    # 10. RANKINGS + RATINGS + COMMENTARY
    # FIX — fps passé à generate_match_story pour les minutes correctes
    # ─────────────────────────────────────────
    print("Step 10 : Ratings + Commentary...")
    ratings    = {}
    mvp        = None
    commentary = []
    story      = ""

    try:
        ranked_highlights = rank_highlights(events)
        ratings           = compute_player_ratings(events)
        mvp               = get_mvp(ratings)
        commentary        = generate_commentary(ranked_highlights[:10])
        story             = generate_match_story(events, fps=fps)  # FIX — fps
        print(f"  OK MVP={mvp[0] if mvp else '?'} | commentary={len(commentary)} lines")
    except Exception as e:
        print(f"  Ratings error : {e}")

    # ─────────────────────────────────────────
    # 11. SUMMARY
    # ─────────────────────────────────────────
    print("Step 11 : Summary...")
    summary = compute_match_summary(events, stats, total_frames, fps)

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
    # FIX — jersey_map + mvp passés correctement
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
                    "mvp":            str(mvp[0]) if mvp else None,
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
        "summary":      summary,
        "events":       events,
        "stats":        stats,
        "highlights":   highlights,
        "jersey_map":   jersey_map,
        "heatmaps":     heatmaps,
        "heatmap":      heatmap_path,
        "reel":         reel_path,
        "montage":      montage_path,
        "annotated":    annotated_path,
        "ai_summary":   ai_summary,
        "pdf":          pdf_path,
        "sport":        sport,
        "calib":        calib,
        "fps":          fps,
        "total_frames": total_frames,
        "teams":        teams,
        "formation":    formation,
        "pressing":     pressing,
        "phases":       phases,
        "tactical":     tactical,
        "pass_network": pass_network,
        "offsides":     offsides,
        "dominance":    dominance,
        "key_moments":  key_moments,
        "ratings":      ratings,
        "mvp":          str(mvp[0]) if mvp else None,
        "commentary":   commentary,
        "story":        story
    }

    # FIX — sanitize avant json.dump
    result = sanitize_for_json(result)

    with open(os.path.join(output_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nPIPELINE DONE")
    print(f"  {summary['goals']} buts | {summary['shots']} tirs | "
          f"xG: {summary['total_xg']} | {summary['players']} joueurs")
    print(f"  Formation: {formation} | Style: {tactical.get('style','?')}")
    print(f"  MVP: {mvp[0] if mvp else '?'}")

    return result