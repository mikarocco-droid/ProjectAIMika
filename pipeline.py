# pipeline.py
# -*- coding: utf-8 -*-

import os
import json

import config
from main import process_video
from analytics.stats import compute_stats
from analytics.heatmap import generate_all_heatmaps
from video_utils import create_highlights, create_highlight_reel
from video.montage import create_montage
from ai.claude import summarize
from export.pdf import generate_pdf
from vision.calibration import calibrate


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
# xG
# ─────────────────────────────────────────
def compute_xg(x, y, frame_width=None):
    frame_width  = frame_width or config.FRAME_WIDTH
    normalized_x = x / frame_width
    dist_to_goal = 1.0 - normalized_x
    xg = max(0.02, min(0.9, 1.0 - dist_to_goal * 1.5))
    return round(xg, 3)


def enrich_events_with_xg(events, frame_width=None):
    for e in events:
        if e.get("type") == "shot":
            x = e.get("x")
            y = e.get("y")
            e["xg"] = compute_xg(x, y, frame_width) if x and y else 0.1
    return events


# ─────────────────────────────────────────
# MATCH SUMMARY
# ─────────────────────────────────────────
def compute_match_summary(events, stats, total_frames=0, fps=30):

    total_xg = sum(
        e.get("xg", 0)
        for e in events
        if e.get("type") == "shot"
    )

    duration_sec = int(total_frames / fps) if fps > 0 else 0
    minutes      = duration_sec // 60
    seconds      = duration_sec % 60

    return {
        "total_events":  len(events),
        "goals":         sum(1 for e in events if e.get("type") in ["goal", "score"]),
        "shots":         sum(1 for e in events if e.get("type") == "shot"),
        "passes":        sum(1 for e in events if e.get("type") == "pass"),
        "interceptions": sum(1 for e in events if e.get("type") == "interception"),
        "dribbles":      sum(1 for e in events if e.get("type") == "dribble"),
        "total_xg":      round(total_xg, 2),
        "players":       len(stats),
        "duration":      f"{minutes:02d}:{seconds:02d}",
        "total_frames":  total_frames,
        "fps":           fps
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
    # 0. CALIBRATION AUTOMATIQUE
    # ─────────────────────────────────────────
    print("Step 0 : Calibration camera...")
    calib      = None
    shot_zones = None

    try:
        calib = calibrate(video_path, sport)

        # Mettre à jour la zone de jeu dans le detector
        if calib and calib.get("play_zone"):
            from vision.detector import PLAY_ZONES
            PLAY_ZONES[sport] = calib["play_zone"]
            print(f"  Zone de jeu mise a jour : {calib['play_zone']}")

        # Récupérer les zones de tir calibrées
        if calib and calib.get("shot_zones"):
            shot_zones = calib["shot_zones"]
            print(f"  Shot zones calibrees : {shot_zones}")

    except Exception as e:
        print(f"  Calibration ignoree : {e}")

    # ─────────────────────────────────────────
    # 1. TRACKING + EVENTS
    # ─────────────────────────────────────────
    print("Step 1 : Tracking + Events...")

    annotated_path = os.path.join(output_dir, "annotated.mp4") \
        if save_annotated else None

    events, jersey_map, fps, total_frames = process_video(
        video_path        = video_path,
        sport             = sport,
        progress_callback = progress,
        save_annotated    = save_annotated,
        annotated_path    = annotated_path,
        shot_zones        = shot_zones      # ← calibration passée ici
    )

    print(f"  OK {len(events)} events | {len(jersey_map)} maillots")

    # ─────────────────────────────────────────
    # 2. xG
    # ─────────────────────────────────────────
    print("Step 2 : xG...")
    events = enrich_events_with_xg(events, frame_width=config.FRAME_WIDTH)

    # ─────────────────────────────────────────
    # 3. STATS
    # ─────────────────────────────────────────
    print("Step 3 : Stats...")
    stats = compute_stats(events)
    print(f"  OK {len(stats)} joueurs")

    # ─────────────────────────────────────────
    # 4. HEATMAPS
    # ─────────────────────────────────────────
    print("Step 4 : Heatmaps...")
    heatmaps = generate_all_heatmaps(
        events     = events,
        output_dir = os.path.join(output_dir, "heatmaps"),
        width      = config.FRAME_WIDTH,
        height     = config.FRAME_HEIGHT,
        sport      = sport
    )
    heatmap_path = heatmaps.get("global")
    print(f"  OK {len(heatmaps)} heatmaps")

    # ─────────────────────────────────────────
    # 5. HIGHLIGHTS
    # ─────────────────────────────────────────
    print("Step 5 : Highlights...")
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

    # ─────────────────────────────────────────
    # 6. MONTAGE
    # ─────────────────────────────────────────
    print("Step 6 : Montage...")
    montage_path = create_montage(
        highlights  = highlights,
        video_path  = video_path,
        output_path = os.path.join(output_dir, "montage.mp4"),
        title       = f"Analyse {sport.capitalize()}",
        with_intro  = config.MONTAGE_WITH_INTRO,
        with_labels = config.MONTAGE_WITH_LABELS,
        with_fades  = config.MONTAGE_WITH_FADES
    )
    print(f"  OK Montage -> {montage_path}")

    # ─────────────────────────────────────────
    # 7. SUMMARY
    # ─────────────────────────────────────────
    print("Step 7 : Summary...")
    summary = compute_match_summary(events, stats, total_frames, fps)

    # ─────────────────────────────────────────
    # 8. AI
    # ─────────────────────────────────────────
    print("Step 8 : AI summary...")
    ai_summary = None
    if config.CLAUDE_API_KEY:
        try:
            ai_summary = summarize(highlights)
            print("  OK Resume IA genere")
        except Exception as e:
            print(f"  AI error : {e}")
    else:
        print("  CLAUDE_API_KEY manquante")

    # ─────────────────────────────────────────
    # 9. PDF
    # ─────────────────────────────────────────
    print("Step 9 : PDF...")
    pdf_path = None
    if config.PLANS.get(plan, {}).get("pdf", False):
        try:
            pdf_path = generate_pdf(
                result      = {
                    "summary":    summary,
                    "stats":      stats,
                    "highlights": highlights,
                    "jersey_map": jersey_map,
                    "heatmap":    heatmap_path,
                    "ai_summary": ai_summary
                },
                output_path = os.path.join(output_dir, "rapport.pdf"),
                sport       = sport
            )
            print(f"  OK PDF -> {pdf_path}")
        except Exception as e:
            print(f"  PDF error : {e}")
    else:
        print("  PDF non disponible sur ce plan")

    # ─────────────────────────────────────────
    # 10. SAVE JSON
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
        "total_frames": total_frames
    }

    with open(os.path.join(output_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nPIPELINE DONE")
    print(f"  {summary['goals']} buts | {summary['shots']} tirs | xG: {summary['total_xg']}")

    return result