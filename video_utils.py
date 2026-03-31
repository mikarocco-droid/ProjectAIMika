# video_utils.py
# -*- coding: utf-8 -*-

import os
import subprocess


# ─────────────────────────────────────────
# SCORE DES EVENTS
# ─────────────────────────────────────────
def score_event(e):
    scores = {
        "score":        10,
        "goal":         10,
        "shot":          6,
        "interception":  5,
        "dribble":       4,
        "pass":          2,
        "possession":    1
    }
    return scores.get(e.get("type", ""), 1)


def frame_to_time(frame, fps=30):
    return frame / fps


# ─────────────────────────────────────────
# MERGE EVENTS PROCHES
# ─────────────────────────────────────────
def merge_close_events(events, window=8, fps=30):
    merged = []
    events = sorted(events, key=lambda e: e.get("frame", 0))

    for e in events:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        if abs(e.get("frame", 0) - last.get("frame", 0)) < window * fps:
            if score_event(e) > score_event(last):
                merged[-1] = e
        else:
            merged.append(e)

    return merged


# ─────────────────────────────────────────
# EXTRACTION CLIP
# ─────────────────────────────────────────
def cut_clip(video_path, start, end, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────
# HIGHLIGHTS PRINCIPAL
# FIX — retourne des dicts avec time_start/time_end/main_type/file
#       cohérents avec normalize_highlights() et create_highlight_reel()
# ─────────────────────────────────────────
def create_highlights(
    video_path,
    events,
    output_dir = "outputs/highlights",
    fps        = 30,
    max_clips  = 20
):
    os.makedirs(output_dir, exist_ok=True)

    key_events = [
        e for e in events
        if e.get("type") in ["shot", "goal", "score", "interception", "dribble"]
    ]

    if not key_events:
        return []

    key_events = sorted(key_events, key=score_event, reverse=True)
    key_events = key_events[:max_clips]
    key_events = merge_close_events(key_events, fps=fps)

    highlights = []

    for i, e in enumerate(key_events):
        t          = frame_to_time(e.get("frame", 0), fps)
        time_start = max(0, t - 6)
        time_end   = t + 4

        filename    = f"highlight_{i+1}_{e.get('type','action')}.mp4"
        output_path = os.path.join(output_dir, filename)

        cut_clip(video_path, time_start, time_end, output_path)

        # FIX — structure unifiée attendue par pipeline + normalize_highlights
        highlights.append({
            "file":       output_path,   # pour create_highlight_reel
            "main_type":  e.get("type", "action"),
            "time_start": round(time_start, 2),
            "time_end":   round(time_end,   2),
            "score":      float(score_event(e)),
            "player":     e.get("player"),
            "team":       e.get("team"),
            "frame":      e.get("frame", 0)
        })

    return highlights


# ─────────────────────────────────────────
# CREATE REEL FINAL
# FIX — utilise h["file"] de façon cohérente
#       stderr supprimé pour éviter pollution logs
# ─────────────────────────────────────────
def create_highlight_reel(highlights, output_path="outputs/reel.mp4"):
    if not highlights:
        return None

    # Filtrer les clips qui existent vraiment
    valid = [h for h in highlights if os.path.exists(h.get("file", ""))]

    if not valid:
        print("  ⚠️  Aucun clip valide pour le reel")
        return None

    list_file = output_path + "_clips.txt"

    with open(list_file, "w") as f:
        for h in valid:
            f.write(f"file '{os.path.abspath(h['file'])}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Nettoyage fichier temporaire
    try:
        os.remove(list_file)
    except:
        pass

    return output_path if os.path.exists(output_path) else None