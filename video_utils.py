# video_utils.py
# -*- coding: utf-8 -*-

import os
import subprocess


# ─────────────────────────────────────────
# SCORE DES EVENTS
# ─────────────────────────────────────────
def score_event(e, mode="match"):
    if mode == "player":
        scores = {
            "goal":           10,
            "score":          10,
            "shot":            7,
            "dribble":         6,
            "fast_break":      6,
            "progressive_run": 5,
            "interception":    4,
        }
    else:
        scores = {
            "goal":  10,
            "score": 10,
            "shot":   6,
        }
    return scores.get(e.get("type", ""), 0)


def frame_to_time(frame, fps=25):
    return frame / fps if fps > 0 else 0


# ─────────────────────────────────────────
# MERGE EVENTS PROCHES
# ─────────────────────────────────────────
def merge_close_events(events, window=8, fps=25, mode="match"):
    merged = []
    events = sorted(events, key=lambda e: e.get("frame", 0))

    for e in events:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        if abs(e.get("frame", 0) - last.get("frame", 0)) < window * fps:
            if score_event(e, mode) > score_event(last, mode):
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
        "-ss", str(max(0, start - 0.5)),
        "-to", str(end),
        "-i", video_path,
        "-ss", "0.5",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────
# HIGHLIGHTS
# mode="match"  → goals + tirs uniquement
# mode="player" → goals + tirs + dribbles + actions individuelles
# ─────────────────────────────────────────
def create_highlights(
    video_path,
    events,
    output_dir = "outputs/highlights",
    fps        = 25,
    max_clips  = 20,
    mode       = "match",  # "match" | "player"
    player_id  = None      # ID tracker du joueur ciblé (mode player)
):
    os.makedirs(output_dir, exist_ok=True)

    # Types autorisés selon le mode
    if mode == "player":
        allowed_types = [
            "goal", "score", "shot",
            "dribble", "progressive_run",
            "interception", "fast_break"
        ]
    else:
        allowed_types = ["goal", "score", "shot"]

    key_events = [
        e for e in events
        if e.get("type") in allowed_types
        and e.get("frame") is not None
        and e.get("frame", 0) > 0
        and e.get("gemini_type") not in ["touche", "corner", "none"]
    ]

    # Filtrer par joueur si mode player
    if mode == "player" and player_id is not None:
        key_events = [
            e for e in key_events
            if str(e.get("player")) == str(player_id)
        ]

    if not key_events:
        print("  ⚠️ Aucun event valide pour les highlights")
        return []

    # Trier : goals en premier, puis par score décroissant
    key_events = sorted(
        key_events,
        key=lambda e: (
            e.get("type") in ["goal", "score"],
            score_event(e, mode),
            e.get("xg", 0)
        ),
        reverse=True
    )
    key_events = key_events[:max_clips * 2]
    key_events = merge_close_events(key_events, fps=fps, mode=mode)
    key_events = key_events[:max_clips]

    highlights = []

    for i, e in enumerate(key_events):
        frame      = e.get("frame", 0)
        t          = frame_to_time(frame, fps)

        # Contexte selon type d'action et sport
        is_goal    = e.get("type") in ["goal", "score"]
        time_start = max(0, t - context_before)
        time_end   = t + (context_goal if is_goal else context_after)

        filename    = f"highlight_{i+1}_{e.get('type','shot')}.mp4"
        output_path = os.path.join(output_dir, filename)

        cut_clip(video_path, time_start, time_end, output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"  ⚠️ Clip {i+1} vide, ignoré")
            continue

        highlights.append({
            "file":       output_path,
            "main_type":  e.get("type", "shot"),
            "time_start": round(time_start, 2),
            "time_end":   round(time_end,   2),
            "score":      float(score_event(e, mode)),
            "player":     e.get("player"),
            "team":       e.get("team"),
            "frame":      frame,
            "xg":         e.get("xg", 0)
        })

        mins = int(t // 60)
        secs = int(t % 60)
        print(f"  Clip {i+1} : {e.get('type')} à {mins:02d}:{secs:02d} "
              f"(xG={e.get('xg', 0):.2f})")

    return highlights


# ─────────────────────────────────────────
# CREATE REEL FINAL
# ─────────────────────────────────────────
def create_highlight_reel(highlights, output_path="outputs/reel.mp4"):
    if not highlights:
        return None

    valid = [h for h in highlights if os.path.exists(h.get("file", ""))]
    if not valid:
        print("  ⚠️ Aucun clip valide pour le reel")
        return None

    list_file = output_path + "_clips.txt"
    with open(list_file, "w") as f:
        for h in valid:
            f.write(f"file '{os.path.abspath(h['file'])}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        os.remove(list_file)
    except:
        pass

    return output_path if os.path.exists(output_path) else None