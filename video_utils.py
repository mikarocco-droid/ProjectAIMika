# video_utils.py
# -*- coding: utf-8 -*-

import os
import subprocess


# ─────────────────────────────────────────
# SCORE DES EVENTS
# ─────────────────────────────────────────
def score_event(e):
    scores = {
        "goal":          10,
        "score":         10,
        "shot":           6,
        "interception":   5,
        "dribble":        4,
        "pass":           2,
        "possession":     1
    }
    return scores.get(e.get("type", ""), 1)


def frame_to_time(frame, fps=25):
    """FIX — fps=25 aligné sur la vidéo source."""
    return frame / fps if fps > 0 else 0


# ─────────────────────────────────────────
# MERGE EVENTS PROCHES
# ─────────────────────────────────────────
def merge_close_events(events, window=8, fps=25):
    merged = []
    events = sorted(events, key=lambda e: e.get("frame", 0))

    for e in events:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        # FIX — fenêtre en frames (8 secondes * fps)
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
        "-ss", str(max(0, start)),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────
# HIGHLIGHTS PRINCIPAL
# ─────────────────────────────────────────
def create_highlights(
    video_path,
    events,
    output_dir = "outputs/highlights",
    fps        = 25,
    max_clips  = 20
):
    os.makedirs(output_dir, exist_ok=True)

    # FIX — inclure "goal" explicitement
    key_events = [
        e for e in events
        if e.get("type") in ["goal", "score", "shot", "interception", "dribble"]
        and e.get("frame") is not None   # FIX — ignorer events sans frame
        and e.get("frame", 0) > 0        # FIX — ignorer frame 0
    ]

    if not key_events:
        print("  ⚠️ Aucun event clé avec timestamp valide")
        return []

    # Trier par importance puis dédupliquer
    key_events = sorted(key_events, key=score_event, reverse=True)
    key_events = key_events[:max_clips * 2]  # marge avant merge
    key_events = merge_close_events(key_events, fps=fps)
    key_events = key_events[:max_clips]

    highlights = []

    for i, e in enumerate(key_events):
        frame      = e.get("frame", 0)
        t          = frame_to_time(frame, fps)

        # FIX — contexte autour de l'action
        time_start = max(0, t - 5)
        time_end   = t + 4

        filename    = f"highlight_{i+1}_{e.get('type','action')}.mp4"
        output_path = os.path.join(output_dir, filename)

        cut_clip(video_path, time_start, time_end, output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"  ⚠️ Clip {i+1} vide, ignoré")
            continue

        highlights.append({
            "file":       output_path,
            "main_type":  e.get("type", "action"),
            "time_start": round(time_start, 2),
            "time_end":   round(time_end,   2),
            "score":      float(score_event(e)),
            "player":     e.get("player"),
            "team":       e.get("team"),
            "frame":      frame
        })

        mins = int(t // 60)
        secs = int(t % 60)
        print(f"  Clip {i+1} : {e.get('type')} à {mins:02d}:{secs:02d}")

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
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        os.remove(list_file)
    except:
        pass

    return output_path if os.path.exists(output_path) else None