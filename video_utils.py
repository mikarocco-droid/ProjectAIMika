# ─────────────────────────────────────────
# VIDEO V6 — HIGHLIGHTS INTELLIGENTS
# ─────────────────────────────────────────

import os
import subprocess


# ─────────────────────────────────────────
# SCORE DES EVENTS (importance)
# ─────────────────────────────────────────
def score_event(e):

    scores = {
        "score": 10,
        "goal": 10,
        "shot": 6,
        "interception": 5,
        "dribble": 4,
        "pass": 2,
        "possession": 1
    }

    return scores.get(e["type"], 1)


# ─────────────────────────────────────────
# CONVERT FRAME → SECONDES
# ─────────────────────────────────────────
def frame_to_time(frame, fps=30):
    return frame / fps


# ─────────────────────────────────────────
# MERGE EVENTS PROCHES (évite clips doublons)
# ─────────────────────────────────────────
def merge_close_events(events, window=8):

    merged = []

    events = sorted(events, key=lambda e: e["frame"])

    for e in events:

        if not merged:
            merged.append(e)
            continue

        last = merged[-1]

        if abs(e["frame"] - last["frame"]) < window * 30:
            # garder le plus important
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
        "ffmpeg",
        "-y",
        "-ss", str(start),
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
    output_dir="outputs/highlights",
    fps=30,
    max_clips=20
):

    os.makedirs(output_dir, exist_ok=True)

    # 🔥 filtrer events intéressants
    key_events = [
        e for e in events
        if e["type"] in ["shot", "score", "interception", "dribble"]
    ]

    if not key_events:
        return []

    # 🔥 trier par importance
    key_events = sorted(key_events, key=score_event, reverse=True)

    # 🔥 limiter nombre
    key_events = key_events[:max_clips]

    # 🔥 éviter doublons proches
    key_events = merge_close_events(key_events)

    highlights = []

    for i, e in enumerate(key_events):

        t = frame_to_time(e["frame"], fps)

        # 🎯 contexte autour action
        start = max(0, t - 6)
        end   = t + 4

        filename = f"highlight_{i+1}_{e['type']}.mp4"
        output_path = os.path.join(output_dir, filename)

        cut_clip(video_path, start, end, output_path)

        highlights.append({
            "file": output_path,
            "type": e["type"],
            "time": round(t, 2)
        })

    return highlights


# ─────────────────────────────────────────
# CREATE REEL FINAL (montage)
# ─────────────────────────────────────────
def create_highlight_reel(highlights, output_path="outputs/reel.mp4"):

    if not highlights:
        return None

    list_file = "clips.txt"

    with open(list_file, "w") as f:
        for h in highlights:
            f.write(f"file '{os.path.abspath(h['file'])}'\n")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ])

    return output_path


# ─────────────────────────────────────────
# TEST LOCAL
# ─────────────────────────────────────────
if __name__ == "__main__":

    fake_events = [
        {"type": "shot", "frame": 300},
        {"type": "score", "frame": 900},
        {"type": "interception", "frame": 1500},
        {"type": "dribble", "frame": 1800},
    ]

    video = "match.mp4"

    highlights = create_highlights(video, fake_events)
    reel = create_highlight_reel(highlights)

    print("\n🎬 Highlights générés :")
    for h in highlights:
        print(h)

    print("\n🎥 Reel final :", reel)