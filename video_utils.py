# video_utils.py
# -*- coding: utf-8 -*-

import os
import subprocess

from sports.config import get_sport_config, get_highlight_types

# ─────────────────────────────────────────
# SEUIL xG minimum pour inclure un tir
# 0.50 = tir vraiment dangereux
# Évite les faux positifs quand sklearn pas encore actif
# ─────────────────────────────────────────
XG_MIN_FOR_HIGHLIGHT = 0.50


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
# FILTRE QUALITÉ SHOT
# Un shot est inclus dans le reel si :
#   1. xG >= 0.50 (tir vraiment dangereux)
#   2. on_target = True (tir cadré)
#   3. shot_blocked dans les 3s (gardien)
#   4. Gemini a confirmé "shot"
# Les buts sont TOUJOURS inclus
# ─────────────────────────────────────────
def _shot_qualifies(e, all_events):
    xg        = float(e.get("xg", 0) or 0)
    on_target = bool(e.get("on_target", False))
    gemini_ok = (e.get("gemini_validated", False)
                 and e.get("gemini_type") == "shot")
    t         = e.get("time", 0)

    if xg >= XG_MIN_FOR_HIGHLIGHT:
        return True

    if on_target and xg >= 0.25:
        return True

    has_blocked = any(
        ev.get("type") == "shot_blocked"
        and 0 <= ev.get("time", 0) - t <= 3.0
        for ev in all_events
    )
    if has_blocked:
        return True

    if gemini_ok:
        return True

    return False


# ─────────────────────────────────────────
# MERGE EVENTS PROCHES
# Les buts ne sont JAMAIS fusionnés
# ─────────────────────────────────────────
def merge_close_events(events, window=8, fps=25, mode="match"):
    merged = []
    events = sorted(events, key=lambda e: e.get("frame", 0))

    for e in events:
        if not merged:
            merged.append(e)
            continue

        last         = merged[-1]
        is_goal      = e.get("type") in ["goal", "score"]
        last_is_goal = last.get("type") in ["goal", "score"]

        if is_goal or last_is_goal:
            merged.append(e)
            continue

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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────
# HIGHLIGHTS
# ─────────────────────────────────────────
def create_highlights(
    video_path,
    events,
    output_dir = "outputs/highlights",
    fps        = 25,
    max_clips  = 20,
    mode       = "match",
    player_id  = None,
    sport      = "football"
):
    os.makedirs(output_dir, exist_ok=True)

    allowed_types  = get_highlight_types(sport, mode)
    cfg            = get_sport_config(sport)
    context_before = cfg.get("context_before", 12)
    context_after  = cfg.get("context_after",   4)
    context_goal   = cfg.get("context_goal",     5)

    key_events = []
    for e in events:
        etype = e.get("type", "")

        if e.get("gemini_type") in ["touche", "corner", "none",
                                     "defensive_clearance",
                                     "goalkeeper_hold", "goalkeeper_throw"]:
            continue

        if etype not in allowed_types:
            continue

        if e.get("frame") is None or e.get("frame", 0) <= 0:
            continue

        if mode == "match":
            if etype in ["goal", "score"]:
                key_events.append(e)
            elif etype == "shot":
                if _shot_qualifies(e, events):
                    # Ne pas inclure un tir si un but arrive dans les 15s suivantes
                    t_shot = e.get("time", 0)
                    too_close_to_goal = any(
                        g.get("type") in ["goal", "score"]
                        and 0 <= g.get("time", 0) - t_shot <= 15
                        for g in events
                    )
                    if not too_close_to_goal:
                        key_events.append(e)
        else:
            key_events.append(e)

    if mode == "player" and player_id is not None:
        key_events = [
            e for e in key_events
            if str(e.get("player")) == str(player_id)
        ]

    if not key_events:
        print("  Aucun event valide pour les highlights")
        return []

    n_goals = sum(1 for e in key_events if e.get("type") in ["goal", "score"])
    n_shots = sum(1 for e in key_events if e.get("type") == "shot")
    print(f"  Highlights sélectionnés : {n_goals} buts + {n_shots} tirs qualifiés")

    key_events = sorted(
        key_events,
        key=lambda e: (
            e.get("type") in ["goal", "score"],
            e.get("xg", 0),
            score_event(e, mode),
        ),
        reverse=True
    )
    key_events = key_events[:max_clips * 2]
    key_events = merge_close_events(key_events, fps=fps, mode=mode)
    key_events = sorted(key_events, key=lambda e: e.get("frame", 0))
    key_events = key_events[:max_clips]

    highlights = []

    for i, e in enumerate(key_events):
        frame      = e.get("frame", 0)
        t          = frame_to_time(frame, fps)
        is_goal    = e.get("type") in ["goal", "score"]
        time_start = max(0, t - context_before)
        time_end   = t + (context_goal if is_goal else context_after)

        filename    = f"highlight_{i+1}_{e.get('type','shot')}.mp4"
        output_path = os.path.join(output_dir, filename)

        cut_clip(video_path, time_start, time_end, output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"  Clip {i+1} vide, ignoré")
            continue

        reason = "goal"
        if e.get("type") == "shot":
            xg = float(e.get("xg", 0) or 0)
            if xg >= XG_MIN_FOR_HIGHLIGHT:
                reason = f"xG={xg:.3f}"
            elif e.get("on_target"):
                reason = "on_target"
            elif e.get("gemini_validated") and e.get("gemini_type") == "shot":
                reason = "gemini_shot"
            else:
                reason = "shot_blocked"

        highlights.append({
            "file":       output_path,
            "main_type":  e.get("type", "shot"),
            "time_start": round(time_start, 2),
            "time_end":   round(time_end,   2),
            "score":      float(score_event(e, mode)),
            "player":     e.get("player"),
            "team":       e.get("team"),
            "frame":      frame,
            "xg":         e.get("xg", 0),
            "on_target":  e.get("on_target", False),
            "reason":     reason,
        })

        mins = int(t // 60)
        secs = int(t % 60)
        print(f"  Clip {i+1} : {e.get('type')} à {mins:02d}:{secs:02d} "
              f"(xG={e.get('xg', 0):.3f} | {reason})")

    return highlights


# ─────────────────────────────────────────
# CREATE REEL FINAL
# ─────────────────────────────────────────
def create_highlight_reel(highlights, output_path="outputs/reel.mp4"):
    if not highlights:
        return None

    reel_events = [
        h for h in highlights
        if h.get("main_type") in ["goal", "score"]
        or h.get("xg", 0) >= XG_MIN_FOR_HIGHLIGHT
        or h.get("on_target", False)
    ]

    if not reel_events:
        reel_events = [
            h for h in highlights
            if h.get("main_type") in ["goal", "score"]
        ]

    if not reel_events:
        print("  Aucun clip valide pour le reel")
        return None

    valid = [h for h in reel_events if os.path.exists(h.get("file", ""))]
    if not valid:
        print("  Aucun clip valide pour le reel")
        return None

    list_file = output_path + "_clips.txt"
    with open(list_file, "w") as f:
        for h in valid:
            f.write(f"file '{os.path.abspath(h['file'])}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        os.remove(list_file)
    except Exception:
        pass

    return output_path if os.path.exists(output_path) else None