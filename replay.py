# replay.py
# -*- coding: utf-8 -*-
#
# Replay Engine — rejoue le pipeline à partir d'un cache sauvegardé.
#
# Usage typique dans le notebook Kaggle :
#
#   from replay import save_cache, load_cache, CACHE_AVAILABLE
#
#   # Premier run (complet) :
#   REPLAY = False
#
#   # Runs suivants (rapides) :
#   REPLAY = True
#   CACHE_DIR = "/kaggle/input/scoutia-cache/match_test"
#
# Ce qui est sauvegardé après le tracking (Step 1) :
#   events.pkl       — tous les events bruts
#   frames_data.pkl  — frames avec positions joueurs/ballon
#   jersey_map.pkl   — maillots identifiés
#   meta.pkl         — fps, total_frames, frame_w, frame_h, sport, video_path
#   team_colors.pkl  — couleurs équipes calibrées par TeamColorDetector
#
# Ce qui est REJOUÉ (Steps 1b → fin) :
#   → Gemini validation
#   → possession / stats
#   → buts / highlights
#
# Ce qui est SKIPPÉ en replay :
#   → YOLO (4h sur 1h47)
#   → DeepSORT tracking
#   → goal_posthoc (déjà dans events.pkl)
#   → terminal_events (déjà dans events.pkl)

import os
import pickle
import json
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(
    output_dir,
    events,
    frames_data,
    jersey_map,
    fps,
    total_frames,
    frame_w,
    frame_h,
    sport,
    video_path,
    team_colors=None,
    kickoff_offset=0.0,
    camera_profile=None,
):
    """
    Sauvegarde le cache après le Step 1 (tracking).
    À appeler dans pipeline.py juste avant le Step 1b.
    """
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Events — on fait une copie légère (sans frames_data qui est lourd)
    pickle.dump(events,      open(os.path.join(cache_dir, "events.pkl"),      "wb"), protocol=4)
    pickle.dump(jersey_map,  open(os.path.join(cache_dir, "jersey_map.pkl"),  "wb"), protocol=4)
    pickle.dump(team_colors, open(os.path.join(cache_dir, "team_colors.pkl"), "wb"), protocol=4)

    # frames_data peut être très lourd (~500 MB sur 1h47)
    # On sauvegarde uniquement les champs utiles pour la suite du pipeline
    _frames_lite = _slim_frames_data(frames_data)
    pickle.dump(_frames_lite, open(os.path.join(cache_dir, "frames_data.pkl"), "wb"), protocol=4)

    # Métadonnées
    meta = {
        "fps":            fps,
        "total_frames":   total_frames,
        "frame_w":        frame_w,
        "frame_h":        frame_h,
        "sport":          sport,
        "video_path":     video_path,
        "kickoff_offset": kickoff_offset,
        "camera_profile": camera_profile or {},
        "saved_at":       datetime.now().isoformat(),
        "n_events":       len(events),
        "n_frames":       len(frames_data),
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    _size = _dir_size_mb(cache_dir)
    print(f"  [REPLAY] Cache sauvegardé : {cache_dir}")
    print(f"  [REPLAY] {len(events)} events | {len(_frames_lite)} frames | {_size:.0f} MB")
    print(f"  [REPLAY] Pour rejouer : REPLAY=True, CACHE_DIR='{cache_dir}'")

    return cache_dir


def _slim_frames_data(frames_data):
    """
    Réduit frames_data aux champs strictement nécessaires pour la suite du pipeline.
    Supprime les images brutes (_frame_orig, _frame_rgb) qui gonflent la taille.
    Garde : frame, frame_id, frame_w, frame_h, players (id, bbox, center, team, color),
            ball (center, x, bbox, conf), fps.
    """
    slim = []
    for f in frames_data:
        players_slim = []
        for p in (f.get("players") or []):
            players_slim.append({
                "id":         p.get("id"),
                "player_id":  p.get("player_id"),
                "tracker_id": p.get("tracker_id"),
                "bbox":       p.get("bbox"),
                "center":     p.get("center"),
                "team":       p.get("team"),
                "color":      p.get("color"),
                "jersey":     p.get("jersey"),
                "conf":       p.get("conf"),
            })

        ball = f.get("ball") or {}
        ball_slim = {
            "center":      ball.get("center"),
            "x":           ball.get("x"),
            "bbox":        ball.get("bbox"),
            "conf":        ball.get("conf"),
            "interpolated": ball.get("interpolated"),
            "frame":       ball.get("frame"),
        } if ball else {}

        slim.append({
            "frame":    f.get("frame"),
            "frame_id": f.get("frame_id", f.get("frame")),
            "frame_w":  f.get("frame_w"),
            "frame_h":  f.get("frame_h"),
            "fps":      f.get("fps"),
            "players":  players_slim,
            "ball":     ball_slim,
        })
    return slim


def _dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_dir):
    """
    Charge le cache et retourne un dict avec toutes les données.
    Lève FileNotFoundError si le cache n'existe pas.

    Retourne :
        {
            "events":         [...],
            "frames_data":    [...],
            "jersey_map":     {...},
            "team_colors":    {...},
            "fps":            25.0,
            "total_frames":   ...,
            "frame_w":        1920,
            "frame_h":        1080,
            "sport":          "football",
            "video_path":     "...",
            "kickoff_offset": 0.0,
            "camera_profile": {...},
        }
    """
    required = ["events.pkl", "frames_data.pkl", "jersey_map.pkl", "meta.json"]
    for f in required:
        path = os.path.join(cache_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache incomplet — fichier manquant : {path}\n"
                f"Lance d'abord un run complet avec REPLAY=False."
            )

    events      = pickle.load(open(os.path.join(cache_dir, "events.pkl"),      "rb"))
    frames_data = pickle.load(open(os.path.join(cache_dir, "frames_data.pkl"), "rb"))
    jersey_map  = pickle.load(open(os.path.join(cache_dir, "jersey_map.pkl"),  "rb"))

    team_colors_path = os.path.join(cache_dir, "team_colors.pkl")
    team_colors = pickle.load(open(team_colors_path, "rb")) if os.path.exists(team_colors_path) else {}

    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)

    print(f"  [REPLAY] Cache chargé : {cache_dir}")
    print(f"  [REPLAY] {len(events)} events | {len(frames_data)} frames")
    print(f"  [REPLAY] video={meta.get('video_path')} | fps={meta.get('fps')} | "
          f"sport={meta.get('sport')} | saved={meta.get('saved_at', '?')[:10]}")

    return {
        "events":         events,
        "frames_data":    frames_data,
        "jersey_map":     jersey_map,
        "team_colors":    team_colors or {},
        "fps":            float(meta["fps"]),
        "total_frames":   int(meta["total_frames"]),
        "frame_w":        int(meta["frame_w"]),
        "frame_h":        int(meta["frame_h"]),
        "sport":          meta["sport"],
        "video_path":     meta["video_path"],
        "kickoff_offset": float(meta.get("kickoff_offset", 0.0)),
        "camera_profile": meta.get("camera_profile", {}),
    }


def cache_available(cache_dir):
    """Retourne True si le cache existe et est complet."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return False
    required = ["events.pkl", "frames_data.pkl", "jersey_map.pkl", "meta.json"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required)
