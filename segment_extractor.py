# -*- coding: utf-8 -*-
"""
segment_extractor.py — Extraction et analyse des segments chauds

Remplace le process_video() full-match par :
1. extract_segments()  → clips MP4 via ffmpeg (rapide, -c copy)
2. analyze_segments()  → process_video() uniquement sur ces clips
3. merge_segment_results() → recolle les timestamps

Gain : YOLO/tracking uniquement sur ~10-20% du match.
"""

import os
import subprocess
import math


# ─────────────────────────────────────────
# EXTRACTION FFMPEG
# ─────────────────────────────────────────
def extract_segments(video_path, segments, output_dir, margin_s=0.5):
    """
    Extrait les segments chauds en clips MP4 via ffmpeg (-c copy = ultra rapide).

    Args:
        video_path  : chemin vidéo source
        segments    : [(start_s, end_s), ...]
        output_dir  : dossier de sortie
        margin_s    : marge de sécurité ajoutée de chaque côté

    Returns:
        [(clip_path, start_s, end_s), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Durée totale via ffprobe
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        )
        total_duration = float(probe.stdout.strip())
    except Exception:
        total_duration = 9999.0

    for i, (start, end) in enumerate(segments):
        # Appliquer la marge
        seg_start = max(0.0,           start - margin_s)
        seg_end   = min(total_duration, end   + margin_s)

        out_path = os.path.join(output_dir, f"segment_{i:03d}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.3f}",
            "-to", f"{seg_end:.3f}",
            "-i", video_path,
            "-c", "copy",          # pas de re-encode → ultra rapide
            "-avoid_negative_ts", "make_zero",
            out_path
        ]

        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode == 0 and os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  [EXTRACT] Segment {i+1} : "
                  f"{int(seg_start//60):02d}:{int(seg_start%60):02d} → "
                  f"{int(seg_end//60):02d}:{int(seg_end%60):02d} "
                  f"({seg_end-seg_start:.1f}s | {size_mb:.1f} MB)")
            results.append((out_path, seg_start, seg_end))
        else:
            print(f"  [EXTRACT] ⚠️  Segment {i+1} échoué : {ret.stderr.decode()[:100]}")

    return results


# ─────────────────────────────────────────
# REMAPPING TIMESTAMPS
# ─────────────────────────────────────────
def remap_events(events, seg_start, fps):
    """
    Remappe les timestamps d'un segment vers le temps absolu du match.
    Le clip commence à t=0 mais dans le match il commence à seg_start.
    """
    for e in events:
        e["time"]  = round(e.get("time",  0) + seg_start, 3)
        e["frame"] = int(e.get("frame", 0) + seg_start * fps)
    return events


def remap_frames_data(frames_data, seg_start, fps):
    """Remappe les frames_data d'un segment."""
    for fd in frames_data:
        fd["frame"]      = int(fd.get("frame", 0) + seg_start * fps)
        fd["frame_time"] = round(fd.get("frame_time", 0) + seg_start, 3)
    return frames_data


# ─────────────────────────────────────────
# ANALYSE DES SEGMENTS
# ─────────────────────────────────────────
def _analyze_single_segment(args):
    """Worker pour analyse parallèle d'un segment."""
    clip_path, seg_start, seg_end, sport, shot_zones, frame_skip, batch_size, imgsz, idx, total = args
    try:
        from main import process_video
        events, jersey_map, fps, total_frames, frames_data = process_video(
            video_path       = clip_path,
            sport            = sport,
            save_annotated   = False,
            annotated_path   = None,
            shot_zones       = shot_zones,
            return_frames    = True,
            frame_skip_every = frame_skip,
            batch_size       = batch_size,
            imgsz            = imgsz,
        )
        fps = fps or 25.0
        events     = remap_events(events, seg_start, fps)
        frames_data = remap_frames_data(frames_data, seg_start, fps)
        print(f"  [DEEP] Segment {idx+1}/{total} OK : "
              f"{int(seg_start//60):02d}:{int(seg_start%60):02d} → "
              f"{int(seg_end//60):02d}:{int(seg_end%60):02d} | "
              f"{len(events)} events")
        return events, frames_data, fps, jersey_map
    except Exception as e:
        print(f"  [DEEP] ⚠️  Segment {idx+1} échoué : {e}")
        return [], [], 25.0, {}


def analyze_segments(
    segment_clips,
    sport         = "football",
    shot_zones    = None,
    frame_skip    = 2,
    batch_size    = 8,
    imgsz         = 960,
    max_workers   = 2,    # max 2 sur T4 (VRAM), 4 sur A100
):
    """
    Lance process_video() sur chaque clip en parallèle si possible.
    Recolle les timestamps automatiquement.

    Args:
        segment_clips : [(clip_path, seg_start, seg_end), ...]
        max_workers   : workers parallèles (2=T4, 4=A100)

    Returns:
        (all_events, all_frames_data, fps, jersey_map)
    """
    if not segment_clips:
        return [], [], 25.0, {}

    print(f"\n  [DEEP] {len(segment_clips)} segments | "
          f"workers={min(max_workers, len(segment_clips))} | "
          f"imgsz={imgsz} | skip={frame_skip}")

    all_events  = []
    all_frames  = []
    fps_global  = 25.0
    jersey_map  = {}
    total       = len(segment_clips)

    args_list = [
        (clip_path, seg_start, seg_end, sport, shot_zones,
         frame_skip, batch_size, imgsz, i, total)
        for i, (clip_path, seg_start, seg_end) in enumerate(segment_clips)
    ]

    # Parallélisation si plusieurs segments
    if max_workers > 1 and len(segment_clips) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=min(max_workers, len(segment_clips))) as executor:
                futures = {executor.submit(_analyze_single_segment, args): i
                           for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    events, frames_data, fps, jmap = future.result()
                    all_events.extend(events)
                    all_frames.extend(frames_data)
                    fps_global = fps
                    jersey_map.update(jmap)
        except Exception as e:
            print(f"  [DEEP] Parallélisation échouée ({e}) — mode séquentiel")
            max_workers = 1

    if max_workers <= 1:
        for args in args_list:
            events, frames_data, fps, jmap = _analyze_single_segment(args)
            all_events.extend(events)
            all_frames.extend(frames_data)
            fps_global = fps
            jersey_map.update(jmap)

    all_events.sort(key=lambda e: e.get("time", 0))
    all_frames.sort(key=lambda f: f.get("frame", 0))

    print(f"\n  [DEEP] Total : {len(all_events)} events | "
          f"{len(all_frames)} frames | {len(jersey_map)} maillots")

    return all_events, all_frames, fps_global, jersey_map


# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────
def cleanup_segments(segment_clips):
    """Supprime les clips temporaires après analyse."""
    for clip_path, _, _ in segment_clips:
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
        except Exception:
            pass
