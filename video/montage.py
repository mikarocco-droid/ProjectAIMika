# video/montage.py
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile


# ─────────────────────────────────────────
# VERIFICATION FFMPEG
# ─────────────────────────────────────────
def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ─────────────────────────────────────────
# INTRO TEXTE
# ─────────────────────────────────────────
def create_intro(output_path, title="Match Highlights", duration=3, resolution="1280x720"):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:size={resolution}:rate=30:duration={duration}",
        "-vf", (
            f"drawtext=text='{title}'"
            f":fontcolor=white"
            f":fontsize=48"
            f":x=(w-text_w)/2"
            f":y=(h-text_h)/2"
        ),
        "-c:v", "libx264",
        "-t", str(duration),
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# ─────────────────────────────────────────
# LABEL SUR UN CLIP
# ─────────────────────────────────────────
def add_label_to_clip(input_path, output_path, label, time_start=None):
    label_map = {
        "goal":         "BUT",
        "score":        "BUT",
        "shot":         "TIR",
        "interception": "INTERCEPTION",
        "dribble":      "DRIBBLE",
        "long_pass":    "PASSE LONGUE",
        "pass":         "PASSE",
    }

    display = label_map.get(label, label.upper())

    if time_start is not None:
        minutes = int(time_start // 60)
        seconds = int(time_start % 60)
        display += f"  {minutes:02d}:{seconds:02d}"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            f"drawtext=text='{display}'"
            f":fontcolor=white"
            f":fontsize=28"
            f":x=20"
            f":y=h-th-20"
            f":box=1"
            f":boxcolor=black@0.5"
            f":boxborderw=8"
        ),
        "-c:v", "libx264",
        "-c:a", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# ─────────────────────────────────────────
# FONDU
# ─────────────────────────────────────────
def apply_fade(input_path, output_path, duration=0.3):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            f"fade=t=in:st=0:d={duration},"
            f"fade=t=out:st=0:d={duration}:enable='gte(t,0)'"
        ),
        "-c:v", "libx264",
        "-c:a", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# ─────────────────────────────────────────
# ASSEMBLAGE FINAL
# ─────────────────────────────────────────
def assemble_reel(clip_paths, output_path):
    if not clip_paths:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        list_file = f.name
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.unlink(list_file)
    return output_path


# ─────────────────────────────────────────
# PIPELINE MONTAGE COMPLET
# ─────────────────────────────────────────
def create_montage(
    highlights,
    video_path,
    output_path   = "outputs/montage.mp4",
    title         = "Match Highlights",
    with_intro    = True,
    with_labels   = True,
    with_fades    = True
):
    if not check_ffmpeg():
        print("ffmpeg non trouve - montage impossible")
        return None

    if not highlights:
        print("Aucun highlight a monter")
        return None

    os.makedirs(os.path.dirname(output_path) or "outputs", exist_ok=True)

    tmp_dir = tempfile.mkdtemp()
    clips   = []

    # Intro
    if with_intro:
        intro_path = os.path.join(tmp_dir, "intro.mp4")
        create_intro(intro_path, title=title)
        clips.append(intro_path)

    # Clips
    for i, h in enumerate(highlights):
        raw_path = os.path.join(tmp_dir, f"raw_{i}.mp4")

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(h["time_start"]),
            "-to", str(h["time_end"]),
            "-i", video_path,
            "-c", "copy",
            raw_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        current = raw_path

        if with_labels:
            labeled_path = os.path.join(tmp_dir, f"labeled_{i}.mp4")
            add_label_to_clip(
                current, labeled_path,
                label      = h.get("main_type", ""),
                time_start = h.get("time_start")
            )
            current = labeled_path

        if with_fades:
            faded_path = os.path.join(tmp_dir, f"faded_{i}.mp4")
            apply_fade(current, faded_path)
            current = faded_path

        clips.append(current)

    # Assemblage
    print(f"Assemblage de {len(clips)} clips...")
    result = assemble_reel(clips, output_path)

    if result:
        print(f"Montage termine -> {output_path}")
    else:
        print("Echec assemblage")

    return result