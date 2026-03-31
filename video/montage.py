# video/montage.py
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile


# ─────────────────────────────────────────
# CONFIG V17
# ─────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
FPS    = 30


# ─────────────────────────────────────────
# CHECK FFMPEG
# ─────────────────────────────────────────
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except:
        return False


# ─────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────
def normalize_clip(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# ZOOM INTELLIGENT (effet broadcast)
# ─────────────────────────────────────────
def add_zoom(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        "zoompan=z='min(zoom+0.0015,1.15)':d=125"
        ":x='iw/2-(iw/zoom/2)'"
        ":y='ih/2-(ih/zoom/2)'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# LABEL PRO (haut écran)
# ─────────────────────────────────────────
def add_label(inp, out, label, t=None):
    txt = label.upper()

    if t:
        m = int(t // 60)
        s = int(t % 60)
        txt += f"  {m:02d}:{s:02d}"

    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        (
            f"drawtext=text='{txt}'"
            ":fontcolor=white"
            ":fontsize=36"
            ":x=(w-text_w)/2"
            ":y=40"
            ":box=1"
            ":boxcolor=black@0.6"
            ":boxborderw=12"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# FADE
# ─────────────────────────────────────────
def fade(inp, out, d=0.4):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        f"fade=t=in:st=0:d={d},fade=t=out:st=0:d={d}:enable='gte(t,0)'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# INTRO
# ─────────────────────────────────────────
def intro(out, title="Scout IA"):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-t", "3",
        "-vf",
        (
            "fade=t=in:st=0:d=1,"
            "fade=t=out:st=2:d=1,"
            f"drawtext=text='{title}'"
            ":fontcolor=white:fontsize=60"
            ":x=(w-text_w)/2"
            ":y=(h-text_h)/2"
        ),
        "-c:v", "libx264",
        "-crf", "23",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# CONCAT AVEC CROSSFADE 🔥
# ─────────────────────────────────────────
def crossfade_concat(clips, output):
    if len(clips) == 1:
        os.rename(clips[0], output)
        return output

    inputs = []
    filter_complex = ""
    offset = 0

    for i, c in enumerate(clips):
        inputs += ["-i", c]

    for i in range(len(clips) - 1):
        filter_complex += (
            f"[{i}:v][{i+1}:v]xfade=transition=fade:duration=0.5:"
            f"offset={i*2}[v{i}];"
        )

    last = f"[v{len(clips)-2}]"

    subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex.rstrip(";"),
        "-map", last,
        "-c:v", "libx264",
        "-crf", "23",
        output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output


# ─────────────────────────────────────────
# FORMAT VERTICAL (TikTok)
# ─────────────────────────────────────────
def to_vertical(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf", "scale=720:1280,setsar=1",
        "-c:v", "libx264", "-crf", "23",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# PIPELINE V17
# ─────────────────────────────────────────
def create_montage(
    highlights,
    video_path,
    output="outputs/montage.mp4",
    title="Scout IA Highlights",
    vertical=False
):
    if not check_ffmpeg():
        print("❌ ffmpeg absent")
        return None

    tmp = tempfile.mkdtemp()
    clips = []

    # INTRO
    intro_path = os.path.join(tmp, "intro.mp4")
    intro(intro_path, title)
    clips.append(intro_path)

    # HIGHLIGHTS
    for i, h in enumerate(highlights):

        raw = os.path.join(tmp, f"raw_{i}.mp4")

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(h["time_start"]),
            "-to", str(h["time_end"]),
            "-i", video_path,
            "-c", "copy",
            raw
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        step1 = normalize_clip(raw, os.path.join(tmp, f"norm_{i}.mp4"))
        step2 = add_zoom(step1, os.path.join(tmp, f"zoom_{i}.mp4"))
        step3 = add_label(step2, os.path.join(tmp, f"label_{i}.mp4"),
                          h.get("main_type", ""), h.get("time_start"))
        step4 = fade(step3, os.path.join(tmp, f"fade_{i}.mp4"))

        clips.append(step4)

    # CONCAT PRO
    merged = os.path.join(tmp, "merged.mp4")
    crossfade_concat(clips, merged)

    # VERTICAL SI DEMANDE
    if vertical:
        final = to_vertical(merged, output)
    else:
        os.rename(merged, output)
        final = output

    print(f"🔥 Montage V17 prêt → {final}")
    return final