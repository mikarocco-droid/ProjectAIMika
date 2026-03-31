# video/montage.py
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile


# ─────────────────────────────────────────
# CONFIG V18
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
# DURÉE D'UN CLIP
# ─────────────────────────────────────────
def get_duration(path):
    """Retourne la durée en secondes d'un fichier vidéo."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 2.0  # fallback


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
    txt = label.upper().replace("'", "\\'")  # échapper les apostrophes

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
    dur = get_duration(inp)
    fade_out_st = max(0, dur - d)
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        f"fade=t=in:st=0:d={d},fade=t=out:st={fade_out_st:.3f}:d={d}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# INTRO
# ─────────────────────────────────────────
def intro(out, title="Scout IA"):
    safe_title = title.replace("'", "\\'")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-t", "3",
        "-vf",
        (
            "fade=t=in:st=0:d=1,"
            "fade=t=out:st=2:d=1,"
            f"drawtext=text='{safe_title}'"
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
# CONCAT AVEC CROSSFADE — corrigé V18
# ─────────────────────────────────────────
def crossfade_concat(clips, output):
    if len(clips) == 1:
        import shutil
        shutil.copy2(clips[0], output)  # copy au lieu de rename (tmp → output)
        return output

    FADE_DUR = 0.5

    inputs = []
    for c in clips:
        inputs += ["-i", c]

    # Calcul des offsets réels basés sur les durées
    durations = [get_duration(c) for c in clips]

    filter_parts = []
    prev_v = "[0:v]"
    prev_a = "[0:a]"
    offset = 0.0

    for i in range(1, len(clips)):
        offset += durations[i - 1] - FADE_DUR
        out_v = f"[v{i}]"
        out_a = f"[a{i}]"

        filter_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:duration={FADE_DUR}:offset={offset:.3f}{out_v}"
        )
        filter_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={FADE_DUR}{out_a}"
        )
        prev_v = out_v
        prev_a = out_a

    subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", prev_v,
        "-map", prev_a,
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac",
        output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output


# ─────────────────────────────────────────
# FORMAT VERTICAL (TikTok)
# ─────────────────────────────────────────
def to_vertical(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,"
               "pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# PIPELINE V18
# ─────────────────────────────────────────
def create_montage(
    highlights,
    video_path,
    output="outputs/montage.mp4",       # ← paramètre = "output" (pas "output_path")
    title="Scout IA Highlights",
    vertical=False
):
    if not check_ffmpeg():
        print("❌ ffmpeg absent")
        return None

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tmp   = tempfile.mkdtemp()
    clips = []

    # INTRO
    intro_path = os.path.join(tmp, "intro.mp4")
    intro(intro_path, title)
    if os.path.exists(intro_path):
        clips.append(intro_path)

    # HIGHLIGHTS
    for i, h in enumerate(highlights):
        raw = os.path.join(tmp, f"raw_{i}.mp4")

        ret = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(h["time_start"]),
            "-to", str(h["time_end"]),
            "-i", video_path,
            "-c", "copy",
            raw
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(raw) or os.path.getsize(raw) == 0:
            print(f"  ⚠️ Clip {i} vide, ignoré")
            continue

        step1 = normalize_clip(raw,   os.path.join(tmp, f"norm_{i}.mp4"))
        step2 = add_zoom(step1,       os.path.join(tmp, f"zoom_{i}.mp4"))
        step3 = add_label(step2,      os.path.join(tmp, f"label_{i}.mp4"),
                          h.get("main_type", ""), h.get("time_start"))
        step4 = fade(step3,           os.path.join(tmp, f"fade_{i}.mp4"))

        if os.path.exists(step4):
            clips.append(step4)

    if not clips:
        print("❌ Aucun clip valide")
        return None

    # CONCAT PRO
    merged = os.path.join(tmp, "merged.mp4")
    crossfade_concat(clips, merged)

    if not os.path.exists(merged):
        print("❌ Échec du montage final")
        return None

    # VERTICAL SI DEMANDE
    if vertical:
        final = to_vertical(merged, output)
    else:
        import shutil
        shutil.copy2(merged, output)
        final = output

    print(f"🔥 Montage V18 prêt → {final}")
    return final