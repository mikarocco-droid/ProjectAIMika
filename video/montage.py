# video/montage.py
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import shutil


# ─────────────────────────────────────────
# CONFIG V19 GPU
# ─────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
FPS    = 25   # aligné sur la vidéo source 25fps


# ─────────────────────────────────────────
# DÉTECTION ENCODEUR
# Priorité : h264_nvenc (GPU) → libx264 (CPU fallback)
# ─────────────────────────────────────────
def detect_encoder():
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True
    )
    if "h264_nvenc" in result.stdout:
        print("  Encodeur : h264_nvenc (GPU Tesla T4) ✅")
        return "h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23"]
    else:
        print("  Encodeur : libx264 (CPU fallback)")
        return "libx264", ["-preset", "fast", "-crf", "23"]


ENCODER, ENCODER_OPTS = detect_encoder()


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
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 2.0


# ─────────────────────────────────────────
# NORMALISATION GPU
# ─────────────────────────────────────────
def normalize_clip(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
        "-c:v", ENCODER, *ENCODER_OPTS,
        "-c:a", "aac",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# ZOOM INTELLIGENT (effet broadcast)
# Activé sur GPU — rapide avec nvenc
# ─────────────────────────────────────────
def add_zoom(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        "zoompan=z='min(zoom+0.0015,1.15)':d=125"
        ":x='iw/2-(iw/zoom/2)'"
        ":y='ih/2-(ih/zoom/2)'",
        "-c:v", ENCODER, *ENCODER_OPTS,
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# LABEL PRO (haut écran)
# ─────────────────────────────────────────
def add_label(inp, out, label, t=None):
    txt = (label or "").upper().replace("'", "\\'")

    if t:
        m = int(t // 60)
        s = int(t % 60)
        txt += f"  {m:02d}:{s:02d}"

    if not txt.strip():
        shutil.copy2(inp, out)
        return out

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
        "-c:v", ENCODER, *ENCODER_OPTS,
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# FADE
# ─────────────────────────────────────────
def fade(inp, out, d=0.4):
    dur         = get_duration(inp)
    fade_out_st = max(0, dur - d)
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf",
        f"fade=t=in:st=0:d={d},fade=t=out:st={fade_out_st:.3f}:d={d}",
        "-c:v", ENCODER, *ENCODER_OPTS,
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
        "-c:v", ENCODER, *ENCODER_OPTS,
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def crossfade_concat(clips, output):
    if not clips:
        return None

    if len(clips) == 1:
        shutil.copy2(clips[0], output)
        return output

    # FIX — xfade échoue avec trop de clips (filtre_complex trop long)
    # Au-delà de 8 clips, on fait un concat simple direct qui est plus stable
    if len(clips) > 8:
        print(f"  {len(clips)} clips → concat simple (xfade limité à 8)")
        return _concat_simple(clips, output)

    FADE_DUR   = 0.5
    tmp_dir    = tempfile.mkdtemp()
    safe_clips = []

    for i, c in enumerate(clips):
        safe = os.path.join(tmp_dir, f"safe_{i}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", c,
            "-c:v", ENCODER, *ENCODER_OPTS,
            "-c:a", "aac",
            "-vsync", "cfr",
            "-af", "aresample=async=1",
            safe
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(safe) and os.path.getsize(safe) > 0:
            safe_clips.append(safe)

    if not safe_clips:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    inputs       = []
    for c in safe_clips:
        inputs += ["-i", c]

    durations    = [get_duration(c) for c in safe_clips]
    filter_parts = []
    prev_v = "[0:v]"
    prev_a = "[0:a]"
    offset = 0.0

    for i in range(1, len(safe_clips)):
        offset += durations[i - 1] - FADE_DUR
        out_v   = f"[v{i}]"
        out_a   = f"[a{i}]"
        filter_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:duration={FADE_DUR}:offset={offset:.3f}{out_v}"
        )
        filter_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={FADE_DUR}{out_a}"
        )
        prev_v = out_v
        prev_a = out_a

    ret = subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", prev_v,
        "-map", prev_a,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        output
    ], capture_output=True, text=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if ret.returncode != 0 or not os.path.exists(output) or os.path.getsize(output) == 0:
        print("  ⚠️ xfade échoué — fallback concat simple")
        return _concat_simple(clips, output)

    return output


def _concat_simple(clips, output):
    """Concat simple sans transitions — stable peu importe le nombre de clips."""
    list_file = output + "_list.txt"
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", ENCODER, *ENCODER_OPTS,
        "-c:a", "aac",
        "-movflags", "+faststart",
        output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.remove(list_file)
    except:
        pass
    return output if os.path.exists(output) and os.path.getsize(output) > 0 else None


# ─────────────────────────────────────────
# FORMAT VERTICAL (TikTok)
# ─────────────────────────────────────────
def to_vertical(inp, out):
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,"
               "pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", ENCODER, *ENCODER_OPTS,
        "-c:a", "copy",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ─────────────────────────────────────────
# PIPELINE V19 GPU
# ─────────────────────────────────────────
def create_montage(
    highlights,
    video_path,
    output   = "outputs/montage.mp4",
    title    = "Scout IA Highlights",
    vertical = False
):
    if not check_ffmpeg():
        print("❌ ffmpeg absent")
        return None

    if not highlights:
        print("  ⚠️ Aucun highlight pour le montage")
        return None

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tmp   = tempfile.mkdtemp()
    clips = []

    print(f"  Montage GPU : {len(highlights)} clips | encodeur={ENCODER}")

    # INTRO
    intro_path = os.path.join(tmp, "intro.mp4")
    intro(intro_path, title)
    if os.path.exists(intro_path) and os.path.getsize(intro_path) > 0:
        clips.append(intro_path)

    # HIGHLIGHTS — pipeline complet qualité max
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

        if not os.path.exists(raw) or os.path.getsize(raw) == 0:
            print(f"  ⚠️ Clip {i} vide, ignoré")
            continue

        step1 = normalize_clip(raw,   os.path.join(tmp, f"norm_{i}.mp4"))
        step2 = add_zoom(step1,       os.path.join(tmp, f"zoom_{i}.mp4"))
        step3 = add_label(step2,      os.path.join(tmp, f"label_{i}.mp4"),
                          h.get("main_type", ""), h.get("time_start"))
        step4 = fade(step3,           os.path.join(tmp, f"fade_{i}.mp4"))

        if os.path.exists(step4) and os.path.getsize(step4) > 0:
            clips.append(step4)
            print(f"  ✅ Clip {i+1}/{len(highlights)} OK")
        else:
            print(f"  ⚠️ Clip {i+1} échoué, ignoré")

    if not clips:
        print("❌ Aucun clip valide")
        return None

    # CONCAT avec crossfade
    merged = os.path.join(tmp, "merged.mp4")
    crossfade_concat(clips, merged)

    if not os.path.exists(merged) or os.path.getsize(merged) == 0:
        print("❌ Échec concat final")
        return None

    # VERTICAL SI DEMANDE
    if vertical:
        final = to_vertical(merged, output)
    else:
        shutil.copy2(merged, output)
        final = output

    # Nettoyage tmp
    try:
        shutil.rmtree(tmp)
    except:
        pass

    print(f"🔥 Montage V19 GPU prêt → {final}")
    return final