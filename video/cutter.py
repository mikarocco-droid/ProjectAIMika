# video/cutter.py (V15)

import subprocess


def cut_clip(input_video, start, end, output, reencode=False):
    """
    Coupe un clip vidéo.
    
    reencode=False  → rapide (copy)
    reencode=True   → propre (re-encode)
    """

    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_video,
            "-c", "copy",
            output
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output