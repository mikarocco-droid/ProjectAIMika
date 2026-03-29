# video/cutter.py

import subprocess

def cut_clip(input_video, start, end, output):
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_video,
        "-c", "copy",
        output
    ])