# kaggle_test.py
# -*- coding: utf-8 -*-
"""
Script de test pipeline sur Kaggle/GPU
Sans interface Flask — test direct du pipeline
"""

import os
import sys
import json

# Config GPU
os.environ["PYTHONIOENCODING"] = "utf-8"

# Vérifier GPU
import torch
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Mode CPU")

# ─────────────────────────────────────────
# CONFIG TEST
# ─────────────────────────────────────────
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
SPORT      = sys.argv[2] if len(sys.argv) > 2 else "football"
OUTPUT_DIR = "outputs/kaggle_test"

print(f"\nTest pipeline :")
print(f"  Video  : {VIDEO_PATH}")
print(f"  Sport  : {SPORT}")
print(f"  Output : {OUTPUT_DIR}")

# ─────────────────────────────────────────
# LANCER LE PIPELINE
# ─────────────────────────────────────────
import time
from pipeline import run_pipeline

start = time.time()

result = run_pipeline(
    video_path     = VIDEO_PATH,
    sport          = SPORT,
    output_dir     = OUTPUT_DIR,
    save_annotated = True,   # générer vidéo annotée
    plan           = "pro"   # plan pro pour tout tester
)

elapsed = time.time() - start

# ─────────────────────────────────────────
# RAPPORT DE TEST
# ─────────────────────────────────────────
print("\n" + "="*50)
print("RAPPORT DE TEST")
print("="*50)

summary = result.get("summary", {})
print(f"\nMatch :")
print(f"  Durée analysée  : {summary.get('duration', '--')}")
print(f"  Temps traitement: {elapsed:.1f}s")
print(f"  Ratio temps réel: {elapsed / max(1, sum(1 for _ in open(VIDEO_PATH, 'rb'))):.2f}x")

print(f"\nStats :")
print(f"  Events détectés : {summary.get('total_events', 0)}")
print(f"  Buts            : {summary.get('goals', 0)}")
print(f"  Tirs            : {summary.get('shots', 0)}")
print(f"  Passes          : {summary.get('passes', 0)}")
print(f"  Joueurs         : {summary.get('players', 0)}")

print(f"\nOutputs générés :")
for key in ["reel", "montage", "annotated", "pdf"]:
    val = result.get(key)
    if val and os.path.exists(val):
        size = os.path.getsize(val) / 1024 / 1024
        print(f"  {key:12s} : {val} ({size:.1f} MB)")
    else:
        print(f"  {key:12s} : non généré")

print(f"\nHeatmaps :")
for name, path in result.get("heatmaps", {}).items():
    if path and os.path.exists(path):
        print(f"  {name:12s} : OK")

print(f"\nHighlights : {len(result.get('highlights', []))}")
for h in result.get("highlights", [])[:5]:
    mins = int(h.get("time_start", 0) // 60)
    secs = int(h.get("time_start", 0) % 60)
    print(f"  {mins:02d}:{secs:02d} — {h.get('main_type', '?')} "
          f"(score: {h.get('score', 0):.1f})")

print("\n" + "="*50)
print(f"TEST TERMINE en {elapsed:.1f}s")
print("="*50)

# Sauvegarder le rapport
report = {
    "gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "elapsed_sec":  round(elapsed, 2),
    "video":        VIDEO_PATH,
    "sport":        SPORT,
    "summary":      summary,
    "highlights_count": len(result.get("highlights", [])),
}

with open(f"{OUTPUT_DIR}/test_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nRapport sauvegarde : {OUTPUT_DIR}/test_report.json")
