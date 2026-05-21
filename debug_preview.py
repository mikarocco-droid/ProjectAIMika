# ════════════════════════════════════════════════════════════
# DEBUG detect_teams_preview — coller dans une cellule Colab
# ════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, r'D:\ProjetAIMika')
os.chdir(r'D:\ProjetAIMika')

import cv2
import numpy as np
from collections import defaultdict

VIDEO_PATH = r"D:\ProjetAIMika\uploads\Resume_Bullange_-_Stavelot_B_0-1.mp4"  # ← adapter si besoin

# ── 1. Tester extract_jersey_color_strict sur une frame ──────────────────────
print("=== TEST 1 : extract_jersey_color_strict ===")
from analysis.detect_teams_preview import extract_jersey_color_strict, bgr_to_name

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (960, 540))
    # Tester sur une bbox joueur fictive au centre
    test_bboxes = [
        (200, 100, 280, 260),
        (400, 120, 480, 300),
        (600, 80, 680, 280),
    ]
    for bbox in test_bboxes:
        color = extract_jersey_color_strict(small, bbox)
        if color is not None:
            bgr = tuple(int(x) for x in color)
            print(f"  bbox={bbox} → BGR{bgr} → {bgr_to_name(bgr)}")
        else:
            print(f"  bbox={bbox} → None (rien détecté)")
else:
    print("  ❌ Impossible de lire la vidéo")

# ── 2. Tester le tracker sur 30s ─────────────────────────────────────────────
print("\n=== TEST 2 : Tracker 30s + mémoire PID ===")

from vision.detector import Detector
from vision.tracker  import Tracker
import config

detector = Detector(sport="football")
tracker  = Tracker()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
max_frame = int(30 * fps)   # 30 secondes
skip = max(1, int(fps / 8))

pid_colors = defaultdict(list)
frame_id   = 0
n_tracked  = 0

while frame_id < max_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id % skip != 0:
        continue

    small = cv2.resize(frame, (960, 540))

    try:
        results = detector.model([small], conf=0.4, verbose=False,
                                  imgsz=int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ)))
    except Exception as e:
        print(f"  YOLO error frame {frame_id}: {e}")
        continue

    players = []
    for box in results[0].boxes:
        if int(box.cls[0]) != detector.player_cls:
            continue
        if float(box.conf[0]) < 0.4:
            continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        bh = y2 - y1
        if bh < 540 * 0.12:
            continue
        players.append({"bbox":[x1,y1,x2,y2],
                         "center":[(x1+x2)/2,(y1+y2)/2],
                         "conf":float(box.conf[0])})

    tracked = tracker.update(players, small)
    n_tracked += len(tracked)

    for p in tracked:
        pid  = str(p.get("id") or p.get("player_id",""))
        bbox = p.get("bbox")
        if not pid or not bbox:
            continue
        color = extract_jersey_color_strict(small, bbox)
        if color is not None:
            pid_colors[pid].append(color)

cap.release()

print(f"  {frame_id} frames lues | {n_tracked} détections")
print(f"  {len(pid_colors)} PIDs vus")

# Afficher les PIDs stables
stable = {pid: colors for pid, colors in pid_colors.items() if len(colors) >= 5}
print(f"  {len(stable)} PIDs stables (>= 5 obs)")

for pid, colors in list(stable.items())[:10]:
    arr    = np.array(colors, dtype=np.float32)
    median = np.median(arr, axis=0)
    bgr    = tuple(int(x) for x in median)
    print(f"    PID {pid:>4} : {len(colors):3d} obs → BGR{bgr} → {bgr_to_name(bgr)}")

# ── 3. KMeans sur joueurs stables ────────────────────────────────────────────
if len(stable) >= 4:
    print("\n=== TEST 3 : KMeans équipes ===")
    stable_colors = []
    for pid, colors in stable.items():
        arr    = np.array(colors, dtype=np.float32)
        median = np.median(arr, axis=0)
        stable_colors.append(median)

    samples  = np.array(stable_colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centroids = cv2.kmeans(
        samples, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    c0 = tuple(int(x) for x in centroids[0])
    c1 = tuple(int(x) for x in centroids[1])
    n0 = int((labels==0).sum())
    n1 = int((labels==1).sum())
    dist = float(np.linalg.norm(centroids[0] - centroids[1]))
    print(f"  Team 0: BGR{c0} → {bgr_to_name(c0)} ({n0} joueurs)")
    print(f"  Team 1: BGR{c1} → {bgr_to_name(c1)} ({n1} joueurs)")
    print(f"  Distance: {dist:.1f}")
else:
    print(f"\n  ❌ Pas assez de joueurs stables pour KMeans")
