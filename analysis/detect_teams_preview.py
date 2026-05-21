# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-
#
# Détection équipes ROBUSTE — vrai tracker YOLO+DeepSort sur 90s
#
# Philosophie :
#   - Tracking réel sur les 90 premières secondes
#   - Mémoire couleur par PID (30-50 obs par joueur)
#   - Médiane robuste par joueur
#   - KMeans sur profils joueurs stables
#
# Résultat attendu : ~90% de fiabilité vs ~60% avant

import os
import cv2
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────
# EXTRACTION COULEUR MAILLOT STRICTE
# ─────────────────────────────────────────
def extract_jersey_color_strict(frame, bbox):
    """
    Zone stricte 18%→42% hauteur, 25%→75% largeur.
    Filtre HSV : S > 80, V entre 40 et 220.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 25 or w < 10:
        return None
    jersey = crop[int(h*0.18):int(h*0.42), int(w*0.25):int(w*0.75)]
    if jersey.size == 0:
        return None
    try:
        hsv  = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        S, V = hsv[:,:,1], hsv[:,:,2]
        mask = (S > 80) & (V > 40) & (V < 220)
        if mask.sum() >= 8:
            return jersey[mask].mean(axis=0).astype(float)
        mask2 = (S > 40) & (V > 30) & (V < 230)
        if mask2.sum() >= 5:
            return jersey[mask2].mean(axis=0).astype(float)
    except Exception:
        pass
    return None


def bgr_to_name(bgr):
    """Nom couleur depuis BGR."""
    if not bgr:
        return "inconnu"
    try:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        pixel   = np.uint8([[[b, g, r]]])
        hsv     = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h = int(hsv[0]) * 2
        s = int(hsv[1] / 255 * 100)
        v = int(hsv[2] / 255 * 100)
    except Exception:
        return "inconnu"
    if s < 15: return "blanc" if v > 70 else ("gris" if v > 30 else "noir")
    if v < 15: return "noir"
    if 0 <= h < 20 or 340 <= h <= 360:
        return "bordeaux foncé" if v < 35 else ("bordeaux" if v < 60 else "rouge")
    if 20 <= h < 35:  return "orange"
    if 35 <= h < 75:  return "jaune"
    if 75 <= h < 165: return "vert foncé" if v < 50 else "vert"
    if 155 <= h < 185: return "cyan"
    if 185 <= h < 265:
        return "bleu foncé" if v < 35 else ("bleu marine" if v < 55 else "bleu")
    if 265 <= h < 295: return "violet"
    if 295 <= h < 340: return "bordeaux" if v < 45 else "rose"
    return "inconnu"


def save_preview_frame(frame, team_id, output_dir, color_bgr=None):
    """Sauvegarde une frame représentative avec bandeau couleur."""
    os.makedirs(output_dir, exist_ok=True)
    preview = frame.copy()
    h, w    = preview.shape[:2]
    if color_bgr:
        b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
        cv2.rectangle(preview, (0, h-30), (w, h), (b,g,r), -1)
        cv2.putText(preview, bgr_to_name(color_bgr).upper(),
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2)
    path = os.path.join(output_dir, f"team_{team_id}_preview.jpg")
    cv2.imwrite(path, preview)
    return path


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE — VRAI TRACKER
# ─────────────────────────────────────────
def detect_teams_preview(video_path, output_dir="outputs/preview",
                          bootstrap_duration=90.0, sport="football"):
    """
    Détecte les 2 équipes avec le vrai tracker YOLO+DeepSort
    sur les `bootstrap_duration` premières secondes.

    Retourne :
    {
        "success": True,
        "team_0": {"color_bgr": ..., "color_name": ..., "preview_frame": ...},
        "team_1": {...},
        "n_players_analyzed": N,
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Ouvrir la vidéo ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Impossible d'ouvrir la vidéo"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame    = min(int(bootstrap_duration * fps), total_frames)

    PROCESS_W, PROCESS_H = 960, 540
    scale_x = w_orig / PROCESS_W
    scale_y = h_orig / PROCESS_H

    print(f"  [PREVIEW] Tracker sur {bootstrap_duration:.0f}s "
          f"({max_frame} frames) | {w_orig}x{h_orig}")

    # ── Initialiser tracker et détecteur ─────────────────────────────────────
    try:
        from vision.detector import Detector
        from vision.tracker  import Tracker
        detector = Detector(sport=sport)
        tracker  = Tracker()
        print(f"  [PREVIEW] YOLO+DeepSort initialisés")
    except Exception as e:
        cap.release()
        return {"success": False, "error": f"Tracker non disponible : {e}"}

    # ── Mémoire couleur par PID ───────────────────────────────────────────────
    pid_colors     = defaultdict(list)   # pid → [color_bgr, ...]
    pid_first_seen = {}
    pid_last_seen  = {}
    best_frames    = {}   # team_id → frame pour la preview
    frame_id       = 0
    n_obs_total    = 0
    prev_gray      = None

    import config
    skip = max(1, int(fps / 8))   # ~8 frames/s analysées

    while frame_id < max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % skip != 0:
            continue

        t = frame_id / fps

        # Resize pour YOLO
        small = cv2.resize(frame, (PROCESS_W, PROCESS_H),
                           interpolation=cv2.INTER_LINEAR)

        # ── Masque mouvement (éliminer fond statique) ─────────────────────
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
            motion = cv2.dilate(motion, kernel, iterations=2)
        else:
            motion = None
        prev_gray = gray

        # ── Détection YOLO ────────────────────────────────────────────────
        try:
            results = detector.model(
                [small],
                conf    = config.YOLO_CONFIDENCE,
                verbose = False,
                imgsz   = int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ))
            )
        except Exception:
            continue

        players = []
        for box in results[0].boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != detector.player_cls or conf < 0.4:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bh = y2 - y1
            # Filtrer les petits joueurs (lointains = trop de gazon dans crop)
            if bh < PROCESS_H * 0.12:
                continue
            players.append({"bbox": [x1,y1,x2,y2],
                             "center": [(x1+x2)/2, (y1+y2)/2],
                             "conf": conf})

        if not players:
            continue

        # ── Tracking DeepSort ─────────────────────────────────────────────
        tracked = tracker.update(players, small)

        # ── Accumuler couleurs par PID ────────────────────────────────────
        for p in tracked:
            pid  = str(p.get("id") or p.get("player_id", ""))
            bbox = p.get("bbox")
            if not pid or not bbox:
                continue

            # Vérifier que le joueur est en mouvement (pas le gazon)
            if motion is not None:
                x1,y1,x2,y2 = map(int, bbox)
                roi = motion[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if roi.size > 0 and roi.mean() < 15:
                    continue

            color = extract_jersey_color_strict(small, bbox)
            if color is None:
                continue

            pid_colors[pid].append(color)
            if pid not in pid_first_seen:
                pid_first_seen[pid] = t
            pid_last_seen[pid] = t
            n_obs_total += 1

    cap.release()

    print(f"  [PREVIEW] {len(pid_colors)} PIDs | "
          f"{n_obs_total} observations totales")

    # ── Tentative 1 : récupérer les centroids depuis PlayerReID ──────────────
    # PlayerReID._calibrate_teams() a déjà fait un KMeans robuste
    # sur toutes les couleurs pendant le tracking — utilisons-le directement
    try:
        from analysis.player_reid import get_team_colors
        reid_colors = get_team_colors()
        if reid_colors and len(reid_colors) >= 2:
            c0_reid = tuple(int(x) for x in reid_colors.get(0, (0,0,0)))
            c1_reid = tuple(int(x) for x in reid_colors.get(1, (0,0,0)))
            dist_reid = float(np.linalg.norm(
                np.array(c0_reid) - np.array(c1_reid)
            ))
            print(f"  [PREVIEW] PlayerReID colors: {c0_reid}→{bgr_to_name(c0_reid)} "
                  f"| {c1_reid}→{bgr_to_name(c1_reid)} | dist={dist_reid:.1f}")
    except Exception as _e:
        reid_colors = {}
        print(f"  [PREVIEW] PlayerReID non disponible : {_e}")

    # ── Joueurs stables (>= 10 observations) ─────────────────────────────────
    MIN_OBS = 10
    stable_colors = []
    stable_pids   = []

    for pid, colors in pid_colors.items():
        if len(colors) < MIN_OBS:
            continue
        arr    = np.array(colors, dtype=np.float32)
        median = np.median(arr, axis=0)

        # Filtrer outliers : garder seulement les obs proches de la médiane
        dists  = np.linalg.norm(arr - median, axis=1)
        thresh = np.percentile(dists, 70)   # garder 70% les plus proches
        clean  = arr[dists <= thresh]

        if len(clean) >= 3:
            final_color = np.median(clean, axis=0)
        else:
            final_color = median

        stable_colors.append(final_color)
        stable_pids.append(pid)

    n_stable = len(stable_colors)
    print(f"  [PREVIEW] {n_stable} joueurs stables (>= {MIN_OBS} obs)")

    if n_stable < 4:
        return {
            "success": False,
            "error":   f"Pas assez de joueurs stables ({n_stable}/4 minimum)"
        }

    # ── KMeans sur profils joueurs ────────────────────────────────────────────
    samples  = np.array(stable_colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centroids = cv2.kmeans(
        samples, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    dist = float(np.linalg.norm(centroids[0] - centroids[1]))
    n0   = int((labels == 0).sum())
    n1   = int((labels == 1).sum())

    c0 = tuple(int(x) for x in centroids[0])
    c1 = tuple(int(x) for x in centroids[1])

    # Utiliser PlayerReID si sa distance est meilleure
    if reid_colors and len(reid_colors) >= 2:
        c0_r = tuple(int(x) for x in reid_colors.get(0, c0))
        c1_r = tuple(int(x) for x in reid_colors.get(1, c1))
        dist_r = float(np.linalg.norm(np.array(c0_r) - np.array(c1_r)))
        if dist_r > dist:
            print(f"  [PREVIEW] Utilisation PlayerReID (dist={dist_r:.1f} > {dist:.1f})")
            c0, c1, dist = c0_r, c1_r, dist_r

    print(f"  [PREVIEW] Distance: {dist:.1f} | "
          f"Team0: {c0}→{bgr_to_name(c0)} ({n0}j) | "
          f"Team1: {c1}→{bgr_to_name(c1)} ({n1}j)")

    # ── Preview frames ────────────────────────────────────────────────────────
    # Reprendre la vidéo pour trouver les meilleures frames
    cap = cv2.VideoCapture(video_path)
    best_frame_0 = best_frame_1 = None
    best_score_0 = best_score_1 = 0

    sample_frames = np.linspace(int(fps*5), max_frame, 20, dtype=int)

    for fid in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        try:
            results  = detector.model([small], conf=0.4, verbose=False,
                                       imgsz=int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ)))
            tracked2 = tracker.update(
                [{"bbox": b.xyxy[0].tolist(),
                  "center": [(b.xyxy[0][0]+b.xyxy[0][2])/2,
                              (b.xyxy[0][1]+b.xyxy[0][3])/2],
                  "conf": float(b.conf[0])}
                 for b in results[0].boxes
                 if int(b.cls[0]) == detector.player_cls], small)
        except Exception:
            continue

        s0 = s1 = 0
        for p in tracked2:
            pid  = str(p.get("id",""))
            bbox = p.get("bbox")
            if not pid or not bbox: continue
            color = extract_jersey_color_strict(small, bbox)
            if color is None: continue
            c = np.array(color, dtype=np.float32)
            d0 = np.linalg.norm(c - centroids[0])
            d1 = np.linalg.norm(c - centroids[1])
            if d0 < d1: s0 += 1
            else:        s1 += 1

        if s0 > best_score_0: best_score_0 = s0; best_frame_0 = frame.copy()
        if s1 > best_score_1: best_score_1 = s1; best_frame_1 = frame.copy()

    cap.release()

    preview_0 = save_preview_frame(best_frame_0, 0, output_dir, c0) if best_frame_0 is not None else None
    preview_1 = save_preview_frame(best_frame_1, 1, output_dir, c1) if best_frame_1 is not None else None

    return {
        "success":            True,
        "n_players_analyzed": n_stable,
        "team_0": {
            "color_bgr":     list(c0),
            "color_name":    bgr_to_name(c0),
            "short_bgr":     None,
            "preview_frame": preview_0,
        },
        "team_1": {
            "color_bgr":     list(c1),
            "color_name":    bgr_to_name(c1),
            "short_bgr":     None,
            "preview_frame": preview_1,
        },
    }