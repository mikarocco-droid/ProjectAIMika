# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from collections import defaultdict


def bgr_to_name(bgr):
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
    if 20 <= h < 35:   return "orange"
    if 35 <= h < 75:   return "jaune"
    if 75 <= h < 165:  return "vert foncé" if v < 50 else "vert"
    if 165 <= h < 185: return "cyan"
    if 185 <= h < 265:
        return "bleu foncé" if v < 35 else ("bleu marine" if v < 55 else "bleu")
    if 265 <= h < 295: return "violet"
    if 295 <= h < 340: return "bordeaux" if v < 45 else "rose"
    return "inconnu"


def extract_jersey_color_strict(frame, bbox):
    try:
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
        hsv   = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        S     = hsv[:, :, 1]
        V     = hsv[:, :, 2]
        total = float(S.size)
        mask1  = (S > 80) & (V > 40) & (V < 220)
        ratio1 = mask1.sum() / max(total, 1)
        if ratio1 >= 0.30 and mask1.sum() >= 8:
            color = jersey[mask1].mean(axis=0).astype(float)
        elif mask2.sum() >= 8 and (mask2.sum()/max(total,1)) >= 0.40:
            color = jersey[mask2].mean(axis=0).astype(float)
        else:
            return None

        # Vérifier que la couleur dominante n'est pas du gazon/tribune
        # H=35-90 (jaune-vert) avec V>100 = gazon ou tribunes → rejeter
        try:
            px  = np.uint8([[[int(color[0]), int(color[1]), int(color[2])]]])
            hsv_c = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
            h_c = int(hsv_c[0]) * 2
            v_c = int(hsv_c[2])
            if 35 <= h_c <= 90 and v_c > 100:
                return None   # gazon clair ou tribunes jaune-vert
        except Exception:
            pass

        return color
    except Exception:
        return None


def save_preview_frame(frame, team_id, output_dir, color_bgr=None):
    os.makedirs(output_dir, exist_ok=True)
    preview = frame.copy()
    h, w    = preview.shape[:2]
    if color_bgr:
        b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
        cv2.rectangle(preview, (0, h-30), (w, h), (b, g, r), -1)
        cv2.putText(preview, bgr_to_name(color_bgr).upper(),
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    path = os.path.join(output_dir, f"team_{team_id}_preview.jpg")
    cv2.imwrite(path, preview)
    return path


def detect_teams_preview(video_path, output_dir="outputs/preview",
                          bootstrap_duration=90.0, sport="football",
                          n_frames=60, analysis_duration=120.0):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Impossible d'ouvrir la vidéo"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame    = min(int(bootstrap_duration * fps), total_frames)

    PROCESS_W, PROCESS_H = 960, 540

    print(f"  [PREVIEW] Tracker sur {bootstrap_duration:.0f}s "
          f"({max_frame} frames) | {w_orig}x{h_orig}")

    try:
        from vision.detector import Detector
        from vision.tracker  import Tracker
        import config
        detector = Detector(sport=sport)
        tracker  = Tracker()
        print(f"  [PREVIEW] YOLO+DeepSort initialisés")
    except Exception as e:
        cap.release()
        import traceback; traceback.print_exc()
        return {"success": False, "error": f"Tracker non disponible : {e}"}

    pid_colors  = defaultdict(list)
    n_obs_total = 0
    n_rejected  = 0
    frame_id    = 0
    prev_gray   = None
    skip        = max(1, int(fps / 8))

    print(f"  [PREVIEW] Analyse 1 frame / {skip} (skip={skip})")

    while frame_id < max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % skip != 0:
            continue

        # Log progression
        if frame_id % 300 == 0:
            print(f"  [PREVIEW] frame {frame_id}/{max_frame} | "
                  f"pids={len(pid_colors)} obs={n_obs_total}")

        try:
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H),
                               interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"  [PREVIEW] resize error f{frame_id}: {e}")
            continue

        # Masque mouvement
        motion = None
        try:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
                motion = cv2.dilate(motion, kernel, iterations=2)
            prev_gray = gray
        except Exception as e:
            print(f"  [PREVIEW] motion error f{frame_id}: {e}")
            prev_gray = None

        # Détection YOLO
        try:
            results = detector.model(
                [small], conf=0.4, verbose=False,
                imgsz=int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ))
            )
        except Exception as e:
            print(f"  [PREVIEW] YOLO error f{frame_id}: {e}")
            continue

        players = []
        for box in results[0].boxes:
            if int(box.cls[0]) != detector.player_cls:
                continue
            if float(box.conf[0]) < 0.4:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (y2 - y1) < PROCESS_H * 0.12:
                continue
            players.append({
                "bbox":   [x1, y1, x2, y2],
                "center": [(x1+x2)/2, (y1+y2)/2],
                "conf":   float(box.conf[0])
            })

        if not players:
            continue

        # Tracking
        try:
            tracked = tracker.update(players, small)
        except Exception as e:
            print(f"  [PREVIEW] tracker error f{frame_id}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Accumuler couleurs
        for p in tracked:
            pid  = str(p.get("id") or p.get("player_id", ""))
            bbox = p.get("bbox")
            if not pid or not bbox:
                continue

            if motion is not None:
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    roi = motion[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if roi.size > 0 and roi.mean() < 15:
                        continue
                except Exception:
                    pass

            color = extract_jersey_color_strict(small, bbox)
            if color is None:
                n_rejected += 1
                continue

            pid_colors[pid].append(color)
            n_obs_total += 1

    cap.release()
    print(f"  [PREVIEW] Boucle terminée : {frame_id} frames | "
          f"{len(pid_colors)} PIDs | "
          f"{n_obs_total} obs valides | {n_rejected} rejetées")

    # Joueurs stables
    MIN_OBS = 8
    stable_colors = []
    stable_pids   = []

    for pid, colors in pid_colors.items():
        if len(colors) < MIN_OBS:
            continue
        arr = np.array(colors, dtype=np.float32)

        # Extraire les teintes H pour chaque observation
        hues = []
        for c in arr:
            try:
                px  = np.uint8([[[int(c[0]), int(c[1]), int(c[2])]]])
                h_v = int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0][0]) * 2
                hues.append(h_v)
            except Exception:
                hues.append(0)
        hues = np.array(hues, dtype=np.float32)

        if len(arr) >= 12:
            # KMeans sur les teintes → trouver le cluster dominant
            try:
                h_2d     = hues.reshape(-1, 1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                _, labels, centers = cv2.kmeans(
                    h_2d, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
                )
                labels = labels.flatten()
                n0, n1 = (labels==0).sum(), (labels==1).sum()
                # Cluster dominant = maillot
                dom_label = 0 if n0 >= n1 else 1
                dom_mask  = labels == dom_label
                if dom_mask.sum() >= 4:
                    final = np.median(arr[dom_mask], axis=0)
                else:
                    final = np.median(arr, axis=0)
            except Exception:
                final = np.median(arr, axis=0)
        else:
            # Peu d'obs → médiane simple avec filtre outliers
            median = np.median(arr, axis=0)
            dists  = np.linalg.norm(arr - median, axis=1)
            thresh = np.percentile(dists, 70)
            clean  = arr[dists <= thresh]
            final  = np.median(clean, axis=0) if len(clean) >= 3 else median

        stable_colors.append(final)
        stable_pids.append(pid)

    n_stable = len(stable_colors)
    print(f"  [PREVIEW] {n_stable} joueurs stables (>= {MIN_OBS} obs)")

    # Fallback PlayerReID
    if n_stable < 4:
        print(f"  [PREVIEW] Pas assez → fallback PlayerReID")
        try:
            from analysis.player_reid import get_team_colors
            reid = get_team_colors()
            if reid and len(reid) >= 2:
                c0 = tuple(int(x) for x in reid[0])
                c1 = tuple(int(x) for x in reid[1])
                print(f"  [PREVIEW] PlayerReID: {c0}→{bgr_to_name(c0)} | "
                      f"{c1}→{bgr_to_name(c1)}")
                return {
                    "success": True,
                    "n_players_analyzed": 0,
                    "team_0": {"color_bgr": list(c0), "color_name": bgr_to_name(c0),
                               "short_bgr": None, "preview_frame": None},
                    "team_1": {"color_bgr": list(c1), "color_name": bgr_to_name(c1),
                               "short_bgr": None, "preview_frame": None},
                }
        except Exception as e:
            print(f"  [PREVIEW] PlayerReID échoué : {e}")
        return {"success": False,
                "error": f"Pas assez de joueurs stables ({n_stable})"}

    # KMeans
    samples  = np.array(stable_colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centroids = cv2.kmeans(
        samples, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    n0  = int((labels == 0).sum())
    n1  = int((labels == 1).sum())
    c0  = tuple(int(x) for x in centroids[0])
    c1  = tuple(int(x) for x in centroids[1])
    dist = float(np.linalg.norm(centroids[0] - centroids[1]))

    print(f"  [PREVIEW] KMeans: dist={dist:.1f} | "
          f"Team0:{c0}→{bgr_to_name(c0)}({n0}j) | "
          f"Team1:{c1}→{bgr_to_name(c1)}({n1}j)")

    # Preview frames
    preview_0 = preview_1 = None
    try:
        cap2 = cv2.VideoCapture(video_path)
        best_f0 = best_f1 = None
        best_s0 = best_s1 = 0
        for fid in np.linspace(int(fps*5), max_frame, 12, dtype=int):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frm = cap2.read()
            if not ret: continue
            sm = cv2.resize(frm, (PROCESS_W, PROCESS_H))
            try:
                res = detector.model([sm], conf=0.4, verbose=False,
                                      imgsz=int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ)))
                pls = [{"bbox": b.xyxy[0].tolist(),
                        "center": [(b.xyxy[0][0]+b.xyxy[0][2])/2,
                                   (b.xyxy[0][1]+b.xyxy[0][3])/2],
                        "conf": float(b.conf[0])}
                       for b in res[0].boxes
                       if int(b.cls[0]) == detector.player_cls]
                tr2 = tracker.update(pls, sm)
            except Exception:
                continue
            s0 = s1 = 0
            for p in tr2:
                col = extract_jersey_color_strict(sm, p.get("bbox", []))
                if col is None: continue
                c = np.array(col, dtype=np.float32)
                if np.linalg.norm(c-centroids[0]) < np.linalg.norm(c-centroids[1]):
                    s0 += 1
                else:
                    s1 += 1
            if s0 > best_s0: best_s0 = s0; best_f0 = frm.copy()
            if s1 > best_s1: best_s1 = s1; best_f1 = frm.copy()
        cap2.release()
        if best_f0 is not None:
            preview_0 = save_preview_frame(best_f0, 0, output_dir, c0)
        if best_f1 is not None:
            preview_1 = save_preview_frame(best_f1, 1, output_dir, c1)
    except Exception as e:
        print(f"  [PREVIEW] Preview frames échouées : {e}")

    return {
        "success":            True,
        "n_players_analyzed": n0 + n1,
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