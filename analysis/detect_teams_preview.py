# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-
#
# V4 — Histogramme HSV par joueur + KMeans sur joueurs
#
# Pipeline :
#   1. Tracker YOLO+DeepSort sur 30s
#   2. Par bbox : histogramme HSV 24D (16 bins maillot + 8 bins short)
#   3. Accumulation par PID
#   4. KMeans SUR joueurs (2 clusters)
#   5. Couleurs BGR récupérées séparément pour l'affichage

import os
import cv2
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────
# FEATURES COULEUR : HSV hist + LAB mean
# ─────────────────────────────────────────
_H_BINS_JERSEY = 16
_H_BINS_SHORT  = 8


def _h_histogram(zone, bins):
    """Histogramme de teinte H normalisé sur les pixels colorés."""
    if zone is None or zone.size == 0 or zone.shape[0] < 4 or zone.shape[1] < 4:
        return None
    try:
        hsv  = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        S, V = hsv[:,:,1], hsv[:,:,2]
        H    = hsv[:,:,0]
        mask = (S > 40) & (V > 20) & (V < 240)
        if mask.sum() < 5:
            mask = V < 230
            if mask.sum() < 5:
                return None
        h_vals = H[mask].astype(np.float32)
        hist   = np.histogram(h_vals, bins=bins, range=(0, 180))[0].astype(np.float32)
        total  = hist.sum()
        if total > 0:
            hist /= total
        return hist
    except Exception:
        return None


def _ellipse_mask(zone):
    """Masque elliptique central — garde uniquement le cœur du maillot."""
    h, w = zone.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx = max(1, int(w * 0.35))   # rayon horizontal 35% largeur
    ry = max(1, int(h * 0.45))   # rayon vertical 45% hauteur
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask > 0


def _lab_feature(zone):
    """
    Mean LAB normalisé sur le cœur du maillot (masque elliptique).
    LAB est plus stable que HSV sous mauvaise lumière.
    """
    if zone is None or zone.size == 0 or zone.shape[0] < 6 or zone.shape[1] < 6:
        return None
    try:
        lab  = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = _ellipse_mask(zone)
        if mask.sum() < 5:
            return None
        mean_lab = lab[mask].mean(axis=0)  # [L, A, B]
        # Normaliser : L→0-1, A→-1-1, B→-1-1
        return np.array([
            mean_lab[0] / 255.0,
            (mean_lab[1] - 128.0) / 128.0,
            (mean_lab[2] - 128.0) / 128.0,
        ], dtype=np.float32)
    except Exception:
        return None


def _mean_bgr(zone, accept_dark=False):
    """Couleur moyenne BGR d'une zone pour l'affichage."""
    if zone is None or zone.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        S, V = hsv[:,:,1], hsv[:,:,2]
        if accept_dark:
            mask = V < 230
        else:
            mask = (S > 40) & (V > 30) & (V < 230)
        if mask.sum() < 5:
            mask = np.ones(S.shape, dtype=bool)
        return zone[mask].mean(axis=0).astype(float)
    except Exception:
        return zone.mean(axis=(0,1)).astype(float)


def _trim_bbox_bottom(crop):
    """
    Recadre le bas de la bbox pour exclure le gazon.
    Si les dernières lignes sont vertes → c'est du sol, pas des jambes.
    Retourne le crop recadré.
    """
    h, w = crop.shape[:2]
    if h < 20:
        return crop
    # Tester les 15% du bas
    bottom = crop[int(h*0.85):, :]
    if bottom.size == 0:
        return crop
    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    H, S = hsv[:,:,0]*2, hsv[:,:,1]
    # Si >40% des pixels du bas sont du gazon (H=30-90) → recadrer
    gazon_mask = (H >= 30) & (H <= 90) & (S > 40)
    if gazon_mask.mean() > 0.40:
        # Couper les 15% du bas
        new_h = int(h * 0.85)
        return crop[:new_h, :]
    return crop


def extract_player_feature(frame, bbox):
    """
    Retourne vecteur 27D = [hist_H_maillot(16D), hist_H_short(8D), LAB_maillot(3D)].
    - Masque elliptique sur le torse central → élimine bras/fond
    - LAB pour robustesse sous mauvaise lumière
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 30 or w < 12:
            return None

        crop = _trim_bbox_bottom(crop)
        h, w = crop.shape[:2]
        if h < 30:
            return None

        # Zone torse uniquement : 20%→55%, centre horizontal 20%→80%
        # PAS de short → élimine contamination jambes/gazon
        torso = crop[int(h*0.20):int(h*0.55), int(w*0.20):int(w*0.80)]

        h_jersey = _h_histogram(torso, bins=_H_BINS_JERSEY)
        lab_feat = _lab_feature(torso)

        if h_jersey is None:
            return None

        lab_part = lab_feat if lab_feat is not None else np.zeros(3, dtype=np.float32)

        # LAB × 4 : discriminant principal (A sépare vert vs rouge/bordeaux)
        return np.concatenate([h_jersey, lab_part * 4.0])  # 19D

    except Exception:
        return None


def _is_gazon_color(bgr):
    """Retourne True si la couleur ressemble au gazon (H=30-90, V>60)."""
    if bgr is None: return True
    try:
        b,g,r = int(bgr[0]),int(bgr[1]),int(bgr[2])
        px  = np.uint8([[[b,g,r]]])
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        h = int(hsv[0])*2; s = int(hsv[1]); v = int(hsv[2])
        if 15 <= h <= 100: return True   # jaune/vert = gazon
        return False
    except Exception:
        return True


def extract_player_bgr(frame, bbox):
    """
    Retourne (jersey_bgr, short_bgr) pour l'affichage.
    Le short n'est extrait que sur les grandes bboxes (joueurs proches).
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        h, w = crop.shape[:2]
        if h < 30 or w < 12:
            return None, None

        torso   = crop[int(h*0.20):int(h*0.45), int(w*0.35):int(w*0.65)]
        j_bgr   = _mean_bgr(torso)

        # Short : seulement sur les joueurs assez grands (h >= 80px)
        # et vérifier que la couleur n'est pas du gazon
        s_bgr = None
        if h >= 80:
            short_z = crop[int(h*0.50):int(h*0.75), int(w*0.25):int(w*0.75)]
            s_candidate = _mean_bgr(short_z, accept_dark=True)
            if s_candidate is not None and not _is_gazon_color(s_candidate):
                s_bgr = s_candidate

        return j_bgr, s_bgr
    except Exception:
        return None, None


# ─────────────────────────────────────────
# NOM COULEUR
# ─────────────────────────────────────────
def bgr_to_name(bgr):
    if bgr is None:
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


def lab_to_color_name(bgr, short_bgr=None):
    """
    Naming couleur robuste basé sur LAB A/B.
    Priorité : canal A (vert vs rouge/bordeaux), puis HSV pour affiner.
    Si maillot ambigu → utiliser le short comme fallback.
    """
    if bgr is None:
        return "inconnu", 0.0

    try:
        b,g,r = int(bgr[0]),int(bgr[1]),int(bgr[2])
        px  = np.uint8([[[b,g,r]]])
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0]
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]

        A = (int(lab[1]) - 128) / 128.0   # -1 à +1
        B = (int(lab[2]) - 128) / 128.0
        L = int(lab[0]) / 255.0
        H = int(hsv[0]) * 2
        S = int(hsv[1]) / 255.0
        V = int(hsv[2]) / 255.0
        conf = abs(A)   # confiance = force du signal A

        # Très sombre → noir
        if V < 0.20:
            return "noir", 0.8

        # Très peu saturé → gris/blanc
        if S < 0.12:
            return "blanc" if V > 0.70 else "gris", 0.5

        # LAB A discriminant principal
        if A < -0.08:
            name = "vert foncé" if V < 0.45 else "vert"
            return name, conf

        if A > 0.08:
            if H < 20 or H >= 340:
                name = "bordeaux foncé" if V < 0.35 else ("bordeaux" if V < 0.55 else "rouge")
            elif 20 <= H < 35:
                name = "orange"
            elif 185 <= H < 265:
                name = "bleu marine" if V < 0.55 else "bleu"
            elif 265 <= H < 295:
                name = "violet"
            else:
                name = "bordeaux"
            return name, conf

        # Zone ambiguë (|A| < 0.08)
        # 1. Essayer le short
        if short_bgr is not None:
            s_b,s_g,s_r = int(short_bgr[0]),int(short_bgr[1]),int(short_bgr[2])
            px_s  = np.uint8([[[s_b,s_g,s_r]]])
            lab_s = cv2.cvtColor(px_s, cv2.COLOR_BGR2LAB)[0][0]
            A_s   = (int(lab_s[1]) - 128) / 128.0
            if A_s < -0.05:
                return "vert", abs(A_s) * 0.7
            elif A_s > 0.05:
                return "rouge", abs(A_s) * 0.7

        # 2. R-B fort → rouge/chaud même si A≈0 (compression désature)
        r_minus_b = r - b
        if r_minus_b > 35:
            return "rouge", min(r_minus_b / 100.0, 0.5)

        # 3. B-R fort → bleu/froid
        if b - r > 20:
            return "bleu marine" if V < 0.55 else "bleu", min((b - r) / 100.0, 0.5)

        # Fallback HSV
        return bgr_to_name(bgr), 0.2

    except Exception:
        return bgr_to_name(bgr) if bgr else "inconnu", 0.1


def save_preview_frame(frame, team_id, output_dir, color_bgr=None):
    os.makedirs(output_dir, exist_ok=True)
    preview = frame.copy()
    h, w    = preview.shape[:2]
    if color_bgr:
        b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
        cv2.rectangle(preview, (0, h-30), (w, h), (b, g, r), -1)
        cv2.putText(preview, bgr_to_name(color_bgr).upper(),
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    path = os.path.join(output_dir, f"team_{team_id}_preview.jpg")
    cv2.imwrite(path, preview)
    return path


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE
# ─────────────────────────────────────────
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

    PROCESS_W, PROCESS_H = 1088, 612

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

    pid_features  = defaultdict(list)   # pid → features 24D
    pid_team      = {}                  # pid → équipe assignée par ReID
    pid_bgr_j     = defaultdict(list)   # pid → jersey BGR
    pid_bgr_s     = defaultdict(list)   # pid → short BGR
    n_obs_total   = 0
    n_rejected    = 0
    frame_id      = 0
    skip          = max(1, int(fps / 8))

    print(f"  [PREVIEW] skip={skip} (~8fps)")

    # ── Trouver la meilleure fenêtre de 30s ──────────────────────────────────
    # On cherche une fenêtre avec beaucoup de joueurs grands et statiques
    # En scannant légèrement la vidéo (toutes les 5s sur les 5 premières minutes)
    best_start_frame = 0
    best_window_score = -1
    scan_limit = min(int(fps * 120), total_frames)  # 2 min max

    print(f"  [PREVIEW] Recherche meilleure fenêtre...")
    for scan_f in range(0, scan_limit, int(fps * 5)):   # toutes les 5s
        cap.set(cv2.CAP_PROP_POS_FRAMES, scan_f)
        ret_s, frame_s = cap.read()
        if not ret_s:
            break
        sm_s = cv2.resize(frame_s, (PROCESS_W, PROCESS_H),
                           interpolation=cv2.INTER_LINEAR)
        try:
            res_s = detector.model([sm_s], conf=0.4, verbose=False,
                                    imgsz=int(os.environ.get('YOLO_IMGSZ',
                                             config.YOLO_IMGSZ)))
            # Score = somme des hauteurs des grandes bboxes
            score_s = sum(
                float(b.xyxy[0][3] - b.xyxy[0][1])
                for b in res_s[0].boxes
                if int(b.cls[0]) == detector.player_cls
                and float(b.conf[0]) >= 0.4
                and float(b.xyxy[0][3] - b.xyxy[0][1]) >= PROCESS_H * 0.15
            )
            if score_s > best_window_score:
                best_window_score = score_s
                best_start_frame  = scan_f
        except Exception:
            continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset
    max_frame = min(best_start_frame + int(bootstrap_duration * fps), total_frames)
    print(f"  [PREVIEW] Meilleure fenêtre: t={best_start_frame/fps:.0f}s "
          f"(score={best_window_score:.0f}) → analyse t={best_start_frame/fps:.0f}s "
          f"à t={max_frame/fps:.0f}s")

    # Repositionner la capture au bon endroit
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_frame)
    frame_id = best_start_frame

    while frame_id < max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % skip != 0:
            continue

        if frame_id % 300 == 0:
            print(f"  [PREVIEW] frame {frame_id}/{max_frame} | "
                  f"pids={len(pid_features)} obs={n_obs_total}")

        small = cv2.resize(frame, (PROCESS_W, PROCESS_H),
                           interpolation=cv2.INTER_LINEAR)

        try:
            results = detector.model(
                [small], conf=0.4, verbose=False,
                imgsz=int(os.environ.get('YOLO_IMGSZ', config.YOLO_IMGSZ))
            )
        except Exception:
            continue

        players = []
        for box in results[0].boxes:
            if int(box.cls[0]) != detector.player_cls: continue
            if float(box.conf[0]) < 0.4: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bh = y2 - y1
            bw = x2 - x1
            # Seuil agressif : garder seulement les joueurs assez proches
            # → maillot et short bien visibles, peu de contamination gazon
            if bh < PROCESS_H * 0.22: continue   # ~135px sur 612 → joueurs très proches
            if bw < PROCESS_W * 0.03: continue
            players.append({"bbox": [x1,y1,x2,y2],
                             "center": [(x1+x2)/2,(y1+y2)/2],
                             "conf": float(box.conf[0])})

        if not players:
            continue

        # Ne garder que les 4 joueurs les plus grands (plus proches = couleurs plus propres)
        players = sorted(players, key=lambda p: -(p["bbox"][3]-p["bbox"][1]))[:4]

        try:
            tracked = tracker.update(players, small)
        except Exception:
            continue

        for p in tracked:
            pid  = str(p.get("id") or p.get("player_id", ""))
            bbox = p.get("bbox")
            if not pid or not bbox:
                continue

            # Stocker l'équipe assignée par le tracker (ReID)
            team = p.get("team")
            if team is not None:
                pid_team[pid] = int(team)

            # Filtre flou
            try:
                x1b,y1b,x2b,y2b = map(int, bbox)
                patch = small[max(0,y1b):max(0,y2b), max(0,x1b):max(0,x2b)]
                if patch.size > 0:
                    if cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var() < 30:
                        n_rejected += 1
                        continue
            except Exception:
                pass

            feat = extract_player_feature(small, bbox)
            if feat is None:
                n_rejected += 1
                continue

            pid_features[pid].append(feat)
            n_obs_total += 1

            # Couleurs BGR pour affichage (1 fois suffit)
            if len(pid_bgr_j[pid]) < 5:
                j_bgr, s_bgr = extract_player_bgr(small, bbox)
                if j_bgr is not None:
                    pid_bgr_j[pid].append(j_bgr)
                if s_bgr is not None:
                    pid_bgr_s[pid].append(s_bgr)

    cap.release()
    print(f"  [PREVIEW] Boucle terminée : {frame_id} frames | "
          f"{len(pid_features)} PIDs | "
          f"{n_obs_total} obs valides | {n_rejected} rejetées")

    # ── Joueurs stables ───────────────────────────────────────────────────────
    MIN_OBS = 4   # moins d'observations car on garde seulement les joueurs proches
    player_feats = []
    stable_pids  = []
    player_bgr_j = []   # couleur maillot par joueur stable
    player_bgr_s = []   # couleur short par joueur stable

    for pid, feats in pid_features.items():
        if len(feats) < MIN_OBS:
            continue
        arr    = np.array(feats, dtype=np.float32)
        median = np.median(arr, axis=0)
        dists  = np.linalg.norm(arr - median, axis=1)
        thresh = np.percentile(dists, 70)
        clean  = arr[dists <= thresh]
        final  = np.median(clean, axis=0) if len(clean) >= 3 else median
        player_feats.append(final)
        stable_pids.append(pid)

        # Médiane BGR pour affichage
        jl = pid_bgr_j[pid]
        sl = pid_bgr_s[pid]
        player_bgr_j.append(np.median(np.array(jl, dtype=np.float32), axis=0) if jl else None)
        player_bgr_s.append(np.median(np.array(sl, dtype=np.float32), axis=0) if sl else None)

    n_players = len(player_feats)
    print(f"  [PREVIEW] {n_players} joueurs stables (>= {MIN_OBS} obs)")

    if n_players < 4:
        return {"success": False,
                "error": f"Pas assez de joueurs stables ({n_players})"}

    samples = np.array(player_feats, dtype=np.float32)

    # ── Filtre gardiens/arbitres (petits clusters) ────────────────────────────
    n_pre = min(4, n_players // 2, n_players)
    mask_big = np.ones(n_players, dtype=bool)

    if n_pre >= 3 and n_players >= 6:
        try:
            crit_pre = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, lbl_pre, _ = cv2.kmeans(samples, n_pre, None, crit_pre, 5,
                                        cv2.KMEANS_PP_CENTERS)
            lbl_pre   = lbl_pre.flatten()
            cnt_pre   = np.bincount(lbl_pre, minlength=n_pre)
            big       = [i for i, c in enumerate(cnt_pre) if c >= n_players * 0.15]
            if len(big) >= 2:
                mask_big = np.isin(lbl_pre, big)
                excluded = n_players - mask_big.sum()
                print(f"  [PREVIEW] Exclusion petits clusters : "
                      f"{mask_big.sum()}/{n_players} gardés ({excluded} exclus)")
        except Exception:
            pass

    samples_c   = samples[mask_big]
    stable_pids = [stable_pids[i]  for i in range(n_players) if mask_big[i]]
    player_bgr_j = [player_bgr_j[i] for i in range(n_players) if mask_big[i]]
    player_bgr_s = [player_bgr_s[i] for i in range(n_players) if mask_big[i]]

    if len(samples_c) < 4:
        samples_c = samples

    # ── Récupérer couleurs ReID comme seeds KMeans ───────────────────────────
    reid_seeds = None
    try:
        from analysis.player_reid import get_team_colors
        reid_colors = get_team_colors()
        if reid_colors and len(reid_colors) >= 2:
            c0_r = np.array(reid_colors[0], dtype=np.float32)
            c1_r = np.array(reid_colors[1], dtype=np.float32)
            # Convertir BGR en histogramme approximatif pour initialisation
            # (juste utiliser comme hint de direction)
            reid_seeds = np.array([c0_r[:3], c1_r[:3]])
            print(f"  [PREVIEW] ReID seeds: {tuple(int(x) for x in c0_r[:3])} | "
                  f"{tuple(int(x) for x in c1_r[:3])}")
    except Exception:
        pass

    # ── KMeans sur joueurs — 8 runs, meilleur équilibre×distance ─────────────
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    best_labels = best_score = None

    for _ in range(8):
        _, lbl, ctr = cv2.kmeans(samples_c, 2, None, criteria, 1,
                                  cv2.KMEANS_PP_CENTERS)
        lbl   = lbl.flatten()
        na, nb = int((lbl==0).sum()), int((lbl==1).sum())
        balance = 1.0 - abs(na-nb) / max(na+nb, 1)
        d = float(np.linalg.norm(ctr[0] - ctr[1]))
        score = balance * d
        if best_score is None or score > best_score:
            best_score  = score
            best_labels = lbl

    labels = best_labels
    n0 = int((labels==0).sum())
    n1 = int((labels==1).sum())
    print(f"  [PREVIEW] Balance: {n0}j vs {n1}j (score={best_score:.1f})")

    # ── Cohérence avec ReID ───────────────────────────────────────────────────
    # Si le ReID a assigné des équipes, vérifier si le KMeans est aligné
    if pid_team:
        reid_votes = {0: {0:0, 1:0}, 1: {0:0, 1:0}}
        for i, pid in enumerate(stable_pids):
            if i >= len(labels): continue
            km_label  = int(labels[i])
            reid_label = pid_team.get(pid)
            if reid_label is not None:
                reid_votes[km_label][reid_label] += 1

        # Si KMeans cluster 0 = majorité ReID équipe 1 → inverser
        v00 = reid_votes[0].get(0, 0)
        v01 = reid_votes[0].get(1, 0)
        if v01 > v00:
            labels = 1 - labels  # inverser
            n0, n1 = n1, n0
            print(f"  [PREVIEW] Labels inversés pour cohérence avec ReID")

    # ── Couleurs BGR par cluster ──────────────────────────────────────────────
    def cluster_bgr(ci, bgr_list):
        vals = [bgr_list[i] for i in range(len(labels))
                if labels[i] == ci and bgr_list[i] is not None]
        if not vals:
            return None
        return tuple(int(x) for x in np.median(
            np.array(vals, dtype=np.float32), axis=0))

    c0_j = cluster_bgr(0, player_bgr_j)
    c1_j = cluster_bgr(1, player_bgr_j)
    c0_s = cluster_bgr(0, player_bgr_s)
    c1_s = cluster_bgr(1, player_bgr_s)

    c0_j = c0_j or (100, 100, 100)
    c1_j = c1_j or (50, 50, 50)

    def make_name(j_bgr, s_bgr):
        name, conf = lab_to_color_name(j_bgr, s_bgr)
        s_name = bgr_to_name(s_bgr) if s_bgr else None
        # Ajouter le short au nom si utile
        if s_name and s_name not in ("inconnu", "gris", name):
            return f"{name}/{s_name}", conf
        return name, conf

    name0, conf0 = make_name(c0_j, c0_s)
    name1, conf1 = make_name(c1_j, c1_s)

    # Log LAB pour debug
    def lab_a(bgr):
        if bgr is None: return 0.0
        try:
            px = np.uint8([[[int(bgr[0]),int(bgr[1]),int(bgr[2])]]])
            lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0]
            return round((int(lab[1])-128)/128.0, 3)
        except: return 0.0

    print(f"  [PREVIEW] Team0: {c0_j}→{name0} (A={lab_a(c0_j):+.3f} conf={conf0:.2f}) short={c0_s} ({n0}j)")
    print(f"  [PREVIEW] Team1: {c1_j}→{name1} (A={lab_a(c1_j):+.3f} conf={conf1:.2f}) short={c1_s} ({n1}j)")

    # ── Preview frames ────────────────────────────────────────────────────────
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
                                      imgsz=int(os.environ.get('YOLO_IMGSZ',
                                               config.YOLO_IMGSZ)))
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
                feat = extract_player_feature(sm, p.get("bbox", []))
                if feat is None: continue
                # Attribuer au cluster le plus proche (par histogramme)
                f = feat.reshape(1, -1).astype(np.float32)
                d0 = float(cv2.norm(f - samples_c[labels==0].mean(axis=0)))
                d1 = float(cv2.norm(f - samples_c[labels==1].mean(axis=0)))
                if d0 < d1: s0 += 1
                else:        s1 += 1
            if s0 > best_s0: best_s0 = s0; best_f0 = frm.copy()
            if s1 > best_s1: best_s1 = s1; best_f1 = frm.copy()

        cap2.release()
        if best_f0 is not None:
            preview_0 = save_preview_frame(best_f0, 0, output_dir, c0_j)
        if best_f1 is not None:
            preview_1 = save_preview_frame(best_f1, 1, output_dir, c1_j)
    except Exception as e:
        print(f"  [PREVIEW] Preview frames échouées : {e}")

    return {
        "success":            True,
        "n_players_analyzed": n0 + n1,
        "team_0": {
            "color_bgr":     list(c0_j),
            "color_name":    name0,
            "short_bgr":     list(c0_s) if c0_s else None,
            "preview_frame": preview_0,
        },
        "team_1": {
            "color_bgr":     list(c1_j),
            "color_name":    name1,
            "short_bgr":     list(c1_s) if c1_s else None,
            "preview_frame": preview_1,
        },
    }