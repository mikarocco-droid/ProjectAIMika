# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-
#
# V3 — KMeans PAR JOUEUR (pas par pixel)
#
# Pipeline :
#   1. Tracker YOLO+DeepSort sur 30s (coup d'envoi)
#   2. Par bbox : mini-KMeans(3) sur zone torse stricte
#      → couleur dominante maillot (cluster le plus saturé)
#   3. Accumulation par PID (20+ obs)
#   4. Médiane robuste par joueur
#   5. KMeans SUR joueurs (2 clusters)
#   6. Filtrage gardiens/arbitres avant clustering

import os
import cv2
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────
# COULEUR DOMINANTE MAILLOT PAR BBOX
# ─────────────────────────────────────────
def _best_cluster_color(zone):
    """
    Mini-KMeans(3) sur une zone → retourne la couleur du cluster
    le plus compact et saturé. Retourne None si qualité insuffisante.
    """
    if zone is None or zone.size == 0 or zone.shape[0] < 5 or zone.shape[1] < 5:
        return None
    try:
        pixels = zone.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 9:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        labels_flat  = labels.flatten()
        label_counts = np.bincount(labels_flat, minlength=3)
        cluster_stds = []
        for i in range(3):
            mask_i = labels_flat == i
            if mask_i.sum() < 3:
                cluster_stds.append(999.0)
                continue
            cluster_stds.append(float(np.mean(np.std(pixels[mask_i], axis=0))))

        best_center = None
        best_score  = -1
        for i, center in enumerate(centers):
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            if (b + g + r) / 3 < 30: continue   # noir pur
            if (b + g + r) / 3 > 240: continue  # blanc pur
            px  = np.uint8([[[b, g, r]]])
            sat = int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0][1])
            compacity  = max(0, 50 - cluster_stds[i])
            score      = label_counts[i] * (sat + 10) * (compacity + 1)
            if score > best_score:
                best_score  = score
                best_center = center

        if best_center is None:
            return None
        b, g, r = int(best_center[0]), int(best_center[1]), int(best_center[2])
        px  = np.uint8([[[b, g, r]]])
        if int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0][1]) < 35:
            return None
        return best_center.astype(float)
    except Exception:
        return None


def _best_cluster_color_short(zone):
    """
    Comme _best_cluster_color mais accepte aussi les couleurs sombres/noires.
    Le noir est une couleur valide pour un short.
    """
    if zone is None or zone.size == 0 or zone.shape[0] < 4 or zone.shape[1] < 4:
        return None
    try:
        pixels = zone.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 6:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        n_clusters = min(3, len(pixels) // 3)
        if n_clusters < 2:
            return pixels.mean(axis=0)
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        labels_flat  = labels.flatten()
        label_counts = np.bincount(labels_flat, minlength=n_clusters)

        # Cluster dominant = le plus grand (pas de contrainte saturation)
        best_idx    = int(np.argmax(label_counts))
        best_center = centers[best_idx]

        b, g, r = int(best_center[0]), int(best_center[1]), int(best_center[2])
        # Rejeter blanc pur (fond, ciel)
        if (b + g + r) / 3 > 220:
            return None
        return best_center.astype(float)
    except Exception:
        return None


def dominant_jersey_color(frame, bbox):
    """
    Extrait vecteur 6D [maillot_BGR, short_BGR].
    Maillot : zone 20%→45% hauteur, 35%→65% largeur (torse central).
    Short   : zone 50%→75% hauteur, 25%→75% largeur.
    Mini-KMeans(3) par zone → cluster le plus compact et saturé.
    Retourne None si qualité insuffisante.
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

        # Zone torse stricte
        torso = crop[int(h*0.20):int(h*0.45),
                     int(w*0.35):int(w*0.65)]
        if torso.size == 0 or torso.shape[0] < 5 or torso.shape[1] < 5:
            return None

        # ── Mini-KMeans(3) sur le torse ──────────────────────────────────
        pixels = torso.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 9:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        labels_flat  = labels.flatten()
        label_counts = np.bincount(labels_flat, minlength=3)

        # ── Compacité des clusters (variance intra-cluster) ───────────────
        # Un cluster compact = maillot uniforme
        # Un cluster diffus  = gazon/fond mélangé
        # On mesure l'écart-type des pixels de chaque cluster
        cluster_stds = []
        for i in range(3):
            mask_i = labels_flat == i
            if mask_i.sum() < 3:
                cluster_stds.append(999.0)
                continue
            cluster_pixels = pixels[mask_i]
            std_i = float(np.mean(np.std(cluster_pixels, axis=0)))
            cluster_stds.append(std_i)

        # ── Choisir le cluster dominant le plus compact ───────────────────
        # Score = taille × compacité × saturation
        # Pas de règle sur la couleur (vert maillot autorisé)
        best_center = None
        best_score  = -1

        for i, center in enumerate(centers):
            b, g, r = int(center[0]), int(center[1]), int(center[2])

            # Rejeter noir pur / ombres profondes
            if (b + g + r) / 3 < 30:
                continue
            # Rejeter blanc pur / surexposition
            if (b + g + r) / 3 > 240:
                continue

            px  = np.uint8([[[b, g, r]]])
            hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
            sat = int(hsv[1])

            n_cluster  = label_counts[i]
            compacity  = max(0, 50 - cluster_stds[i])  # plus compact = meilleur
            score      = n_cluster * (sat + 10) * (compacity + 1)

            if score > best_score:
                best_score  = score
                best_center = center

        if best_center is None:
            return None

        # Rejeter si trop peu saturé (gazon dilué, béton, fond gris)
        b, g, r = int(best_center[0]), int(best_center[1]), int(best_center[2])
        px  = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        if int(hsv[1]) < 35:
            return None

        c_jersey = best_center.astype(float)

        # ── Extraire couleur short ────────────────────────────────────────
        # Zone 50%→75% hauteur, 25%→75% largeur
        short_zone = crop[int(h*0.50):int(h*0.75),
                         int(w*0.25):int(w*0.75)]
        c_short = _best_cluster_color_short(short_zone)

        if c_short is not None:
            return np.concatenate([c_jersey, c_short])  # vecteur 6D
        else:
            return c_jersey  # fallback 3D

    except Exception:
        return None


# ─────────────────────────────────────────
# NOM COULEUR
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE
# ─────────────────────────────────────────
def player_bgr_colors(frame, bbox):
    """Retourne (jersey_bgr, short_bgr) pour l'affichage — séparé du clustering."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1=max(0,x1); y1=max(0,y1)
        x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None, None
        h, w = crop.shape[:2]
        if h < 30 or w < 12: return None, None
        torso   = crop[int(h*0.20):int(h*0.45), int(w*0.35):int(w*0.65)]
        short_z = crop[int(h*0.50):int(h*0.75), int(w*0.25):int(w*0.75)]
        return _mean_bgr(torso), _mean_bgr(short_z, accept_dark=True)
    except Exception:
        return None, None


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

    PROCESS_W, PROCESS_H = 1088, 612  # compromis vitesse/qualité (~3min)
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

    # pid → liste de features (histogrammes 24D)
    pid_colors       = defaultdict(list)
    # pid → (jersey_bgr, short_bgr) pour l'affichage
    bgr_colors_by_pid = {}
    n_obs_total = 0
    n_rejected  = 0
    frame_id    = 0
    skip        = max(1, int(fps / 8))

    print(f"  [PREVIEW] skip={skip} (~8fps)")

    while frame_id < max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % skip != 0:
            continue

        if frame_id % 300 == 0:
            print(f"  [PREVIEW] frame {frame_id}/{max_frame} | "
                  f"pids={len(pid_colors)} obs={n_obs_total}")

        small = cv2.resize(frame, (PROCESS_W, PROCESS_H),
                           interpolation=cv2.INTER_LINEAR)

        # Détection YOLO
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
            if bh < PROCESS_H * 0.10: continue  # trop loin (adapté 720p)
            if bw < PROCESS_W * 0.02: continue  # trop étroit
            players.append({
                "bbox":   [x1, y1, x2, y2],
                "center": [(x1+x2)/2, (y1+y2)/2],
                "conf":   float(box.conf[0])
            })

        if not players:
            continue

        try:
            tracked = tracker.update(players, small)
        except Exception as e:
            print(f"  [PREVIEW] tracker error f{frame_id}: {e}")
            continue

        for p in tracked:
            pid  = str(p.get("id") or p.get("player_id", ""))
            bbox = p.get("bbox")
            if not pid or not bbox:
                continue

            # Filtre flou : ignorer les crops flous (motion blur)
            try:
                x1b, y1b, x2b, y2b = map(int, bbox)
                patch = small[max(0,y1b):max(0,y2b), max(0,x1b):max(0,x2b)]
                if patch.size > 0:
                    gray_p  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray_p, cv2.CV_64F).var()
                    if sharpness < 30:   # crop trop flou
                        n_rejected += 1
                        continue
            except Exception:
                pass

            color = dominant_jersey_color(small, bbox)
            if color is None:
                n_rejected += 1
                continue

            pid_colors[pid].append(color)
            n_obs_total += 1

            # Stocker couleurs BGR pour l'affichage (accumulation par PID)
            j_bgr, s_bgr = player_bgr_colors(small, bbox)
            if j_bgr is not None:
                if pid not in bgr_colors_by_pid:
                    bgr_colors_by_pid[pid] = ([], [])
                bgr_colors_by_pid[pid][0].append(j_bgr)
                if s_bgr is not None:
                    bgr_colors_by_pid[pid][1].append(s_bgr)

    cap.release()
    print(f"  [PREVIEW] Boucle terminée : {frame_id} frames | "
          f"{len(pid_colors)} PIDs | "
          f"{n_obs_total} obs valides | {n_rejected} rejetées")

    # ── Médiane robuste par joueur ────────────────────────────────────────────
    MIN_OBS = 6
    player_colors = []  # une feature (histogramme 24D) par joueur stable
    stable_pids   = []  # pid correspondant

    for pid, colors in pid_colors.items():
        if len(colors) < MIN_OBS:
            continue

        # Garder seulement les vecteurs 24D (histogrammes complets)
        colors_filtered = [c for c in colors if len(c) == 24]
        if len(colors_filtered) < MIN_OBS:
            # Fallback : accepter aussi les partiels
            colors_filtered = colors
        if len(colors_filtered) < MIN_OBS:
            continue

        arr    = np.array(colors_filtered, dtype=np.float32)
        # Filtre outliers : garder 70% proches de la médiane
        median = np.median(arr, axis=0)
        dists  = np.linalg.norm(arr - median, axis=1)
        thresh = np.percentile(dists, 70)
        clean  = arr[dists <= thresh]
        final  = np.median(clean, axis=0) if len(clean) >= 3 else median
        player_colors.append(final)
        stable_pids.append(pid)

    n_players = len(player_colors)
    print(f"  [PREVIEW] {n_players} joueurs stables (>= {MIN_OBS} obs)")

    # Calculer médiane BGR par PID
    bgr_median_by_pid = {}
    for pid, (j_list, s_list) in bgr_colors_by_pid.items():
        if j_list:
            bgr_median_by_pid[pid] = (
                np.median(np.array(j_list), axis=0),
                np.median(np.array(s_list), axis=0) if s_list else None
            )
    bgr_colors_by_pid = bgr_median_by_pid

    if n_players < 4:
        return {"success": False,
                "error": f"Pas assez de joueurs stables ({n_players})"}

    # ── KMeans préliminaire pour exclure gardiens/arbitres ──────────────────
    # Principe : sur un match, 2 grandes équipes + petits clusters isolés
    # On fait un KMeans(4) puis on ne garde que les 2 plus grands clusters
    samples = np.array(player_colors, dtype=np.float32)

    n_pre_clusters = min(4, n_players // 2, n_players)
    samples_clean  = samples
    n_filtered     = n_players

    if n_pre_clusters >= 3 and n_players >= 6:
        try:
            criteria_pre = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels_pre, _ = cv2.kmeans(
                samples, n_pre_clusters, None, criteria_pre, 5,
                cv2.KMEANS_PP_CENTERS
            )
            labels_pre   = labels_pre.flatten()
            counts_pre   = np.bincount(labels_pre, minlength=n_pre_clusters)
            # Garder seulement les clusters dont la taille > 15% du total
            threshold    = n_players * 0.15
            big_clusters = [i for i, c in enumerate(counts_pre) if c >= threshold]
            if len(big_clusters) >= 2:
                mask_big   = np.isin(labels_pre, big_clusters)
                samples_clean = samples[mask_big]
                n_filtered    = mask_big.sum()
                excluded = n_players - n_filtered
                print(f"  [PREVIEW] Exclusion petits clusters : "
                      f"{n_filtered}/{n_players} joueurs gardés "
                      f"({excluded} gardiens/arbitres exclus)")
        except Exception as _ep:
            print(f"  [PREVIEW] Filtre clusters ignoré : {_ep}")

    if n_filtered < 4:
        samples_clean = samples
        n_filtered    = n_players

    # ── KMeans SUR joueurs (3D ou 6D selon présence du short) ───────────────
    dim = samples_clean.shape[1]  # 3 ou 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)

    # Lancer KMeans plusieurs fois et garder la solution la plus équilibrée
    # Un match = 2 équipes de taille similaire → ratio idéal proche de 50/50
    best_labels = best_centroids = None
    best_balance_score = -1

    for _ in range(8):
        _, lbl, ctr = cv2.kmeans(
            samples_clean, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )
        lbl = lbl.flatten()
        n_a = int((lbl == 0).sum())
        n_b = int((lbl == 1).sum())
        total = n_a + n_b
        # Score d'équilibre : 1.0 = parfait 50/50, 0 = tout dans un cluster
        balance = 1.0 - abs(n_a - n_b) / max(total, 1)
        # Score de distance : grande distance = bonne séparation
        d = float(np.linalg.norm(ctr[0] - ctr[1]))
        score = balance * d
        if score > best_balance_score:
            best_balance_score = score
            best_labels    = lbl
            best_centroids = ctr

    labels    = best_labels
    centroids = best_centroids
    labels = labels.flatten()
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    print(f"  [PREVIEW] Balance: {n0}j vs {n1}j (score={best_balance_score:.1f})")

    dist = float(np.linalg.norm(centroids[0] - centroids[1]))

    # Recalculer couleurs représentatives sur les joueurs les plus proches
    # du centroid (top 40%) pour éviter les valeurs polluées
    # Les features sont des histogrammes — récupérer les vraies couleurs BGR
    # depuis les PIDs de chaque cluster en relisant les frames
    # Pour simplifier : utiliser les couleurs stockées dans bgr_colors_by_pid
    c0_j = c0_s = c1_j = c1_s = None

    pids_list = stable_pids
    for ci, (cj_ref, cs_ref) in enumerate([(0, 0), (1, 1)]):
        cluster_pids = [pids_list[i] for i in range(len(pids_list))
                        if i < len(labels) and labels[i] == ci]
        bgr_j_list = [bgr_colors_by_pid[p][0] for p in cluster_pids
                      if p in bgr_colors_by_pid and bgr_colors_by_pid[p][0] is not None]
        bgr_s_list = [bgr_colors_by_pid[p][1] for p in cluster_pids
                      if p in bgr_colors_by_pid and bgr_colors_by_pid[p][1] is not None]
        if bgr_j_list:
            med_j = np.median(np.array(bgr_j_list), axis=0)
            if ci == 0: c0_j = tuple(int(x) for x in med_j)
            else:        c1_j = tuple(int(x) for x in med_j)
        if bgr_s_list:
            med_s = np.median(np.array(bgr_s_list), axis=0)
            if ci == 0: c0_s = tuple(int(x) for x in med_s)
            else:        c1_s = tuple(int(x) for x in med_s)

    # Fallbacks
    c0_j = c0_j or (128,128,128)
    c1_j = c1_j or (64,64,64)
    c0   = c0_j
    c1   = c1_j

    # Nommer par le short si sa couleur est plus fiable (plus saturée)
    # Le short est souvent moins contaminé que le maillot
    def best_name(jersey_bgr, short_bgr):
        if short_bgr is None:
            return bgr_to_name(jersey_bgr)
        import numpy as _np
        # Comparer saturation des deux
        def sat(bgr):
            px = _np.uint8([[[int(bgr[0]),int(bgr[1]),int(bgr[2])]]])
            return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0][1])
        s_j = sat(jersey_bgr)
        s_s = sat(short_bgr)
        # Utiliser le plus saturé comme nom principal
        if s_s > s_j + 20:
            return f"{bgr_to_name(short_bgr)} (short)/{bgr_to_name(jersey_bgr)}"
        return bgr_to_name(jersey_bgr) + f"/{bgr_to_name(short_bgr)}"

    name0 = best_name(c0_j, c0_s)
    name1 = best_name(c1_j, c1_s)

    print(f"  [PREVIEW] KMeans hist: dist={dist:.3f} | "
          f"Team0:{c0_j}→{name0}({n0}j) | "
          f"Team1:{c1_j}→{name1}({n1}j)")
    if c0_s:
        print(f"  [PREVIEW] Shorts → Team0:{c0_s}→{bgr_to_name(c0_s)} | "
              f"Team1:{c1_s}→{bgr_to_name(c1_s)}")

    # ── Vérification cohérence : si noms trop similaires → KMeans HSV ────────
    # Sur vecteur 6D le short discrimine déjà → pas besoin du fallback HSV
    similar_names = False
    if dim == 3:
        similar_names = (bgr_to_name(c0_j) == bgr_to_name(c1_j)) or (
            bgr_to_name(c0_j).replace(" foncé","") ==
            bgr_to_name(c1_j).replace(" foncé","")
        )

    if dist < 80 or similar_names:
        print(f"  [PREVIEW] Séparation faible → KMeans sin/cos (distance circulaire)")
        try:
            # Encoder H avec sin/cos pour respecter la circularité (0°=360°)
            circ_colors = []
            for sc in samples_clean:
                px  = np.uint8([[[int(sc[0]), int(sc[1]), int(sc[2])]]])
                hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].astype(float)
                h_rad = hsv[0] * 2 * np.pi / 180.0   # H OpenCV 0-180 → radians
                s_norm = hsv[1] / 255.0
                v_norm = hsv[2] / 255.0
                # Feature : [sin(H)×S, cos(H)×S, V] — pondère saturation
                circ_colors.append([
                    np.sin(h_rad) * s_norm * 2,
                    np.cos(h_rad) * s_norm * 2,
                    v_norm
                ])

            samples_circ = np.array(circ_colors, dtype=np.float32)
            _, labels_c, _ = cv2.kmeans(
                samples_circ, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS
            )
            labels_c = labels_c.flatten()
            n0_c = int((labels_c==0).sum())
            n1_c = int((labels_c==1).sum())

            # Centroids BGR réels = moyenne des membres
            c0_c = tuple(int(x) for x in samples_clean[labels_c==0].mean(axis=0))
            c1_c = tuple(int(x) for x in samples_clean[labels_c==1].mean(axis=0))
            dist_c = float(np.linalg.norm(np.array(c0_c) - np.array(c1_c)))

            print(f"  [PREVIEW] KMeans circ: dist={dist_c:.1f} | "
                  f"Team0:{c0_c}→{bgr_to_name(c0_c)}({n0_c}j) | "
                  f"Team1:{c1_c}→{bgr_to_name(c1_c)}({n1_c}j)")

            if dist_c > dist or similar_names:
                c0, c1, n0, n1, dist = c0_c, c1_c, n0_c, n1_c, dist_c
                print(f"  [PREVIEW] KMeans circulaire retenu")
        except Exception as _eh:
            print(f"  [PREVIEW] KMeans circulaire échoué : {_eh}")

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
                col = dominant_jersey_color(sm, p.get("bbox", []))
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