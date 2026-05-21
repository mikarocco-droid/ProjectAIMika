# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-
#
# Pré-analyse légère — détecte les 2 équipes en ~30s
# sans lancer le pipeline complet.
#
# Retourne :
#   {
#     "team_0": {"color_bgr": (b,g,r), "color_name": "bordeaux",
#                "preview_frame": "/tmp/team0.jpg"},
#     "team_1": {"color_bgr": (b,g,r), "color_name": "rouge",
#                "preview_frame": "/tmp/team1.jpg"},
#   }
#
# Usage depuis l'API Flask :
#   from analysis.detect_teams_preview import detect_teams_preview
#   teams = detect_teams_preview(video_path, output_dir)

import os
import cv2
import numpy as np


# ─────────────────────────────────────────
# COULEUR BGR → NOM LISIBLE
# ─────────────────────────────────────────
def bgr_to_name(bgr):
    """Convertit BGR en nom couleur lisible via OpenCV HSV."""
    if not bgr:
        return "inconnue"
    try:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        pixel   = np.uint8([[[b, g, r]]])
        hsv     = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h = int(hsv[0]) * 2   # 0..358
        s = int(hsv[1] / 255 * 100)
        v = int(hsv[2] / 255 * 100)
    except Exception:
        return "inconnue"

    if s < 15:
        return "blanc" if v > 70 else ("gris" if v > 30 else "noir")
    if v < 20:
        return "noir"
    if 0 <= h < 20 or 340 <= h <= 360:
        return "bordeaux" if v < 50 else "rouge"
    if 20 <= h < 35:   return "orange"
    if 35 <= h < 75:   return "jaune"
    if 75 <= h < 155:  return "vert"
    if 155 <= h < 185: return "cyan"
    if 185 <= h < 265: return "bleu marine" if v < 45 else "bleu"
    if 265 <= h < 295: return "violet"
    if 295 <= h < 340: return "rose"
    return "inconnue"


# ─────────────────────────────────────────
# EXTRACTION COULEUR TORSE (avec filtre saturation)
# ─────────────────────────────────────────
def extract_jersey_color(frame, bbox):
    """
    Extrait la couleur dominante du maillot (torse uniquement).
    Filtre les pixels peu saturés (pelouse, peau, ombres).
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h = crop.shape[0]
    # Zone torse : 15%→45% (évite tête et short)
    torse = crop[int(h * 0.15):int(h * 0.45), :]
    if torse.size == 0:
        return None

    # Filtre saturation HSV → garder pixels maillot coloré uniquement
    try:
        hsv  = cv2.cvtColor(torse, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 60   # saturation > 24%
        if mask.sum() >= 10:
            return torse[mask].mean(axis=0).astype(float)
    except Exception:
        pass

    return torse.mean(axis=(0, 1)).astype(float)


# ─────────────────────────────────────────
# DÉTECTION JOUEURS SIMPLE (sans YOLO)
# Basée sur la détection de contours verticaux
# suffisamment rapide pour la pré-analyse
# ─────────────────────────────────────────
def detect_player_bboxes_simple(frame, min_area=800):
    """
    Détecte des bboxes joueurs par soustraction de fond simple.
    Rapide mais approximatif — suffisant pour la calibration couleur.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Masquer le haut de l'image (ciel, tribunes)
    h, w = thresh.shape
    thresh[:int(h * 0.35), :] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        # Filtre : taille joueur typique + ratio vertical
        if area < min_area:
            continue
        aspect = bh / max(bw, 1)
        if aspect < 1.2 or aspect > 5.0:
            continue
        bboxes.append((x, y, x + bw, y + bh))

    return bboxes


# ─────────────────────────────────────────
# SAUVEGARDE FRAME PREVIEW
# ─────────────────────────────────────────
def save_preview_frame(frame, team_id, output_dir, team_color_bgr=None):
    """
    Sauvegarde une frame représentative d'une équipe.
    Ajoute un bandeau couleur en bas pour identification visuelle.
    """
    preview = frame.copy()
    h, w    = preview.shape[:2]

    # Bandeau couleur en bas (30px)
    if team_color_bgr is not None:
        b, g, r = int(team_color_bgr[0]), int(team_color_bgr[1]), int(team_color_bgr[2])
        cv2.rectangle(preview, (0, h - 30), (w, h), (b, g, r), -1)
        # Texte couleur sur le bandeau
        color_name = bgr_to_name(team_color_bgr)
        cv2.putText(preview, color_name.upper(), (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    path = os.path.join(output_dir, f"team_{team_id}_preview.jpg")
    cv2.imwrite(path, preview)
    return path


# ─────────────────────────────────────────
# DÉTECTION PRINCIPALE
# ─────────────────────────────────────────
def detect_teams_preview(video_path, output_dir="outputs/preview",
                          n_frames=60, analysis_duration=120.0):
    """
    Analyse légère pour détecter les 2 équipes.

    Args:
        video_path        : chemin de la vidéo
        output_dir        : dossier pour les frames preview
        n_frames          : nombre de frames à analyser (défaut 60)
        analysis_duration : durée analysée en secondes (défaut 2min)

    Returns:
        dict avec team_0 et team_1 :
        {
            "team_0": {
                "color_bgr":     (71, 79, 153),
                "color_name":    "bordeaux",
                "preview_frame": "/path/team_0_preview.jpg",
            },
            "team_1": { ... },
            "success": True,
            "n_players_analyzed": 124,
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Impossible d'ouvrir la vidéo"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame    = min(int(analysis_duration * fps), total_frames)

    # Frames à analyser : réparties sur les 2 premières minutes
    frame_indices = np.linspace(
        int(fps * 5),     # Commencer à 5s (éviter le kickoff)
        max_frame,
        n_frames,
        dtype=int
    )

    # Collecter les couleurs de tous les joueurs détectés
    all_colors        = []
    best_frames       = {}   # cluster_id → meilleure frame
    n_players_total   = 0

    print(f"  [PREVIEW] Analyse {n_frames} frames sur {analysis_duration:.0f}s...")

    for fid in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize pour accélérer la détection
        h, w   = frame.shape[:2]
        scale  = min(1.0, 960 / w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Détecter les joueurs
        bboxes = detect_player_bboxes_simple(frame)
        if not bboxes:
            continue

        for bbox in bboxes[:12]:   # max 12 joueurs/frame
            color = extract_jersey_color(frame, bbox)
            if color is not None and np.any(color > 10):
                all_colors.append(color)
                n_players_total += 1

    cap.release()

    if len(all_colors) < 10:
        return {
            "success": False,
            "error":   f"Pas assez de joueurs détectés ({len(all_colors)})"
        }

    print(f"  [PREVIEW] {n_players_total} joueurs analysés → KMeans...")

    # ── KMeans → 2 clusters couleur équipes ──────────────────────────────────
    samples  = np.array(all_colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centroids = cv2.kmeans(
        samples, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    c0 = tuple(int(x) for x in centroids[0])
    c1 = tuple(int(x) for x in centroids[1])

    print(f"  [PREVIEW] Team 0: BGR{c0} → {bgr_to_name(c0)}")
    print(f"  [PREVIEW] Team 1: BGR{c1} → {bgr_to_name(c1)}")

    # ── Sélectionner les meilleures frames preview ────────────────────────────
    # Une frame avec beaucoup de joueurs du cluster dominant = bonne preview
    cap = cv2.VideoCapture(video_path)
    best_frame_0  = None
    best_frame_1  = None
    best_score_0  = 0
    best_score_1  = 0

    # Analyser un sous-ensemble de frames pour trouver les meilleures
    preview_indices = frame_indices[::3]  # 1 frame sur 3

    for fid in preview_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            continue

        h, w  = frame.shape[:2]
        scale = min(1.0, 960 / w)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        bboxes = detect_player_bboxes_simple(small)
        if not bboxes:
            continue

        score_0, score_1 = 0, 0
        for bbox in bboxes[:12]:
            color = extract_jersey_color(small, bbox)
            if color is None:
                continue
            c = np.array(color, dtype=np.float32)
            d0 = np.linalg.norm(c - centroids[0])
            d1 = np.linalg.norm(c - centroids[1])
            if d0 < d1:
                score_0 += 1
            else:
                score_1 += 1

        if score_0 > best_score_0:
            best_score_0 = score_0
            best_frame_0 = frame

        if score_1 > best_score_1:
            best_score_1 = score_1
            best_frame_1 = frame

    cap.release()

    # ── Sauvegarder les previews ──────────────────────────────────────────────
    preview_0 = None
    preview_1 = None

    if best_frame_0 is not None:
        preview_0 = save_preview_frame(best_frame_0, 0, output_dir, c0)

    if best_frame_1 is not None:
        preview_1 = save_preview_frame(best_frame_1, 1, output_dir, c1)

    return {
        "success":            True,
        "n_players_analyzed": n_players_total,
        "team_0": {
            "color_bgr":     c0,
            "color_name":    bgr_to_name(c0),
            "preview_frame": preview_0,
        },
        "team_1": {
            "color_bgr":     c1,
            "color_name":    bgr_to_name(c1),
            "preview_frame": preview_1,
        },
    }
