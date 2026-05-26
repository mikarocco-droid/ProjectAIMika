# kickoff_detector.py
# -*- coding: utf-8 -*-
#
# Détection automatique du coup d'envoi dans une vidéo de football.
#
# Principe :
#   On cherche le frame où le ballon est au centre du terrain et les joueurs
#   sont répartis dans les deux moitiés — signal caractéristique d'un kickoff.
#
# Retourne :
#   kickoff_time_s  : timestamp (secondes) du coup d'envoi dans le fichier vidéo
#                     ou 0.0 si non trouvé (pas de correction appliquée)
#
# Usage dans pipeline.py :
#   from analysis.kickoff_detector import detect_kickoff_offset
#   kickoff_offset = detect_kickoff_offset(video_path, fps, search_window=600)
#   # Ensuite soustraire kickoff_offset de tous les timestamps events

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

# Fenêtre de recherche maximale depuis le début du fichier (secondes)
DEFAULT_SEARCH_WINDOW_S = 600   # 10 minutes max

# Frame skip pendant la recherche (on n'a pas besoin d'analyser chaque frame)
SEARCH_FRAME_SKIP = 15          # ~0.5s à 30fps

# Tolérance pour "ballon au centre" : fraction de la largeur du terrain
CENTER_X_TOLERANCE = 0.12       # ±12% de la largeur = zone centrale
CENTER_Y_TOLERANCE = 0.15       # ±15% de la hauteur

# Taille minimale du ballon pour être crédible (pixels, résolution 960x540)
BALL_MIN_AREA = 20
BALL_MAX_AREA = 2000

# Score minimum pour valider un kickoff
KICKOFF_SCORE_MIN = 3.0

# Nombre de frames consécutives avec score suffisant pour confirmer
CONSECUTIVE_FRAMES_MIN = 2


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION BALLON PAR HSV (léger, sans YOLO)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ball_hsv(frame_bgr, frame_w, frame_h):
    """
    Détecte le ballon par couleur HSV.
    Retourne (cx_norm, cy_norm) normalisés [0,1] ou None.
    On cherche une zone blanche ou très claire (ballon typique).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Masque blanc/clair (ballon blanc ou jaune clair)
    masks = []

    # Blanc
    masks.append(cv2.inRange(hsv,
        np.array([0,   0, 200]),
        np.array([180, 40, 255])
    ))
    # Jaune clair
    masks.append(cv2.inRange(hsv,
        np.array([20, 80, 180]),
        np.array([40, 255, 255])
    ))
    # Orange (ballon coloré)
    masks.append(cv2.inRange(hsv,
        np.array([5, 100, 150]),
        np.array([20, 255, 255])
    ))

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Chercher contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA):
            continue

        # Circularité
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        score = circularity * area
        if score > best_score:
            best_score = score
            best = (cx / frame_w, cy / frame_h)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# SCORE KICKOFF POUR UN FRAME
# ─────────────────────────────────────────────────────────────────────────────

def _score_kickoff_frame(frame_bgr, frame_w, frame_h, play_zone_y_min=0.3):
    """
    Calcule un score de probabilité de coup d'envoi pour ce frame.

    Score basé sur :
    +2.0  ballon détecté dans la zone centrale (±12% x, ±15% y)
    +1.0  ballon dans la moitié basse du terrain (zone de jeu)
    +1.5  présence de la ligne médiane (ligne blanche verticale au centre)
    +1.0  terrain vert dominant (pas un arrêt de jeu/hors terrain)
    -1.0  ballon dans un coin (probablement corner ou touche)

    Retourne (score, ball_pos_norm or None)
    """
    score = 0.0

    # ── Terrain vert dominant ───────────────────────────────────────────
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv,
        np.array([35,  40,  40]),
        np.array([85, 255, 255])
    )
    green_ratio = np.sum(green_mask > 0) / (frame_w * frame_h)
    if green_ratio > 0.25:
        score += 1.0

    # ── Détection ballon ────────────────────────────────────────────────
    ball = _detect_ball_hsv(frame_bgr, frame_w, frame_h)

    if ball is not None:
        bx, by = ball

        # Dans la zone de jeu (pas le haut du frame = tribunes)
        if by >= play_zone_y_min:
            score += 1.0

        # Zone centrale x
        in_center_x = abs(bx - 0.5) <= CENTER_X_TOLERANCE
        # Zone centrale y (milieu du terrain de jeu)
        play_h = 1.0 - play_zone_y_min
        center_y_norm = play_zone_y_min + play_h * 0.5
        in_center_y = abs(by - center_y_norm) <= CENTER_Y_TOLERANCE * play_h

        if in_center_x and in_center_y:
            score += 2.0

        # Pénalité coin
        if (bx < 0.08 or bx > 0.92) and (by > 0.7):
            score -= 1.0

    # ── Ligne médiane ───────────────────────────────────────────────────
    # On cherche une ligne blanche verticale proche du centre horizontal
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Colonne centrale ±5% de la largeur
    cx_lo = int(frame_w * 0.45)
    cx_hi = int(frame_w * 0.55)
    col_strip = thresh[:, cx_lo:cx_hi]

    # Ligne de jeu seulement (ignorer tribunes)
    y_start = int(frame_h * play_zone_y_min)
    col_strip = col_strip[y_start:, :]

    white_cols = np.sum(col_strip > 0, axis=0)
    # Si au moins une colonne a >15% de pixels blancs = ligne médiane probable
    if np.any(white_cols > col_strip.shape[0] * 0.15):
        score += 1.5

    return score, ball


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def detect_kickoff_offset(
    video_path,
    fps             = 25.0,
    search_window_s = DEFAULT_SEARCH_WINDOW_S,
    play_zone_y_min = 0.3,
    verbose         = True,
):
    """
    Analyse les premières `search_window_s` secondes de la vidéo pour trouver
    le coup d'envoi.

    Paramètres
    ----------
    video_path      : chemin vers la vidéo
    fps             : FPS de la vidéo (pour calculer les timestamps)
    search_window_s : fenêtre de recherche en secondes (défaut 600 = 10 min)
    play_zone_y_min : fraction de hauteur au-dessus de laquelle on ignore
                      (tribunes) — détecté automatiquement si possible
    verbose         : afficher les logs de détection

    Retourne
    --------
    kickoff_time_s  : float, timestamp du coup d'envoi en secondes
                      0.0 si non détecté (pas de correction)
    confidence      : float [0,1], confiance de la détection
    """
    if not os.path.exists(video_path):
        logger.warning(f"[KICKOFF] Vidéo introuvable : {video_path}")
        return 0.0, 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"[KICKOFF] Impossible d'ouvrir : {video_path}")
        return 0.0, 0.0

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame    = min(total_frames, int(search_window_s * video_fps))

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Réduire la résolution pour la recherche (rapide)
    target_w = 480
    scale    = target_w / max(frame_w, 1)
    proc_w   = target_w
    proc_h   = int(frame_h * scale)

    if verbose:
        print(f"  [KICKOFF] Recherche dans les {search_window_s:.0f}s "
              f"({max_frame} frames, skip={SEARCH_FRAME_SKIP})")

    # ── Détecter play_zone_y_min automatiquement depuis le premier frame ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        first_small = cv2.resize(first_frame, (proc_w, proc_h))
        hsv_f = cv2.cvtColor(first_small, cv2.COLOR_BGR2HSV)
        green_m = cv2.inRange(hsv_f,
            np.array([35, 40, 40]),
            np.array([85, 255, 255])
        )
        # Trouver la première ligne avec >30% de vert = début du terrain
        for row in range(proc_h):
            if np.sum(green_m[row] > 0) > proc_w * 0.30:
                play_zone_y_min = max(0.1, row / proc_h - 0.05)
                break

    # ── Scan des frames ──────────────────────────────────────────────────
    candidates   = []   # (frame_idx, score, ball_pos)
    consecutive  = 0
    best_frame   = -1
    best_score   = 0.0

    frame_idx = 0
    while frame_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (proc_w, proc_h))
        score, ball = _score_kickoff_frame(small, proc_w, proc_h, play_zone_y_min)

        if score >= KICKOFF_SCORE_MIN:
            consecutive += 1
            candidates.append((frame_idx, score, ball))

            if score > best_score:
                best_score = score
                best_frame = frame_idx

            if consecutive >= CONSECUTIVE_FRAMES_MIN:
                # On a trouvé le kickoff — prendre le premier frame de la série
                kickoff_frame = candidates[-consecutive][0]
                kickoff_time  = kickoff_frame / video_fps
                confidence    = min(1.0, best_score / 6.0)

                if verbose:
                    mm = int(kickoff_time // 60)
                    ss = int(kickoff_time % 60)
                    print(f"  [KICKOFF] ✅ Coup d'envoi détecté à "
                          f"{mm:02d}:{ss:02d} "
                          f"(frame={kickoff_frame}, score={best_score:.1f}, "
                          f"conf={confidence:.2f})")

                cap.release()
                return kickoff_time, confidence
        else:
            consecutive = 0
            # Garder les candidats récents seulement (fenêtre 5s)
            cutoff = frame_idx - int(5 * video_fps)
            candidates = [(f, s, b) for f, s, b in candidates if f > cutoff]

        frame_idx += SEARCH_FRAME_SKIP

    cap.release()

    # Pas de séquence consécutive — prendre le meilleur candidat isolé
    if best_frame >= 0 and best_score >= KICKOFF_SCORE_MIN + 0.5:
        kickoff_time = best_frame / video_fps
        confidence   = min(0.6, best_score / 8.0)   # confiance réduite

        if verbose:
            mm = int(kickoff_time // 60)
            ss = int(kickoff_time % 60)
            print(f"  [KICKOFF] ⚠️  Coup d'envoi probable à "
                  f"{mm:02d}:{ss:02d} "
                  f"(frame={best_frame}, score={best_score:.1f}, "
                  f"conf={confidence:.2f}, non-confirmé)")

        return kickoff_time, confidence

    if verbose:
        print(f"  [KICKOFF] ❌ Coup d'envoi non détecté "
              f"— timestamps relatifs au début du fichier")

    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# APPLIQUER L'OFFSET SUR LES EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_kickoff_offset(events, kickoff_offset_s, fps=25.0):
    """
    Soustrait kickoff_offset_s de tous les timestamps des events.
    Les events avant le coup d'envoi (temps négatif) sont supprimés.

    Paramètres
    ----------
    events           : liste de dicts events du pipeline
    kickoff_offset_s : float, timestamp du coup d'envoi en secondes
    fps              : FPS de la vidéo

    Retourne
    --------
    events_adjusted : liste d'events avec timestamps corrigés
    n_removed       : nombre d'events supprimés (avant le coup d'envoi)
    """
    if kickoff_offset_s <= 0:
        return events, 0

    adjusted  = []
    n_removed = 0

    for e in events:
        t_raw = float(e.get("time", 0) or 0)
        t_adj = t_raw - kickoff_offset_s

        if t_adj < -5.0:
            # Event clairement avant le coup d'envoi → supprimer
            n_removed += 1
            continue

        e = dict(e)  # copie pour ne pas modifier l'original
        e["time"] = max(0.0, t_adj)

        # Corriger aussi le champ frame si présent
        if "frame" in e:
            frame_raw = int(e["frame"] or 0)
            frame_adj = frame_raw - int(kickoff_offset_s * fps)
            e["frame"] = max(0, frame_adj)

        adjusted.append(e)

    if n_removed > 0:
        print(f"  [KICKOFF] {n_removed} events supprimés (avant le coup d'envoi)")

    return adjusted, n_removed


# ─────────────────────────────────────────────────────────────────────────────
# APPLIQUER L'OFFSET SUR LES FRAMES_DATA
# ─────────────────────────────────────────────────────────────────────────────

def apply_kickoff_offset_frames(frames_data, kickoff_offset_s, fps=25.0):
    """
    Soustrait kickoff_offset_s des frames dans frames_data.
    Utile pour que les heatmaps et stats ne comptent que le match réel.

    Retourne frames_data filtré (frames avant coup d'envoi retirées).
    """
    if kickoff_offset_s <= 0:
        return frames_data

    cutoff_frame = int(kickoff_offset_s * fps)
    filtered = []

    for fd in frames_data:
        f = int(fd.get("frame", 0) or 0)
        if f < cutoff_frame:
            continue
        fd = dict(fd)
        fd["frame"] = f - cutoff_frame
        filtered.append(fd)

    print(f"  [KICKOFF] frames_data : {len(frames_data)} → {len(filtered)} "
          f"(supprimé {len(frames_data)-len(filtered)} frames pré-match)")

    return filtered
