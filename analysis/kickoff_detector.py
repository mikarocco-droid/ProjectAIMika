# kickoff_detector.py
# -*- coding: utf-8 -*-
#
# Détection automatique du coup d'envoi dans une vidéo de football.
#
# Logique en 2 phases :
#
#   Phase 1 — Scan des 5 premières minutes :
#     Si le ballon traverse la ligne médiane ET les deux équipes sont mélangées
#     → jeu réel déjà en cours → coup d'envoi raté → offset = 0.0
#     Si ballon actif mais chaque équipe reste dans sa moitié → échauffement
#     → continuer à chercher
#
#   Phase 2 — Scan jusqu'à 20 minutes max :
#     Chercher le signal coup d'envoi : ballon au rond central + joueurs en
#     position de départ + premier mouvement depuis le centre
#     Trouvé → offset = kickoff_time
#     Pas trouvé → offset = 0.0 (pas de correction)
#
# Note : l'échauffement est distingué du vrai jeu par le fait que chaque équipe
# reste dans sa moitié pendant l'échauffement, alors qu'en match les joueurs
# sont mélangés et le ballon traverse librement la ligne médiane.

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1 : fenêtre de détection "jeu déjà en cours"
EARLY_GAME_WINDOW_S  = 300      # 5 premières minutes

# Phase 2 : fenêtre de recherche du coup d'envoi
# On cherche jusqu'à 50% de la durée totale — au-delà c'est probablement
# la mi-temps ou autre chose, pas le coup d'envoi initial
KICKOFF_SEARCH_MAX_RATIO = 0.50  # 50% de la durée vidéo

# Frame skip pendant le scan (rapide, pas besoin d'analyser chaque frame)
SEARCH_FRAME_SKIP    = 15       # ~0.5s à 30fps

# Seuil "ballon traverse la ligne médiane"
CENTER_X_BAND        = 0.12     # ±12% autour du centre horizontal

# Seuil "ballon au rond central" (pour kickoff)
KICKOFF_CENTER_X     = 0.10     # ±10% x
KICKOFF_CENTER_Y     = 0.18     # ±18% y autour du centre de jeu

# Nombre de frames consécutives avec score kickoff pour confirmer
KICKOFF_CONSECUTIVE  = 2

# Score minimum kickoff
KICKOFF_SCORE_MIN    = 3.0

# Pour Phase 1 : nombre de frames avec "jeu mélangé" pour conclure
MIXED_PLAY_FRAMES    = 3


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION BALLON (HSV léger, sans YOLO)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ball_hsv(frame_bgr, frame_w, frame_h):
    """
    Retourne (cx_norm, cy_norm) en [0,1] ou None.
    Détecte ballon blanc, jaune ou orange par HSV.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    masks = [
        cv2.inRange(hsv, np.array([0,   0, 200]), np.array([180, 40, 255])),   # blanc
        cv2.inRange(hsv, np.array([20, 80, 180]), np.array([40, 255, 255])),   # jaune
        cv2.inRange(hsv, np.array([5, 100, 150]), np.array([20, 255, 255])),   # orange
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best       = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (20 <= area <= 2000):
            continue
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
# DÉTECTION TERRAIN + LIGNE MÉDIANE
# ─────────────────────────────────────────────────────────────────────────────

def _get_play_zone_y(frame_bgr, frame_w, frame_h):
    """
    Retourne la fraction y à partir de laquelle commence le terrain (ignore tribunes).
    Cherche la première ligne avec >30% de vert.
    """
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_m = cv2.inRange(hsv,
        np.array([35, 40, 40]),
        np.array([85, 255, 255])
    )
    for row in range(frame_h):
        if np.sum(green_m[row] > 0) > frame_w * 0.30:
            return max(0.1, row / frame_h - 0.05)
    return 0.3  # fallback


def _detect_midline(frame_bgr, frame_w, frame_h, play_zone_y):
    """
    Retourne True si une ligne blanche verticale est visible au centre horizontal.
    Signal = ligne médiane du terrain.
    """
    gray        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh   = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cx_lo       = int(frame_w * 0.45)
    cx_hi       = int(frame_w * 0.55)
    y_start     = int(frame_h * play_zone_y)
    col_strip   = thresh[y_start:, cx_lo:cx_hi]
    white_cols  = np.sum(col_strip > 0, axis=0)
    return bool(np.any(white_cols > col_strip.shape[0] * 0.15))


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION JOUEURS PAR COULEUR (simplifié)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_player_positions(frame_bgr, frame_w, frame_h, play_zone_y):
    """
    Retourne une liste de positions normalisées (cx_norm, cy_norm) des joueurs
    détectés par clustering de couleurs non-vertes dans la zone de jeu.
    Rapide et sans YOLO.
    """
    y_start = int(frame_h * play_zone_y)
    roi     = frame_bgr[y_start:, :]

    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Masque non-vert (joueurs, lignes, ballon)
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    non_green = cv2.bitwise_not(green)

    # Enlever le blanc (lignes terrain)
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    mask  = cv2.bitwise_and(non_green, cv2.bitwise_not(white))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (200 <= area <= 8000):   # taille joueur raisonnable
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"] / frame_w
        cy = (M["m01"] / M["m00"] + y_start) / frame_h
        positions.append((cx, cy))

    return positions


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — JEU DÉJÀ EN COURS ?
# ─────────────────────────────────────────────────────────────────────────────

def _is_mixed_play(frame_bgr, frame_w, frame_h, play_zone_y, ball_pos):
    """
    Retourne True si le jeu est clairement en cours :
    - Ballon traverse la ligne médiane (dans la bande centrale)
    - ET joueurs des deux équipes dans les deux moitiés du terrain
      (distingués par le fait qu'on voit des joueurs des deux côtés x<0.5 ET x>0.5)

    Distingue du simple échauffement où chaque équipe reste dans sa moitié.
    """
    if ball_pos is None:
        return False

    bx, by = ball_pos

    # Ballon doit être dans la zone de jeu
    if by < play_zone_y:
        return False

    # Ballon dans la bande centrale (traverse ou est proche de la ligne médiane)
    ball_near_center = abs(bx - 0.5) <= CENTER_X_BAND

    # Détecter joueurs des deux côtés du terrain
    positions = _detect_player_positions(frame_bgr, frame_w, frame_h, play_zone_y)

    if len(positions) < 4:
        return False

    left_count  = sum(1 for (cx, cy) in positions if cx < 0.42)
    right_count = sum(1 for (cx, cy) in positions if cx > 0.58)

    # Les deux moitiés ont des joueurs ET le ballon est dans la zone de jeu
    teams_mixed = left_count >= 2 and right_count >= 2

    # Signal fort : ballon traverse le centre + joueurs mélangés
    if ball_near_center and teams_mixed:
        return True

    # Signal fort alternatif : beaucoup de joueurs dans les deux moitiés
    # même sans ballon au centre (jeu en cours, caméra suit l'action)
    if teams_mixed and left_count >= 3 and right_count >= 3:
        # Vérifier que ce n'est pas l'échauffement :
        # pendant l'échauffement les joueurs restent dans leur moitié
        # → on le distingue par la présence du ballon dans le camp adverse
        if bx < 0.35 and right_count >= 3:   # ballon côté gauche, joueurs à droite
            return True
        if bx > 0.65 and left_count >= 3:    # ballon côté droit, joueurs à gauche
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — SCORE COUP D'ENVOI
# ─────────────────────────────────────────────────────────────────────────────

def _score_kickoff_frame(frame_bgr, frame_w, frame_h, play_zone_y):
    """
    Calcule un score de probabilité de coup d'envoi.

    +2.0  ballon au rond central (±10% x, ±18% y autour du centre terrain)
    +1.5  ligne médiane visible
    +1.0  terrain vert dominant
    +1.0  joueurs des deux côtés en position symétrique
    -1.0  ballon dans un coin

    Retourne (score, ball_pos_norm or None)
    """
    score = 0.0

    # Terrain vert
    hsv         = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask  = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / (frame_w * frame_h)
    if green_ratio > 0.25:
        score += 1.0

    # Ballon
    ball = _detect_ball_hsv(frame_bgr, frame_w, frame_h)

    if ball is not None:
        bx, by = ball

        play_h         = 1.0 - play_zone_y
        center_y_norm  = play_zone_y + play_h * 0.5
        in_center_x    = abs(bx - 0.5)       <= KICKOFF_CENTER_X
        in_center_y    = abs(by - center_y_norm) <= KICKOFF_CENTER_Y * play_h
        in_play_zone   = by >= play_zone_y

        if in_play_zone:
            score += 0.5
        if in_center_x and in_center_y:
            score += 2.0
        if (bx < 0.08 or bx > 0.92) and by > 0.7:
            score -= 1.0   # coin = pas un coup d'envoi

    # Ligne médiane
    if _detect_midline(frame_bgr, frame_w, frame_h, play_zone_y):
        score += 1.5

    # Joueurs symétriques (signe de position de départ)
    positions   = _detect_player_positions(frame_bgr, frame_w, frame_h, play_zone_y)
    left_count  = sum(1 for (cx, cy) in positions if cx < 0.45)
    right_count = sum(1 for (cx, cy) in positions if cx > 0.55)
    if left_count >= 2 and right_count >= 2:
        score += 1.0

    return score, ball


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def detect_kickoff_offset(
    video_path,
    fps             = 25.0,
    verbose         = True,
):
    """
    Détecte le coup d'envoi en 2 phases.

    Retourne
    --------
    kickoff_time_s : float — timestamp du coup d'envoi (0.0 si non détecté)
    confidence     : float [0,1]
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
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_duration_s = total_frames / max(video_fps, 1)
    search_window_s  = video_duration_s * KICKOFF_SEARCH_MAX_RATIO

    # Résolution réduite pour la recherche
    proc_w = 480
    scale  = proc_w / max(frame_w, 1)
    proc_h = int(frame_h * scale)

    # Détecter play_zone_y depuis le premier frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    play_zone_y = 0.3
    if ret:
        small      = cv2.resize(first_frame, (proc_w, proc_h))
        play_zone_y = _get_play_zone_y(small, proc_w, proc_h)

    if verbose:
        print(f"  [KICKOFF] play_zone_y={play_zone_y:.2f} | "
              f"vidéo={video_duration_s:.0f}s | "
              f"recherche jusqu'à {search_window_s:.0f}s ({KICKOFF_SEARCH_MAX_RATIO*100:.0f}% de la vidéo)")

    # ── PHASE 1 : les 5 premières minutes — jeu déjà en cours ? ─────────────
    early_max_frame  = min(total_frames, int(EARLY_GAME_WINDOW_S * video_fps))
    mixed_play_count = 0

    if verbose:
        print(f"  [KICKOFF] Phase 1 : scan {EARLY_GAME_WINDOW_S//60} min "
              f"({early_max_frame} frames) — jeu déjà en cours ?")

    frame_idx = 0
    while frame_idx < early_max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small    = cv2.resize(frame, (proc_w, proc_h))
        ball_pos = _detect_ball_hsv(small, proc_w, proc_h)

        if _is_mixed_play(small, proc_w, proc_h, play_zone_y, ball_pos):
            mixed_play_count += 1
            if mixed_play_count >= MIXED_PLAY_FRAMES:
                t = frame_idx / video_fps
                if verbose:
                    mm = int(t // 60)
                    ss = int(t % 60)
                    print(f"  [KICKOFF] ⚡ Jeu déjà en cours à {mm:02d}:{ss:02d} "
                          f"(coup d'envoi raté) → offset=0s")
                cap.release()
                return 0.0, 0.0
        else:
            mixed_play_count = max(0, mixed_play_count - 1)

        frame_idx += SEARCH_FRAME_SKIP

    if verbose:
        print(f"  [KICKOFF] Phase 1 : pas de jeu actif dans les "
              f"{EARLY_GAME_WINDOW_S//60} min → recherche coup d'envoi...")

    # ── PHASE 2 : chercher le coup d'envoi jusqu'à KICKOFF_SEARCH_MAX_S ─────
    max_frame   = min(total_frames, int(search_window_s * video_fps))
    consecutive = 0
    best_frame  = -1
    best_score  = 0.0
    candidates  = []

    if verbose:
        print(f"  [KICKOFF] Phase 2 : scan jusqu'à {search_window_s/60:.1f} min")

    frame_idx = 0
    while frame_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small        = cv2.resize(frame, (proc_w, proc_h))
        score, ball  = _score_kickoff_frame(small, proc_w, proc_h, play_zone_y)

        if score >= KICKOFF_SCORE_MIN:
            consecutive += 1
            candidates.append((frame_idx, score, ball))

            if score > best_score:
                best_score = score
                best_frame = frame_idx

            if consecutive >= KICKOFF_CONSECUTIVE:
                kickoff_frame = candidates[-consecutive][0]
                kickoff_time  = kickoff_frame / video_fps

                # PATCH : si le coup d'envoi est dans les 60 premières secondes,
                # c'est soit un vrai coup d'envoi sans pré-match (offset inutile),
                # soit un faux positif (but → kickoff d'après-but).
                # Dans les deux cas, offset=0 est la bonne réponse.
                if kickoff_time < 60.0:
                    if verbose:
                        mm = int(kickoff_time // 60)
                        ss = int(kickoff_time % 60)
                        print(f"  [KICKOFF] ⚡ Kickoff à {mm:02d}:{ss:02d} < 60s "
                              f"→ match commence dès le début → offset=0s")
                    cap.release()
                    return 0.0, 0.0

                confidence    = min(1.0, best_score / 6.0)

                if verbose:
                    mm = int(kickoff_time // 60)
                    ss = int(kickoff_time % 60)
                    print(f"  [KICKOFF] ✅ Coup d'envoi détecté à {mm:02d}:{ss:02d} "
                          f"(score={best_score:.1f}, conf={confidence:.2f})")

                cap.release()
                return kickoff_time, confidence
        else:
            consecutive = 0
            cutoff      = frame_idx - int(5 * video_fps)
            candidates  = [(f, s, b) for f, s, b in candidates if f > cutoff]

        frame_idx += SEARCH_FRAME_SKIP

    cap.release()

    # Meilleur candidat isolé si score suffisant
    if best_frame >= 0 and best_score >= KICKOFF_SCORE_MIN + 0.5:
        kickoff_time = best_frame / video_fps
        confidence   = min(0.6, best_score / 8.0)
        if verbose:
            mm = int(kickoff_time // 60)
            ss = int(kickoff_time % 60)
            print(f"  [KICKOFF] ⚠️  Coup d'envoi probable à {mm:02d}:{ss:02d} "
                  f"(score={best_score:.1f}, conf={confidence:.2f}, non-confirmé)")
        return kickoff_time, confidence

    if verbose:
        print(f"  [KICKOFF] ❌ Coup d'envoi non détecté → offset=0s")

    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# APPLIQUER L'OFFSET SUR LES EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_kickoff_offset(events, kickoff_offset_s, fps=25.0):
    """
    Soustrait kickoff_offset_s de tous les timestamps.
    Supprime les events avant le coup d'envoi (temps < -5s).
    """
    if kickoff_offset_s <= 0:
        return events, 0

    adjusted  = []
    n_removed = 0

    for e in events:
        t_raw = float(e.get("time", 0) or 0)
        t_adj = t_raw - kickoff_offset_s

        if t_adj < -5.0:
            n_removed += 1
            continue

        e          = dict(e)
        e["time"]  = max(0.0, t_adj)

        if "frame" in e:
            e["frame"] = max(0, int(e["frame"]) - int(kickoff_offset_s * fps))

        adjusted.append(e)

    if n_removed > 0:
        print(f"  [KICKOFF] {n_removed} events supprimés (avant coup d'envoi)")

    return adjusted, n_removed


# ─────────────────────────────────────────────────────────────────────────────
# APPLIQUER L'OFFSET SUR LES FRAMES_DATA
# ─────────────────────────────────────────────────────────────────────────────

def apply_kickoff_offset_frames(frames_data, kickoff_offset_s, fps=25.0):
    """
    Filtre frames_data pour ne garder que les frames après le coup d'envoi.
    Soustrait l'offset des numéros de frame.
    """
    if kickoff_offset_s <= 0:
        return frames_data

    cutoff_frame = int(kickoff_offset_s * fps)
    filtered     = []

    for fd in frames_data:
        f = int(fd.get("frame", 0) or 0)
        if f < cutoff_frame:
            continue
        fd          = dict(fd)
        fd["frame"] = f - cutoff_frame
        filtered.append(fd)

    print(f"  [KICKOFF] frames_data : {len(frames_data)} → {len(filtered)} "
          f"({len(frames_data)-len(filtered)} frames pré-match retirées)")

    return filtered