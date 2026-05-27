# kickoff_detector.py
# -*- coding: utf-8 -*-
#
# Détection automatique du coup d'envoi — algorithme en 3 phases :
#
#   Phase 1 — Scan grossier (toutes les 30s)
#     On cherche la première tranche où le "jeu est en cours"
#     (ballon actif des deux côtés, joueurs mélangés).
#     Dès qu'on le trouve → le coup d'envoi est dans la tranche précédente.
#
#   Phase 2 — Scan fin (toutes les 0.5s) sur la tranche précédente
#     On cherche le signal précis du coup d'envoi dans les 30-60s avant
#     le moment où le jeu a été détecté comme actif.
#     Signal strict : ballon au rond central + ligne médiane + joueurs symétriques.
#
#   Phase 3 — Fallback
#     Si la phase 1 ne trouve jamais de jeu actif dans la première moitié
#     de la vidéo → offset=0 (pas de pré-match ou match commence dès le début).
#
# Gain de vitesse : x30-60 vs scan linéaire frame par frame.

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1 : intervalle du scan grossier
COARSE_INTERVAL_S    = 30.0    # une frame toutes les 30s

# Phase 2 : intervalle du scan fin
FINE_INTERVAL_S      = 0.5     # une frame toutes les 0.5s

# Phase 2 : fenêtre de recherche arrière depuis le moment où le jeu est détecté
FINE_WINDOW_S        = 90.0    # chercher dans les 90s précédant le jeu actif

# Seuil minimum kickoff pour le scan fin (strict)
KICKOFF_SCORE_MIN    = 3.5

# Nombre de frames consécutives avec score suffisant pour confirmer
KICKOFF_CONSECUTIVE  = 2

# Si kickoff détecté < 60s → match commence dès le début → offset=0
KICKOFF_MIN_OFFSET_S = 60.0

# Résolution de traitement (plus petit = plus rapide)
PROC_W = 320


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES VISION
# ─────────────────────────────────────────────────────────────────────────────

def _get_play_zone_y(frame_bgr, frame_w, frame_h):
    """Retourne la fraction y de début du terrain (ignore tribunes)."""
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_m = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    for row in range(frame_h):
        if np.sum(green_m[row] > 0) > frame_w * 0.30:
            return max(0.1, row / frame_h - 0.05)
    return 0.3


def _detect_ball_hsv(frame_bgr, frame_w, frame_h):
    """Retourne (cx_norm, cy_norm) en [0,1] ou None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0,   0, 200]), np.array([180, 40, 255])),
        cv2.inRange(hsv, np.array([20, 80, 180]), np.array([40, 255, 255])),
        cv2.inRange(hsv, np.array([5, 100, 150]), np.array([20, 255, 255])),
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (15 <= area <= 1500):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter ** 2)
        if circ < 0.35:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        score = circ * area
        if score > best_score:
            best_score = score
            best = (cx / frame_w, cy / frame_h)
    return best


def _detect_player_positions(frame_bgr, frame_w, frame_h, play_zone_y):
    """Retourne liste de (cx_norm, cy_norm) des joueurs détectés."""
    y_start = int(frame_h * play_zone_y)
    roi     = frame_bgr[y_start:, :]
    hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green   = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    white   = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    mask    = cv2.bitwise_and(cv2.bitwise_not(green), cv2.bitwise_not(white))
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (150 <= area <= 6000):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"] / frame_w
        cy = (M["m01"] / M["m00"] + y_start) / frame_h
        positions.append((cx, cy))
    return positions


def _detect_midline(frame_bgr, frame_w, frame_h, play_zone_y):
    """Retourne True si ligne médiane visible au centre horizontal."""
    gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cx_lo     = int(frame_w * 0.44)
    cx_hi     = int(frame_w * 0.56)
    y_start   = int(frame_h * play_zone_y)
    strip     = thresh[y_start:, cx_lo:cx_hi]
    white_cols = np.sum(strip > 0, axis=0)
    return bool(np.any(white_cols > strip.shape[0] * 0.12))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DÉTECTION "JEU EN COURS" (scan grossier)
# ─────────────────────────────────────────────────────────────────────────────

def _is_game_active(cap, frame_idx, video_fps, frame_w_orig, frame_h_orig,
                    play_zone_y, proc_w, proc_h):
    """
    Retourne True si le jeu est actif à ce moment de la vidéo.

    Méthode : comparer 5 frames espacées de 2s autour de frame_idx.
    On exige un mouvement soutenu (score élevé sur plusieurs comparaisons)
    pour distinguer le vrai jeu du mouvement sporadic du pré-match
    (tribunes, caméramans, joueurs qui s'échauffent statiquement).

    Critères :
    - Mouvement moyen > 15.0 sur au moins 3 des 4 comparaisons
    - OU mouvement moyen > 25.0 sur au moins 2 comparaisons
    """
    MOTION_HIGH   = 25.0   # mouvement intense (jeu actif certain)
    MOTION_MEDIUM = 15.0   # mouvement moyen (jeu probable si soutenu)

    step = int(2.0 * video_fps)  # 2 secondes entre chaque frame
    frames_gray = []

    for offset in [-2*step, -step, 0, step, 2*step]:
        fidx = max(0, frame_idx + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        small   = cv2.resize(f, (proc_w, proc_h))
        roi     = small[int(proc_h * play_zone_y):, :]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)

    if len(frames_gray) < 3:
        return False

    motion_scores = []
    for i in range(len(frames_gray) - 1):
        diff  = cv2.absdiff(frames_gray[i], frames_gray[i + 1])
        score = float(np.mean(diff))
        motion_scores.append(score)

    high_count   = sum(1 for s in motion_scores if s > MOTION_HIGH)
    medium_count = sum(1 for s in motion_scores if s > MOTION_MEDIUM)

    # Jeu actif si mouvement intense sur 2+ comparaisons
    # OU mouvement moyen soutenu sur 3+ comparaisons
    return high_count >= 2 or medium_count >= 3


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — SCORE COUP D'ENVOI STRICT (scan fin)
# ─────────────────────────────────────────────────────────────────────────────

def _score_kickoff_strict(frame_bgr, frame_w, frame_h, play_zone_y):
    """
    Score strict du coup d'envoi — critères plus exigeants que la version
    précédente pour éviter les faux positifs.

    +3.0  ballon dans zone centrale stricte (±8% x, ±12% y autour du centre terrain)
    +1.5  ligne médiane visible
    +1.0  terrain vert dominant
    +1.0  joueurs symétriques (≥2 de chaque côté)
    +0.5  ballon dans la zone de jeu
    -2.0  ballon dans un coin
    -1.0  ballon trop proche d'un but (probablement pas un kickoff)

    Retourne (score, ball_pos)
    """
    score = 0.0

    # Terrain vert
    hsv        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    if np.sum(green_mask > 0) / (frame_w * frame_h) > 0.25:
        score += 1.0

    ball = _detect_ball_hsv(frame_bgr, frame_w, frame_h)

    if ball is not None:
        bx, by = ball

        play_h        = 1.0 - play_zone_y
        center_y_norm = play_zone_y + play_h * 0.5

        in_play = by >= play_zone_y
        # Zone centrale stricte
        in_cx   = abs(bx - 0.5) <= 0.08
        in_cy   = abs(by - center_y_norm) <= 0.12 * play_h

        if in_play:
            score += 0.5
        if in_cx and in_cy:
            score += 3.0  # signal fort
        elif in_cx:
            score += 1.0  # ballon au centre x mais pas y

        # Pénalités
        if (bx < 0.07 or bx > 0.93) and by > 0.65:
            score -= 2.0  # coin
        if (bx < 0.12 or bx > 0.88) and in_play:
            score -= 1.0  # proche d'un but

    # Ligne médiane (signal important pour kickoff)
    if _detect_midline(frame_bgr, frame_w, frame_h, play_zone_y):
        score += 1.5

    # Joueurs symétriques
    positions   = _detect_player_positions(frame_bgr, frame_w, frame_h, play_zone_y)
    left_count  = sum(1 for (cx, cy) in positions if cx < 0.45)
    right_count = sum(1 for (cx, cy) in positions if cx > 0.55)
    if left_count >= 2 and right_count >= 2:
        score += 1.0

    return score, ball


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def detect_kickoff_offset(video_path, fps=25.0, verbose=True):
    """
    Détecte le coup d'envoi en 2 phases (grossier → fin).

    Retourne (kickoff_time_s, confidence).
    kickoff_time_s = 0.0 si pas de correction nécessaire.
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
    frame_w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_duration_s = total_frames / max(video_fps, 1)
    search_max_s     = video_duration_s * 0.50  # chercher dans la 1ère moitié max

    scale  = PROC_W / max(frame_w_orig, 1)
    proc_h = int(frame_h_orig * scale)

    # Détecter play_zone_y sur le premier frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first = cap.read()
    play_zone_y = 0.3
    if ret:
        small       = cv2.resize(first, (PROC_W, proc_h))
        play_zone_y = _get_play_zone_y(small, PROC_W, proc_h)

    coarse_step = int(COARSE_INTERVAL_S * video_fps)
    max_frame   = min(total_frames, int(search_max_s * video_fps))

    if verbose:
        print(f"  [KICKOFF] Scan grossier toutes les {COARSE_INTERVAL_S:.0f}s "
              f"sur {search_max_s/60:.1f} min | vidéo={video_duration_s/60:.1f} min")

    # ── PHASE 1 : scan grossier ───────────────────────────────────────────────
    game_active_time = None   # timestamp où le jeu est détecté comme actif

    frame_idx = coarse_step   # on commence à 30s, pas à 0
    while frame_idx < max_frame:
        t     = frame_idx / video_fps

        if _is_game_active(cap, frame_idx, video_fps, frame_w_orig, frame_h_orig,
                           play_zone_y, PROC_W, proc_h):
            game_active_time = t
            mm = int(t // 60)
            ss = int(t % 60)
            if verbose:
                print(f"  [KICKOFF] Jeu actif détecté à {mm:02d}:{ss:02d} "
                      f"→ scan fin des {FINE_WINDOW_S:.0f}s précédentes...")
            break

        frame_idx += coarse_step

    if game_active_time is None:
        cap.release()
        if verbose:
            print(f"  [KICKOFF] ❌ Pas de coup d'envoi détecté → t=0 inchangé")
        return 0.0, 0.0

    # ── PHASE 2 : scan fin dans la fenêtre précédant le jeu actif ────────────
    fine_start_s = max(0.0, game_active_time - FINE_WINDOW_S)
    fine_end_s   = game_active_time
    fine_step    = int(FINE_INTERVAL_S * video_fps)

    if verbose:
        ms = int(fine_start_s // 60); ss_s = int(fine_start_s % 60)
        me = int(fine_end_s   // 60); ss_e = int(fine_end_s   % 60)
        print(f"  [KICKOFF] Scan fin : {ms:02d}:{ss_s:02d} → {me:02d}:{ss_e:02d} "
              f"(pas={FINE_INTERVAL_S:.1f}s)")

    fine_start_frame = int(fine_start_s * video_fps)
    fine_end_frame   = int(fine_end_s   * video_fps)

    consecutive  = 0
    best_frame   = -1
    best_score   = 0.0
    candidates   = []

    frame_idx = fine_start_frame
    while frame_idx <= fine_end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small        = cv2.resize(frame, (PROC_W, proc_h))
        score, ball  = _score_kickoff_strict(small, PROC_W, proc_h, play_zone_y)

        if score >= KICKOFF_SCORE_MIN:
            consecutive += 1
            candidates.append((frame_idx, score))
            if score > best_score:
                best_score = score
                best_frame = frame_idx
            if consecutive >= KICKOFF_CONSECUTIVE:
                kickoff_frame = candidates[-consecutive][0]
                kickoff_time  = kickoff_frame / video_fps
                # Kickoff < 60s = match qui commence dès le début
                if kickoff_time < KICKOFF_MIN_OFFSET_S:
                    if verbose:
                        print(f"  [KICKOFF] ❌ Pas de coup d'envoi détecté "
                              f"— match commence dès le début du fichier → t=0 inchangé")
                    cap.release()
                    return 0.0, 0.0
                confidence = min(1.0, best_score / 6.0)
                mm = int(kickoff_time // 60); ss_k = int(kickoff_time % 60)
                if verbose:
                    print(f"  [KICKOFF] ✅ Coup d'envoi détecté à {mm:02d}:{ss_k:02d} "
                          f"→ t=0 (conf={confidence:.2f}, score={best_score:.1f})")
                    print(f"  [KICKOFF] Timestamps corrigés : t_video - {kickoff_time:.0f}s")
                cap.release()
                return kickoff_time, confidence
        else:
            consecutive = 0
            cutoff = frame_idx - int(10 * video_fps)
            candidates = [(f, s) for f, s in candidates if f > cutoff]

        frame_idx += fine_step

    cap.release()

    # Pas trouvé dans le scan fin — utiliser le début de la fenêtre comme approximation
    if best_frame >= 0 and best_score >= KICKOFF_SCORE_MIN - 0.5:
        kickoff_time = best_frame / video_fps
        if kickoff_time < KICKOFF_MIN_OFFSET_S:
            if verbose:
                print(f"  [KICKOFF] ❌ Pas de coup d'envoi détecté "
                      f"— match commence dès le début du fichier → t=0 inchangé")
            return 0.0, 0.0
        confidence = min(0.5, best_score / 8.0)
        mm = int(kickoff_time // 60); ss_k = int(kickoff_time % 60)
        if verbose:
            print(f"  [KICKOFF] ⚠️  Coup d'envoi probable à {mm:02d}:{ss_k:02d} "
                  f"→ t=0 (conf={confidence:.2f}, non-confirmé)")
            print(f"  [KICKOFF] Timestamps corrigés : t_video - {kickoff_time:.0f}s")
        return kickoff_time, confidence

    # Aucun signal kickoff dans la fenêtre fine — utiliser le début du jeu actif
    # comme approximation (on sait que le coup d'envoi était juste avant)
    kickoff_time = max(KICKOFF_MIN_OFFSET_S, game_active_time - COARSE_INTERVAL_S)
    if kickoff_time < KICKOFF_MIN_OFFSET_S:
        if verbose:
            print(f"  [KICKOFF] ❌ Pas de coup d'envoi détecté → t=0 inchangé")
        return 0.0, 0.0

    mm = int(kickoff_time // 60); ss_k = int(kickoff_time % 60)
    if verbose:
        print(f"  [KICKOFF] ⚠️  Coup d'envoi approximatif à {mm:02d}:{ss_k:02d} "
              f"→ t=0 (conf=0.30, signal visuel non trouvé, approximation par jeu actif)")
        print(f"  [KICKOFF] Timestamps corrigés : t_video - {kickoff_time:.0f}s")
    return kickoff_time, 0.30


# ─────────────────────────────────────────────────────────────────────────────
# APPLIQUER L'OFFSET SUR LES EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_kickoff_offset(events, kickoff_offset_s, fps=25.0):
    """Soustrait kickoff_offset_s de tous les timestamps. Supprime les events avant t=0."""
    if kickoff_offset_s <= 0:
        return events, 0

    adjusted, n_removed = [], 0
    for e in events:
        t_adj = float(e.get("time", 0) or 0) - kickoff_offset_s
        if t_adj < -5.0:
            n_removed += 1
            continue
        e = dict(e)
        e["time"] = max(0.0, t_adj)
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
    """Filtre frames_data pour ne garder que les frames après le coup d'envoi."""
    if kickoff_offset_s <= 0:
        return frames_data

    cutoff_frame = int(kickoff_offset_s * fps)
    filtered     = []
    for fd in frames_data:
        f = int(fd.get("frame", 0) or 0)
        if f < cutoff_frame:
            continue
        fd = dict(fd)
        fd["frame"] = f - cutoff_frame
        filtered.append(fd)

    print(f"  [KICKOFF] frames_data : {len(frames_data)} → {len(filtered)} "
          f"({len(frames_data)-len(filtered)} frames pré-match retirées)")
    return filtered