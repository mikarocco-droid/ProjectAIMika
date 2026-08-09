"""
recalibrate_colors.py
========================
Recherche, dans TOUTE la vidéo (pas seulement le début), la meilleure
fenêtre pour calibrer les couleurs des 2 équipes — puis reclassifie
rétroactivement tous les joueurs de frames_data avec cette calibration.

Ne modifie PAS rendering/overlay.py. Script séparé, tourne APRÈS
process_video(), en relisant le fichier vidéo aux frames déjà connues
(via frames_data) pour en extraire les couleurs de maillot.

Critère de sélection d'une fenêtre, comme discuté :
  1. assez de joueurs présents
  2. 2 groupes de couleur bien séparés
  3. couleurs STABLES dans le temps (pas juste séparées une fois)
Aucun seuil de décision figé ici — le script calcule les scores pour
TOUTES les fenêtres candidates, à inspecter avant de choisir un seuil.
"""

import cv2
import numpy as np


def _extract_jersey_color(frame, bbox):
    """Identique à TeamColorDetector._extract_jersey_color (overlay.py),
    dupliqué ici pour ne pas dépendre du pipeline de production."""
    try:
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch = crop.shape[0]
        torse = crop[int(ch * 0.15):int(ch * 0.45), :]
        if torse.size == 0:
            return None
        hsv = cv2.cvtColor(torse, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 60
        if mask.sum() >= 10:
            color = torse[mask].mean(axis=0).astype(np.float32)
        else:
            color = torse.mean(axis=(0, 1)).astype(np.float32)
        return color
    except Exception:
        return None


def _kmeans_2(colors):
    """KMeans k=2 sur une liste de couleurs BGR. Retourne (centroids, distance) ou (None, 0)."""
    if len(colors) < 10:
        return None, 0.0
    samples = np.array(colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centroids = cv2.kmeans(samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    dist = float(np.linalg.norm(centroids[0] - centroids[1]))
    return centroids, dist


def _valid_bbox(bbox, frame_w, frame_h):
    """Rejette un bbox entièrement hors du cadre réel de l'image — trace
    fantôme du tracker après occlusion prolongée (même bug diagnostiqué
    sur Andrimont t=200s dans detect_c_capitaines.py : bbox jusqu'à
    x=2774 sur une image de 1920px)."""
    if bbox is None or frame_w is None or frame_h is None:
        return True  # pas assez d'info pour juger, on laisse passer
    x1, y1, x2, y2 = bbox
    if x2 < 0 or x1 > frame_w or y2 < 0 or y1 > frame_h:
        return False
    return True


def scan_calibration_windows(video_path, frames_data, fps, min_players=10,
                               window_s=15.0, step_s=5.0):
    """
    Scanne toute la vidéo par fenêtres glissantes de window_s secondes
    (pas de step_s), et pour chaque fenêtre calcule :
      - n_samples : nb de couleurs de torse collectées
      - separation : distance entre les 2 centroïdes sur TOUTE la fenêtre
      - stability : à quel point les centroïdes de la 1ere moitié et de
        la 2e moitié de la fenêtre se ressemblent (distance faible = stable)

    Optimisation : chaque frame vidéo n'est lue et traitée qu'UNE SEULE
    FOIS (mise en cache), même si elle appartient à plusieurs fenêtres
    qui se chevauchent — nécessaire pour un scan rapide sur des matchs
    longs (fenêtres de 15s tous les 5s = 66% de chevauchement).

    Ne prend AUCUNE décision — retourne la liste complète pour inspection.
    """
    frames_by_num = {}
    for fd in frames_data:
        frame_w, frame_h = fd.get("frame_w"), fd.get("frame_h")
        players = [p for p in (fd.get("players") or [])
                   if _valid_bbox(p.get("bbox"), frame_w, frame_h)]
        if len(players) < min_players:
            continue
        frames_by_num[fd["frame"]] = players

    if not frames_by_num:
        return []

    frame_nums = sorted(frames_by_num.keys())
    t_max = frame_nums[-1] / fps

    # --- Passe unique de lecture vidéo : une couleur par joueur, par frame ---
    cap = cv2.VideoCapture(video_path)
    colors_by_frame = {}  # frame_num -> [couleurs de tous les joueurs de cette frame]
    for fn in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        colors = []
        for p in frames_by_num[fn]:
            bbox = p.get("bbox")
            if not bbox:
                continue
            c = _extract_jersey_color(frame, bbox)
            if c is not None:
                colors.append(c)
        colors_by_frame[fn] = colors
    cap.release()

    # --- Construction des fenêtres à partir du cache (aucune relecture vidéo) ---
    results = []
    t = 0.0
    while t + window_s <= t_max:
        window_frame_nums = [f for f in frame_nums if t <= f / fps < t + window_s]
        if len(window_frame_nums) < 3:
            t += step_s
            continue

        mid = len(window_frame_nums) // 2
        first_half_nums = window_frame_nums[:mid]
        second_half_nums = window_frame_nums[mid:]

        def gather(nums):
            out = []
            for fn in nums:
                out.extend(colors_by_frame.get(fn, []))
            return out

        colors_all = gather(window_frame_nums)
        colors_first = gather(first_half_nums)
        colors_second = gather(second_half_nums)

        centroids_all, sep_all = _kmeans_2(colors_all)
        centroids_first, _ = _kmeans_2(colors_first)
        centroids_second, _ = _kmeans_2(colors_second)

        stability = None
        if centroids_first is not None and centroids_second is not None:
            d_direct = (np.linalg.norm(centroids_first[0]-centroids_second[0]) +
                        np.linalg.norm(centroids_first[1]-centroids_second[1]))
            d_croise = (np.linalg.norm(centroids_first[0]-centroids_second[1]) +
                        np.linalg.norm(centroids_first[1]-centroids_second[0]))
            stability = float(min(d_direct, d_croise) / 2)

        results.append({
            "t_debut": t, "t_fin": t + window_s,
            "n_samples": len(colors_all),
            "separation": sep_all,
            "stability_drift": stability,
            "centroids": centroids_all.tolist() if centroids_all is not None else None,
        })
        t += step_s

    return results


def apply_calibration(video_path, frames_data, centroids):
    """Reclassifie TOUS les joueurs de frames_data avec les centroïdes
    donnés — relit la vidéo (frames_data ne contient pas les pixels),
    extrait la couleur de chaque joueur, l'assigne à l'équipe la plus
    proche. Modifie frames_data en place (ajoute/écrase 'team')."""
    centroids = np.array(centroids, dtype=np.float32)
    cap = cv2.VideoCapture(video_path)

    for fd in frames_data:
        players = fd.get("players") or []
        if not players:
            continue
        frame_w, frame_h = fd.get("frame_w"), fd.get("frame_h")
        cap.set(cv2.CAP_PROP_POS_FRAMES, fd["frame"])
        ret, frame = cap.read()
        if not ret:
            continue
        for p in players:
            bbox = p.get("bbox")
            if not bbox or not _valid_bbox(bbox, frame_w, frame_h):
                p["team"] = None
                continue
            color = _extract_jersey_color(frame, bbox)
            if color is None:
                p["team"] = None
                continue
            d0 = np.linalg.norm(color - centroids[0])
            d1 = np.linalg.norm(color - centroids[1])
            p["team"] = 0 if d0 < d1 else 1

    cap.release()
    return frames_data


if __name__ == "__main__":
    print(__doc__)
