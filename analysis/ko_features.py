"""
KO_FEATURES — Module pur, pensé pour s'intégrer directement dans le pipeline Scoutia
==============================================================================
AUCUNE dépendance Kaggle (pas de clone, pas de pip install, pas de
process_video() appelé ici). Ce module part du principe que frames_data,
fps et video_path sont DÉJÀ disponibles — exactement ce que votre pipeline
produit déjà via main.process_video().

Usage prévu dans pipeline.py / kickoff_detector.py :

    from analysis.ko_features import extract_features
    row = extract_features(frames_data, fps, frame_w, video_path, t_candidat)

Ne fait AUCUNE décision — retourne une table de signaux, à combiner ensuite
(régression logistique / random forest entraînés séparément, voir
entrainement_modele_ko.py) plutôt que par des poids choisis à la main.

Dépendances externes : uniquement numpy, scipy (déjà présents dans
l'environnement Scoutia via requirements existants) et ffmpeg (déjà utilisé
par le pipeline pour l'extraction vidéo).
"""

import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram


# ═══════════════════════════════════════════════════════
# SIGNAUX GÉOMÉTRIE / TRANSITION / BALLON — logique déjà validée dans ce
# projet (voir kickoff_score_hybride.py et observatoire_signaux.py pour
# l'historique complet des tests). Reproduits ici en autonome pour éviter
# toute dépendance à des fichiers expérimentaux lors de l'intégration.
# ═══════════════════════════════════════════════════════

def _players_at(frames_data, fps, t, tolerance_s=1.0):
    best_fd, best_dist = None, tolerance_s
    for fd in frames_data:
        fd_t = fd.get("frame", 0) / fps
        dist = abs(fd_t - t)
        if dist < best_dist:
            best_dist, best_fd = dist, fd
    if best_fd is None:
        return []
    return [
        {"id": p.get("id") or p.get("player_id") or p.get("tracker_id"),
         "team": p.get("team"), "center": p.get("center")}
        for p in (best_fd.get("players") or []) if p.get("center")
    ]


def _ball_at(frames_data, fps, t, tolerance_s=1.0, max_dist_to_player_px=300.0):
    best_fd, best_dist = None, tolerance_s
    for fd in frames_data:
        fd_t = fd.get("frame", 0) / fps
        dist = abs(fd_t - t)
        if dist < best_dist:
            ball = fd.get("ball") or {}
            if ball.get("interpolated"):
                continue
            best_dist, best_fd = dist, fd
    if best_fd is None:
        return None
    ball = best_fd.get("ball") or {}
    center = ball.get("center")
    if center is None:
        return None
    players = [p.get("center") for p in (best_fd.get("players") or []) if p.get("center")]
    if players:
        min_dist = min(np.hypot(center[0] - p[0], center[1] - p[1]) for p in players)
        if min_dist > max_dist_to_player_px:
            return None
    return center


def _camera_motion_series(frames_data, fps, times):
    players_by_t = {tt: {p["id"]: p["center"] for p in _players_at(frames_data, fps, tt, tolerance_s=0.5)
                          if p["id"] is not None} for tt in times}
    cam_motion = {}
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        p0, p1 = players_by_t[t0], players_by_t[t1]
        common = set(p0) & set(p1)
        if len(common) >= 3:
            dxs = [p1[pid][0] - p0[pid][0] for pid in common]
            dys = [p1[pid][1] - p0[pid][1] for pid in common]
            cam_motion[t1] = (float(np.median(dxs)), float(np.median(dys)))
        else:
            cam_motion[t1] = (0.0, 0.0)
    cumulative = {times[0]: (0.0, 0.0)}
    rx, ry = 0.0, 0.0
    for i in range(1, len(times)):
        dx, dy = cam_motion[times[i]]
        rx, ry = rx + dx, ry + dy
        cumulative[times[i]] = (rx, ry)
    return players_by_t, cumulative


def _camera_offset_between(cumulative, t0, t1):
    cx0, cy0 = cumulative.get(t0, (0.0, 0.0))
    cx1, cy1 = cumulative.get(t1, (0.0, 0.0))
    return cx1 - cx0, cy1 - cy0


def score_geometrie(frames_data, fps, t, frame_w):
    players = _players_at(frames_data, fps, t)
    if len(players) < 6:
        return 0.0, 0
    mid_x = frame_w / 2
    left = sum(1 for p in players if p["center"][0] < mid_x)
    right = len(players) - left
    return 1.0 - abs(left - right) / len(players), len(players)


def _window_features(frames_data, fps, t_center, frame_w, window_s=6.0, sub_step_s=1.0):
    """Vitesse médiane compensée caméra + géométrie, sur une fenêtre centrée."""
    half = window_s / 2
    sub_times = list(np.arange(t_center - half, t_center + half + sub_step_s, sub_step_s))
    players_by_t, cumulative = _camera_motion_series(frames_data, fps, sub_times)
    all_speeds = []
    for i in range(1, len(sub_times)):
        t0, t1 = sub_times[i - 1], sub_times[i]
        p0, p1 = players_by_t[t0], players_by_t[t1]
        common = set(p0) & set(p1)
        if len(common) < 3:
            continue
        cam_dx, cam_dy = _camera_offset_between(cumulative, t0, t1)
        dt = t1 - t0
        for pid in common:
            dx, dy = (p1[pid][0] - p0[pid][0]) - cam_dx, (p1[pid][1] - p0[pid][1]) - cam_dy
            all_speeds.append(np.hypot(dx, dy) / dt if dt > 0 else 0.0)
    mean_speed = float(np.median(all_speeds)) if all_speeds else 0.0
    geo_sep, n_players = score_geometrie(frames_data, fps, t_center, frame_w)
    return mean_speed, geo_sep, n_players


def score_ballon(frames_data, fps, t, window_s=5.0, stable_threshold_px_s=10.0):
    times = list(np.arange(t - window_s, t + window_s + 0.5, 0.5))
    _, cumulative = _camera_motion_series(frames_data, fps, times)
    positions = {}
    for tt in times:
        pos = _ball_at(frames_data, fps, tt, tolerance_s=0.5)
        if pos:
            positions[tt - t] = (tt, pos)
    if len(positions) < 4:
        return 0.0, None, None
    t_sorted = sorted(positions.keys())
    before = [tt for tt in t_sorted if tt <= 0]
    after = [tt for tt in t_sorted if tt > 0]
    if len(before) < 2 or len(after) < 2:
        return 0.0, None, None

    def _speeds(ts):
        sp = []
        for i in range(1, len(ts)):
            t0_, p0 = positions[ts[i - 1]]
            t1_, p1 = positions[ts[i]]
            dt = t1_ - t0_
            if dt > 0:
                cdx, cdy = _camera_offset_between(cumulative, t0_, t1_)
                sp.append(np.hypot((p1[0] - p0[0]) - cdx, (p1[1] - p0[1]) - cdy) / dt)
        return sp

    sb = float(np.mean(_speeds(before))) if len(before) > 1 else None
    sa = float(np.mean(_speeds(after))) if len(after) > 1 else None
    if sb is not None and sa is not None:
        score = 1.0 if (sb < stable_threshold_px_s and sa >= stable_threshold_px_s) else (0.5 if sa >= stable_threshold_px_s else 0.0)
    else:
        score = 0.0
    return score, round(sb, 1) if sb is not None else None, round(sa, 1) if sa is not None else None


def _stability_before(frames_data, fps, frame_w, t, max_lookback_s=60.0, step_s=2.0):
    ts = np.arange(max(0, t - max_lookback_s), t + step_s, step_s)
    speeds, geos = [], []
    for tt in ts:
        s, g, n = _window_features(frames_data, fps, float(tt), frame_w, window_s=6.0)
        speeds.append(s); geos.append(g)
    speeds, geos = np.array(speeds), np.array(geos)
    valid = ~((geos == 0.0) & (speeds == 0.0))
    if not valid.any():
        return 0.0
    med_speed, med_geo = np.median(speeds[valid]), np.median(geos[valid])
    is_a1 = valid & (geos >= med_geo) & (speeds <= med_speed)
    duration = 0.0
    for i in range(len(is_a1) - 1, -1, -1):
        if is_a1[i]:
            duration += step_s
        else:
            break
    return round(duration, 1)


def _activity_after(frames_data, fps, frame_w, t, check_s=10.0):
    ts = np.arange(t, t + check_s, 2.0)
    speeds = [_window_features(frames_data, fps, float(tt), frame_w, window_s=4.0)[0] for tt in ts]
    return round(float(np.mean(speeds)), 1) if speeds else 0.0


def _transition_synchro(frames_data, fps, t, window_s=5.0, immobile_threshold_px_s=8.0):
    """pct_immobile, transition_rate, synchronisation_std_s, mean_speed_change."""
    times = list(np.arange(t - window_s, t + window_s + 0.5, 0.5))
    players_by_t, cumulative = _camera_motion_series(frames_data, fps, times)
    track_history = {}
    for tt in times:
        for pid, center in players_by_t[tt].items():
            track_history.setdefault(pid, {})[tt - t] = (tt, center)
    transition_times, n_immobile, n_tracks, speed_changes = [], 0, 0, []
    for pid, history in track_history.items():
        t_sorted = sorted(history.keys())
        if len(t_sorted) < 0.5 * len(times):
            continue
        n_tracks += 1
        speeds = {}
        for i in range(1, len(t_sorted)):
            rel0, rel1 = t_sorted[i - 1], t_sorted[i]
            t0_, p0 = history[rel0]; t1_, p1 = history[rel1]
            dt = t1_ - t0_
            cdx, cdy = _camera_offset_between(cumulative, t0_, t1_)
            speeds[rel1] = np.hypot((p1[0] - p0[0]) - cdx, (p1[1] - p0[1]) - cdy) / dt if dt > 0 else 0.0
        before_speeds = [s for tt, s in speeds.items() if tt <= -1.0]
        after_speeds = [s for tt, s in speeds.items() if tt > 1.0]
        if before_speeds and after_speeds:
            speed_changes.append(np.median(after_speeds) - np.median(before_speeds))
        if not before_speeds or np.mean(before_speeds) >= immobile_threshold_px_s:
            continue
        n_immobile += 1
        for tt in sorted(t for t in speeds if t > -1.0):
            if speeds[tt] >= immobile_threshold_px_s and tt > 0:
                transition_times.append(tt)
                break
    pct_immobile = n_immobile / n_tracks if n_tracks > 0 else None
    sync_std = float(np.std(transition_times)) if len(transition_times) >= 2 else None
    mean_change = float(np.median(speed_changes)) if speed_changes else None
    return pct_immobile, sync_std, mean_change


# ═══════════════════════════════════════════════════════
# SIFFLET — nouveau signal, seul à nécessiter l'audio (ffmpeg)
# ═══════════════════════════════════════════════════════

def whistle_score(video_path, t, window_s=2.0, baseline_s=8.0,
                   band_low_hz=2000, band_high_hz=4200, tmp_wav="/tmp/_whistle_check.wav"):
    """Score continu (z-score local, pas de seuil absolu) de l'énergie dans
    la bande sifflet (2-4.2kHz) autour de t, comparée au niveau de fond de
    CE clip. None si l'extraction audio échoue (pas de piste audio, etc.)."""
    import os
    start_s = max(0, t - baseline_s / 2)
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-i", video_path, "-t", str(baseline_s),
           "-vn", "-ac", "1", "-ar", "16000", tmp_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, None
    try:
        sr, data = wavfile.read(tmp_wav)
    except Exception:
        return None, None
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(float)
    f, tt, Sxx = spectrogram(data, fs=sr, nperseg=1024, noverlap=768)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    band_mask = (f >= band_low_hz) & (f <= band_high_hz)
    band_energy = Sxx_db[band_mask, :].mean(axis=0)
    t_rel = t - start_s
    interest_mask = np.abs(tt - t_rel) <= window_s
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    if not interest_mask.any():
        return 0.0, None
    baseline_energy, baseline_std = np.median(band_energy), (np.std(band_energy) or 1.0)
    energy_in_window = band_energy[interest_mask]
    tt_in_window = tt[interest_mask]
    peak_idx = np.argmax(energy_in_window)
    score = float((energy_in_window[peak_idx] - baseline_energy) / baseline_std)
    peak_t_abs = start_s + tt_in_window[peak_idx]
    return round(score, 2), round(float(peak_t_abs), 2)


# ═══════════════════════════════════════════════════════
# API PRINCIPALE — celle à appeler depuis le pipeline
# ═══════════════════════════════════════════════════════

def extract_features(frames_data, fps, frame_w, video_path, t, label=None, include_audio=True):
    """UNE ligne de features pour le candidat à l'instant t. N'importe quel
    appelant qui a déjà frames_data (ex: pipeline.py après process_video())
    peut appeler cette fonction directement, sans aucune dépendance Kaggle."""
    geo_sep, n_players = score_geometrie(frames_data, fps, t, frame_w)
    pct_immobile, sync_std, mean_speed_change = _transition_synchro(frames_data, fps, t)
    ball_sc, ball_before, ball_after = score_ballon(frames_data, fps, t)
    stab_before = _stability_before(frames_data, fps, frame_w, t)
    activite_apres = _activity_after(frames_data, fps, frame_w, t)

    row = {
        "t": t, "label": label,
        "geo_separation": round(geo_sep, 3), "n_players": n_players,
        "mean_speed_change_px_s": round(mean_speed_change, 1) if mean_speed_change is not None else None,
        "pct_immobile": round(pct_immobile, 3) if pct_immobile is not None else None,
        "synchronisation_std_s": round(sync_std, 2) if sync_std is not None else None,
        "ball_score": round(ball_sc, 3), "ball_speed_before": ball_before, "ball_speed_after": ball_after,
        "stabilite_avant_s": stab_before, "activite_apres": activite_apres,
        "whistle_score": None, "whistle_peak_t": None,
    }
    if include_audio:
        w_score, w_peak = whistle_score(video_path, t)
        row["whistle_score"], row["whistle_peak_t"] = w_score, w_peak
    return row
