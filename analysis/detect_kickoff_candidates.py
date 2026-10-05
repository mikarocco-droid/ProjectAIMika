"""
detect_kickoff_candidates.py
============================
Détecteur KO géométrique depuis frames_data.
Cherche les transitions : jeu normal → balance équipes monte → stabilité.

Ne distingue pas KO après but / KO début match / reprise.
→ Gemini fait la classification finale.

Usage :
    from analysis.detect_kickoff_candidates import detect_kickoff_candidates
    candidates = detect_kickoff_candidates(frames_data, fps, frame_w, frame_h)
"""

import numpy as np


def _build_series(frames_data, fps, frame_w, frame_h, start_s=0.0):
    """
    Construit la série temporelle des métriques joueurs.
    Returns : list of dict {t_abs, n, n_center, disp, balance}
    """
    series = []
    for fd in frames_data:
        fid     = fd.get("frame", 0)
        t_abs   = start_s + fid / max(fps, 1)
        players = fd.get("players", [])
        if not players:
            continue

        centers = [p.get("center") for p in players if p.get("center")]
        if len(centers) < 5:
            continue

        xs = [c[0] / max(frame_w, 1) for c in centers]

        # Joueurs en zone centrale (x: 35-65%)
        n_center = sum(1 for x in xs if 0.35 <= x <= 0.65)

        # Dispersion (écart-type des positions x)
        dispersion = float(np.std(xs))

        # Balance équipes (min/max ratio)
        teams = [p.get("team") for p in players if p.get("team") is not None]
        t0 = sum(1 for t in teams if t == 0)
        t1 = sum(1 for t in teams if t == 1)
        balance = min(t0, t1) / max(max(t0, t1), 1) if teams else 0.0

        series.append({
            "t_abs":    t_abs,
            "n":        len(players),
            "n_center": n_center,
            "disp":     dispersion,
            "balance":  balance,
        })

    return series


def _mean_in_window(series, t_center, t_before, t_after, key):
    vals = [s[key] for s in series
            if t_center + t_before <= s["t_abs"] <= t_center + t_after]
    return float(np.mean(vals)) if vals else 0.0


def detect_kickoff_candidates(
    frames_data,
    fps,
    frame_w,
    frame_h,
    start_s             = 0.0,
    balance_delta_min   = 0.20,   # hausse minimale de balance pour déclencher
    balance_after_min   = 0.30,   # niveau minimal de balance après
    window_before_s     = 15.0,   # fenêtre avant pour baseline
    window_after_s      = 10.0,   # fenêtre après pour mesure
    min_gap_s           = 60.0,   # distance minimale entre deux candidats
    scan_step_s         = 2.0,    # pas de scan en secondes
):
    """
    Détecte les candidats KO depuis frames_data.

    Returns : list of dict {
        type, time, frame, score,
        balance_before, balance_after, balance_delta,
        n_players_before, n_players_after,
        n_center_before, n_center_after,
        dispersion_before, dispersion_after,
    }
    """
    series = _build_series(frames_data, fps, frame_w, frame_h, start_s)
    if not series:
        return []

    t_min = series[0]["t_abs"]
    t_max = series[-1]["t_abs"]

    # Index pour accès rapide
    series_by_t = {round(s["t_abs"], 1): s for s in series}

    candidates_raw = []

    t = t_min + window_before_s
    while t <= t_max - window_after_s:
        # Moyennes avant / après
        bal_b  = _mean_in_window(series, t, -window_before_s, -2.0,  "balance")
        bal_a  = _mean_in_window(series, t,  0.0,  window_after_s,   "balance")
        delta  = bal_a - bal_b

        if delta >= balance_delta_min and bal_a >= balance_after_min:
            n_b   = _mean_in_window(series, t, -window_before_s, -2.0, "n")
            n_a   = _mean_in_window(series, t,  0.0, window_after_s,   "n")
            nc_b  = _mean_in_window(series, t, -window_before_s, -2.0, "n_center")
            nc_a  = _mean_in_window(series, t,  0.0, window_after_s,   "n_center")
            di_b  = _mean_in_window(series, t, -window_before_s, -2.0, "disp")
            di_a  = _mean_in_window(series, t,  0.0, window_after_s,   "disp")

            # Score composite : delta balance + niveau balance_after
            score = round(delta * 0.6 + bal_a * 0.4, 3)

            # Frame la plus proche de ce timestamp
            closest = min(series, key=lambda s: abs(s["t_abs"] - t))
            frame_approx = frames_data[0].get("frame", 0)
            for fd in frames_data:
                if abs((start_s + fd.get("frame",0)/max(fps,1)) - t) < 0.5:
                    frame_approx = fd.get("frame", 0)
                    break

            candidates_raw.append({
                "type":               "kickoff_candidate",
                "time":               round(t, 2),
                "frame":              frame_approx,
                "score":              score,
                "balance_before":     round(bal_b, 3),
                "balance_after":      round(bal_a, 3),
                "balance_delta":      round(delta, 3),
                "n_players_before":   round(n_b, 1),
                "n_players_after":    round(n_a, 1),
                "n_center_before":    round(nc_b, 1),
                "n_center_after":     round(nc_a, 1),
                "dispersion_before":  round(di_b, 3),
                "dispersion_after":   round(di_a, 3),
            })

        t += scan_step_s

    if not candidates_raw:
        return []

    # Déduplication : garder le meilleur score dans chaque fenêtre min_gap_s
    candidates_raw.sort(key=lambda c: c["time"])
    deduped = []
    for c in candidates_raw:
        if not deduped or c["time"] - deduped[-1]["time"] > min_gap_s:
            deduped.append(c)
        elif c["score"] > deduped[-1]["score"]:
            deduped[-1] = c

    return deduped
