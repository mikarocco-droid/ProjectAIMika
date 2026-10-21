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
    balance_delta_min   = 0.10,   # permissif — rappel prioritaire (5/5 buts couverts)
    balance_after_min   = 0.20,   # niveau minimal de balance après
    window_before_s     = 15.0,   # fenêtre avant pour baseline
    window_after_s      = 10.0,   # fenêtre après pour mesure
    min_gap_s           = 30.0,   # réduit pour ne pas fusionner des KO proches
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

    # Déduplication. 20/10/2026 : trois modes (SCOUTIA_KO_DEDUP), l'historique reste le défaut.
    #   fenetre    : meilleur score dans une fenêtre glissante de min_gap_s (historique). Défaut : un
    #                candidat qui en remplace un autre DÉPLACE la référence -> un épisode distinct mais
    #                mieux noté, à < 30 s, efface le précédent (Franchimont : remise en jeu 80:18 effacée
    #                par 80:46 puis 80:48 -> but de 79:42 perdu).
    #   episodes   : candidats bruts regroupés en épisodes (écart <= 6 s entre bruts consécutifs),
    #                meilleur candidat de chaque épisode.
    #   episodes20 : episodes, puis fusion des épisodes dont les meilleurs candidats sont à < 20 s.
    import os as _os_dd
    _mode = (_os_dd.environ.get("SCOUTIA_KO_DEDUP", "") or "fenetre").strip().lower()
    if _mode not in ("fenetre", "episodes", "episodes20"):
        print(f"  [KO_DETECTION] SCOUTIA_KO_DEDUP='{_mode}' inconnu — mode historique « fenetre »")
        _mode = "fenetre"
    if _mode == "fenetre":
        deduped = []
        for c in candidates_raw:
            if not deduped or c["time"] - deduped[-1]["time"] > min_gap_s:
                deduped.append(c)
            elif c["score"] > deduped[-1]["score"]:
                deduped[-1] = c
    else:
        episodes, cur = [], []
        for c in candidates_raw:
            if cur and c["time"] - cur[-1]["time"] > 6.0:
                episodes.append(max(cur, key=lambda x: x["score"])); cur = []
            cur.append(c)
        if cur:
            episodes.append(max(cur, key=lambda x: x["score"]))
        deduped = []
        for c in episodes:
            if _mode == "episodes20" and deduped and c["time"] - deduped[-1]["time"] < 20.0:
                if c["score"] > deduped[-1]["score"]:
                    deduped[-1] = c
                continue
            deduped.append(c)
    print(f"  [KO_DETECTION] déduplication « {_mode} » : {len(candidates_raw)} candidats bruts → {len(deduped)}")
    return deduped
