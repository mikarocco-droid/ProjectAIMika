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


def _dedupliquer(candidates_raw, _mode, min_gap_s):
    """Déduplication. 20/10/2026 : trois modes (SCOUTIA_KO_DEDUP), l'historique reste le défaut.
      fenetre    : meilleur score dans une fenêtre glissante de min_gap_s (historique). Défaut : un candidat qui en
                   remplace un autre DÉPLACE la référence -> une CHAÎNE de candidats rapprochés finit en un seul
                   (Franchimont : but de 79:42 perdu ; Raeren 15/11 : penalty de 69:14, KO à 69:43, sans doute absorbé
                   par 68:51, la mise en place du penalty).
      episodes   : candidats bruts regroupés en épisodes (écart <= 6 s entre bruts consécutifs), meilleur de chaque.
      episodes20 : episodes, puis fusion des épisodes dont les meilleurs candidats sont à < 20 s.
    (15/11 : sortie en fonction pour la trace ; comportement inchangé.)"""
    if _mode == "fenetre":
        deduped = []
        for c in candidates_raw:
            if not deduped or c["time"] - deduped[-1]["time"] > min_gap_s:
                deduped.append(c)
            elif c["score"] > deduped[-1]["score"]:
                deduped[-1] = c
        return deduped
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
    return deduped


def _trace_candidats(candidates_raw, min_gap_s):
    """15/11 (diagnostic, sans effet sur le résultat) : SCOUTIA_KO_TRACE = "68:00-71:00" (heures comme dans
    « [KO_DETECTION] t=… » du journal ; plusieurs plages séparées par des virgules) → affiche les candidats BRUTS de
    la plage et ce que garde chacun des trois modes de regroupement."""
    import os as _os_tr
    _tr = _os_tr.environ.get("SCOUTIA_KO_TRACE", "").strip()
    if not _tr:
        return
    def _s(h):
        m, sec = h.strip().split(":")
        return int(m) * 60 + float(sec)
    def _h(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"
    for plage in _tr.split(","):
        try:
            a, b = (_s(x) for x in plage.split("-"))
        except Exception:
            print(f"  [KO_TRACE] plage illisible : {plage!r} (format 68:00-71:00)")
            continue
        brut = [c for c in candidates_raw if a <= c["time"] <= b]
        print(f"  [KO_TRACE] {_h(a)}-{_h(b)} : {len(brut)} candidat(s) BRUT(S) (un point toutes les 2 s si le critère est rempli)")
        for c in brut:
            print(f"  [KO_TRACE]    brut {_h(c['time'])} score={c['score']:.3f} hausse={c['balance_delta']:.3f} "
                  f"balance après={c['balance_after']:.3f} joueurs au centre {c['n_center_before']:.1f}→{c['n_center_after']:.1f}")
        for mode in ("fenetre", "episodes", "episodes20"):
            garde = [c for c in _dedupliquer(candidates_raw, mode, min_gap_s) if a <= c["time"] <= b]
            print(f"  [KO_TRACE]    mode « {mode} » garde : " + (", ".join(f"{_h(c['time'])} ({c['score']:.3f})" for c in garde) or "aucun"))


def detect_kickoff_candidates(
    frames_data,
    fps,
    frame_w,
    frame_h,
    start_s             = 0.0,
    balance_delta_min   = 0.07,   # 05/11/2026 : 0.10 → 0.07 (validé sur Raeren, Minerois Ster, Franchimont, Spa, Andrimont)
    balance_after_min   = 0.15,   # 05/11/2026 : 0.20 → 0.15 (idem)
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
    # 05/11/2026 : DÉFAUT PASSÉ À 0.07 / 0.15. Replays comparés au réglage 0.10 / 0.20 : Raeren +1 vrai but (20:08),
    #   Minerois Ster, Franchimont, Spa, Andrimont (cache partiel) : aucun vrai but perdu, aucun faux but ajouté.
    #   Pour revenir à l'ancien réglage : SCOUTIA_KO_DELTA_MIN=0.10 et SCOUTIA_KO_APRES_MIN=0.20.
    # 04/11/2026 : seuils réglables par variables d'environnement.
    #   SCOUTIA_KO_DELTA_MIN : hausse minimale de la balance des équipes (Raeren, run du 04/11 : le replacement après
    #   le but de 20:08 ne montait qu'à +0.07 → aucun candidat ; à 0.07, un candidat 20:23 apparaît, comme au run du 18/10)
    #   SCOUTIA_KO_APRES_MIN : niveau minimal de la balance après.
    import os as _os_seuils
    try:
        balance_delta_min = float(_os_seuils.environ.get("SCOUTIA_KO_DELTA_MIN", "") or balance_delta_min)
        balance_after_min = float(_os_seuils.environ.get("SCOUTIA_KO_APRES_MIN", "") or balance_after_min)
    except ValueError:
        print("  [KO_DETECTION] ⚠️ SCOUTIA_KO_DELTA_MIN / SCOUTIA_KO_APRES_MIN illisible : seuils par défaut")
    if (balance_delta_min, balance_after_min) != (0.07, 0.15):
        print(f"  [KO_DETECTION] seuils NON STANDARD : hausse ≥ {balance_delta_min:g}, balance après ≥ {balance_after_min:g}")

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

    import os as _os_dd
    _mode = (_os_dd.environ.get("SCOUTIA_KO_DEDUP", "") or "fenetre").strip().lower()
    if _mode not in ("fenetre", "episodes", "episodes20"):
        print(f"  [KO_DETECTION] SCOUTIA_KO_DEDUP='{_mode}' inconnu — mode historique « fenetre »")
        _mode = "fenetre"
    deduped = _dedupliquer(candidates_raw, _mode, min_gap_s)
    _trace_candidats(candidates_raw, min_gap_s)
    print(f"  [KO_DETECTION] déduplication « {_mode} » : {len(candidates_raw)} candidats bruts → {len(deduped)}")
    return deduped
