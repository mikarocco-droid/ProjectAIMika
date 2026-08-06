"""
candidates_mining.py
=====================
Objectif UNIQUE : transformer "25 lignes" en "des centaines de lignes",
sans écrire un nouveau détecteur, sans toucher un seul seuil de
ko_features.py (qui reste GELÉ, importé tel quel).

Ce script ne décide RIEN. Il génère des candidats (instants t) sur un
match ENTIER à partir de sources multiples, calcule leurs features avec
ko_features.extract_features(), et propose une étiquette automatique
quand c'est possible (label_auto). Tout candidat non tranchable
automatiquement reçoit "a_verifier" — à checker à l'oeil plutôt que de
gonfler artificiellement le dataset avec des étiquettes devinées.

Sources de candidats (fusionnées puis dédupliquées) :
  1. pics de geo_separation (formations séparées → souvent KO ou
     regroupement, PAS forcément contre-attaque)
  2. pics de mean_speed_change (accélérations fortes → souvent
     contre-attaques, exactement ce qui manquait dans le dataset v1)
  3. pics du signal sifflet (audio, orthogonal aux 2 premiers)
  4. fenêtres autour d'événements déjà loggés par le pipeline
     (buts, tirs, faux positifs de buts) — étiquetage AUTOMATIQUE fiable
  5. le vrai KO connu, si fourni — étiquetage AUTOMATIQUE fiable

Usage prévu (dans un notebook Kaggle, après process_video en cache) :

    from analysis.ko_features import extract_features
    from candidates_mining import mine_candidates

    df = mine_candidates(
        frames_data=frames_data, fps=fps, frame_w=frame_w,
        video_path=video_path, match_name="Andrimont",
        true_kickoff_t=307.0,
        terminal_events=events,   # sortie de process_video() / terminal_events
        video_duration_s=6420.0,
    )
    df.to_csv("candidates_andrimont.csv", index=False)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

try:
    from analysis.ko_features import (
        extract_features, score_geometrie, _window_features,
        _camera_motion_series,
    )
except ImportError:
    from ko_features import (
        extract_features, score_geometrie, _window_features,
        _camera_motion_series,
    )


# ---------------------------------------------------------------------------
# 1. Grilles de signaux bruts sur tout le match (peu coûteux : uniquement
#    des opérations numpy sur frames_data déjà en mémoire, pas de YOLO,
#    pas de ffmpeg répété)
# ---------------------------------------------------------------------------

def _geo_and_speed_grid(frames_data, fps, frame_w, video_duration_s, step_s=4.0):
    """Calcule geo_separation et mean_speed_change sur une grille régulière
    de tout le match. Réutilise directement les fonctions de ko_features
    (gelées, non modifiées)."""
    times = np.arange(step_s, video_duration_s - step_s, step_s)
    geo_vals, speed_vals, n_players_vals = [], [], []
    for t in times:
        speed, geo, n_players = _window_features(frames_data, fps, float(t), frame_w, window_s=6.0)
        geo_vals.append(geo)
        speed_vals.append(speed)
        n_players_vals.append(n_players)
    return times, np.array(geo_vals), np.array(speed_vals), np.array(n_players_vals)


def _peaks(times, values, top_n, min_distance_s, step_s):
    """Pics locaux, pas juste top-N brut (sinon on récupère 10 fois le
    même événement à 4s d'écart)."""
    if len(values) == 0:
        return []
    min_dist_samples = max(1, int(min_distance_s / step_s))
    idx, _ = find_peaks(values, distance=min_dist_samples)
    if len(idx) == 0:
        idx = np.arange(len(values))
    idx_sorted = idx[np.argsort(-values[idx])][:top_n]
    return sorted(times[idx_sorted].tolist())


# ---------------------------------------------------------------------------
# 2. Signal sifflet sur tout le match — UNE extraction audio, pas une par
#    candidat (le whistle_score() de ko_features fait un ffmpeg par appel,
#    inutilisable en boucle sur des centaines de candidats)
# ---------------------------------------------------------------------------

def _whistle_series_full_match(video_path, video_duration_s,
                                band_low_hz=2000, band_high_hz=4200,
                                tmp_wav="/tmp/_whistle_full.wav"):
    """Extrait l'audio UNE fois pour tout le match, retourne la série
    temporelle d'énergie dans la bande sifflet. None si pas d'audio
    (vidéo trimée avec -an, cf. section 11 du résumé projet)."""
    import subprocess, os
    from scipy.io import wavfile
    from scipy.signal import spectrogram

    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", tmp_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [whistle] pas d'audio exploitable ({video_path}) — vérifier -c:a aac au découpage")
        return None, None
    try:
        sr, data = wavfile.read(tmp_wav)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(float)
    f, tt, Sxx = spectrogram(data, fs=sr, nperseg=1024, noverlap=768)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    band_mask = (f >= band_low_hz) & (f <= band_high_hz)
    band_energy = Sxx_db[band_mask, :].mean(axis=0)
    return tt, band_energy


def _whistle_peaks(tt, band_energy, top_n, min_distance_s=3.0):
    """Z-score GLOBAL sur tout le match (approximation : ko_features.py
    utilise un baseline LOCAL de 8s par candidat, plus précis mais trop
    coûteux à répéter ici). Bon pour repérer des candidats ; le score
    précis sera recalculé par extract_features() sur les candidats retenus."""
    if tt is None:
        return []
    baseline, std = np.median(band_energy), (np.std(band_energy) or 1.0)
    zscore = (band_energy - baseline) / std
    min_dist_samples = max(1, int(min_distance_s / np.median(np.diff(tt))))
    idx, _ = find_peaks(zscore, distance=min_dist_samples, height=1.5)
    idx_sorted = idx[np.argsort(-zscore[idx])][:top_n]
    return sorted(tt[idx_sorted].tolist())


# ---------------------------------------------------------------------------
# 3. Étiquetage automatique — uniquement quand c'est fiable, sinon
#    "a_verifier" (pas d'étiquette devinée)
# ---------------------------------------------------------------------------

def _auto_label(t, true_kickoff_t, terminal_events, ko_tolerance_s=3.0, event_tolerance_s=5.0):
    if true_kickoff_t is not None and abs(t - true_kickoff_t) <= ko_tolerance_s:
        return "KO"
    if terminal_events:
        for ev in terminal_events:
            ev_t = ev.get("t") or ev.get("time") or ev.get("frame_time")
            ev_type = ev.get("type") or ev.get("event") or "evenement"
            if ev_t is not None and abs(t - ev_t) <= event_tolerance_s:
                return f"connu:{ev_type}"
    return "a_verifier"


# ---------------------------------------------------------------------------
# 4. Fusion + dédoublonnage
# ---------------------------------------------------------------------------

def _merge_candidates(*groups, dedup_tolerance_s=3.0):
    """Fusionne plusieurs listes de candidats (avec leur source), fusionne
    les candidats trop proches (garde la première source rencontrée,
    ajoute les sources supplémentaires en note)."""
    all_c = []
    for source_name, times in groups:
        for t in times:
            all_c.append((t, source_name))
    all_c.sort(key=lambda x: x[0])

    merged = []
    for t, source in all_c:
        if merged and abs(t - merged[-1]["t"]) <= dedup_tolerance_s:
            if source not in merged[-1]["sources"]:
                merged[-1]["sources"].append(source)
        else:
            merged.append({"t": t, "sources": [source]})
    return merged


# ---------------------------------------------------------------------------
# API PRINCIPALE
# ---------------------------------------------------------------------------

def mine_candidates(frames_data, fps, frame_w, video_path, match_name,
                     video_duration_s, true_kickoff_t=None, terminal_events=None,
                     top_n_geo=15, top_n_speed=15, top_n_whistle=10,
                     event_window_s=8.0, grid_step_s=4.0):
    """Génère un DataFrame de candidats pour UN match, avec features
    complètes (ko_features.extract_features, gelé) et label_auto.

    Ne modifie AUCUN seuil de ko_features.py. N'écrit aucune nouvelle
    heuristique de décision — seulement de la génération de candidats.
    """
    print(f"[{match_name}] grille geo/vitesse sur {video_duration_s:.0f}s (pas={grid_step_s}s)...")
    times, geo_vals, speed_vals, n_players_vals = _geo_and_speed_grid(
        frames_data, fps, frame_w, video_duration_s, step_s=grid_step_s
    )

    geo_candidates = _peaks(times, geo_vals, top_n_geo, min_distance_s=20.0, step_s=grid_step_s)
    speed_candidates = _peaks(times, speed_vals, top_n_speed, min_distance_s=20.0, step_s=grid_step_s)

    print(f"[{match_name}] extraction audio complète pour le sifflet...")
    tt_audio, band_energy = _whistle_series_full_match(video_path, video_duration_s)
    whistle_candidates = _whistle_peaks(tt_audio, band_energy, top_n_whistle)

    event_candidates = []
    if terminal_events:
        for ev in terminal_events:
            ev_t = ev.get("t") or ev.get("time") or ev.get("frame_time")
            if ev_t is not None:
                event_candidates.append(ev_t)

    ko_candidates = [true_kickoff_t] if true_kickoff_t is not None else []

    merged = _merge_candidates(
        ("geo_peak", geo_candidates),
        ("speed_peak", speed_candidates),
        ("whistle_peak", whistle_candidates),
        ("evenement_connu", event_candidates),
        ("vrai_ko_connu", ko_candidates),
        dedup_tolerance_s=3.0,
    )
    print(f"[{match_name}] {len(merged)} candidats uniques après fusion "
          f"(geo={len(geo_candidates)}, speed={len(speed_candidates)}, "
          f"whistle={len(whistle_candidates)}, events={len(event_candidates)})")

    rows = []
    for i, cand in enumerate(merged):
        t = cand["t"]
        if t < 3 or t > video_duration_s - 3:
            continue  # trop près des bords, features non fiables
        row = extract_features(frames_data, fps, frame_w, video_path, t, include_audio=False)
        row["match"] = match_name
        row["sources"] = "+".join(cand["sources"])
        row["label_auto"] = _auto_label(t, true_kickoff_t, terminal_events,
                                          ko_tolerance_s=event_window_s / 2,
                                          event_tolerance_s=event_window_s)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ...features calculées pour {i + 1}/{len(merged)} candidats")

    df = pd.DataFrame(rows)
    n_ko = (df["label_auto"] == "KO").sum() if len(df) else 0
    n_connus = df["label_auto"].str.startswith("connu:").sum() if len(df) else 0
    n_a_verifier = (df["label_auto"] == "a_verifier").sum() if len(df) else 0
    print(f"[{match_name}] TERMINÉ : {len(df)} candidats — "
          f"KO={n_ko}, connus(auto)={n_connus}, à_vérifier={n_a_verifier}")
    return df


if __name__ == "__main__":
    print(__doc__)
    print("\nCe script est une librairie à importer dans un notebook Kaggle,")
    print("pas un script à lancer seul (il a besoin de frames_data en mémoire).")
