# analysis/match_boundaries_v2.py
# -*- coding: utf-8 -*-
#
# V5.2 - Phase A : detection KO2 / Fin1MT / FinMatch, INDEPENDANTE du
# tracking (process_video n'est pas touche). Reutilise :
#   - la cascade Gemini Q1/Q2 deja validee pour KO1 (via
#     kickoff_gemini_cascade.detect_kickoff_gemini_avec_retry), avec
#     un filtre "deux_equipes_visibles" ajoute et une gestion de retry
#     sur erreur API - voir detect_ko2_gemini() ci-dessous.
#   - le detecteur audio de transition de regime (proportion bande
#     sifflet 3400-4300Hz + platitude spectrale, zone tampon 4s)
#     valide 8/9 pour Fin1MT et FinMatch, avec un critere de
#     persistance supplementaire pour FinMatch (corrige le cas
#     MineroisSter, cri isole pres du micro).
#
# Contrat :
#   - Fin1MT et FinMatch retournent None si aucun candidat credible
#     n'est trouve (jamais de faux timestamp), coherent avec le
#     contrat deja etabli pour KO1 (AUTO_CONFIRMED/NOT_FOUND/ERROR).
#   - KO2 est INDEPENDANT de Fin1MT (pas de dependance circulaire) -
#     cherche uniquement a partir de KO1, fenetre parametrique sur
#     half_duration_min (voir doc find_ko2_gemini).
#   - Fin1MT et FinMatch dependent tous les deux de KO2 (fenetres
#     relatives), mais sont independants l'un de l'autre - peuvent
#     etre calcules dans n'importe quel ordre une fois KO2 connu.

import os
import subprocess
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


# ─────────────────────────────────────────────────────────────────────────
# KO2 — cascade Gemini Q1/Q2 (reutilise celle de KO1), fenetre parametrique
# ─────────────────────────────────────────────────────────────────────────

def find_ko2_gemini(video_path, ko1_s, half_duration_min=45, marge_avant_min=4,
                     marge_apres_min=23, max_retries=3):
    """
    Cherche KO2 (coup d'envoi 2e mi-temps) via la MEME cascade Gemini
    Q1/Q2 deja validee pour KO1 - structure visuelle identique (joueurs
    en formation, ballon au centre).

    ⚠️ Necessite le patch de kickoff_gemini_cascade.py (V5.2 Phase A) qui
    ajoute le parametre t_debut (defaut=60, inchange pour KO1) - SANS ce
    patch, cette fonction ne peut pas demarrer la recherche a un point
    avance de la video (verifie sur le vrai fichier de production,
    _rechercher_kickoff demarrait a t=60 en dur avant patch).

    Fenetre PARAMETRIQUE (pas un intervalle absolu fixe) :
        [KO1 + half_duration_min + marge_avant_min,
         KO1 + half_duration_min + marge_apres_min]

    Valeurs par defaut (marge_avant_min=4, marge_apres_min=23) calibrees
    sur les 9 matchs de reference (tous a half_duration_min=45,
    football amateur reglementaire) :
        MIN(KO2-KO1) observe = 53.6min = 45+8.6min
        MAX(KO2-KO1) observe = 62.5min = 45+17.5min
        -> fenetre retenue avec marge de securite : [45+4min, 45+23min]

    ⚠️ Ces marges (+4/+23min) encodent des habitudes de temps
    additionnel et de duree de pause mesurees UNIQUEMENT sur du
    football amateur a mi-temps de 45min. Si half_duration_min differe
    fortement (ex: match jeunes a 25-30min), ces memes marges absolues
    n'ont PAS ete validees - a revalider avec de vraies donnees avant
    usage en production sur ce type de match.

    ⚠️ NE REUTILISE PAS le filtre "deux_equipes_visibles" (teste et
    valide dans les notebooks d'exploration pour corriger un faux
    positif specifique - une seule equipe en echauffement pres de la
    touche) - ce filtre n'existe PAS dans kickoff_gemini_cascade.py de
    production, et ne doit PAS y etre ajoute silencieusement (prompt
    partage avec KO1, deja valide 9/9 sans ce filtre - toute
    modification du prompt necessite une revalidation complete sur les
    9 matchs KO1 avant d'etre deployee, voir avertissement en tete de
    kickoff_gemini_cascade.py). A proposer et valider separement.

    Retourne un dict {"status": ..., "ko2_s": float|None, "reason": str}
    - meme contrat que detect_kickoff_gemini_avec_retry.
    """
    from analysis.kickoff_gemini_cascade import detect_kickoff_gemini_avec_retry

    t_debut_recherche = ko1_s + (half_duration_min + marge_avant_min) * 60
    t_fin_recherche = ko1_s + (half_duration_min + marge_apres_min) * 60

    resultat = detect_kickoff_gemini_avec_retry(
        video_path,
        max_search_s = t_fin_recherche,   # t_max est ABSOLU dans _rechercher_kickoff,
                                            # pas relatif a t_debut - verifie dans le
                                            # code source (boucle "while t <= t_max")
        t_debut      = t_debut_recherche,
        max_retries  = max_retries,
    )

    if resultat["status"] == "AUTO_CONFIRMED":
        return {"status": "AUTO_CONFIRMED", "ko2_s": resultat["kickoff_s"], "reason": None}
    else:
        return {"status": resultat["status"], "ko2_s": None,
                 "reason": resultat.get("reason", "raison non precisee")}


# ─────────────────────────────────────────────────────────────────────────
# Detecteur audio commun (Fin1MT et FinMatch) — transition de regime
# ─────────────────────────────────────────────────────────────────────────

BANDE_SIFFLET = (3400, 4300)
BANDE_REFERENCE = (200, 6000)
FENETRE_COMPARAISON_S = 30
TAMPON_S = 4.0


def _extraire_audio(video_path, t_debut, duree, chemin_wav):
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(max(0, t_debut)), "-t", str(duree), "-i", video_path,
        "-vn", "-ac", "1", "-ar", "22050", chemin_wav
    ], check=True, capture_output=True)


def _calculer_series(chemin_wav):
    y, sr = librosa.load(chemin_wav, sr=22050)
    n_fft, hop = 2048, 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    temps = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    idx_sifflet = np.where((freqs >= BANDE_SIFFLET[0]) & (freqs <= BANDE_SIFFLET[1]))[0]
    idx_ref = np.where(((freqs >= BANDE_REFERENCE[0]) & (freqs < BANDE_SIFFLET[0])) |
                         ((freqs > BANDE_SIFFLET[1]) & (freqs <= BANDE_REFERENCE[1])))[0]
    energie_sifflet = np.sum(S[idx_sifflet, :] ** 2, axis=0)
    energie_totale = np.sum(S ** 2, axis=0) + 1e-10
    proportion_sifflet = energie_sifflet / energie_totale
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    return temps, proportion_sifflet, flatness


def _score_transition(temps, prop_sifflet, flatness, avec_persistance=False):
    """Score de transition. avec_persistance=True ajoute le critere de
    persistance valide pour FinMatch (corrige les faux positifs de
    bruit bref/isole, ex: cri de supporter pres du micro)."""
    dt = temps[1] - temps[0]
    n_pts = max(1, int(FENETRE_COMPARAISON_S / dt))
    n_tampon = max(1, int(TAMPON_S / dt))
    scores = np.full(len(temps), -np.inf)

    for i in range(n_pts, len(temps) - n_pts - n_tampon):
        avant_prop = np.mean(prop_sifflet[i-n_pts:i])
        apres_prop = np.mean(prop_sifflet[i+n_tampon:i+n_tampon+n_pts])
        avant_flat = np.mean(flatness[i-n_pts:i])
        apres_flat_segment = flatness[i+n_tampon:i+n_tampon+n_pts]
        apres_flat = np.mean(apres_flat_segment)

        baisse_prop = (avant_prop - apres_prop) / (avant_prop + 1e-10)
        hausse_flat = (apres_flat - avant_flat) / (avant_flat + 1e-10)
        score = baisse_prop + hausse_flat

        if avec_persistance:
            seuil_local = avant_flat * 1.5
            fraction_persistante = np.mean(apres_flat_segment > seuil_local)
            score *= fraction_persistante

        scores[i] = score

    return scores


def find_fin1mt_audio(video_path, ko2_s, marge_avant_min=16, marge_apres_min=8,
                       tmp_dir="/tmp"):
    """
    Cherche Fin1MT par transition de regime audio, dans la fenetre
    [KO2-marge_avant_min, KO2-marge_apres_min] (recherche EN ARRIERE
    depuis KO2 - independant de KO2 uniquement, PAS de circularite avec
    Fin1MT lui-meme). Valide 8/9 a <=19s sur les 9 matchs de reference.

    Retourne float (timestamp absolu) ou None si aucun signal credible
    (score au maximum <= 0, jamais de faux timestamp).
    """
    t_debut = ko2_s - marge_avant_min * 60
    duree = (marge_avant_min - marge_apres_min) * 60
    if duree <= 0:
        return None

    chemin_wav = os.path.join(tmp_dir, "_fin1mt_audio_tmp.wav")
    _extraire_audio(video_path, t_debut, duree, chemin_wav)
    temps, prop_sifflet, flatness = _calculer_series(chemin_wav)
    scores = _score_transition(temps, prop_sifflet, flatness, avec_persistance=False)

    idx_max = np.argmax(scores)
    if not np.isfinite(scores[idx_max]) or scores[idx_max] <= 0:
        return None
    return t_debut + temps[idx_max]


def find_finmatch_audio(video_path, ko2_s, marge_debut_min=40, marge_fin_min=55,
                          tmp_dir="/tmp"):
    """
    Cherche FinMatch par transition de regime audio + critere de
    persistance, dans la fenetre [KO2+marge_debut_min, KO2+marge_fin_min]
    (recherche EN AVANT depuis KO2). Valide 8/9 a <=10s sur les 9 matchs
    de reference (limite connue : Raeren, +62s, signal faible du a du
    vent + retour progressif des joueurs).

    Retourne float (timestamp absolu) ou None si aucun signal credible.
    """
    t_debut = ko2_s + marge_debut_min * 60
    duree = (marge_fin_min - marge_debut_min) * 60

    chemin_wav = os.path.join(tmp_dir, "_finmatch_audio_tmp.wav")
    _extraire_audio(video_path, t_debut, duree, chemin_wav)
    temps, prop_sifflet, flatness = _calculer_series(chemin_wav)
    scores = _score_transition(temps, prop_sifflet, flatness, avec_persistance=True)

    idx_max = np.argmax(scores)
    if not np.isfinite(scores[idx_max]) or scores[idx_max] <= 0:
        return None
    return t_debut + temps[idx_max]
