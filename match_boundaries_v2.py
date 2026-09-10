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
                     marge_apres_min=23, max_retries=3, nom_fonction_q1="vote_economique",
                     model_name="gemini-3.1-pro-preview", nom_fonction_q2="standard", tmp_dir="/tmp"):
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
    from analysis.kickoff_gemini_cascade import detect_kickoff_gemini_avec_retry, _q1_une_lecture_ko2, _q1_une_lecture_ko2_vote_economique, _q2_une_lecture, _q2_avant_apres_une_lecture

    fonction_q1_choisie = {
        "standard": _q1_une_lecture_ko2,
        "vote_economique": _q1_une_lecture_ko2_vote_economique,
    }[nom_fonction_q1]

    fonction_q2_choisie = {
        "standard": _q2_une_lecture,
        "avant_apres": _q2_avant_apres_une_lecture,
    }[nom_fonction_q2]

    t_debut_recherche = ko1_s + (half_duration_min + marge_avant_min) * 60
    t_fin_recherche = ko1_s + (half_duration_min + marge_apres_min) * 60

    resultat = detect_kickoff_gemini_avec_retry(
        video_path,
        max_search_s = t_fin_recherche,   # t_max est ABSOLU dans _rechercher_kickoff,
                                            # pas relatif a t_debut - verifie dans le
                                            # code source (boucle "while t <= t_max")
        t_debut      = t_debut_recherche,
        pas_scan     = 20,   # V5.2 : pas plus fin que le defaut KO1 (60s) - valide
                             # empiriquement sur Franchimont (la camera ne se stabilise
                             # sur le centre que brievement lors de la reprise de 2e
                             # mi-temps, un pas de 60s peut sauter par-dessus). Diagnostic
                             # confirme en production : sans ce fix, Franchimont et
                             # Stembert donnaient des erreurs KO2 de plusieurs minutes.
        delai_verif_q2 = 30, # V5.2 FIX (09/09/2026) : filet de securite tardif,
                             # PASSE DE 22 A 30 suite au diagnostic Spa (candidat
                             # legitime a t=3798s rejete par une verification tombee
                             # a 2s pres de la vraie transition 3820->3822s) et Melen
                             # (candidats legitimes a 4255s/4277s rejetes par +22s).
                             # Valide sans regression sur 11 matchs (9 reference +
                             # Melen + Spa) : gain net sur Wanze (4734->4712s, plus
                             # proche du vrai 4735s) et Melen (4499->4255s, gain de
                             # 244s !), 0 regression ailleurs. cf.
                             # notebook_test_delai_ko2_10matchs.py pour le detail
                             # complet du test (candidat par candidat).
        delai_verif_q2_precoce = 22, # V5.2 : premiere verification, inchangee -
                             # c'etait deja la seule verification avant aujourd'hui.
                             # Valeur choisie empiriquement (test Franchimont : bascule
                             # nette False->True entre 20s et 21s ; test Stembert :
                             # comportement NON MONOTONE observe (True a 6-10s, False a
                             # nouveau a 15s, True a nouveau a 20s+) - aucun delai fixe
                             # n'est parfaitement fiable partout, 22s est un compromis
                             # raisonnable avec un peu de marge sur Franchimont, pas une
                             # garantie universelle. Le systeme garde une redondance
                             # (reprise de scan si Q2 rejette a tort) qui absorbe une
                             # partie de ce risque residuel.
        max_retries  = max_retries,
        model_name   = model_name,
        tmp_dir      = tmp_dir,  # V5.2 FIX : dossier temporaire dedie,
                             # evite toute collision de nom de fichier
                             # avec un autre detecteur/match tournant
                             # dans le meme processus (fichiers nommes
                             # uniquement par timestamp arrondi -
                             # risque reel si deux detecteurs
                             # interrogent par coincidence le meme
                             # timestamp arrondi dans /tmp partage).
        fonction_q1  = fonction_q1_choisie,  # V5.2 : "standard" (defaut, deja
                             # deja valide dependait de ce filtre (deux equipes
                             # visibles obligatoire) - jamais porte en production
                             # avant ce fix, cause reelle des divergences P1Minerois/
                             # Goe/Stembert observees lors du premier test d'integration.
                             # KO1 continue d'utiliser _q1_une_lecture (original),
                             # non touche.
        taille_lot   = 1,   # V5.2 TEST ISOLE : le notebook qui a valide 9/9 scannait
                             # sequentiellement (1 candidat a la fois), PAS par lots de
                             # 6 en parallele comme le fait la production par defaut.
                             # Seule variable changee dans ce test, pour isoler
                             # precisement si le scan par lots affecte le resultat.
                             # KO1 continue d'utiliser TAILLE_LOT_Q1=6 (non touche).
        fonction_q2  = fonction_q2_choisie,  # V5.2 : "standard" (defaut, ancien
                             # signal "match deja commence") ou "avant_apres"
                             # (nouveau signal AVANT/TRANSITION/APRES/INCERTAIN,
                             # moins fragile - experience de caracterisation sur
                             # 9 matchs a montre 6/9 sequences parfaitement
                             # monotones contre une non-monotonie averee de
                             # l'ancien signal sur au moins 1 cas, Stembert).
                             # KO1 continue d'utiliser _q2_une_lecture (original),
                             # non touche.
    )

    if resultat["status"] == "AUTO_CONFIRMED":
        return {"status": "AUTO_CONFIRMED", "ko2_s": resultat["kickoff_s"], "reason": None,
                 "n_appels_gemini": resultat.get("n_appels_gemini")}
    else:
        return {"status": resultat["status"], "ko2_s": None,
                 "reason": resultat.get("reason", "raison non precisee"),
                 "n_appels_gemini": resultat.get("n_appels_gemini")}


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
                       tmp_dir="/tmp", fin1mt_verite_debug=None):
    """
    Cherche Fin1MT par transition de regime audio, dans la fenetre
    [KO2-marge_avant_min, KO2-marge_apres_min] (recherche EN ARRIERE
    depuis KO2 - independant de KO2 uniquement, PAS de circularite avec
    Fin1MT lui-meme). Valide 8/9 a <=19s sur les 9 matchs de reference.

    ⚠️ V5.2 FIX (diagnostic production, Andrimont/Goe) : extrait un
    BUFFER supplementaire de 60s de chaque cote au-dela de la fenetre
    officielle - le calcul de score lui-meme a besoin d'une marge
    interne (~34s = fenetre de comparaison 30s + tampon 4s) de chaque
    cote du segment pour etre calculable. Sans ce buffer, un vrai
    Fin1MT situe pres du bord de la fenetre officielle (observe a
    ~19s du bord sur 2 des 9 matchs) tombe dans la zone -inf non
    calculable, forcant le detecteur a choisir un autre point, moins
    bon. Le buffer ne change PAS la fenetre officielle (deja validee
    9/9 pour contenir le vrai Fin1MT), il donne juste au calcul la
    place d'atteindre ses bords.

    fin1mt_verite_debug : si fourni (tests uniquement), affiche le score
    au point de verite terrain pour comparer au candidat retenu.

    Retourne float (timestamp absolu) ou None si aucun signal credible
    (score au maximum <= 0, jamais de faux timestamp).
    """
    BUFFER_S = 60.0
    t_debut_officiel = ko2_s - marge_avant_min * 60
    t_fin_officiel = ko2_s - marge_apres_min * 60
    duree_officielle = t_fin_officiel - t_debut_officiel
    if duree_officielle <= 0:
        return None

    t_debut_extraction = t_debut_officiel - BUFFER_S
    duree_extraction = duree_officielle + 2 * BUFFER_S

    chemin_wav = os.path.join(tmp_dir, "_fin1mt_audio_tmp.wav")
    _extraire_audio(video_path, t_debut_extraction, duree_extraction, chemin_wav)
    temps, prop_sifflet, flatness = _calculer_series(chemin_wav)
    scores = _score_transition(temps, prop_sifflet, flatness, avec_persistance=False)

    # Ne considerer que les points DANS la fenetre officielle (le buffer
    # sert seulement a rendre leur score calculable, pas a etendre la
    # zone de recherche elle-meme)
    idx_officiel = np.where((temps >= BUFFER_S) & (temps <= BUFFER_S + duree_officielle))[0]
    if len(idx_officiel) == 0:
        return None
    idx_max = idx_officiel[np.argmax(scores[idx_officiel])]

    print(f"    [DEBUG FIN1MT] fenêtre officielle : [{t_debut_officiel:.0f}s, {t_fin_officiel:.0f}s] "
          f"(extraction avec buffer : [{t_debut_extraction:.0f}s, {t_debut_extraction+duree_extraction:.0f}s])")
    print(f"    [DEBUG FIN1MT] candidat : t_relatif={temps[idx_max]-BUFFER_S:.1f}s (dans la fenêtre officielle), "
          f"score={scores[idx_max]:.4f}")
    if fin1mt_verite_debug is not None:
        t_verite_relatif = fin1mt_verite_debug - t_debut_extraction
        idx_verite = np.argmin(np.abs(temps - t_verite_relatif))
        print(f"    [DEBUG FIN1MT] score AU VRAI Fin1MT (t_relatif={t_verite_relatif-BUFFER_S:.1f}s) : "
              f"{scores[idx_verite]:.4f}")

    if not np.isfinite(scores[idx_max]) or scores[idx_max] <= 0:
        return None
    return t_debut_extraction + temps[idx_max]


def find_finmatch_audio(video_path, ko2_s, marge_debut_min=40, marge_fin_min=55,
                          tmp_dir="/tmp"):
    """
    Cherche FinMatch par transition de regime audio + critere de
    persistance, dans la fenetre [KO2+marge_debut_min, KO2+marge_fin_min]
    (recherche EN AVANT depuis KO2). Valide 8/9 a <=10s sur les 9 matchs
    de reference (limite connue : Raeren, +62s, signal faible du a du
    vent + retour progressif des joueurs).

    ⚠️ V5.2 FIX (meme correctif que find_fin1mt_audio, par coherence) :
    extrait un buffer de 60s de chaque cote au-dela de la fenetre
    officielle, pour que le calcul de score (qui a besoin de ~34s de
    marge interne) puisse atteindre les bords de la fenetre officielle.
    Le buffer ne change pas la fenetre officielle elle-meme.

    Retourne float (timestamp absolu) ou None si aucun signal credible.
    """
    BUFFER_S = 60.0
    t_debut_officiel = ko2_s + marge_debut_min * 60
    t_fin_officiel = ko2_s + marge_fin_min * 60
    duree_officielle = t_fin_officiel - t_debut_officiel
    if duree_officielle <= 0:
        return None

    t_debut_extraction = t_debut_officiel - BUFFER_S
    duree_extraction = duree_officielle + 2 * BUFFER_S

    chemin_wav = os.path.join(tmp_dir, "_finmatch_audio_tmp.wav")
    _extraire_audio(video_path, t_debut_extraction, duree_extraction, chemin_wav)
    temps, prop_sifflet, flatness = _calculer_series(chemin_wav)
    scores = _score_transition(temps, prop_sifflet, flatness, avec_persistance=True)

    idx_officiel = np.where((temps >= BUFFER_S) & (temps <= BUFFER_S + duree_officielle))[0]
    if len(idx_officiel) == 0:
        return None
    idx_max = idx_officiel[np.argmax(scores[idx_officiel])]

    if not np.isfinite(scores[idx_max]) or scores[idx_max] <= 0:
        return None
    return t_debut_extraction + temps[idx_max]
