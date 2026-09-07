"""
fin1mt_gemini_cascade.py
============================
V5.2 - Detection Fin1MT (fin de 1ere mi-temps) par vision Gemini, meme
architecture AVANT/TRANSITION/APRES/INCERTAIN validee sur KO2 aujourd'hui
(8/9 <=3s), plutot que la detection audio existante (2 echecs a >200s
sur Franchimont/Stembert, faux positifs acoustiques plus forts que le
vrai signal).

Recherche DANS LA FENETRE DEJA VALIDEE [KO2-marge_avant_min,
KO2-marge_apres_min] (validee 9/9 pour contenir le vrai Fin1MT dans le
travail R&D anterieur) - on ne change pas la fenetre de recherche,
seulement la methode de detection a l'interieur.

Reutilise l'infrastructure generique de kickoff_gemini_cascade.py
(_appeler_json_robuste, _EtatRecherche, executor, PALIERS_RECHERCHE_FINE)
- AUCUNE modification de ce fichier, uniquement des imports.
"""

import time
import concurrent.futures

from analysis.kickoff_gemini_cascade import (
    _EtatRecherche,
    _appeler_json_robuste,
    PALIERS_RECHERCHE_FINE,
    MAX_GEMINI_CALLS_DEFAUT,
    MAX_WALLCLOCK_S_DEFAUT,
)

MODEL_NAME_DEFAUT = "gemini-3.5-flash"

# ─────────────────────────────────────────────────────────────────────────
# PROMPT — meme structure AVANT/TRANSITION/APRES/INCERTAIN que KO2,
# adaptee a la transition "jeu actif -> mi-temps sifflee"
# ─────────────────────────────────────────────────────────────────────────
PROMPT_FIN1MT = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur, autour du moment supposé de la fin de la première mi-temps.

OBJECTIF : classifier cette image par rapport à la fin de la première mi-temps - le jeu est-il encore en cours (AVANT la fin), ou la mi-temps a-t-elle clairement commencé (APRÈS la fin) ?

═══════════════════════════════════════════════════
CATÉGORIES POSSIBLES
═══════════════════════════════════════════════════

AVANT : le jeu est manifestement encore en cours - joueurs actifs sur le terrain, ballon en jeu, action identifiable (passe, duel, course), même un arrêt de jeu ponctuel (touche, corner, coup franc) qui fait partie du match en cours.

TRANSITION : l'instant est ambigu, semble être exactement au moment du coup de sifflet de fin de mi-temps ou juste après (joueurs qui s'arrêtent, commencent tout juste à se regrouper).

APRES : la mi-temps est clairement en cours - joueurs qui quittent le terrain vers la ligne de touche/les vestiaires, se regroupent en dehors du jeu, marchent calmement sans ballon en jeu, arbitre qui s'éloigne du terrain, absence prolongée d'action de jeu.

INCERTAIN : l'image ne permet vraiment pas de juger (cadrage, flou, éléments masqués empêchant toute conclusion).

═══════════════════════════════════════════════════
INDICES À CONSIDÉRER ENSEMBLE (aucun n'est éliminatoire à lui seul)
═══════════════════════════════════════════════════

- Joueurs dispersés sur le terrain en action de jeu → AVANT
- Arrêt de jeu ponctuel reconnaissable comme faisant partie du match (touche, corner, coup franc, faute) → AVANT (ça compte comme jeu en cours)
- Joueurs qui marchent vers la ligne de touche ou les vestiaires, sans ballon en jeu → APRES
- Joueurs regroupés en dehors du terrain ou à l'arrêt sans logique de jeu → APRES
- Arbitre qui s'éloigne du centre du terrain vers la sortie → APRES
- Absence de ballon visible ne signifie PAS automatiquement APRES - regarde la position et l'attitude des joueurs pour juger quand même

Réponds STRICTEMENT en JSON, avec un raisonnement bref :
{"classification": "AVANT"|"TRANSITION"|"APRES"|"INCERTAIN", "raisonnement": "..."}"""


def _classifier_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME_DEFAUT, max_retry_incertain=2):
    """Classifie un point AVANT/APRES (mappage booleen), avec retry sur
    INCERTAIN a des offsets voisins - meme logique que
    _q2_avant_apres_une_lecture pour KO2."""
    for tentative in range(max_retry_incertain + 1):
        tt = t if tentative == 0 else t + tentative
        result = _appeler_json_robuste(client, video_path, tt, tmp_dir, PROMPT_FIN1MT, etat, model_name=model_name)
        if result is None:
            continue
        classification = result.get("classification", "INCERTAIN")
        if classification == "AVANT":
            return False
        elif classification in ("APRES", "TRANSITION"):
            return True
    return None


def _voter_classification(client, video_path, t, tmp_dir, etat, max_appels=3, model_name=MODEL_NAME_DEFAUT):
    """Vote majoritaire avec arret anticipe - meme principe que _voter_q2
    de KO2. Confirme un point avant de l'utiliser comme borne de
    dichotomie, pour eviter qu'un seul appel malchanceux ne fasse
    derailler la recherche."""
    votes = []
    for _ in range(max_appels):
        v = _classifier_une_lecture(client, video_path, t, tmp_dir, etat, model_name=model_name)
        if v is None:
            if not votes:
                return None
            break
        votes.append(v)
        n_true = sum(votes)
        n_false = len(votes) - n_true
        restants = max_appels - len(votes)
        if n_true > n_false + restants or n_false > n_true + restants:
            break
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def _recherche_fine_fin1mt(client, video_path, tmp_dir, etat, t_avant, t_apres, model_name=MODEL_NAME_DEFAUT):
    """Meme dichotomie 15/5/1s que KO2 (_recherche_fine), adaptee au
    signal AVANT/APRES local. t_avant : dernier point confirme AVANT.
    t_apres : premier point confirme APRES."""
    t_bas, t_haut = t_avant, t_apres
    for pas in PALIERS_RECHERCHE_FINE:
        tt = t_bas + pas
        dernier_avant = t_bas
        while tt < t_haut:
            d = _classifier_une_lecture(client, video_path, tt, tmp_dir, etat, model_name=model_name)
            print(f"    [FIN1MT FINE pas={pas}s] t={tt:.0f}s : {'APRES' if d else 'AVANT' if d is not None else 'ERREUR'}")
            if d:
                t_haut = tt
                t_bas = dernier_avant
                break
            dernier_avant = tt
            tt += pas
        else:
            t_bas = dernier_avant
    return t_haut


def find_fin1mt_gemini(video_path, ko2_s, marge_avant_min=16, marge_apres_min=8,
                        pas_scan=20, model_name=MODEL_NAME_DEFAUT,
                        max_gemini_calls=MAX_GEMINI_CALLS_DEFAUT,
                        max_wallclock_s=MAX_WALLCLOCK_S_DEFAUT, tmp_dir="/tmp"):
    """
    Cherche Fin1MT par vision Gemini (signal AVANT/TRANSITION/APRES),
    dans la fenetre [KO2-marge_avant_min, KO2-marge_apres_min] - MEME
    fenetre officielle que find_fin1mt_audio, deja validee 9/9 pour
    contenir le vrai Fin1MT. Seule la methode de detection change.

    Scan en avant depuis le debut de la fenetre, pas_scan par pas_scan,
    jusqu'a trouver un premier point confirme APRES (vote majoritaire a
    3 voix). Dichotomie 15/5/1s ensuite entre le dernier AVANT et ce
    premier APRES confirme.

    Retourne float (timestamp absolu) ou None si aucune transition
    trouvee dans la fenetre (budget epuise ou fenetre entierement
    AVANT/entierement APRES).
    """
    from google import genai
    client = genai.Client()

    etat = _EtatRecherche(max_gemini_calls, max_wallclock_s)
    try:
        t_debut = ko2_s - marge_avant_min * 60
        t_fin = ko2_s - marge_apres_min * 60
        if t_fin <= t_debut:
            return None

        t = t_debut
        dernier_avant_confirme = t_debut
        while t <= t_fin:
            raison_arret = etat.budget_epuise()
            if raison_arret:
                print(f"  [FIN1MT_GEMINI] arrêt : {raison_arret}")
                return None

            print(f"  [FIN1MT_GEMINI] scan t={t:.0f}s (fenêtre [{t_debut:.0f}s, {t_fin:.0f}s], pas={pas_scan}s)")
            decision = _voter_classification(client, video_path, t, tmp_dir, etat, model_name=model_name)

            if decision is None:
                # echec/incertitude persistante sur ce point : on avance sans
                # pouvoir le classer, ni AVANT ni APRES confirme
                t += pas_scan
                continue

            if not decision:
                dernier_avant_confirme = t
                t += pas_scan
                continue

            # Premier APRES confirme : dichotomie entre dernier_avant_confirme et t
            print(f"  [FIN1MT_GEMINI] premier APRES confirmé à t={t:.0f}s, "
                  f"dichotomie depuis dernier AVANT confirmé à t={dernier_avant_confirme:.0f}s")
            resultat = _recherche_fine_fin1mt(client, video_path, tmp_dir, etat, dernier_avant_confirme, t, model_name=model_name)
            print(f"  [FIN1MT_GEMINI] Fin1MT détecté à t={resultat:.0f}s")
            return float(resultat)

        print(f"  [FIN1MT_GEMINI] aucune transition APRES trouvée dans la fenêtre")
        return None
    finally:
        etat.fermer()
