"""
finmatch_gemini_cascade.py
===============================
V5.2 - Detection FinMatch (fin de match / fin de 2e mi-temps) par
vision Gemini, architecture IDENTIQUE a fin1mt_gemini_cascade.py (Q1
signal directionnel/degarni + Q2 verification terrain vide, dichotomie
Q1 elargie jusqu'au dernier NON confirme).

⚠️ Ne gere PAS les prolongations, tirs au but, etc. - fenetre calibree
pour un temps reglementaire standard (2x periodes). A traiter separement
si necessaire (cf. items GELES du roadmap V5.2).

Recherche dans la fenetre [KO2+marge_avant_min, KO2+marge_apres_min] -
fenetre officielle historique [KO2+40min, KO2+55min], deja validee 9/9
pour contenir le vrai FinMatch, verifiee avec >=248s de marge sur les 9
matchs de reference (pas besoin de reduction comme pour Fin1MT, la
marge etait deja largement suffisante).

Reutilise l'infrastructure generique de kickoff_gemini_cascade.py -
AUCUNE modification de ce fichier ni de fin1mt_gemini_cascade.py,
uniquement des imports.
"""

from analysis.kickoff_gemini_cascade import (
    _EtatRecherche,
    _appeler_json_robuste,
    PALIERS_RECHERCHE_FINE,
    MAX_GEMINI_CALLS_DEFAUT,
    MAX_WALLCLOCK_S_DEFAUT,
)

MODEL_NAME_DEFAUT = "gemini-3.5-flash"

# ─────────────────────────────────────────────────────────────────────────
# PROMPTS — IDENTIQUES a fin1mt_gemini_cascade.py (meme transition
# conceptuelle : jeu actif -> joueurs quittent le terrain)
# ─────────────────────────────────────────────────────────────────────────
PROMPT_Q1_FINMATCH = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur, autour du moment supposé de la fin du match (fin de la 2e mi-temps).

OBJECTIF : détecter un signe précoce que la fin du match vient d'être sifflée - PAS déterminer si le terrain est déjà vide, juste si un mouvement de sortie a commencé.

═══════════════════════════════════════════════════
CRITÈRE PRINCIPAL — SENS DE LA MARCHE, OU TERRAIN DÉJÀ DÉGARNI
═══════════════════════════════════════════════════

Le critère décisif n'est PAS "les joueurs sont-ils calmes" (un arrêt de jeu normal, ou un instant juste après un but marqué, montrent aussi des joueurs calmes, y compris près d'un but). Réponds OUI si L'UN OU L'AUTRE des deux signaux suivants est présent :

SIGNAL A — MOUVEMENT DE SORTIE : plusieurs joueurs (idéalement des deux équipes) sont clairement orientés/en mouvement vers une ligne de touche (peu importe laquelle), plutôt que de rester sur le terrain ou de se diriger vers son centre.

SIGNAL B — TERRAIN DÉJÀ DÉGARNI : le nombre de joueurs visibles sur le terrain est nettement inférieur à un effectif de match complet (moins de la moitié des joueurs habituels), ET ceux qui restent ne sont pas dans une configuration de jeu actif (pas de ballon disputé, pas de course de jeu). Ce signal capte le cas où la sortie a déjà eu lieu avant cette image - tu n'as pas besoin de voir le mouvement lui-même, juste constater que le terrain est déjà nettement plus vide qu'un terrain de match normal.

SIGNAL C — POIGNÉES DE MAIN / SALUT DE FIN DE MATCH : plusieurs joueurs des deux équipes se serrent la main, se félicitent, s'alignent pour se saluer, ou se regroupent de manière non liée au jeu (accolades, échanges de maillots) - même s'ils sont encore regroupés au centre ou n'ont pas commencé à marcher vers la sortie. C'est un rituel de fin de match qui survient généralement juste après le coup de sifflet final, avant même que le mouvement de sortie ne commence. Ce signal est spécifique à la fin de MATCH (n'existe pas à la mi-temps).

Si AUCUN des 3 signaux n'est présent (terrain avec un effectif normal, joueurs qui restent sur le terrain en configuration de jeu, pas de rituel de fin visible) → NON.
Un seul joueur qui s'éloigne (ex: pour une touche, un ballon sorti) ne suffit pas pour le signal A - il faut un mouvement collectif. Mais un terrain visiblement clairsemé suffit pour le signal B, et une poignée de main collective suffit pour le signal C, même sans mouvement de sortie visible.

Réponds STRICTEMENT en JSON, en précisant lequel des 3 signaux (A, B, C, plusieurs, ou aucun) a motivé ta réponse :
{"signal_sortie_detecte": true/false, "signal_utilise": "A"|"B"|"C"|"aucun", "raisonnement": "..."}"""

PROMPT_Q2_FINMATCH = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur, pour vérifier si la fin du match est bien confirmée.

OBJECTIF : déterminer si le terrain est maintenant vide, ou quasiment vide, DES JOUEURS DES DEUX ÉQUIPES DU MATCH (celles visibles avant cet instant, en tenue de match) - PAS déterminer si le terrain est totalement désert.

═══════════════════════════════════════════════════
IMPORTANT — TOLÉRANCE EXPLICITE
═══════════════════════════════════════════════════

D'autres personnes peuvent être présentes sur ou près du terrain SANS que cela invalide la fin du match :
- jeunes joueurs (enfants) qui utilisent le terrain après le match
- pom-pom girls, animation, présentateur
- personnel du club, arbitres assistants, remplaçants
Leur présence NE COMPTE PAS comme "le match est en cours" - seule la présence ou l'absence des JOUEURS DES DEUX ÉQUIPES DU MATCH (en tenue de match, ceux qui jouaient) compte pour ce critère.

═══════════════════════════════════════════════════
CRITÈRES
═══════════════════════════════════════════════════

- Terrain vide ou quasiment vide des joueurs des deux équipes du match → OUI (fin de match confirmée)
- Encore plusieurs joueurs des deux équipes du match visibles sur le terrain, en position de jeu ou clairement encore engagés dans le match → NON
- Uniquement d'autres personnes (enfants, pom-pom girls, staff) visibles, aucun joueur des équipes du match → OUI (fin de match confirmée)

Réponds STRICTEMENT en JSON, avec un raisonnement bref :
{"terrain_vide_des_2_equipes": true/false, "raisonnement": "..."}"""


def _q1_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME_DEFAUT):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q1_FINMATCH, etat, model_name=model_name)
    if result is None:
        return None, "échec API"
    signal_utilise = result.get("signal_utilise", "?")
    raisonnement = f"[signal={signal_utilise}] {result.get('raisonnement', 'non fourni')}"
    return bool(result.get("signal_sortie_detecte", False)), raisonnement


def _q2_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME_DEFAUT):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q2_FINMATCH, etat, model_name=model_name)
    if result is None:
        return None, "échec API"
    return bool(result.get("terrain_vide_des_2_equipes", False)), result.get("raisonnement", "non fourni")


def _voter(client, video_path, t, tmp_dir, etat, fonction_lecture, max_appels=3, model_name=MODEL_NAME_DEFAUT):
    """Vote majoritaire avec arret anticipe - identique a fin1mt."""
    votes = []
    dernier_raisonnement = None
    for _ in range(max_appels):
        v, raisonnement = fonction_lecture(client, video_path, t, tmp_dir, etat, model_name=model_name)
        dernier_raisonnement = raisonnement
        if v is None:
            if not votes:
                return None, raisonnement
            break
        votes.append(v)
        n_true = sum(votes)
        n_false = len(votes) - n_true
        restants = max_appels - len(votes)
        if n_true > n_false + restants or n_false > n_true + restants:
            break
    if not votes:
        return None, dernier_raisonnement
    return (sum(votes) > len(votes) / 2), dernier_raisonnement


def _recherche_fine_finmatch(client, video_path, tmp_dir, etat, t_avant, t_apres, model_name=MODEL_NAME_DEFAUT):
    """Dichotomie 15/5/1s utilisant Q1 (signal large), pas Q2 (trop
    strict) - identique a fin1mt_gemini_cascade.py."""
    t_bas, t_haut = t_avant, t_apres
    for pas in PALIERS_RECHERCHE_FINE:
        tt = t_bas + pas
        dernier_non = t_bas
        while tt < t_haut:
            d, raisonnement = _q1_une_lecture(client, video_path, tt, tmp_dir, etat, model_name=model_name)
            print(f"    [FINMATCH FINE pas={pas}s] t={tt:.0f}s : {'SORTIE' if d else 'PAS_ENCORE' if d is not None else 'ERREUR'} — {raisonnement}")
            if d:
                t_haut = tt
                t_bas = dernier_non
                break
            dernier_non = tt
            tt += pas
        else:
            t_bas = dernier_non
    return t_haut


def find_finmatch_gemini(video_path, ko2_s, marge_avant_min=40, marge_apres_min=55,
                          pas_scan=60, delai_verif_q2=90, model_name=MODEL_NAME_DEFAUT,
                          max_gemini_calls=MAX_GEMINI_CALLS_DEFAUT,
                          max_wallclock_s=MAX_WALLCLOCK_S_DEFAUT, tmp_dir="/tmp"):
    """
    Cherche FinMatch par vision Gemini - architecture identique a
    find_fin1mt_gemini (Q1 large + Q2 verification + dichotomie Q1).

    ⚠️ Ne gere PAS prolongations/tirs au but - fenetre calibree pour un
    temps reglementaire standard.

    Fenetre [KO2+marge_avant_min, KO2+marge_apres_min] - fenetre
    officielle historique, verifiee >=248s de marge sur les 9 matchs de
    reference (pas de reduction necessaire, contrairement a Fin1MT).

    Retourne un dict {"finmatch_s": float|None, "n_appels_gemini": int}.
    """
    from google import genai
    client = genai.Client()

    etat = _EtatRecherche(max_gemini_calls, max_wallclock_s)
    try:
        t_debut = ko2_s + marge_avant_min * 60
        t_fin = ko2_s + marge_apres_min * 60
        if t_fin <= t_debut:
            return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}

        t = t_debut
        dernier_non_confirme = t_debut
        while t <= t_fin:
            raison_arret = etat.budget_epuise()
            if raison_arret:
                print(f"  [FINMATCH_GEMINI] arrêt : {raison_arret}")
                return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}

            print(f"  [FINMATCH_GEMINI] scan Q1 t={t:.0f}s (fenêtre [{t_debut:.0f}s, {t_fin:.0f}s], pas={pas_scan}s)")
            decision_q1, raisonnement_q1 = _voter(client, video_path, t, tmp_dir, etat, _q1_une_lecture, model_name=model_name)
            print(f"  [FINMATCH_GEMINI] Q1 à t={t:.0f}s : {'SIGNAL_SORTIE' if decision_q1 else 'NON' if decision_q1 is not None else 'ERREUR'} — {raisonnement_q1}")

            if not decision_q1:
                dernier_non_confirme = t
                t += pas_scan
                continue

            t_verif = t + delai_verif_q2
            if t_verif > t_fin:
                print(f"  [FINMATCH_GEMINI] candidat à t={t:.0f}s mais vérification hors limite")
                return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}

            print(f"  [FINMATCH_GEMINI] candidat Q1 à t={t:.0f}s, vérif Q2 à t={t_verif:.0f}s...")
            decision_q2, raisonnement_q2 = _voter(client, video_path, t_verif, tmp_dir, etat, _q2_une_lecture, model_name=model_name)
            print(f"  [FINMATCH_GEMINI] Q2 à t={t_verif:.0f}s : {'VIDE' if decision_q2 else 'PAS_VIDE' if decision_q2 is not None else 'ERREUR'} — {raisonnement_q2}")

            if not decision_q2:
                print(f"  [FINMATCH_GEMINI] candidat rejeté, reprise à t={t_verif:.0f}s")
                t = t_verif
                continue

            print(f"  [FINMATCH_GEMINI] confirmé, recherche fine dans "
                  f"[{dernier_non_confirme:.0f}s (dernier Q1=NON), {t_verif:.0f}s]...")
            resultat = _recherche_fine_finmatch(client, video_path, tmp_dir, etat, dernier_non_confirme, t_verif, model_name=model_name)
            print(f"  [FINMATCH_GEMINI] FinMatch détecté à t={resultat:.0f}s")
            return {"finmatch_s": float(resultat), "n_appels_gemini": etat.n_appels}

        print(f"  [FINMATCH_GEMINI] aucun candidat Q1 trouvé dans la fenêtre")
        return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}
    finally:
        etat.fermer()
