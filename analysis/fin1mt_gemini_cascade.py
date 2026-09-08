"""
fin1mt_gemini_cascade.py
============================
V5.2 - Detection Fin1MT (fin de 1ere mi-temps) par vision Gemini,
architecture Q1/Q2 identique a KO2 (pas juste un signal AVANT/APRES
unique - premiere version abandonnee suite a un retour terrain :
"but visible" n'est pas un signal fiable, le vrai Fin1MT peut se jouer
pres d'un but aussi bien qu'un faux positif apres-but).

Q1 : signal DIRECTIONNEL - beaucoup de joueurs qui marchent vers le
bord du terrain (touche), plutot que de rester sur le terrain ou vers
le centre. C'est le signal specifiquement identifie comme fiable par
observation directe de 2 images comparees (vrai Fin1MT vs faux positif
apres-but - les deux avaient un but visible et des joueurs calmes,
SEULE la direction de marche differait).

Q2 : verification - le terrain est-il maintenant vide (ou quasiment)
DES JOUEURS DES DEUX EQUIPES DU MATCH specifiquement - tolere
explicitement la presence d'autres personnes (jeunes joueurs,
pom-pom girls, staff) qui peuvent occuper le terrain pendant la vraie
pause, sans que ca invalide la detection.

Recherche DANS LA FENETRE DEJA VALIDEE [KO2-marge_avant_min,
KO2-marge_apres_min] (validee 9/9 pour contenir le vrai Fin1MT dans le
travail R&D anterieur).

Reutilise l'infrastructure generique de kickoff_gemini_cascade.py -
AUCUNE modification de ce fichier, uniquement des imports.
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
# PROMPT Q1 — signal directionnel (candidat)
# ─────────────────────────────────────────────────────────────────────────
PROMPT_Q1_FIN1MT = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur, autour du moment supposé de la fin de la première mi-temps.

OBJECTIF : détecter un signe précoce que la fin de la première mi-temps vient d'être sifflée - PAS déterminer si le terrain est déjà vide, juste si un mouvement de sortie a commencé.

═══════════════════════════════════════════════════
CRITÈRE PRINCIPAL — SENS DE LA MARCHE
═══════════════════════════════════════════════════

Le critère décisif n'est PAS "les joueurs sont-ils calmes" (un arrêt de jeu normal, ou un instant juste après un but marqué, montrent aussi des joueurs calmes, y compris près d'un but). Le critère est : **PLUSIEURS joueurs marchent-ils vers le BORD du terrain (ligne de touche), plutôt que de rester sur le terrain ou de se diriger vers son centre ?**

- Si plusieurs joueurs (idéalement des deux équipes) sont clairement orientés/en mouvement vers une ligne de touche (peu importe laquelle) → OUI, signal de fin de mi-temps.
- Si les joueurs sont dispersés mais restent globalement SUR le terrain, ou se dirigent vers le centre (ex: après un but, pour se replacer) → NON, ce n'est probablement pas la fin de la mi-temps.
- Un seul joueur qui s'éloigne (ex: pour une touche, un ballon sorti) ne suffit pas - il faut un mouvement collectif vers la sortie.

Réponds STRICTEMENT en JSON, avec un raisonnement bref décrivant le sens de marche observé :
{"signal_sortie_detecte": true/false, "raisonnement": "..."}"""

# ─────────────────────────────────────────────────────────────────────────
# PROMPT Q2 — verification (terrain vide des 2 equipes du match)
# ─────────────────────────────────────────────────────────────────────────
PROMPT_Q2_FIN1MT = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur, pour vérifier si la pause de mi-temps est bien en cours.

OBJECTIF : déterminer si le terrain est maintenant vide, ou quasiment vide, DES JOUEURS DES DEUX ÉQUIPES DU MATCH (celles visibles avant cet instant, en tenue de match) - PAS déterminer si le terrain est totalement désert.

═══════════════════════════════════════════════════
IMPORTANT — TOLÉRANCE EXPLICITE
═══════════════════════════════════════════════════

D'autres personnes peuvent être présentes sur ou près du terrain SANS que cela invalide la pause de mi-temps :
- jeunes joueurs (enfants) qui utilisent le terrain pendant la pause
- pom-pom girls, animation, présentateur
- personnel du club, arbitres assistants, remplaçants au repos
Leur présence NE COMPTE PAS comme "le match est en cours" - seule la présence ou l'absence des JOUEURS DES DEUX ÉQUIPES DU MATCH (en tenue de match, ceux qui jouaient) compte pour ce critère.

═══════════════════════════════════════════════════
CRITÈRES
═══════════════════════════════════════════════════

- Terrain vide ou quasiment vide des joueurs des deux équipes du match → OUI (pause confirmée)
- Encore plusieurs joueurs des deux équipes du match visibles sur le terrain, en position de jeu ou clairement encore engagés dans le match → NON
- Uniquement d'autres personnes (enfants, pom-pom girls, staff) visibles, aucun joueur des équipes du match → OUI (pause confirmée, le terrain leur appartient pendant la pause)

Réponds STRICTEMENT en JSON, avec un raisonnement bref :
{"terrain_vide_des_2_equipes": true/false, "raisonnement": "..."}"""


def _q1_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME_DEFAUT):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q1_FIN1MT, etat, model_name=model_name)
    if result is None:
        return None, "échec API"
    return bool(result.get("signal_sortie_detecte", False)), result.get("raisonnement", "non fourni")


def _q2_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME_DEFAUT):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q2_FIN1MT, etat, model_name=model_name)
    if result is None:
        return None, "échec API"
    return bool(result.get("terrain_vide_des_2_equipes", False)), result.get("raisonnement", "non fourni")


def _voter(client, video_path, t, tmp_dir, etat, fonction_lecture, max_appels=3, model_name=MODEL_NAME_DEFAUT):
    """Vote majoritaire avec arret anticipe - meme principe que
    _voter_q2 de KO2."""
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


def _recherche_fine_fin1mt(client, video_path, tmp_dir, etat, t_avant, t_apres, model_name=MODEL_NAME_DEFAUT):
    """Dichotomie 15/5/1s, utilise le signal Q2 (terrain vide des 2
    equipes) - meme structure que _recherche_fine de KO2."""
    t_bas, t_haut = t_avant, t_apres
    for pas in PALIERS_RECHERCHE_FINE:
        tt = t_bas + pas
        dernier_non = t_bas
        while tt < t_haut:
            d, raisonnement = _q2_une_lecture(client, video_path, tt, tmp_dir, etat, model_name=model_name)
            print(f"    [FIN1MT FINE pas={pas}s] t={tt:.0f}s : {'VIDE' if d else 'PAS_VIDE' if d is not None else 'ERREUR'} — {raisonnement}")
            if d:
                t_haut = tt
                t_bas = dernier_non
                break
            dernier_non = tt
            tt += pas
        else:
            t_bas = dernier_non
    return t_haut


def find_fin1mt_gemini(video_path, ko2_s, marge_avant_min=16, marge_apres_min=5,
                        pas_scan=60, delai_verif_q2=60, model_name=MODEL_NAME_DEFAUT,
                        max_gemini_calls=MAX_GEMINI_CALLS_DEFAUT,
                        max_wallclock_s=MAX_WALLCLOCK_S_DEFAUT, tmp_dir="/tmp"):
    """
    Cherche Fin1MT par vision Gemini, architecture Q1 (signal
    directionnel de sortie) + Q2 (verification terrain vide des 2
    equipes) + dichotomie fine - meme structure que detect KO2.

    Fenetre [KO2-marge_avant_min, KO2-marge_apres_min] - PROCHE de la
    fenetre officielle historique (marge_apres_min=8 pour l'audio), mais
    REDUITE a 5 min : diagnostic reel (Andrimont, Goe) a montre que
    marge_apres_min=8 ne laissait que 16-18s apres le vrai Fin1MT avant
    la coupure de la fenetre - pas assez pour qu'un mouvement de sortie
    collectif ait le temps de devenir visuellement net. marge_apres_min=5
    garantit >=196s de marge sur les 9 matchs de reference (verifie).

    Retourne float (timestamp absolu) ou None si aucune transition
    confirmee trouvee dans la fenetre.
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
        dernier_non_confirme = t_debut
        while t <= t_fin:
            raison_arret = etat.budget_epuise()
            if raison_arret:
                print(f"  [FIN1MT_GEMINI] arrêt : {raison_arret}")
                return None

            print(f"  [FIN1MT_GEMINI] scan Q1 t={t:.0f}s (fenêtre [{t_debut:.0f}s, {t_fin:.0f}s], pas={pas_scan}s)")
            decision_q1, raisonnement_q1 = _voter(client, video_path, t, tmp_dir, etat, _q1_une_lecture, model_name=model_name)
            print(f"  [FIN1MT_GEMINI] Q1 à t={t:.0f}s : {'SIGNAL_SORTIE' if decision_q1 else 'NON' if decision_q1 is not None else 'ERREUR'} — {raisonnement_q1}")

            if not decision_q1:
                dernier_non_confirme = t
                t += pas_scan
                continue

            t_verif = t + delai_verif_q2
            if t_verif > t_fin:
                print(f"  [FIN1MT_GEMINI] candidat à t={t:.0f}s mais vérification hors limite")
                return None

            print(f"  [FIN1MT_GEMINI] candidat Q1 à t={t:.0f}s, vérif Q2 à t={t_verif:.0f}s...")
            decision_q2, raisonnement_q2 = _voter(client, video_path, t_verif, tmp_dir, etat, _q2_une_lecture, model_name=model_name)
            print(f"  [FIN1MT_GEMINI] Q2 à t={t_verif:.0f}s : {'VIDE' if decision_q2 else 'PAS_VIDE' if decision_q2 is not None else 'ERREUR'} — {raisonnement_q2}")

            if not decision_q2:
                print(f"  [FIN1MT_GEMINI] candidat rejeté, reprise à t={t_verif:.0f}s")
                t = t_verif
                continue

            # V5.2 FIX : la dichotomie doit pouvoir remonter AVANT le point
            # Q1 lui-meme (ou Q1 a detecte le mouvement collectif) jusqu'au
            # dernier point ou Q1 disait encore NON - car Q2 (terrain vide)
            # peut tres bien etre deja vrai plus tot que le moment ou le
            # mouvement de sortie devient assez net pour declencher Q1.
            # Diagnostic reel (Andrimont) : Q1 declenche a t=3064s mais le
            # vrai coup de sifflet est a t=3028s (36s plus tot) - sans ce
            # fix, la dichotomie ne peut jamais explorer cette marge.
            print(f"  [FIN1MT_GEMINI] confirmé, recherche fine dans "
                  f"[{dernier_non_confirme:.0f}s (dernier Q1=NON), {t_verif:.0f}s]...")
            resultat = _recherche_fine_fin1mt(client, video_path, tmp_dir, etat, dernier_non_confirme, t_verif, model_name=model_name)
            print(f"  [FIN1MT_GEMINI] Fin1MT détecté à t={resultat:.0f}s")
            return float(resultat)

        print(f"  [FIN1MT_GEMINI] aucun candidat Q1 trouvé dans la fenêtre")
        return None
    finally:
        etat.fermer()
