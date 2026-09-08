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
    obtenir_duree_video,
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
- Uniquement d'autres personnes (enfants, pom-pom girls, staff) visibles, aucun joueur des équipes du match → OUI (fin de match confirmée)
- DEUX ballons ou plus visibles simultanément sur le terrain → OUI (fin de match confirmée). Un vrai match ne se joue qu'avec UN SEUL ballon - la présence de plusieurs ballons est une preuve objective et certaine qu'il ne s'agit pas d'une phase de jeu réelle (échauffement informel, jeu libre après le match), quel que soit le nombre de joueurs présents ou leur niveau d'activité.
- MOINS DE 2 JOUEURS d'une des deux équipes visibles (une équipe a plusieurs joueurs présents, mais l'autre équipe n'a plus qu'un seul joueur isolé, ou zéro) → OUI (fin de match confirmée). Un vrai match implique la présence de PLUSIEURS joueurs de CHAQUE équipe simultanément - un seul joueur isolé d'une équipe (même très visible) ne compte PAS comme "l'équipe est encore là", c'est probablement un retardataire ou quelqu'un qui traîne, pas un signe que le match continue. Il faut au moins 2-3 joueurs identifiables de chaque équipe pour considérer que les deux équipes sont réellement encore présentes.
- AUCUN gardien visible à proximité des buts (si un but ou sa zone est visible dans l'image) → signe FAIBLE, à ne considérer QUE combiné avec d'autres signes (effectif réduit, un seul ballon, etc.) - JAMAIS suffisant à lui seul. ⚠️ Un gardien peut être temporairement absent de sa zone pour des raisons de jeu tout à fait normales (aller chercher le ballon sorti en corner pour un renvoi/coup de pied de but, dégagement lointain, etc.) - ce n'est PAS un signe de fin de match dans ces cas-là. N'utilise ce critère que si l'absence de gardien s'accompagne d'autres signes clairs (peu de joueurs, pas d'action de jeu généralisée).
- AUCUN arbitre visible nulle part dans l'image → signe FAIBLE, à ne considérer QUE combiné avec d'autres signes, JAMAIS suffisant à lui seul (l'arbitre peut être hors-cadre à un instant donné pendant un vrai match).
- Joueurs des deux équipes REGROUPÉS ENSEMBLE en un seul point du terrain pour discuter/socialiser (pas répartis sur le terrain en formation de jeu) → signe en faveur de OUI, même si plusieurs joueurs de chaque équipe et un seul ballon sont visibles. De vrais joueurs en match, même à l'arrêt (touche, faute), restent globalement RÉPARTIS sur le terrain selon leurs positions - un attroupement compact et informel de joueurs des deux équipes qui discutent ensemble ressemble plutôt à un rassemblement social (fin de match, pause) qu'à une phase de jeu, même sans ballon multiple ni effectif réduit.
- Effectif à peu près normal/complet, avec PLUSIEURS joueurs identifiables de CHAQUE équipe RÉPARTIS SUR LE TERRAIN en formation de jeu (pas regroupés ensemble pour discuter), UN SEUL ballon ou aucun visible (même sans action de jeu à cet instant précis - un match peut avoir des moments calmes : touche, discussion, arrêt de jeu) → NON (le match est très probablement encore en cours)
- Effectif visiblement REDUIT (nettement moins de joueurs qu'un effectif complet) ET aucune action de jeu active → OUI (fin de match confirmée)

⚠️ IMPORTANT : ne réponds OUI sur la base de "pas d'action de jeu" QUE SI le nombre de joueurs visibles est ÉGALEMENT nettement réduit par rapport à un effectif complet (SAUF si le critère des 2 ballons ou celui d'une seule équipe visible ci-dessus s'applique, chacun décisif à lui seul). Un effectif complet ou quasi-complet DES DEUX ÉQUIPES avec un seul ballon, même immobile à cet instant précis, ne suffit PAS à conclure à la fin du match - ça peut être un simple flottement de jeu.

Réponds STRICTEMENT en JSON, avec un raisonnement bref précisant : le nombre approximatif de joueurs visibles par équipe, s'ils sont répartis en formation de jeu ou regroupés/attroupés ensemble, le nombre de ballons visibles, la présence ou non d'un gardien près d'un but visible, la présence ou non d'un arbitre, ET s'il y a une action de jeu :
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

        # V5.2 FIX : ne jamais chercher au-dela de la duree reelle de la
        # video - decouvert en production (MineroisSter) : la fenetre
        # [KO2+40min, KO2+55min] peut depasser la duree reelle du
        # fichier, une extraction hors bornes fait echouer ffmpeg
        # silencieusement. Marge de securite de 5s.
        duree_video = obtenir_duree_video(video_path)
        if duree_video is not None and t_fin > duree_video - 5:
            t_fin_originale = t_fin
            t_fin = max(t_debut, duree_video - 5)
            print(f"  [FINMATCH_GEMINI] fenêtre limitée par la durée réelle de la vidéo "
                  f"({duree_video:.0f}s) : t_fin {t_fin_originale:.0f}s → {t_fin:.0f}s")

        if t_fin <= t_debut:
            return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}

        t = t_debut
        dernier_non_confirme = t_debut
        premier_signal_jamais_vu = None  # V5.2 : trace le tout premier
        # SIGNAL_SORTIE vu (meme rejete ensuite par Q2) - sert de
        # meilleure borne inferieure de repli que dernier_non_confirme,
        # qui peut se retrouver APRES le vrai evenement si le scan
        # continue au-dela (cas Goe : dernier NON a t=7411s, mais vrai
        # FinMatch=7401s, donc dernier_non_confirme seul aurait rate la
        # fenetre utile).
        while t <= t_fin:
            raison_arret = etat.budget_epuise()
            if raison_arret:
                print(f"  [FINMATCH_GEMINI] arrêt : {raison_arret}")
                return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}

            print(f"  [FINMATCH_GEMINI] scan Q1 t={t:.0f}s (fenêtre [{t_debut:.0f}s, {t_fin:.0f}s], pas={pas_scan}s)")
            decision_q1, raisonnement_q1 = _voter(client, video_path, t, tmp_dir, etat, _q1_une_lecture, model_name=model_name)
            print(f"  [FINMATCH_GEMINI] Q1 à t={t:.0f}s : {'SIGNAL_SORTIE' if decision_q1 else 'NON' if decision_q1 is not None else 'ERREUR'} — {raisonnement_q1}")

            if decision_q1 and premier_signal_jamais_vu is None:
                premier_signal_jamais_vu = t

            if not decision_q1:
                dernier_non_confirme = t
                t += pas_scan
                continue

            t_verif = t + delai_verif_q2
            if t_verif > t_fin:
                # V5.2 FIX : plutot que de rejeter purement et simplement
                # (perte du candidat, meme s'il etait bon - observe en
                # production sur MineroisSter : candidat legitime a
                # t=7240s, tres proche du vrai FinMatch=7192s, rejete a
                # tort faute des 90s complets avant la fin reelle de la
                # video), on verifie au plus pres de la fin disponible.
                # MARGE_MIN_VERIF_S : sous ce seuil, vraiment pas assez
                # de marge pour verifier quoi que ce soit d'utile.
                MARGE_MIN_VERIF_S = 5
                if t_fin - t < MARGE_MIN_VERIF_S:
                    print(f"  [FINMATCH_GEMINI] candidat à t={t:.0f}s mais marge insuffisante "
                          f"même en plafonnant ({t_fin-t:.0f}s < {MARGE_MIN_VERIF_S}s)")
                    return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}
                print(f"  [FINMATCH_GEMINI] délai de vérification plafonné à la fin de la vidéo : "
                      f"t_verif {t_verif:.0f}s → {t_fin:.0f}s ({t_fin-t:.0f}s de marge au lieu de {delai_verif_q2}s)")
                t_verif = t_fin

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

        print(f"  [FINMATCH_GEMINI] aucun candidat Q1 confirmé dans la fenêtre")

        # V5.2 FIX : plutot que d'essayer de deviner une borne inferieure
        # fiable (le "vrai" instant n'est de toute facon jamais observable
        # directement, seulement deductible de ce qui se passe apres),
        # on balaie simplement en AVANT avec Q2 (pas Q1) depuis le
        # dernier point connu jusqu'a la fin de la video, a la recherche
        # du premier moment ou le terrain est reellement vide. On accepte
        # un leger biais en retard plutot que de deviner une fourchette
        # de dichotomie qui peut etre completement fausse (cas Goe :
        # premier_signal_jamais_vu=6841s etait beaucoup trop loin en
        # arriere pour etre une bonne borne).
        if duree_video is not None and dernier_non_confirme < duree_video:
            borne_sup = min(duree_video - 5, duree_video)
            if borne_sup > dernier_non_confirme:
                print(f"  [FINMATCH_GEMINI] repli : balayage Q2 en avant depuis "
                      f"{dernier_non_confirme:.0f}s jusqu'à {borne_sup:.0f}s")
                PAS_BALAYAGE_REPLI = 15
                dernier_pas_vide = dernier_non_confirme
                tt = dernier_non_confirme + PAS_BALAYAGE_REPLI
                trouve = None
                while tt <= borne_sup:
                    d, raisonnement = _q2_une_lecture(client, video_path, tt, tmp_dir, etat, model_name=model_name)
                    print(f"    [REPLI pas={PAS_BALAYAGE_REPLI}s] t={tt:.0f}s : "
                          f"{'VIDE' if d else 'PAS_VIDE' if d is not None else 'ERREUR'} — {raisonnement}")
                    if d:
                        trouve = tt
                        break
                    dernier_pas_vide = tt
                    tt += PAS_BALAYAGE_REPLI
                if trouve is not None:
                    # affinage 5s/1s entre dernier_pas_vide et trouve, EN
                    # UTILISANT Q2 (pas Q1) - coherent avec le balayage
                    # ci-dessus, qui cherche "terrain vide", pas "signal
                    # de sortie".
                    t_bas, t_haut = dernier_pas_vide, trouve
                    for pas in (5, 1):
                        tt2 = t_bas + pas
                        dernier_non2 = t_bas
                        while tt2 < t_haut:
                            d2, raisonnement2 = _q2_une_lecture(client, video_path, tt2, tmp_dir, etat, model_name=model_name)
                            print(f"    [REPLI FINE pas={pas}s] t={tt2:.0f}s : "
                                  f"{'VIDE' if d2 else 'PAS_VIDE' if d2 is not None else 'ERREUR'} — {raisonnement2}")
                            if d2:
                                t_haut = tt2
                                t_bas = dernier_non2
                                break
                            dernier_non2 = tt2
                            tt2 += pas
                        else:
                            t_bas = dernier_non2
                    resultat = t_haut
                    print(f"  [FINMATCH_GEMINI] FinMatch détecté (via repli) à t={resultat:.0f}s")
                    return {"finmatch_s": float(resultat), "n_appels_gemini": etat.n_appels}
                print(f"  [FINMATCH_GEMINI] repli : jamais VIDE jusqu'à la fin de la vidéo, "
                      f"utilisation de la fin de la vidéo ({duree_video:.0f}s) comme estimation")
                return {"finmatch_s": duree_video, "n_appels_gemini": etat.n_appels}
            print(f"  [FINMATCH_GEMINI] repli : utilisation de la fin de la vidéo "
                  f"({duree_video:.0f}s) comme estimation de FinMatch (pas de marge)")
            return {"finmatch_s": duree_video, "n_appels_gemini": etat.n_appels}

        return {"finmatch_s": None, "n_appels_gemini": etat.n_appels}
    finally:
        etat.fermer()
