# analysis/kickoff_gemini_cascade.py
# -*- coding: utf-8 -*-
#
# Détection du premier coup d'envoi par cascade Gemini — V5.2.
#
# Remplace kickoff_detector.py (LEGACY, désactivé — heuristique
# "groups[-1]" destructive documentée, voir V5_2_FIABILITE_ROADMAP.md §7).
#
# Architecture validée sur les 9 matchs de référence, 2 runs indépendants
# (9/9 les deux fois, écart max 5,2s, jamais en avance) — voir
# V5_2_FIABILITE_ROADMAP.md §12. Code directement dérivé de
# notebook_test_rigoureux_rapide_9matchs.py, prompts et seuils identiques.
#
# ⚠️ NE PAS REFORMULER LES PROMPTS (PROMPT_Q1, PROMPT_Q2) SANS RAISON
# PRÉCISE ET DÉMONTRÉE. Formulations éprouvées sur 9 matchs x 2 runs —
# une reformulation "pour simplifier" a déjà fait perdre un run complet
# payant (Franchimont/Goe jamais détectés avec un texte raccourci) et
# une régression a fait rater le KO de plus de 100s sur 3 matchs
# (Q2 trop strict, ne comptait pas corner/coup franc comme "match
# commencé"). Voir V5_2_FIABILITE_ROADMAP.md §12.2 pour l'historique
# complet des itérations et leurs échecs.
#
# CONTRAT V1 (§12.9) :
#   - Premier coup d'envoi uniquement. PAS le second (reprise après
#     mi-temps) — hors périmètre, module séparé à construire plus tard.
#   - Statuts explicites : AUTO_CONFIRMED / NOT_FOUND / ERROR.
#     Pas de MANUAL_REQUIRED (aucun cas métier défini pour le distinguer
#     de NOT_FOUND). Pas de champ confidence (jamais calculé dans cette
#     architecture, ne pas promettre ce qui n'existe pas).
#   - NOT_FOUND ou ERROR ne doivent JAMAIS être transformés en un faux
#     timestamp par l'appelant. C'est le principe central de ce module :
#     ne pas fabriquer une réponse quand les preuves sont insuffisantes.

import os
import time
import json
import subprocess
import concurrent.futures

MODEL_NAME = "gemini-3.1-pro-preview"  # PAS gemini-2.5-flash (jamais validé, voir §12.2)
TEMPERATURE = 0.0

# ── Garde-fous de production (§12.9, §12.12) ────────────────────────────
# Constantes fixes, non exposées à l'utilisateur — protègent le pipeline
# et le budget, indépendamment de max_search_s (choix utilisateur).
MAX_GEMINI_CALLS_DEFAUT = 300   # budget d'appels ; le retry robuste peut
                                 # multiplier les appels par checkpoint
MAX_WALLCLOCK_S_DEFAUT  = 900   # 15min ; independant du budget d'appels -
                                 # protege contre une video lente qui reste
                                 # dans son budget mais bloque le pipeline

TIMEOUT_S    = 60   # par tentative d'appel Gemini
MAX_WORKERS  = 32   # PAS 8 : la parallelisation du scan Q1 combinee au
                     # retry robuste peut demander jusqu'a 36 requetes
                     # simultanees dans le pire cas - 8 causait une
                     # cascade de faux timeouts (§12.7)

SEUIL_Q1 = 3  # sur 4 criteres, PAS un ET strict (5/5 donnait 0 OUI sur
              # tout un match, voir §12.2)

PALIERS_RECHERCHE_FINE = [15, 5, 1]
TAILLE_LOT_Q1 = 6


# ─────────────────────────────────────────────────────────────────────────
# PROMPTS — NE PAS REFORMULER (voir avertissement en tête de fichier)
# ─────────────────────────────────────────────────────────────────────────

PROMPT_Q1_RIGOUREUX = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur.

OBJECTIF : déterminer si cette image montre une scène d'AVANT-MATCH EN COURS DE MISE EN PLACE vers un coup d'envoi - PAS nécessairement l'instant exact où tout est déjà prêt. L'image peut être prise plusieurs dizaines de secondes avant le coup d'envoi réel, pendant que les joueurs arrivent encore ou se placent progressivement - c'est normal et acceptable.

Ne rejette PAS une image simplement parce que :
- des joueurs sont encore en train d'arriver ou de se placer
- l'effectif n'est pas encore complet
- la scène n'est pas parfaitement stable/figée

═══════════════════════════════════════════════════
VÉRIFICATIONS PRÉALABLES OBLIGATOIRES — À FAIRE EN PREMIER
═══════════════════════════════════════════════════

Si l'une des deux réponses ci-dessous est "oui", réponds directement "NON" à tout le reste, sans analyser davantage.

QUESTION PRÉALABLE 1 — Un but (cage, poteaux, filet) est-il visible dans l'image ?
Si oui → NON automatique.

QUESTION PRÉALABLE 2 — Le jeu est-il MANIFESTEMENT déjà actif (joueurs en mouvement de jeu réel, ballon en circulation loin du centre) ?
Si oui → NON automatique (ce n'est plus une scène d'avant-match, c'est déjà du jeu).

═══════════════════════════════════════════════════
CRITÈRES (jugés ensemble, pas un ET strict)
═══════════════════════════════════════════════════

1. ZONE CENTRALE PLAUSIBLE : la scène se situe-t-elle dans ou près de la zone centrale du terrain (rond central visible, OU ligne médiane visible, OU absence d'éléments indiquant clairement une autre zone comme un but/corner/surface de réparation) ?

2. CARACTÈRE "AVANT-MATCH" GÉNÉRAL : la scène ressemble-t-elle à une phase de préparation avant un début de jeu (joueurs qui se placent, arrivent, s'organisent, discutent, ou attendent), plutôt qu'à une phase de jeu actif ou un arrêt de jeu localisé (corner, touche, coup franc excentré) ?

3. AU MOINS UNE AMORCE DE SÉPARATION PAR ÉQUIPE : y a-t-il un signe, même partiel, que les deux équipes tendent à se répartir chacune de leur côté (pas obligatoirement terminé, mais visible en tendance) - plutôt que les deux équipes complètement mélangées au même endroit ?

4. PAS UNE AUTRE REMISE EN JEU LOCALISÉE : la scène n'est pas clairement un corner, une touche, ou un coup franc loin du centre.

Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :
{"zone_centrale_plausible": true/false, "caractere_avant_match": true/false, "amorce_separation": true/false, "pas_autre_remise_en_jeu": true/false}"""


# ─────────────────────────────────────────────────────────────────────────
# V5.2 Phase A — variante Q1 DEDIEE A KO2, PAS utilisee pour KO1
# ─────────────────────────────────────────────────────────────────────────
# ⚠️ Le prompt Q1 original (ci-dessus) est PARTAGE et deja valide 9/9 pour
# KO1 - ne pas le modifier sans revalidation complete (cf. avertissement
# en tete de fichier : une reformulation a deja fait perdre un run
# complet).
#
# Le critere "deux_equipes_visibles" ci-dessous a ete decouvert et valide
# specifiquement sur des faux positifs KO2 (P1Minerois : une seule equipe
# en echauffement pres de la touche satisfaisait les 4 criteres
# originaux). CE CRITERE FAISAIT PARTIE DU TEST QUI A VALIDE KO2 9/9 -
# ce n'est pas une amelioration optionnelle, c'est une piece necessaire
# du comportement deja valide, qui n'avait par erreur jamais ete portee
# en production avant ce fix (bug trouve suite a un audit demande
# apres un run de production divergent du test).
#
# Utilise UNIQUEMENT par find_ko2_gemini (match_boundaries_v2.py) - KO1
# continue d'utiliser PROMPT_Q1_RIGOUREUX + _q1_une_lecture originaux,
# intacts.
PROMPT_Q1_KO2 = PROMPT_Q1_RIGOUREUX.replace(
    'Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :\n{"zone_centrale_plausible": true/false, "caractere_avant_match": true/false, "amorce_separation": true/false, "pas_autre_remise_en_jeu": true/false}',
    '''═══════════════════════════════════════════════════
CRITÈRE OBLIGATOIRE SUPPLÉMENTAIRE (à part, pas optionnel)
═══════════════════════════════════════════════════

5. DEUX ÉQUIPES DISTINCTES VISIBLES : vois-tu clairement des joueurs des DEUX équipes (deux couleurs de maillot différentes, hors gardien) ? Une scène montrant une seule équipe en train de s'échauffer ou de se replacer (même avec son propre gardien dans une 3e couleur) NE COMPTE PAS - il faut voir les deux équipes adverses en même temps. Si tu ne vois qu'une seule couleur de maillot de champ (+ éventuellement un gardien), réponds false ici, même si les autres critères semblent satisfaits.

Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :
{"zone_centrale_plausible": true/false, "caractere_avant_match": true/false, "amorce_separation": true/false, "pas_autre_remise_en_jeu": true/false, "deux_equipes_visibles": true/false}'''
).replace(
    # V5.2 FIX (vigilance signalee sur le JSON strict) : la formulation
    # originale "reponds directement NON a tout le reste" pouvait etre
    # mal interpretee par le modele comme "renvoie le texte NON" au lieu
    # du JSON attendu, ce qui aurait fait echouer le parsing JSON en
    # aval. Clarifie UNIQUEMENT pour KO2 - le prompt Q1 original (KO1)
    # n'est pas touche, cette formulation potentiellement ambigue y
    # reste identique a ce qui est deja valide 9/9.
    'Si l\'une des deux réponses ci-dessous est "oui", réponds directement "NON" à tout le reste, sans analyser davantage.',
    'Si l\'une des deux réponses ci-dessous est "oui", réponds immédiatement avec un JSON complet où TOUTES les valeurs (y compris deux_equipes_visibles) sont à false - ne réponds JAMAIS par le simple texte "NON", toujours le JSON structuré complet.'
).replace(
    # V5.2 FIX : ajout d'une 3e verification prealable - fanion de corner
    # visible = signal visuel net et concret (comme le but), decouvert
    # sur un faux positif reel (P1Minerois t=3652s : fanion de corner
    # visible en arriere-plan, scene de coup franc pres de la touche
    # acceptee a tort comme candidat KO2). UNIQUEMENT pour KO2 - prompt
    # KO1 non touche.
    "QUESTION PRÉALABLE 2 — Le jeu est-il MANIFESTEMENT déjà actif (joueurs en mouvement de jeu réel, ballon en circulation loin du centre) ?\nSi oui → NON automatique (ce n'est plus une scène d'avant-match, c'est déjà du jeu).",
    "QUESTION PRÉALABLE 2 — Le jeu est-il MANIFESTEMENT déjà actif (joueurs en mouvement de jeu réel, ballon en circulation loin du centre) ?\nSi oui → NON automatique (ce n'est plus une scène d'avant-match, c'est déjà du jeu).\n\nQUESTION PRÉALABLE 3 — Un fanion de corner APPARTENANT AU TERRAIN SUR LEQUEL SE JOUE LE MATCH (piquet avec petit drapeau, à l'un des 4 coins DE CE terrain) est-il visible dans l'image ? ⚠️ Attention : si plusieurs terrains sont visibles côte à côte (complexe multi-terrains), ignore les fanions d'un terrain VOISIN - seul un fanion appartenant au terrain où se déroule CE match compte. En cas de doute sur l'appartenance du fanion à ce terrain précis, ne le compte pas.\nSi un fanion du terrain de jeu est identifié avec certitude → NON automatique (la scène se situe près d'un coin de ce terrain, pas au centre)."
).replace(
    # V5.2 FIX : ajout d'un critere de NOMBRE DE JOUEURS - decouvert sur un
    # faux positif reel (Franchimont/Flash t=3438s : seulement 3 personnes
    # visibles - 2 joueurs de la MEME equipe en echauffement + 1 gardien
    # confondu avec un arbitre - acceptees a tort comme "deux equipes
    # separees de part et d'autre du centre"). Une vraie mise en place de
    # coup d'envoi implique un nombre significatif de joueurs des deux
    # equipes (typiquement plusieurs par equipe qui convergent/se
    # placent), pas 2-3 personnes isolees qui peuvent trivialement
    # sembler "deux equipes" par la seule couleur. UNIQUEMENT KO2 - KO1
    # non touche.
    'Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :\n{"zone_centrale_plausible": true/false, "caractere_avant_match": true/false, "amorce_separation": true/false, "pas_autre_remise_en_jeu": true/false, "deux_equipes_visibles": true/false}',
    '''6. NOMBRE DE JOUEURS SUFFISANT : compte le nombre total de joueurs visibles dans l'image (hors arbitre). Une vraie mise en place de coup d'envoi implique la présence d'un nombre significatif de joueurs des deux équipes convergeant ou se plaçant vers le centre - PAS seulement 1 ou 2 joueurs isolés par équipe. Si tu ne comptes que 2-3 joueurs au total sur toute l'image (même avec des couleurs différentes), ce n'est PAS une mise en place de coup d'envoi complète - c'est probablement un échauffement localisé, une discussion entre quelques joueurs, ou une scène similaire. Indique le nombre de joueurs compté, et réponds false à ce critère si tu comptes moins de 5 joueurs au total.

Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :
{"zone_centrale_plausible": true/false, "caractere_avant_match": true/false, "amorce_separation": true/false, "pas_autre_remise_en_jeu": true/false, "deux_equipes_visibles": true/false, "nombre_joueurs_compte": <entier>, "nombre_joueurs_suffisant": true/false}'''
)
assert PROMPT_Q1_KO2 != PROMPT_Q1_RIGOUREUX, "Le remplacement du prompt KO2 a échoué (texte cible introuvable) - vérifier que PROMPT_Q1_RIGOUREUX n'a pas changé de formulation"

PROMPT_Q2_RIGOUREUX = """Tu vas analyser UNE SEULE image extraite d'une vidéo de match de football amateur.

OBJECTIF : déterminer si le jeu est MANIFESTEMENT en cours dans cette image - y compris si le jeu est momentanément à l'arrêt pour une raison de match (pas nécessairement en mouvement à cet instant précis). Un jugement strict, pas une impression.

═══════════════════════════════════════════════════
CRITÈRES D'ACCEPTATION (au moins un doit être clairement observable)
═══════════════════════════════════════════════════

1. Le ballon est visiblement en mouvement, loin de la zone centrale.

2. Un arrêt de jeu RECONNAISSABLE comme faisant partie du match en cours - ces situations COMPTENT comme "jeu déjà commencé", même si personne ne bouge à cet instant précis :
   - CORNER : ballon posé près d'un coin du terrain
   - COUP FRANC : joueurs formant un mur, ballon fixe loin du centre
   - TOUCHE : ballon ou joueur près d'une ligne de touche, joueur s'apprêtant à relancer à deux mains au-dessus de la tête
   - PENALTY : ballon sur le point de penalty, gardien seul dans sa surface face au tireur
   - faute sifflée, joueur au sol, célébration après un but

3. Un seul ballon est visible sur le terrain ET les joueurs sont clairement en action de jeu ou en position de reprise (pas dispersés sans logique de jeu).

═══════════════════════════════════════════════════
INDICE IMPORTANT - NOMBRE DE BALLONS
═══════════════════════════════════════════════════

Si tu observes PLUSIEURS ballons sur le terrain simultanément, c'est un signe fort d'ÉCHAUFFEMENT, pas de jeu réel (un match n'utilise qu'un seul ballon à la fois). Dans ce cas, réponds NON même si les joueurs semblent actifs ou en mouvement.

═══════════════════════════════════════════════════
NE RÉPONDS PAS "oui" SIMPLEMENT PARCE QUE
═══════════════════════════════════════════════════

- les joueurs sont dispersés sans formation claire (peut être de l'échauffement, une entrée sur le terrain, ou une animation d'avant-match)
- la scène montre de l'agitation générale sans preuve claire qu'il s'agit du match lui-même (présentation, célébration hors-jeu, public, pom-pom girls)
- un seul joueur bouge légèrement

Réponds "non" si la scène ressemble à : échauffement dispersé (souvent plusieurs ballons visibles), entrée des joueurs sur le terrain, présentation ou animation avant-match, ou une formation de coup d'envoi en préparation.

Réponds STRICTEMENT en JSON, sans texte avant ni après, sans balises markdown :
{"match_deja_commence": true/false}"""


# ─────────────────────────────────────────────────────────────────────────
# ÉTAT INTERNE DE RECHERCHE — regroupe compteurs/horloge/executor pour
# éviter les globals (ce module peut être appelé plusieurs fois en
# parallèle sur des matchs différents dans le pipeline)
# ─────────────────────────────────────────────────────────────────────────

class _EtatRecherche:
    def __init__(self, max_gemini_calls, max_wallclock_s):
        self.n_appels = 0
        self.t_debut = time.monotonic()
        self.max_gemini_calls = max_gemini_calls
        self.max_wallclock_s = max_wallclock_s
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def budget_epuise(self):
        if self.n_appels >= self.max_gemini_calls:
            return "CALL_BUDGET_REACHED"
        if time.monotonic() - self.t_debut >= self.max_wallclock_s:
            return "WALLCLOCK_LIMIT_REACHED"
        return None

    def fermer(self):
        self.executor.shutdown(wait=False)


def _extraire_frame(video_path, t_secondes, tmp_dir):
    t_secondes = max(0, t_secondes)
    chemin = os.path.join(tmp_dir, f"ko_frame_{t_secondes:.1f}.jpg")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(t_secondes), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", chemin
    ], check=True, capture_output=True)
    with open(chemin, "rb") as f:
        data = f.read()
    try:
        os.remove(chemin)
    except OSError:
        pass
    return data


def _appeler_gemini_json(client, image_bytes, prompt, etat, model_name=MODEL_NAME):
    from google.genai import types
    etat.n_appels += 1

    def _appel():
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(temperature=TEMPERATURE),
        )
        texte = response.text.strip()
        if texte.startswith("```"):
            texte = texte.split("```")[1]
            if texte.startswith("json"):
                texte = texte[4:]
        return json.loads(texte.strip())

    try:
        future = etat.executor.submit(_appel)
        return future.result(timeout=TIMEOUT_S)
    except Exception:
        return None  # traite comme "pas d'avis" par l'appelant, retry gere plus haut


def _appeler_json_robuste(client, video_path, t, tmp_dir, prompt, etat, model_name=MODEL_NAME):
    """2 tentatives a t exact, puis t+1s, puis t-1s (chacun 2 tentatives) -
    jusqu'a 6 appels sur 3 images differentes avant d'abandonner ce
    checkpoint. Un echec silencieux casserait la garantie de couverture
    exhaustive du scan (cf. V5_2_FIABILITE_ROADMAP.md §12.6bis).

    V5.2 : logging ajoute pour visibilite - si le retry se declenche
    frequemment, c'est un signe d'instabilite API reelle a surveiller."""
    for delta in (0, 1, -1):
        tt = max(0, t + delta)
        image_bytes = _extraire_frame(video_path, tt, tmp_dir)
        for _tentative in range(2):
            resultat = _appeler_gemini_json(client, image_bytes, prompt, etat, model_name=model_name)
            if resultat is not None:
                if delta != 0 or _tentative != 0:
                    print(f"    [RETRY] succès à t={tt:.0f}s après {delta:+d}s/tentative {_tentative+1} "
                          f"(échec initial à t={t:.0f}s)")
                return resultat
            print(f"    [RETRY] échec à t={tt:.0f}s (delta={delta:+d}, tentative {_tentative+1}/2)")
    print(f"    [RETRY] ÉCHEC TOTAL sur checkpoint t={t:.0f}s (6 appels, 3 frames, tous échoués)")
    return None  # echec total, checkpoint reellement perdu (rare)


def _q1_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q1_RIGOUREUX, etat, model_name=model_name)
    if result is None:
        return None
    criteres = [
        result.get("zone_centrale_plausible", False),
        result.get("caractere_avant_match", False),
        result.get("amorce_separation", False),
        result.get("pas_autre_remise_en_jeu", False),
    ]
    return sum(criteres) >= SEUIL_Q1


def _q1_une_lecture_ko2(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME):
    """V5.2 Phase A : variante Q1 pour KO2 UNIQUEMENT - meme 4 criteres +
    deux_equipes_visibles + nombre_joueurs_suffisant, tous les 2
    OBLIGATOIRES (voir PROMPT_Q1_KO2 ci-dessus).
    KO1 continue d'utiliser _q1_une_lecture (original), intact.

    model_name : V5.2 - parametrable pour comparer Pro vs Flash sur KO2,
    sans toucher au defaut (Pro) utilise partout ailleurs y compris KO1."""
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q1_KO2, etat, model_name=model_name)
    if result is None:
        return None
    criteres = [
        result.get("zone_centrale_plausible", False),
        result.get("caractere_avant_match", False),
        result.get("amorce_separation", False),
        result.get("pas_autre_remise_en_jeu", False),
    ]
    deux_equipes = result.get("deux_equipes_visibles", False)
    nombre_suffisant = result.get("nombre_joueurs_suffisant", False)
    return (sum(criteres) >= SEUIL_Q1) and deux_equipes and nombre_suffisant


def _q1_une_lecture_ko2_vote_economique(client, video_path, t, tmp_dir, etat, max_confirmations=2, model_name=MODEL_NAME):
    """V5.2 : variante ECONOMIQUE du vote majoritaire, appliquee SEULEMENT
    quand le premier appel dit OUI - pas sur chaque point scanne (la
    plupart disent NON correctement et n'ont pas besoin de confirmation
    couteuse). Motivation : diagnostic reel (P1Minerois, t=3652s) a montre
    2 appels sur la MEME image donnant 2 verdicts opposes (OUI puis NON)
    - un seul "OUI" chanceux peut faire derailler toute la recherche vers
    un faux Q2+recherche fine couteux, alors qu'un "NON" incorrect n'a
    qu'un cout faible (on continue simplement le scan).

    Cout ajoute : +max_confirmations appels UNIQUEMENT sur les points ou
    le 1er appel dit deja OUI (rares compares aux NON) - pas un
    multiplicateur global sur tout le scan."""
    premier = _q1_une_lecture_ko2(client, video_path, t, tmp_dir, etat, model_name=model_name)
    if not premier:
        return premier  # NON ou erreur : pas de confirmation, cout inchange

    votes = [premier]
    for _ in range(max_confirmations):
        v = _q1_une_lecture_ko2(client, video_path, t, tmp_dir, etat, model_name=model_name)
        if v is not None:
            votes.append(v)
    return sum(votes) > len(votes) / 2


def _q2_une_lecture(client, video_path, t, tmp_dir, etat, model_name=MODEL_NAME):
    result = _appeler_json_robuste(client, video_path, t, tmp_dir, PROMPT_Q2_RIGOUREUX, etat, model_name=model_name)
    if result is None:
        return None
    return bool(result.get("match_deja_commence", False))


def _voter_q2(client, video_path, t, tmp_dir, etat, max_appels=3, model_name=MODEL_NAME):
    """Vote majoritaire avec arret anticipe - meme principe que
    detect_ko_vision.py::voter_transition(), max_appels=3 (pas 5, pour
    maitriser le cout sur ce point de decision critique)."""
    votes = []
    for _ in range(max_appels):
        v = _q2_une_lecture(client, video_path, t, tmp_dir, etat, model_name=model_name)
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


def _scan_q1_par_lots(client, video_path, tmp_dir, etat, t_debut, t_max, pas=60, taille_lot=TAILLE_LOT_Q1, fonction_q1=_q1_une_lecture, model_name=MODEL_NAME):
    """Scanne Q1 par lots parallèles, traite dans l'ordre chronologique -
    identique en decision au scan sequentiel, juste plus rapide.
    Verifie le budget (appels/wallclock) avant chaque lot.

    fonction_q1 : V5.2 Phase A, defaut=_q1_une_lecture (KO1, inchange).
    Permet de passer _q1_une_lecture_ko2 pour la recherche KO2.
    model_name : V5.2, parametrable pour comparer Pro vs Flash sur KO2."""
    t = t_debut
    while t <= t_max:
        raison_arret = etat.budget_epuise()
        if raison_arret:
            return None, t, raison_arret

        lot = [t + i * pas for i in range(taille_lot) if t + i * pas <= t_max]
        futures = {tt: etat.executor.submit(fonction_q1, client, video_path, tt, tmp_dir, etat, model_name=model_name) for tt in lot}

        for tt in lot:
            decision = futures[tt].result()
            if decision:
                return tt, tt, None

        t += taille_lot * pas

    return None, t, None


def _recherche_fine(client, video_path, tmp_dir, etat, premier_oui, t_verif, model_name=MODEL_NAME):
    """Affine entre premier_oui (Q2=NON, deja verifie) et t_verif (Q2=OUI,
    deja verifie) par paliers decroissants. t_bas et t_haut ne sont
    jamais reredemandes (deja connus a chaque palier)."""
    t_bas, t_haut = premier_oui, t_verif
    for i_pas, pas in enumerate(PALIERS_RECHERCHE_FINE):
        tt = t_bas + pas
        dernier_non = t_bas
        while tt < t_haut:
            d = _q2_une_lecture(client, video_path, tt, tmp_dir, etat, model_name=model_name)
            if d:
                t_haut = tt
                t_bas = dernier_non
                break
            dernier_non = tt
            tt += pas
        else:
            t_bas = dernier_non
    return t_haut


def _rechercher_kickoff(client, video_path, tmp_dir, etat, t_max, t_debut=60, pas_scan=60, fonction_q1=_q1_une_lecture, taille_lot=TAILLE_LOT_Q1, model_name=MODEL_NAME, delai_verif_q2=60):
    # V5.2 Phase A : t_debut parametrable (defaut=60, comportement KO1
    # inchange) - necessaire pour reutiliser cette meme cascade pour KO2,
    # qui doit demarrer sa recherche a KO1+quelque chose, pas a t=60s.
    # pas_scan parametrable (defaut=60, comportement KO1 inchange) -
    # KO2 a besoin d'un pas plus fin (20s, valide empiriquement sur
    # Franchimont : la camera ne se stabilise sur le centre que
    # brievement lors de la reprise, un pas de 60s peut sauter par-dessus
    # cette fenetre - observe concretement en production, cf. diagnostic
    # Franchimont/Stembert KO2 errone).
    # model_name parametrable (defaut=MODEL_NAME=Pro, inchange pour KO1) -
    # permet de comparer Pro vs Flash pour KO2 sans toucher KO1.
    # delai_verif_q2 parametrable (defaut=60, comportement KO1 inchange) -
    # ce delai etait calibre pour pas_scan=60 (KO1) : verifier Q2 "un pas
    # de scan plus loin". Avec pas_scan=20 pour KO2, un delai fixe de 60s
    # verifie 3 pas de scan plus loin au lieu de 1, ce qui peut rendre le
    # controle "fenetre degeneree" (Q2 deja vrai au point premier_oui)
    # trop sensible - suspicion signalee, jamais teste avant ce fix.
    # AUCUN changement de logique/prompts/seuils, uniquement le pas, le
    # point de depart du scan, le modele, et ce delai.
    t = t_debut  # t=0 (ou avant t_debut) toujours "avant-match" pour KO1,
                 # mais pour KO2 t_debut sera deja loin dans la video
    while t <= t_max:
        print(f"  [KICKOFF_GEMINI] scan Q1 depuis t={t:.0f}s (max={t_max:.0f}s, pas={pas_scan}s, modele={model_name})")
        premier_oui, t, raison_arret = _scan_q1_par_lots(client, video_path, tmp_dir, etat, t, t_max, pas=pas_scan, fonction_q1=fonction_q1, taille_lot=taille_lot, model_name=model_name)
        if raison_arret:
            print(f"  [KICKOFF_GEMINI] arrêt : {raison_arret}")
            return {"status": "NOT_FOUND", "kickoff_s": None, "reason": raison_arret}
        if premier_oui is None:
            print(f"  [KICKOFF_GEMINI] aucun candidat Q1 trouvé jusqu'à t={t_max:.0f}s")
            return {"status": "NOT_FOUND", "kickoff_s": None, "reason": "VIDEO_EXHAUSTED"}

        t_verif = premier_oui + delai_verif_q2
        if t_verif > t_max:
            print(f"  [KICKOFF_GEMINI] candidat à t={premier_oui:.0f}s mais vérification hors limite")
            return {"status": "NOT_FOUND", "kickoff_s": None, "reason": "VIDEO_EXHAUSTED"}

        print(f"  [KICKOFF_GEMINI] candidat Q1 à t={premier_oui:.0f}s, vote Q2 à t={t_verif:.0f}s...")
        decision_q2 = _voter_q2(client, video_path, t_verif, tmp_dir, etat, model_name=model_name)
        print(f"  [KICKOFF_GEMINI] vote Q2 : {'OUI' if decision_q2 else 'NON' if decision_q2 is not None else 'ERREUR'}")

        raison_arret = etat.budget_epuise()
        if raison_arret:
            print(f"  [KICKOFF_GEMINI] arrêt : {raison_arret}")
            return {"status": "NOT_FOUND", "kickoff_s": None, "reason": raison_arret}

        if not decision_q2:
            print(f"  [KICKOFF_GEMINI] candidat rejeté, reprise à t={t_verif:.0f}s")
            t = t_verif  # reprend le scan Q1 ici, pas t_verif+60 (trou de couverture, §12.3)
            continue

        # Garde de securite : si Q2 est deja vrai au point de depart, la
        # fenetre [premier_oui, t_verif] est invalide (vrai KO probablement
        # avant premier_oui) - ne jamais deviner, signaler NOT_FOUND.
        premier_check = _q2_une_lecture(client, video_path, premier_oui, tmp_dir, etat, model_name=model_name)
        if premier_check:
            print(f"  [KICKOFF_GEMINI] fenêtre dégénérée détectée (Q2 déjà vrai à t={premier_oui:.0f}s)")
            return {"status": "NOT_FOUND", "kickoff_s": None, "reason": "DEGENERATE_WINDOW"}

        print(f"  [KICKOFF_GEMINI] confirmé, recherche fine dans [{premier_oui:.0f}s, {t_verif:.0f}s]...")
        kickoff_s = _recherche_fine(client, video_path, tmp_dir, etat, premier_oui, t_verif, model_name=model_name)
        print(f"  [KICKOFF_GEMINI] KO détecté à t={kickoff_s:.0f}s")
        return {"status": "AUTO_CONFIRMED", "kickoff_s": float(kickoff_s), "reason": None}

    return {"status": "NOT_FOUND", "kickoff_s": None, "reason": "VIDEO_EXHAUSTED"}


def detect_kickoff_gemini(video_path, max_search_s,
                            max_gemini_calls=MAX_GEMINI_CALLS_DEFAUT,
                            max_wallclock_s=MAX_WALLCLOCK_S_DEFAUT,
                            tmp_dir="/tmp", t_debut=60, pas_scan=60, fonction_q1=_q1_une_lecture, taille_lot=TAILLE_LOT_Q1, model_name=MODEL_NAME, delai_verif_q2=60):
    """
    Détecte le premier coup d'envoi d'un match par cascade Gemini
    (Q1 scan 60s -> Q2 confirmation -> recherche fine 15/5/1s).

    Contrat V1 (V5_2_FIABILITE_ROADMAP.md §12.9) :
    - Premier KO uniquement (pas la reprise après mi-temps).
    - Ne fabrique JAMAIS un timestamp sur une recherche incertaine ou
      épuisée - retourne NOT_FOUND explicitement dans ce cas.

    Paramètres
    ----------
    video_path : str
        Chemin de la vidéo à analyser.
    max_search_s : float
        Limite de recherche (secondes), déjà bornée par l'appelant
        (typiquement min(video_duration_s, formule half_duration_min), §12.10).
    max_gemini_calls : int
        Budget d'appels Gemini, garde-fou technique fixe (pas exposé à
        l'utilisateur final).
    max_wallclock_s : float
        Limite de temps réel (secondes), garde-fou technique fixe,
        indépendant du budget d'appels.
    tmp_dir : str
        Répertoire pour les frames temporaires extraites.
    t_debut : float
        V5.2 Phase A : point de départ du scan (défaut=60, comportement
        KO1 identique à avant). Permet de réutiliser cette cascade pour
        KO2 en démarrant le scan à un point avancé de la vidéo (ex:
        KO1+49min) plutôt qu'au tout début.

    Retourne
    --------
    dict avec au minimum :
        {"status": "AUTO_CONFIRMED" | "NOT_FOUND" | "ERROR",
         "kickoff_s": float | None,
         "reason": str | None,
         "n_appels_gemini": int,
         "duree_recherche_s": float}
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"status": "ERROR", "kickoff_s": None, "reason": "MISSING_API_KEY",
                "n_appels_gemini": 0, "duree_recherche_s": 0.0}

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {"status": "ERROR", "kickoff_s": None, "reason": f"CLIENT_INIT_FAILED: {e}",
                "n_appels_gemini": 0, "duree_recherche_s": 0.0}

    os.makedirs(tmp_dir, exist_ok=True)
    etat = _EtatRecherche(max_gemini_calls, max_wallclock_s)

    try:
        resultat = _rechercher_kickoff(client, video_path, tmp_dir, etat, max_search_s, t_debut=t_debut, pas_scan=pas_scan, fonction_q1=fonction_q1, taille_lot=taille_lot, model_name=model_name, delai_verif_q2=delai_verif_q2)
    except Exception as e:
        resultat = {"status": "ERROR", "kickoff_s": None, "reason": f"UNEXPECTED_EXCEPTION: {e}"}
    finally:
        etat.fermer()

    resultat["n_appels_gemini"] = etat.n_appels
    resultat["duree_recherche_s"] = time.monotonic() - etat.t_debut
    return resultat


def detect_kickoff_gemini_avec_retry(video_path, max_search_s,
                                       max_retries=3, t_debut=60, **kwargs):
    """
    Enveloppe detect_kickoff_gemini() avec un retry automatique -
    UNIQUEMENT sur NOT_FOUND, pas sur ERROR.

    Justification : NOT_FOUND (notamment DEGENERATE_WINDOW) peut venir
    de la variance residuelle de Gemini documentee tout au long de
    V5_2_FIABILITE_ROADMAP.md §12 - observe concretement : meme match
    (Andrimont), 2 appels consecutifs a ce module -> 1x NOT_FOUND/
    DEGENERATE_WINDOW, 1x AUTO_CONFIRMED (310s). Un nouvel essai complet
    peut recuperer le bon resultat sans intervention manuelle.

    ERROR n'est PAS retente : ces conditions (cle API manquante, client
    mal initialise, exception inattendue) sont plus probablement
    structurelles que transitoires - reessayer ne les resoudrait pas et
    gaspillerait du budget/temps pour rien.

    Cout dans le pire cas : jusqu'a max_retries x le cout d'un essai
    normal (appels + temps). A surveiller en production - si NOT_FOUND
    est frequent, ce multiplicateur peut devenir significatif.

    t_debut : voir detect_kickoff_gemini (V5.2 Phase A, defaut=60,
    permet la reutilisation pour KO2).
    """
    dernier_resultat = None
    for tentative in range(1, max_retries + 1):
        print(f"[KICKOFF_GEMINI] tentative {tentative}/{max_retries}")
        resultat = detect_kickoff_gemini(video_path, max_search_s, t_debut=t_debut, **kwargs)
        resultat["tentative"] = tentative
        print(f"[KICKOFF_GEMINI] tentative {tentative} → status={resultat['status']} "
              f"kickoff_s={resultat['kickoff_s']} reason={resultat.get('reason')}")
        if resultat["status"] != "NOT_FOUND":
            return resultat  # AUTO_CONFIRMED ou ERROR : on s'arrete la
        dernier_resultat = resultat

    print(f"[KICKOFF_GEMINI] échec après {max_retries} tentatives (NOT_FOUND persistant)")
    return dernier_resultat  # NOT_FOUND apres max_retries tentatives