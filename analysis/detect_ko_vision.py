# analysis/detect_ko_vision.py
# -*- coding: utf-8 -*-
#
# Détection du coup d'envoi par IA vision — étage final du pipeline.
#
# Architecture validée sur les 10 matchs de test (cf. JALON, sections 13
# à 19) :
#
#   RandomForest (ko_features.py, déjà entraîné)
#         │  filtre à top-10 candidats par match (recall mesuré : 99%,
#         │  cf. jalon section 16 — top-3/top-5 insuffisants et instables)
#         ▼
#   Pour chaque candidat : extraction de 3 images
#   (juste_avant t-2s, candidat t0, juste_apres t+2s)
#         │  motif recherché : formation statique → frappe →
#         │  dispersion générale immédiate (validé 8/9 sur matchs Veo,
#         │  1/1 sur caméra manuelle avec confiance dégradée)
#         ▼
#   IA vision (Gemini, même moteur que detect_teams_preview.py)
#         │  compare les 3 images, juge si la bascule est visible
#         ▼
#   Sortie : candidate (t) + confidence + ambiguity
#   (contrat fonctionnel V1, cf. jalon section 2 — ne jamais transformer
#   une ambiguïté réelle en confiance artificielle)
#
# IMPORTANT : ce module ne remplace pas le RandomForest, il le complète.
# Le RF reste responsable de réduire l'espace de recherche (533 candidats
# → 10) ; ce module départage les 10 candidats retenus.

import os
import cv2
import base64
import json


# ─────────────────────────────────────────
# EXTRACTION DES SÉQUENCES À 3 IMAGES
# ─────────────────────────────────────────

def extraire_sequence_candidat(video_path, t_candidat, fps_source=None, offset_s=2.0):
    """
    Extrait 3 images autour d'un instant candidat :
      juste_avant (t-offset_s), candidat (t0), juste_apres (t+offset_s)

    Protocole validé (jalon section 18) avec offset_s=2.0 (valeur par
    défaut, comportement de production INCHANGÉ) : suffisant dans la
    grande majorité des cas. Limite connue : deux candidats à moins de
    5-6s l'un de l'autre peuvent avoir des fenêtres qui se chevauchent
    (cf. l'erreur sur Stembert) — à surveiller si le top-10 contient
    des candidats très rapprochés temporellement.

    offset_s : PARAMÈTRE DE TEST — cf. cas Raeren où le candidat retenu
    (404.8s) était décalé d'environ 4s par rapport au vrai KO (~409s,
    pas 410s comme indiqué dans le référentiel historique). Avec
    offset_s=2.0, la fenêtre (402.8/404.8/406.8) ratait presque
    entièrement le plateau de mouvement réel (406-410s). Un offset plus
    large (ex. 4.0) pourrait capter la bascule même avec ce genre de
    décalage — À VALIDER par benchmark contrôlé avant tout changement
    du défaut de production (cf. jalon, benchmark ±2 vs ±4).

    Retourne un dict {"juste_avant": np.array, "candidat": np.array,
    "juste_apres": np.array} (frames BGR, format OpenCV), ou None pour
    les images qui n'ont pas pu être lues.
    """
    cap = cv2.VideoCapture(video_path)
    if fps_source is None:
        fps_source = cap.get(cv2.CAP_PROP_FPS)

    offsets = {"juste_avant": -offset_s, "candidat": 0.0, "juste_apres": offset_s}
    frames = {}
    for label, offset in offsets.items():
        t = max(0, t_candidat + offset)
        frame_num = int(t * fps_source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        frames[label] = frame if ret else None

    cap.release()
    return frames


def _frame_to_base64(frame):
    """Encode une frame OpenCV (BGR) en base64 JPEG, pour l'API Gemini."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ─────────────────────────────────────────
# APPEL IA VISION (GEMINI)
# ─────────────────────────────────────────
#
# ⚠️ HISTORIQUE CRITIQUE — NE JAMAIS REFORMULER CE PROMPT EN "HISTOIRE
# GLOBALE" SANS RE-VALIDER SUR LES 9 MATCHS AVEC gemini-2.5-flash ⚠️
#
# Ce prompt "checklist" (vérification image par image : image1=statique,
# image2=frappe, image3=dispersion) A DÉJÀ ÉTÉ reformulé une fois en
# version "histoire globale" (jugement holistique des 3 images comme un
# seul récit, plutôt qu'une vérification stricte critère par critère).
# Testé empiriquement sur gemini-2.5-flash (session antérieure), le
# résultat était netament PIRE, pas meilleur :
#
#   Match          | Checklist (ancien) | Histoire globale (testé)
#   Andrimont      | ~2-3 faux positifs/10 | 5 faux positifs/10
#   Stembert       | ~2 faux positifs/10   | 4 faux positifs/10 (+ vrai KO raté)
#   MineroisSter   | ~3 faux positifs/10   | 7 faux positifs/10
#
# Cause : assouplir le critère donne au modèle la permission implicite
# de rationaliser presque n'importe quelle action de jeu comme
# "cohérente avec un début de match", puisque la plupart des
# tirs/contre-attaques ont un moment où plusieurs joueurs avancent
# ensemble. Sur gemini-3.1-pro-preview, le même prompt "histoire
# globale" donnait un résultat different (trop conservateur, pas trop
# permissif) — donc l'effet dépend fortement du modèle, jamais neutre.
#
# CETTE LEÇON A ÉTÉ OUBLIÉE UNE FOIS DÉJÀ pendant cette investigation
# (la version "histoire globale" s'est retrouvée réintroduite dans ce
# fichier sans que quiconque se souvienne du test qui l'avait invalidée
# — découvert seulement après coup, en fouillant les transcripts d'une
# session antérieure). Le prompt "checklist" a été restauré comme
# référence stable.
#
# ITÉRATION SUIVANTE (celle ci-dessous, actuellement en place) :
# structure "OBLIGATOIRE vs SECONDAIRE" explicite, différente à la fois
# du "checklist" original ET de la version "histoire globale" qui avait
# échoué. Motivée par une confabulation précise observée (Andrimont
# t=194.9s : Gemini affirmait "les deux équipes se dispersent" alors
# qu'une seule équipe était présente, probablement un regroupement
# après un but). Distinction clé avec la version qui avait échoué :
# celle-ci donnait la permission d'IGNORER un détail peu clair
# ("ne rejette pas... parce qu'un détail n'est pas parfaitement
# visible") ; celle-ci au contraire liste explicitement 3 faits
# OBLIGATOIRES (deux équipes distinctes, chacune organisée séparément
# en image 1, vraie mise en mouvement des DEUX équipes en image 3) à
# répondre "non" si non vérifiables, et seulement 5 détails SECONDAIRES
# (ballon, couleurs imparfaites, arbitre) qui ne doivent jamais bloquer
# un "oui". Le critère "franchir la ligne médiane" testé dans une
# itération intermédiaire a été retiré (jugé possiblement trop strict
# pour une fenêtre de 2s) au profit d'un critère de mise en mouvement
# réelle des deux équipes, plus réaliste.
#
# ⚠️ CETTE ITÉRATION N'A PAS ENCORE ÉTÉ VALIDÉE EMPIRIQUEMENT — testée
# uniquement par relecture, pas encore relancée sur les 18 candidats de
# référence (10 faux positifs connus avec le prompt checklist simple).
# NE PAS considérer comme acquise avant ce test.
# ─────────────────────────────────────────

PROMPT_KO_VISION = """Tu vas voir 3 images consécutives d'un match de football amateur.

- Image 1 = environ 2 secondes AVANT le candidat
- Image 2 = instant du candidat
- Image 3 = environ 2 secondes APRÈS le candidat

Ta tâche est UNIQUEMENT de déterminer si ces 3 images montrent une
séquence de COUP D'ENVOI suivie du début du jeu.

Ne cherche PAS à déterminer s'il s'agit du premier coup d'envoi du match.
Ne cherche PAS à déterminer s'il s'agit d'un coup d'envoi après un but.
Tu dois uniquement juger la séquence visuelle qui t'est présentée.

IMPORTANT :
Tu dois regarder les 3 images comme UNE SEULE séquence temporelle.
Ne juge pas l'image 1, l'image 2 et l'image 3 séparément.

Pour répondre OUI, il faut que les éléments suivants soient réellement
visibles dans les images :

1. AVANT LE DÉMARRAGE
Dans l'image 1, la scène doit être compatible avec une situation de
coup d'envoi :
- les deux équipes sont reconnaissables ;
- les joueurs sont principalement répartis dans leurs deux moitiés
  de terrain ;
- la scène ne doit pas être un simple attroupement compact ;
- la disposition générale doit ressembler à une situation précédant
  le démarrage du jeu.

2. TRANSITION
Entre l'image 1 et l'image 3, il doit être visible que le jeu commence :
- plusieurs joueurs des DEUX équipes changent réellement de position ;
- il y a une évolution collective de la disposition des joueurs ;
- il ne suffit pas qu'un seul joueur bouge ;
- une simple différence de position entre les images ne suffit pas si
  elle peut être due à un déplacement individuel ou à une scène statique.

3. COUP D'ENVOI
La position du ballon peut aider à confirmer le coup d'envoi, notamment
s'il est visible près du centre du terrain.

MAIS :
- le ballon n'est PAS un critère obligatoire ;
- si le ballon est caché par un joueur, difficile à voir ou absent du
  cadrage, cela ne suffit PAS à répondre NON ;
- ne suppose jamais que le ballon est au centre s'il n'est pas visible.

À L'INVERSE, si le ballon est clairement visible ailleurs sur le terrain
et que la scène correspond manifestement à une phase de jeu, cela constitue
un indice important contre un coup d'envoi.

FAUX POSITIFS À ÉVITER :
- célébration après un but ;
- discussion ou rassemblement de joueurs ;
- photo d'équipe ;
- échauffement ;
- coup franc ;
- remise en jeu ;
- dégagement de but ;
- corner ;
- simple déplacement des joueurs pendant une phase de jeu ;
- toute scène où les joueurs sont déjà en train de jouer.

POINT TRÈS IMPORTANT :
Une scène peut ressembler fortement à un coup d'envoi sans être le début
du match. Ce n'est PAS à toi de résoudre cette distinction.

Si la scène présentée ressemble visuellement à un coup d'envoi suivi du
début du jeu, réponds OUI, même si tu ne peux pas savoir si c'est le
premier coup d'envoi du match.

Ne transforme jamais une supposition en fait observable.

Le raisonnement doit expliquer les éléments VISIBLES qui ont conduit à
la décision. N'utilise pas de phrases génériques comme "cela indique un
début de jeu" sans expliquer ce qui est réellement visible.

Réponds UNIQUEMENT avec ce JSON valide :

{
  "transition_visible": true,
  "confidence": 0.0,
  "raisonnement": "..."
}

"transition_visible" = true uniquement si la séquence montre réellement
une organisation compatible avec un coup d'envoi suivie d'une mise en jeu. sinon false.

"confidence" doit représenter ta confiance dans l'observation visuelle
de cette transition, pas ta confiance qu'il s'agit du premier coup
d'envoi du match. de 0.0 à 1.0
"raisonnement": "1-2 phrases décrivant uniquement les éléments visuellement observables qui justifient la décision"
}
"""


# ─────────────────────────────────────────
# PRÉ-FILTRE RAPIDE (1 image, avant le vote à 3 images)
# ─────────────────────────────────────────
# Objectif de coût : rejeter à bas prix (1 seul appel, pas 2-3) les
# candidats clairement hors sujet (tir, remise en jeu localisée,
# célébration, plan de banc de touche...), sans jamais risquer de
# rejeter à tort un vrai KO. Conçu délibérément PERMISSIF — un faux
# positif ici ne coûte que 2-3 appels de vote en plus (le vote
# tranchera correctement) ; un faux négatif ferait perdre le candidat
# définitivement, sans recours. Asymétrie de coût assumée.
#
# Vérifié visuellement (jalon) : le ballon peut être invisible même sur
# un vrai KO (cas Raeren, candidat t=408s) — donc le ballon n'est
# JAMAIS un critère bloquant ici, exactement comme pour le vote à 3
# images.

PROMPT_PREFILTRE_RAPIDE = """Tu vas voir UNE SEULE image d'un match de football amateur. C'est un premier tri RAPIDE, PAS une décision finale — en cas de doute, réponds "oui" (un examen plus approfondi sur plusieurs images suivra de toute façon).

Question : cette image pourrait-elle correspondre à un instant proche d'un coup d'envoi (début de match, début de mi-temps, ou reprise après un but) ?

Signal typique à rechercher : les joueurs des deux équipes semblent organisés de part et d'autre d'une zone centrale du terrain (pas besoin d'un alignement parfait), et/ou un arbitre est visible sur le terrain. Les joueurs sont généralement immobiles ou en attente.

IMPORTANT : le ballon n'a PAS besoin d'être visible — il est souvent petit, caché par un joueur, ou peu contrasté (éclairage nocturne, plan large). Son absence NE DOIT JAMAIS, à elle seule, faire répondre "non".

Ce filtre doit être PERMISSIF : en cas de doute, d'image ambiguë, de mauvais angle, de mauvaise qualité, ou de joueurs partiellement visibles, réponds "oui". Réponds "non" SEULEMENT si l'image montre CLAIREMENT autre chose : une seule équipe visible (pas de formation à deux camps), une célébration de but, un arrêt de jeu localisé dans un coin du terrain (touche, corner, coup franc excentré), un plan de banc de touche ou de tribune sans vue d'ensemble du terrain, etc.

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
{
  "formation_plausible": true/false,
  "raisonnement": "1 phrase courte"
}
"""


def demander_gemini_prefiltre(image, client, model="gemini-2.5-flash"):
    """
    Envoie UNE SEULE image (typiquement "juste_avant") à Gemini pour un
    tri rapide et permissif, avant le vote complet à 3 images.

    image : frame OpenCV BGR unique (pas un dict, contrairement à
    demander_gemini_ko).

    Retourne {"formation_plausible": bool, "raisonnement": str}, ou
    None si l'appel échoue — à traiter comme "pas d'avis", donc ne
    JAMAIS bloquer sur un None (cf. utilisation dans _evaluer :
    en cas de None, on procède quand même au vote complet, jamais
    l'inverse).
    """
    if image is None:
        return None

    parts = [{"text": PROMPT_PREFILTRE_RAPIDE}]
    img_b64 = _frame_to_base64(image)
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        response = client.models.generate_content(
            model=model,
            contents=[{"parts": parts}]
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return {
            "formation_plausible": bool(result.get("formation_plausible", True)),  # défaut permissif si clé manquante
            "raisonnement": result.get("raisonnement", ""),
        }
    except Exception as e:
        print(f"  [detect_ko_vision] Erreur Gemini (pré-filtre) : {e}")
        return None


def demander_gemini_ko(images_dict, client, model="gemini-2.5-flash"):
    """
    Envoie les 3 images à Gemini avec le prompt structuré, retourne le
    jugement (transition visible ou non, confiance, raisonnement).

    images_dict : {"juste_avant": frame, "candidat": frame, "juste_apres": frame}
    (frames OpenCV BGR, comme retourné par extraire_sequence_candidat)

    Retourne None si un appel échoue (image manquante, erreur API, JSON
    invalide) — à traiter comme "pas d'avis" par l'appelant, pas comme
    un refus catégorique.
    """
    if any(images_dict.get(k) is None for k in ("juste_avant", "candidat", "juste_apres")):
        return None

    parts = [{"text": PROMPT_KO_VISION}]
    for label in ("juste_avant", "candidat", "juste_apres"):
        img_b64 = _frame_to_base64(images_dict[label])
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        response = client.models.generate_content(
            model=model,
            contents=[{"parts": parts}]
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return {
            "transition_visible": bool(result.get("transition_visible", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "raisonnement": result.get("raisonnement", ""),
        }
    except Exception as e:
        print(f"  [detect_ko_vision] Erreur Gemini : {e}")
        return None


# ─────────────────────────────────────────
# VOTE MAJORITAIRE (jusqu'à 3 appels, avec arrêt anticipé)
# ─────────────────────────────────────────
# Justifié par le test de répétabilité (jalon, test 270 appels sur 9
# matchs) : 92% des candidats sont parfaitement stables (3/3 ou 0/3), et
# le vote majoritaire corrige les rares cas de bruit sans jamais valider
# à tort un candidat clairement négatif. Un seul appel suffit dans la
# grande majorité des cas — d'où l'arrêt anticipé après 2 appels dès que
# la majorité est déjà mathématiquement jouée (2 accords sur 2), pour ne
# pas payer un 3e appel inutile.

def voter_transition(images_dict, client, model="gemini-2.5-flash", max_appels=3):
    """
    Fait jusqu'à max_appels appels indépendants à demander_gemini_ko sur
    les MÊMES images, avec arrêt anticipé dès que la majorité est
    acquise (2 votes identiques sur les 2 premiers -> inutile de payer
    un 3e appel, il ne peut plus changer la décision).

    Retourne {"decision": bool, "confidence": float, "n_appels": int,
    "votes": [...détail de chaque appel...]} ou None si le premier appel
    échoue déjà (image manquante notamment).
    """
    votes = []
    for _ in range(max_appels):
        jugement = demander_gemini_ko(images_dict, client, model=model)
        if jugement is None:
            if not votes:
                return None  # échec dès le 1er appel -> pas d'avis du tout
            break  # on garde ce qu'on a déjà obtenu
        votes.append(jugement)

        # Arrêt anticipé : dès 2 votes, si accord total, la majorité sur
        # 3 est déjà acquise quel que soit le 3e appel -> inutile de le payer.
        if len(votes) == 2 and votes[0]["transition_visible"] == votes[1]["transition_visible"]:
            break

    n_true = sum(1 for v in votes if v["transition_visible"])
    n_total = len(votes)
    decision = n_true > n_total / 2  # majorité stricte (2/3, 2/2, ou 1/1 en cas d'échec partiel)

    # Confiance retenue : moyenne des votes allant dans le sens de la décision finale
    confs_majoritaires = [v["confidence"] for v in votes if v["transition_visible"] == decision]
    confidence_moyenne = sum(confs_majoritaires) / len(confs_majoritaires) if confs_majoritaires else 0.0

    return {
        "decision": decision,
        "confidence": confidence_moyenne,
        "n_appels": n_total,
        "votes": votes,
    }


# ─────────────────────────────────────────
# FONCTION PRINCIPALE — CONTRAT candidate + confidence + ambiguity
# ─────────────────────────────────────────

def detecter_ko_par_vision(video_path, top_n_candidats, fps_source=None,
                             model="gemini-2.5-flash", seuil_ambiguite_secondes=10.0,
                             max_appels_par_candidat=3, offset_s=2.0,
                             candidats_deja_evalues=None, utiliser_prefiltre=False,
                             desactiver_arret_anticipe=False):
    """
    Point d'entrée principal. Prend le top-N candidats déjà produit par
    le RandomForest (ko_features.py + le modèle entraîné), départage-les
    par IA vision (vote majoritaire jusqu'à 3 appels par candidat),
    retourne le contrat fonctionnel V1.

    Objectif du projet : le PREMIER coup d'envoi du match (pas "tous les
    KO", pas "le candidat le plus confiant"). Sur une vidéo qui couvre
    plus que la seule phase pré-match (ex. match complet avec but(s)
    marqué(s)), plusieurs candidats peuvent légitimement être de vrais
    coups d'envoi (l'initial + des reprises après but) — dans ce cas,
    seul le plus ancien temporellement est retenu comme timestamp_final.

    TRI CHRONOLOGIQUE PUR (pas par proba RF) : les candidats sont traités
    par ORDRE CHRONOLOGIQUE CROISSANT. Une version hybride (tri par proba
    RF décroissante, avec vérification des antérieurs) a été testée et
    ABANDONNÉE : plus chère (16-20 appels/match contre 8-23 en
    chronologique) sans gain de qualité, car la probabilité RF ne
    corrèle pas avec la position temporelle (jalon section 23). Dès
    qu'un candidat positif est trouvé, seuls les candidats dans la
    fenêtre d'ambiguïté (seuil_ambiguite_secondes) sont encore vérifiés,
    le reste n'est jamais évalué.

    top_n_candidats : liste de dicts [{"t": float, "categorie": str,
    "proba": float}, ...] — l'ordre d'entrée n'importe pas, la fonction
    trie elle-même par t croissant.

    seuil_ambiguite_secondes : si un 2e candidat positif existe à moins
    de ce nombre de secondes du premier retenu, l'ambiguïté est signalée.

    offset_s : décalage (en secondes) des images juste_avant/juste_apres
    par rapport au candidat, transmis à extraire_sequence_candidat.
    Défaut 2.0 = comportement de production validé (jalon section 18-23).
    Valeur PLUS GRANDE en test uniquement (cf. cas Raeren, jalon section
    26) — ne pas changer le défaut avant validation par benchmark
    contrôlé ±2 vs ±4 sur plusieurs matchs.

    candidats_deja_evalues : dict optionnel {t: résultat_évaluation}
    (même structure qu'une entrée de "candidats_evalues" en sortie).
    Si fourni, les candidats déjà présents dans ce dict ne sont PAS
    ré-évalués par Gemini — leur résultat est simplement réutilisé.
    Sert à l'élargissement adaptatif du pool sans jamais payer deux fois
    le même appel (cf. detecter_ko_par_vision_adaptatif ci-dessous).

    utiliser_prefiltre : bool, défaut False (comportement de production
    INCHANGÉ si non spécifié). Si True, un pré-filtre à 1 image (image
    "juste_avant" seule) est appelé AVANT le vote complet à 3 images,
    pour rejeter à bas coût (1 appel au lieu de 2-3) les candidats
    clairement hors sujet. Conçu délibérément permissif (cf.
    PROMPT_PREFILTRE_RAPIDE) — en cas de doute ou d'échec technique,
    procède quand même au vote complet, jamais l'inverse. À valider par
    benchmark avant adoption en défaut de production.

    desactiver_arret_anticipe : bool, défaut False (comportement de
    production INCHANGÉ). Si True, DIAGNOSTIC UNIQUEMENT — évalue TOUS
    les candidats du pool, même après qu'un premier positif ait été
    trouvé (ignore la fenêtre d'ambiguïté pour le SKIP, pas pour le
    choix final). Découverte majeure (jalon section 18) : sur 3 matchs
    testés, le vrai KO n'a JAMAIS été atteint par la politique
    "premier positif = stop", à cause de faux positifs précoces
    (probablement des formations d'avant-match) — ce mode permet de
    vérifier si le vrai KO aurait été correctement accepté s'il avait
    eu l'occasion d'être examiné. Coût : jusqu'à N appels au lieu de
    l'arrêt anticipé habituel, NE PAS utiliser en production.

    Retourne :
    {
        "timestamp_final": float ou None,
        "confidence": "élevée" / "moyenne" / "faible",
        "ambiguity": bool,
        "candidats_evalues": [...détail des candidats RÉELLEMENT évalués,
                               les autres sont listés avec evalue=False...],
    }

    Ne transforme jamais une ambiguïté réelle en confiance élevée
    (contrat fonctionnel V1, jalon section 2).
    """
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("  [detect_ko_vision] GEMINI_API_KEY manquante — impossible de continuer")
            return {
                "timestamp_final": None, "confidence": "faible",
                "ambiguity": True, "candidats_evalues": [],
                "erreur": "GEMINI_API_KEY manquante",
            }
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"  [detect_ko_vision] Gemini indisponible : {e}")
        return {
            "timestamp_final": None, "confidence": "faible",
            "ambiguity": True, "candidats_evalues": [],
            "erreur": str(e),
        }

    candidats_deja_evalues = candidats_deja_evalues or {}

    def _evaluer(c):
        # Si ce candidat a déjà été évalué lors d'une passe précédente
        # (mode adaptatif), réutiliser le résultat SANS rappeler Gemini —
        # évite tout appel redondant lors d'un élargissement de pool.
        if c["t"] in candidats_deja_evalues:
            return candidats_deja_evalues[c["t"]]

        images = extraire_sequence_candidat(video_path, c["t"], fps_source=fps_source, offset_s=offset_s)

        # Pré-filtre optionnel (1 appel), pour rejeter à bas coût les
        # candidats clairement hors sujet avant le vote complet (2-3
        # appels). Permissif par construction : un échec technique ou un
        # doute ("formation_plausible" absent/True) laisse TOUJOURS
        # passer au vote complet — jamais de rejet basé sur ce seul
        # pré-filtre.
        if utiliser_prefiltre and images.get("juste_avant") is not None:
            prefiltre = demander_gemini_prefiltre(images["juste_avant"], client, model=model)
            if prefiltre is not None and prefiltre["formation_plausible"] is False:
                return {
                    "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
                    "evalue": True, "vote": None,
                    "jugement_vision": {
                        "transition_visible": False, "confidence": 0.9,
                        "raisonnement": f"[pré-filtre] {prefiltre['raisonnement']}",
                    },
                }

        vote = voter_transition(images, client, model=model, max_appels=max_appels_par_candidat)
        return {
            "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
            "evalue": True, "vote": vote,
            "jugement_vision": ({
                "transition_visible": vote["decision"],
                "confidence": vote["confidence"],
                "raisonnement": f"vote majoritaire {sum(1 for v in vote['votes'] if v['transition_visible'])}/{vote['n_appels']}",
            } if vote else None),
        }

    # Tri chronologique croissant — condition nécessaire à l'arrêt anticipé
    candidats_tries = sorted(top_n_candidats, key=lambda c: c["t"])

    resultats = []
    candidats_positifs = []
    premier_positif_t = None

    for c in candidats_tries:
        # Arrêt anticipé : au-delà de la fenêtre d'ambiguïté autour du
        # premier positif trouvé, plus rien ne peut changer la décision.
        # DESACTIVABLE en mode diagnostic (desactiver_arret_anticipe=True).
        if (not desactiver_arret_anticipe and premier_positif_t is not None
                and (c["t"] - premier_positif_t) >= seuil_ambiguite_secondes):
            resultats.append({
                "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
                "evalue": False, "vote": None,
            })
            continue

        resultats.append(_evaluer(c))

        if resultats[-1]["vote"] is not None and resultats[-1]["vote"]["decision"]:
            candidats_positifs.append(resultats[-1])
            if premier_positif_t is None:
                premier_positif_t = c["t"]

    if not candidats_positifs:
        meilleur_rf = top_n_candidats[0] if top_n_candidats else None
        return {
            "timestamp_final": meilleur_rf["t"] if meilleur_rf else None,
            "confidence": "faible",
            "ambiguity": True,
            "candidats_evalues": resultats,
            "note": "Aucune transition claire détectée par l'IA vision — repli sur le meilleur candidat RF, à traiter avec prudence.",
        }

    # Déjà trié par t croissant -> le premier positif trouvé est le bon
    meilleur = candidats_positifs[0]

    ambiguity = False
    if len(candidats_positifs) > 1:
        ecart_temporel = candidats_positifs[1]["t"] - meilleur["t"]
        ambiguity = ecart_temporel < seuil_ambiguite_secondes

    conf_valeur = meilleur["vote"]["confidence"]
    if ambiguity:
        confidence_label = "moyenne"
    elif conf_valeur >= 0.75:
        confidence_label = "élevée"
    elif conf_valeur >= 0.4:
        confidence_label = "moyenne"
    else:
        confidence_label = "faible"

    return {
        "timestamp_final": meilleur["t"],
        "confidence": confidence_label,
        "ambiguity": ambiguity,
        "candidats_evalues": resultats,
    }


def detecter_ko_par_vision_adaptatif(video_path, top_n_complet, fps_source=None,
                                       model="gemini-2.5-flash", seuil_ambiguite_secondes=10.0,
                                       max_appels_par_candidat=3, offset_s=2.0,
                                       taille_pool_initial=10, taille_pool_max=30):
    """
    ASTUCE DE COÛT : mode adaptatif à 2 passes, pour élargir le pool de
    candidats (top-10 -> top-30) SANS payer le surcoût dans les cas où
    le top-10 suffit déjà (la majorité des matchs, cf. jalon).

    Passe 1 : detecter_ko_par_vision() sur les `taille_pool_initial`
    premiers candidats (défaut 10, comportement de production actuel,
    coût inchangé).

    Passe 2 : SEULEMENT SI la passe 1 échoue (ambiguity=True et
    confidence='faible', c'est-à-dire aucun positif franc trouvé),
    élargir aux candidats classés jusqu'à `taille_pool_max` (défaut 30).
    Les candidats déjà évalués en passe 1 sont réutilisés tels quels
    (via candidats_deja_evalues) — AUCUN appel Gemini n'est répété.

    Le tri chronologique + arrêt anticipé déjà présents dans
    detecter_ko_par_vision() s'appliquent normalement sur le pool
    élargi : si le bon candidat est proche chronologiquement du début,
    le coût de la passe 2 reste faible même avec un pool à 30.

    top_n_complet : liste COMPLÈTE des candidats disponibles, DÉJÀ
    triée par proba RF décroissante (comme la sortie habituelle du
    scoring RF) — au moins `taille_pool_max` éléments si possible.

    Retourne le même contrat que detecter_ko_par_vision(), avec deux
    clés ajoutées pour la traçabilité :
      "pool_final_utilise" : taille_pool_initial ou taille_pool_max
      "passe2_declenchee" : bool
    """
    pool_initial = top_n_complet[:taille_pool_initial]

    resultat = detecter_ko_par_vision(
        video_path, pool_initial, fps_source=fps_source, model=model,
        seuil_ambiguite_secondes=seuil_ambiguite_secondes,
        max_appels_par_candidat=max_appels_par_candidat, offset_s=offset_s,
    )

    echec_passe1 = (resultat.get("confidence") == "faible" and resultat.get("ambiguity") is True)

    if not echec_passe1 or len(top_n_complet) <= taille_pool_initial:
        resultat["pool_final_utilise"] = len(pool_initial)
        resultat["passe2_declenchee"] = False
        return resultat

    print(f"  [adaptatif] Passe 1 (top-{taille_pool_initial}) non concluante — "
          f"élargissement à top-{taille_pool_max}...")

    # Réutiliser les résultats déjà obtenus en passe 1, indexés par t,
    # pour ne jamais rappeler Gemini sur les mêmes candidats.
    cache = {r["t"]: r for r in resultat["candidats_evalues"] if r.get("evalue")}

    pool_elargi = top_n_complet[:taille_pool_max]
    resultat_final = detecter_ko_par_vision(
        video_path, pool_elargi, fps_source=fps_source, model=model,
        seuil_ambiguite_secondes=seuil_ambiguite_secondes,
        max_appels_par_candidat=max_appels_par_candidat, offset_s=offset_s,
        candidats_deja_evalues=cache,
    )
    resultat_final["pool_final_utilise"] = len(pool_elargi)
    resultat_final["passe2_declenchee"] = True
    return resultat_final
