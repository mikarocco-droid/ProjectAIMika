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

def extraire_sequence_candidat(video_path, t_candidat, fps_source=None):
    """
    Extrait 3 images autour d'un instant candidat :
      juste_avant (t-2s), candidat (t0), juste_apres (t+2s)

    Protocole validé (jalon section 18) : suffisant dans la grande
    majorité des cas. Limite connue : deux candidats à moins de 5-6s
    l'un de l'autre peuvent avoir des fenêtres qui se chevauchent
    (cf. l'erreur sur Stembert) — à surveiller si le top-10 contient
    des candidats très rapprochés temporellement.

    Retourne un dict {"juste_avant": np.array, "candidat": np.array,
    "juste_apres": np.array} (frames BGR, format OpenCV), ou None pour
    les images qui n'ont pas pu être lues.
    """
    cap = cv2.VideoCapture(video_path)
    if fps_source is None:
        fps_source = cap.get(cv2.CAP_PROP_FPS)

    offsets = {"juste_avant": -2.0, "candidat": 0.0, "juste_apres": 2.0}
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

PROMPT_KO_VISION = """Tu vas voir 3 images d'un match de football amateur, extraites à 2 secondes d'intervalle (image 1 = t-2s, image 2 = instant candidat, image 3 = t+2s).

Regarde ces 3 images comme les 3 étapes d'UNE SEULE histoire courte (4 secondes), pas comme 3 vérifications indépendantes. La question est : "est-ce que ces 3 images, prises ensemble, racontent le début d'un match (coup d'envoi) ?" — pas "est-ce que chaque image, séparément, coche toutes les cases d'une checklist".

L'histoire caractéristique d'un VRAI coup d'envoi se lit ainsi : les joueurs des deux équipes sont globalement en position (proche de leur moitié de terrain) au début → un mouvement de jeu démarre au centre → plusieurs joueurs se mettent en mouvement ensemble vers la fin. Le ballon fait partie de cette histoire mais n'a pas besoin d'être visible avec une netteté parfaite sur chaque image individuelle : il peut être petit, partiellement caché par un joueur, ou peu contrasté (éclairage nocturne, plan large) sur une image sans que ça invalide l'histoire globale si le mouvement collectif qui suit est cohérent avec une reprise de jeu.

Ne rejette PAS une séquence uniquement parce qu'un détail précis (comme la position exacte du ballon) n'est pas parfaitement visible sur une seule des 3 images — utilise le contexte des 2 autres images et la cohérence du mouvement général pour juger.

Attention aux pièges déjà identifiés, à lire aussi comme des histoires globales, pas des checklists :
- Un simple rassemblement (joueurs en cercle serré) qui reste une scène statique sur l'ensemble des 3 images, sans qu'aucun mouvement collectif ne démarre, n'est PAS un coup d'envoi.
- Un "faux départ" ressemble au début de l'histoire mais l'histoire ne se termine pas par une vraie dispersion générale (parfois juste un joueur qui ajuste son équipement, rien d'autre ne change).
- Si les deux équipes ont des couleurs de maillot proches ou que l'angle de caméra est peu favorable, base ton jugement sur la trajectoire globale du mouvement plutôt que sur des détails visuels fins impossibles à vérifier avec certitude.

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
{
  "transition_visible": true/false,
  "confidence": 0.0 à 1.0,
  "raisonnement": "1-2 phrases sur ce qui justifie la réponse, en te basant sur l'histoire globale des 3 images plutôt que sur un détail isolé"
}
"""


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
                             max_appels_par_candidat=3):
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

    OPTIMISATION COÛT (validée sur le raisonnement suivant, pas encore
    sur un vrai run à grande échelle — à surveiller) : les candidats
    sont traités PAR ORDRE CHRONOLOGIQUE CROISSANT, pas par proba RF
    décroissante. Dès qu'un candidat obtient une majorité positive, les
    candidats suivants ne sont évalués QUE s'ils sont à moins de
    seuil_ambiguite_secondes de celui-ci (pour détecter une ambiguïté
    réelle) ; au-delà, l'évaluation s'arrête complètement, car aucun
    candidat plus tardif ne pourra jamais être sélectionné (la règle de
    sélection prend toujours le plus ancien parmi les positifs).

    top_n_candidats : liste de dicts [{"t": float, "categorie": str,
    "proba": float}, ...] — l'ordre d'entrée n'importe pas, la fonction
    trie elle-même par t croissant.

    seuil_ambiguite_secondes : si un 2e candidat positif existe à moins
    de ce nombre de secondes du premier retenu, l'ambiguïté est signalée.

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

    # Tri chronologique croissant — condition nécessaire à l'arrêt anticipé
    candidats_tries = sorted(top_n_candidats, key=lambda c: c["t"])

    resultats = []
    candidats_positifs = []
    premier_positif_t = None

    for c in candidats_tries:
        # Arrêt anticipé : au-delà de la fenêtre d'ambiguïté autour du
        # premier positif trouvé, plus rien ne peut changer la décision.
        if premier_positif_t is not None and (c["t"] - premier_positif_t) >= seuil_ambiguite_secondes:
            resultats.append({
                "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
                "evalue": False, "vote": None,
            })
            continue

        images = extraire_sequence_candidat(video_path, c["t"], fps_source=fps_source)
        vote = voter_transition(images, client, model=model, max_appels=max_appels_par_candidat)

        resultats.append({
            "t": c["t"], "categorie_rf": c.get("categorie"), "proba_rf": c.get("proba"),
            "evalue": True, "vote": vote,
            # rétro-compatibilité avec l'ancien format (un seul jugement) :
            "jugement_vision": ({
                "transition_visible": vote["decision"],
                "confidence": vote["confidence"],
                "raisonnement": f"vote majoritaire {sum(1 for v in vote['votes'] if v['transition_visible'])}/{vote['n_appels']}",
            } if vote else None),
        })

        if vote is not None and vote["decision"]:
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
