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
# FONCTION PRINCIPALE — CONTRAT candidate + confidence + ambiguity
# ─────────────────────────────────────────

def detecter_ko_par_vision(video_path, top_n_candidats, fps_source=None,
                             model="gemini-2.5-flash", seuil_ambiguite=0.15):
    """
    Point d'entrée principal. Prend le top-N candidats déjà produit par
    le RandomForest (ko_features.py + le modèle entraîné), départage-les
    par IA vision, retourne le contrat fonctionnel V1.

    top_n_candidats : liste de dicts [{"t": float, "categorie": str,
    "proba": float}, ...] déjà triée par proba RF décroissante — c'est
    exactement le format déjà produit par les scripts d'évaluation du
    projet (cf. notebook_extraction_images_top10.py et similaires).

    Retourne :
    {
        "timestamp_final": float ou None,
        "confidence": "élevée" / "moyenne" / "faible",
        "ambiguity": bool,
        "candidats_evalues": [...détail par candidat...],
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

    resultats = []
    for c in top_n_candidats:
        images = extraire_sequence_candidat(video_path, c["t"], fps_source=fps_source)
        jugement = demander_gemini_ko(images, client, model=model)
        resultats.append({
            "t": c["t"],
            "categorie_rf": c.get("categorie"),
            "proba_rf": c.get("proba"),
            "jugement_vision": jugement,
        })

    # Ne garder que les candidats où l'IA vision voit effectivement la transition
    candidats_positifs = [
        r for r in resultats
        if r["jugement_vision"] is not None and r["jugement_vision"]["transition_visible"]
    ]

    if not candidats_positifs:
        # Aucun candidat ne montre la transition -> ambiguïté réelle,
        # pas de faux positif à fabriquer. Repli sur le meilleur RF pur,
        # avec confiance explicitement faible.
        meilleur_rf = top_n_candidats[0] if top_n_candidats else None
        return {
            "timestamp_final": meilleur_rf["t"] if meilleur_rf else None,
            "confidence": "faible",
            "ambiguity": True,
            "candidats_evalues": resultats,
            "note": "Aucune transition claire détectée par l'IA vision — repli sur le meilleur candidat RF, à traiter avec prudence.",
        }

    # Parmi les candidats positifs, prendre celui à la confiance vision la plus haute
    candidats_positifs.sort(key=lambda r: r["jugement_vision"]["confidence"], reverse=True)
    meilleur = candidats_positifs[0]

    # Ambiguïté : plusieurs candidats positifs avec confiance proche
    ambiguity = False
    if len(candidats_positifs) > 1:
        ecart = meilleur["jugement_vision"]["confidence"] - candidats_positifs[1]["jugement_vision"]["confidence"]
        ambiguity = ecart < seuil_ambiguite

    conf_valeur = meilleur["jugement_vision"]["confidence"]
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
