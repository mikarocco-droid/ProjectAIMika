# ai/gemini_validator.py
# -*- coding: utf-8 -*-
"""
Validation des events (buts/tirs) et lecture des maillots
via Gemini Flash — bien plus fiable que l'heuristique géométrique.
"""

import os
import cv2
import json
import base64
import re
import numpy as np

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("google-genai non installé — validation Gemini désactivée")


# ─────────────────────────────────────────
# INIT CLIENT
# ─────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY manquante dans .env")
        _client = genai.Client(api_key=api_key)
    return _client


# ─────────────────────────────────────────
# ENCODER FRAME → BASE64
# ─────────────────────────────────────────
def encode_frame(frame):
    """Encode une frame numpy BGR en JPEG base64."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────
# EXTRAIRE FRAMES AUTOUR D'UN EVENT
# ─────────────────────────────────────────
def extract_frames_around(video_path, frame_id, fps=25, n_frames=3):
    """
    Extrait n_frames autour du frame_id (avant, pendant, après).
    Retourne une liste de frames numpy.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    offsets = [-int(fps * 0.5), 0, int(fps * 0.5)]  # -0.5s, 0, +0.5s

    for offset in offsets[:n_frames]:
        target = max(0, frame_id + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            # Redimensionner pour réduire la taille
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            frames.append(frame)

    cap.release()
    return frames


# ─────────────────────────────────────────
# VALIDER UN EVENT (BUT / TIR)
# ─────────────────────────────────────────
def validate_event(video_path, event, fps=25, sport="football"):
    """
    Envoie les frames autour de l'event à Gemini Flash
    pour confirmer le type réel.

    Retourne un dict :
    {
        "type":        "goal" | "shot" | "corner" | "touche" | "none",
        "confiance":   0.0 - 1.0,
        "equipe":      0 | 1 | None,
        "description": "texte libre"
    }
    """
    if not GEMINI_AVAILABLE:
        return None

    frame_id = event.get("frame", 0)
    frames   = extract_frames_around(video_path, frame_id, fps, n_frames=3)

    if not frames:
        return None

    try:
        client = get_client()

        # Construire le contenu multimodal
        content = []

        content.append({
            "type": "text",
            "text": (
                f"Tu es un expert analyste de {sport}. "
                f"Voici 3 frames extraites d'un match autour d'un événement suspect "
                f"(frame avant, frame de l'action, frame après).\n\n"
                f"Analyse ces images et détermine exactement ce qui s'est passé.\n"
                f"Réponds UNIQUEMENT en JSON valide sans markdown :\n"
                f'{{"type": "goal|shot|corner|touche|none", '
                f'"confiance": 0.95, '
                f'"equipe": 0, '
                f'"description": "description courte en français"}}\n\n'
                f"- goal : le ballon franchit la ligne de but\n"
                f"- shot : tir cadré ou non cadré\n"
                f"- corner : corner ou remise en jeu dans le coin\n"
                f"- touche : sortie en touche\n"
                f"- none : aucun event notable\n"
                f"equipe : 0 ou 1 selon la couleur du maillot du joueur concerné, null si incertain"
            )
        })

        for i, frame in enumerate(frames):
            content.append({
                "type": "text",
                "text": f"Frame {i+1}/3 :"
            })
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       encode_frame(frame)
                }
            })

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = content
        )

        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        return {
            "type":        result.get("type", "none"),
            "confiance":   float(result.get("confiance", 0.5)),
            "equipe":      result.get("equipe"),
            "description": result.get("description", "")
        }

    except Exception as e:
        print(f"  Gemini validate error : {e}")
        return None


# ─────────────────────────────────────────
# LIRE NUMÉROS DE MAILLOT
# ─────────────────────────────────────────
def read_jersey_numbers(video_path, players_with_frames, fps=25, max_players=20):
    """
    Envoie un batch de crops de joueurs à Gemini
    pour lire les numéros de maillot.

    players_with_frames : list de {
        "id": track_id,
        "frame_id": int,
        "bbox": [x1,y1,x2,y2]
    }

    Retourne : dict {track_id: jersey_number}
    """
    if not GEMINI_AVAILABLE or not players_with_frames:
        return {}

    try:
        client = get_client()
        cap    = cv2.VideoCapture(video_path)
        crops  = []

        for p in players_with_frames[:max_players]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame_id"])
            ret, frame = cap.read()
            if not ret:
                continue

            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            h_f, w_f = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_f, x2); y2 = min(h_f, y2)

            if x2 - x1 < 20 or y2 - y1 < 30:
                continue

            crop = frame[y1:y2, x1:x2]
            # Agrandir pour meilleure lisibilité
            crop = cv2.resize(crop, (80, 160), interpolation=cv2.INTER_CUBIC)
            crops.append((p["id"], crop))

        cap.release()

        if not crops:
            return {}

        content = [{
            "type": "text",
            "text": (
                f"Voici {len(crops)} crops de joueurs de football. "
                f"Pour chaque image, lis le numéro sur le maillot.\n"
                f"Réponds en JSON sans markdown :\n"
                f'{{"jerseys": [{{"index": 0, "numero": 9}}, {{"index": 1, "numero": null}}]}}\n'
                f"- numero : entier 1-99 si visible, null sinon\n"
                f"- index : position dans la liste (commence à 0)"
            )
        }]

        for i, (tid, crop) in enumerate(crops):
            content.append({"type": "text", "text": f"Joueur {i} (ID={tid}) :"})
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       encode_frame(crop)
                }
            })

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = content
        )

        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        jersey_map = {}
        for item in result.get("jerseys", []):
            idx    = item.get("index", -1)
            numero = item.get("numero")
            if 0 <= idx < len(crops) and numero is not None:
                track_id             = crops[idx][0]
                jersey_map[track_id] = int(numero)

        return jersey_map

    except Exception as e:
        print(f"  Gemini jersey error : {e}")
        return {}


# ─────────────────────────────────────────
# PIPELINE VALIDATION COMPLÈTE
# ─────────────────────────────────────────
def validate_events_with_gemini(
    events,
    video_path,
    fps        = 25,
    sport      = "football",
    min_conf   = 0.7,
    batch_size = 5
):
    """
    Valide les events suspects (goal/shot) avec Gemini.
    Filtre les faux positifs et corrige les types.

    Retourne les events corrigés.
    """
    if not GEMINI_AVAILABLE:
        print("  Gemini non disponible — validation ignorée")
        return events

    # Cibler uniquement les goals et shots importants
    candidates = [
        e for e in events
        if e.get("type") in ["goal", "shot"]
        and e.get("frame", 0) > 0
    ]

    if not candidates:
        return events

    print(f"  Validation Gemini : {len(candidates)} candidats...")

    validated  = 0
    corrected  = 0
    removed    = 0

    # Traiter par batch pour limiter les appels API
    for i, event in enumerate(candidates):
        result = validate_event(video_path, event, fps, sport)

        if result is None:
            continue

        validated += 1
        gemini_type = result["type"]
        confiance   = result["confiance"]

        original_type = event["type"]

        if confiance >= min_conf:
            if gemini_type in ["goal", "shot"]:
                # Corriger le type si nécessaire
                if event["type"] != gemini_type:
                    event["type"] = gemini_type
                    corrected += 1
                # Mettre à jour l'équipe si détectée
                if result.get("equipe") is not None:
                    event["team"] = result["equipe"]
            else:
                # Faux positif (touche, corner, none) → marquer pour suppression
                event["_remove"] = True
                removed += 1

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance

    # Supprimer les faux positifs confirmés
    events = [e for e in events if not e.get("_remove", False)]

    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")

    return events