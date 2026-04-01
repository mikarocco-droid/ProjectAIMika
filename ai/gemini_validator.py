# ai/gemini_validator.py
# -*- coding: utf-8 -*-

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
# ENCODER FRAME → BYTES JPEG
# ─────────────────────────────────────────
def encode_frame(frame):
    """Encode une frame numpy BGR en JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def frame_to_part(frame):
    """Convertit une frame numpy en types.Part Gemini."""
    return types.Part.from_bytes(
        data      = encode_frame(frame),
        mime_type = "image/jpeg"
    )


def text_to_part(text):
    """Convertit un texte en types.Part Gemini."""
    return types.Part.from_text(text=text)


# ─────────────────────────────────────────
# EXTRAIRE FRAMES AUTOUR D'UN EVENT
# ─────────────────────────────────────────
def extract_frames_around(video_path, frame_id, fps=25, n_frames=3):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    offsets = [-int(fps * 0.5), 0, int(fps * 0.5)]

    for offset in offsets[:n_frames]:
        target = max(0, frame_id + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
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
    if not GEMINI_AVAILABLE:
        return None

    frame_id = event.get("frame", 0)
    frames   = extract_frames_around(video_path, frame_id, fps, n_frames=3)
    if not frames:
        return None

    try:
        client = get_client()

        # FIX — utiliser types.Part au lieu de dicts
        parts = [text_to_part(
            f"Tu es un expert analyste de {sport}. "
            f"Voici 3 frames autour d'un événement suspect "
            f"(type détecté : {event.get('type','?')}).\n\n"
            f"Détermine exactement ce qui s'est passé.\n"
            f"Réponds UNIQUEMENT en JSON valide sans markdown :\n"
            f'{{"type": "goal|shot|corner|touche|none", '
            f'"confiance": 0.95, '
            f'"equipe": 0, '
            f'"description": "description courte"}}\n\n'
            f"- goal : ballon franchit la ligne de but\n"
            f"- shot : tir cadré ou non\n"
            f"- corner : corner ou remise en coin\n"
            f"- touche : sortie en touche\n"
            f"- none : rien de notable\n"
            f"equipe : 0 ou 1 selon maillot, null si incertain"
        )]

        for i, frame in enumerate(frames):
            parts.append(text_to_part(f"Frame {i+1}/3 :"))
            parts.append(frame_to_part(frame))

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = parts
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
            crop = cv2.resize(crop, (80, 160), interpolation=cv2.INTER_CUBIC)
            crops.append((p["id"], crop))

        cap.release()

        if not crops:
            return {}

        parts = [text_to_part(
            f"Voici {len(crops)} crops de joueurs. "
            f"Pour chaque image, lis le numéro sur le maillot.\n"
            f"JSON sans markdown :\n"
            f'{{"jerseys": [{{"index": 0, "numero": 9}}, {{"index": 1, "numero": null}}]}}\n'
            f"- numero : entier 1-99 si lisible, null sinon"
        )]

        for i, (tid, crop) in enumerate(crops):
            parts.append(text_to_part(f"Joueur {i} (ID={tid}) :"))
            parts.append(frame_to_part(crop))

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = parts
        )

        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        jersey_map = {}
        for item in result.get("jerseys", []):
            idx    = item.get("index", -1)
            numero = item.get("numero")
            if 0 <= idx < len(crops) and numero is not None:
                jersey_map[crops[idx][0]] = int(numero)

        return jersey_map

    except Exception as e:
        print(f"  Gemini jersey error : {e}")
        return {}


# ─────────────────────────────────────────
# VALIDATION COMPLÈTE
# ─────────────────────────────────────────
def validate_events_with_gemini(
    events,
    video_path,
    fps      = 25,
    sport    = "football",
    min_conf = 0.7
):
    if not GEMINI_AVAILABLE:
        print("  Gemini non disponible — validation ignorée")
        return events

    candidates = [
        e for e in events
        if e.get("type") in ["goal", "shot"]
        and e.get("frame", 0) > 0
    ]

    if not candidates:
        return events

    print(f"  Validation Gemini : {len(candidates)} candidats...")
    validated = corrected = removed = 0

    for event in candidates:
        result = validate_event(video_path, event, fps, sport)
        if result is None:
            continue

        validated    += 1
        gemini_type   = result["type"]
        confiance     = result["confiance"]

        if confiance >= min_conf:
            if gemini_type in ["goal", "shot"]:
                if event["type"] != gemini_type:
                    event["type"] = gemini_type
                    corrected += 1
                if result.get("equipe") is not None:
                    event["team"] = result["equipe"]
            else:
                event["_remove"] = True
                removed += 1

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance

    events = [e for e in events if not e.get("_remove", False)]
    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")
    return events