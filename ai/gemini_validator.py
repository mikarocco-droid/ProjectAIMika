# ai/gemini_validator.py
# -*- coding: utf-8 -*-

import os
import cv2
import json
import base64
import re
import time
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
# FLAG QUOTA + FLAG 503
# FIX 4 — deux flags distincts :
#   _quota_exhausted : quota journalier épuisé → arrêt total
#   _gemini_unavailable : 503 temporaire → on marque les events
#     comme non-validés au lieu de les accepter automatiquement
# ─────────────────────────────────────────
_quota_exhausted   = False
_gemini_unavailable = False   # FIX 4 — 503 en cours

def _call_gemini(client, parts, max_retries=2):
    """
    Appel Gemini avec retry limité.
    FIX 4 — sur 503, on lève _gemini_unavailable au lieu
             d'accepter l'event silencieusement.
    """
    global _quota_exhausted, _gemini_unavailable

    if _quota_exhausted:
        return None

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model    = "gemini-2.5-flash",
                contents = parts
            )
            # Succès → réinitialiser le flag 503
            _gemini_unavailable = False
            return response

        except Exception as e:
            err_str = str(e)
            last_err = err_str

            if "RESOURCE_EXHAUSTED" in err_str:
                print("  ⚠️  Quota Gemini journalier épuisé — validation désactivée")
                _quota_exhausted = True
                return None

            elif "503" in err_str or "UNAVAILABLE" in err_str:
                # FIX 4 — 503 : on lève le flag et on retourne None
                # sans accepter l'event
                _gemini_unavailable = True
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None  # après retries → None, event NON validé

            elif "429" in err_str and attempt < max_retries - 1:
                m    = re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(m.group(1)) + 2 if m else 30
                wait = min(wait, 30)
                print(f"  ⏳ Rate limit Gemini — attente {wait}s...")
                time.sleep(wait)
            else:
                raise

    return None


# ─────────────────────────────────────────
# ENCODER FRAME → BYTES JPEG
# ─────────────────────────────────────────
def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

def frame_to_part(frame):
    return types.Part.from_bytes(
        data      = encode_frame(frame),
        mime_type = "image/jpeg"
    )

def text_to_part(text):
    return types.Part.from_text(text=text)


# ─────────────────────────────────────────
# EXTRAIRE FRAMES AUTOUR D'UN EVENT
# FIX 3 — on extrait aussi des frames AVANT l'event
#          pour donner le contexte (gardien qui tient la balle, etc.)
# ─────────────────────────────────────────
def extract_frames_around(video_path, frame_id, fps=25, n_before=2, n_after=2):
    """
    Extrait n_before frames avant l'event + frame event + n_after frames après.
    FIX 3 : le contexte avant est crucial pour distinguer
            gardien-qui-tient-la-balle d'un vrai but.
    """
    cap    = cv2.VideoCapture(video_path)
    frames = []

    # Offsets : -2s, -1s, 0, +0.5s, +1s
    before_offsets = [-int(fps * (n_before - i)) for i in range(n_before)]
    after_offsets  = [int(fps * (i + 1) * 0.5)  for i in range(n_after)]
    offsets        = before_offsets + [0] + after_offsets

    for offset in offsets:
        target = max(0, frame_id + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            frames.append((offset, frame))

    cap.release()
    return frames


# ─────────────────────────────────────────
# VALIDER UN EVENT (BUT / TIR)
# FIX 3 — prompt enrichi avec :
#   - description explicite du cas gardien
#   - frames labellisées avant/après pour le contexte
#   - danger score comme hint
# ─────────────────────────────────────────
def validate_event(video_path, event, fps=25, sport="football"):
    if not GEMINI_AVAILABLE:
        return None

    frame_id     = event.get("frame", 0)
    danger       = event.get("danger", 0)
    event_type   = event.get("type", "?")
    frames_data  = extract_frames_around(video_path, frame_id, fps, n_before=2, n_after=2)

    if not frames_data:
        return None

    try:
        client = get_client()

        # FIX 3 — prompt enrichi avec cas gardien explicitement mentionné
        parts = [text_to_part(
            f"Tu es un expert analyste de {sport}. "
            f"Voici {len(frames_data)} frames chronologiques autour d'un événement suspect "
            f"(type détecté : {event_type}, danger score : {danger:.1f}/10).\n\n"
            f"Les frames AVANT (offset négatif) montrent la situation AVANT l'événement. "
            f"Utilise ce contexte pour ne pas confondre :\n"
            f"- Un GARDIEN QUI TIENT LE BALLON devant son but avec un but\n"
            f"- Une RELANCE À LA MAIN DU GARDIEN avec un tir\n"
            f"- Un DÉGAGEMENT avec un but\n\n"
            f"Détermine exactement ce qui s'est passé.\n"
            f"Réponds UNIQUEMENT en JSON valide sans markdown :\n"
            f'{{"type": "goal|shot|corner|touche|goalkeeper_hold|goalkeeper_throw|none", '
            f'"confiance": 0.95, '
            f'"equipe": 0, '
            f'"description": "description courte"}}\n\n'
            f"Types possibles :\n"
            f"- goal : ballon franchit CLAIREMENT la ligne de but\n"
            f"- shot : tir cadré ou non cadré\n"
            f"- goalkeeper_hold : gardien qui tient/porte le ballon\n"
            f"- goalkeeper_throw : relance à la main ou au pied du gardien\n"
            f"- corner : remise en coin\n"
            f"- touche : sortie en touche\n"
            f"- none : rien de notable\n"
            f"equipe : 0 ou 1 selon couleur maillot, null si incertain"
        )]

        for offset, frame in frames_data:
            label = (
                f"Frame AVANT ({offset//fps:.1f}s)" if offset < 0
                else "Frame ÉVÉNEMENT"              if offset == 0
                else f"Frame APRÈS (+{offset//fps:.1f}s)"
            )
            parts.append(text_to_part(f"{label} :"))
            parts.append(frame_to_part(frame))

        response = _call_gemini(client, parts)

        # FIX 4 — si Gemini indisponible, retourner None
        # (l'appelant devra traiter ce cas explicitement)
        if response is None:
            return None

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

        response = _call_gemini(client, parts)
        if response is None:
            return {}

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
# FIX 4 — sur 503/None :
#   - les goals NON validés par Gemini sont rejetés si danger < 8
#   - les shots NON validés sont conservés (moins critique)
# ─────────────────────────────────────────
def validate_events_with_gemini(
    events,
    video_path,
    fps      = 25,
    sport    = "football",
    min_conf = 0.7
):
    global _gemini_unavailable

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
        if _quota_exhausted:
            print("  ⚠️  Quota épuisé — validation Gemini interrompue")
            break

        result = validate_event(video_path, event, fps, sport)

        # FIX 4 — Gemini indisponible (503 ou None) :
        # on NE valide PAS automatiquement
        if result is None:
            if event.get("type") == "goal":
                danger = event.get("danger", 0) or 0
                if danger < 8.0:
                    # FIX 4 — goal non validé + danger faible → rejeté
                    event["_remove"] = True
                    removed += 1
                    print(f"  FIX4 : goal à t={event.get('time',0):.0f}s rejeté "
                          f"(Gemini indisponible, danger={danger:.1f})")
                else:
                    # Danger très élevé → on garde avec flag d'avertissement
                    event["gemini_validated"] = False
                    event["gemini_unavailable"] = True
            # Pour les shots → on conserve toujours (moins critique)
            continue

        # Pause entre appels pour rester sous la RPM limit
        time.sleep(2)

        validated  += 1
        gemini_type = result["type"]
        confiance   = result["confiance"]

        if confiance >= min_conf:
            # FIX 3 — nouveaux types goalkeeper_hold / goalkeeper_throw → supprimer
            if gemini_type in ["goal", "shot"]:
                if event["type"] != gemini_type:
                    event["type"] = gemini_type
                    corrected += 1
                if result.get("equipe") is not None:
                    event["team"] = result["equipe"]
            elif gemini_type in ["goalkeeper_hold", "goalkeeper_throw",
                                  "corner", "touche", "none"]:
                event["_remove"] = True
                removed += 1
        else:
            # Confiance insuffisante sur un goal → rejeté par précaution
            if event.get("type") == "goal":
                event["_remove"] = True
                removed += 1

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance

    events = [e for e in events if not e.get("_remove", False)]
    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")
    return events