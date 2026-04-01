# ai/highlight_scorer.py
# -*- coding: utf-8 -*-
"""
Scoring intelligent des highlights par Gemini Vision.
Remplace le scoring basique par score_event().
Évalue chaque clip sur : spectaculaire, important, contexte.
"""

import os
import cv2
import json
import re
import subprocess
import tempfile

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ai.gemini_validator import get_client, encode_frame


# ─────────────────────────────────────────
# EXTRAIRE FRAMES D'UN CLIP
# ─────────────────────────────────────────
def extract_clip_frames(video_path, time_start, time_end, n=4):
    """Extrait n frames d'un segment vidéo."""
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []

    duration = time_end - time_start
    for i in range(n):
        t   = time_start + duration * (i + 0.5) / n
        pos = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            frames.append(frame)

    cap.release()
    return frames


# ─────────────────────────────────────────
# SCORER UN HIGHLIGHT
# ─────────────────────────────────────────
def score_highlight(video_path, highlight, sport="football"):
    """
    Score un highlight avec Gemini.

    Retourne un dict :
    {
        "score":         8.5,    # 0-10
        "type_reel":     "goal",
        "spectaculaire": true,
        "important":     true,
        "description":   "Superbe frappe enroulée en lucarne",
        "titre":         "But splendide #9 — 01:47"
    }
    """
    if not GEMINI_AVAILABLE:
        return None

    frames = extract_clip_frames(
        video_path,
        highlight.get("time_start", 0),
        highlight.get("time_end",   10),
        n=4
    )

    if not frames:
        return None

    try:
        client    = get_client()
        t         = highlight.get("time_start", 0)
        mins      = int(t // 60)
        secs      = int(t % 60)
        h_type    = highlight.get("main_type", "action")

        content = [{
            "type": "text",
            "text": (
                f"Tu es un expert en {sport}. "
                f"Voici 4 frames d'un clip à {mins:02d}:{secs:02d} "
                f"(type détecté : {h_type}).\n\n"
                f"Évalue ce moment et réponds en JSON sans markdown :\n"
                f"{{\n"
                f'  "score": 8.5,\n'
                f'  "type_reel": "goal",\n'
                f'  "spectaculaire": true,\n'
                f'  "important": true,\n'
                f'  "description": "Description courte et précise en français",\n'
                f'  "titre": "Titre accrocheur court"\n'
                f"}}\n\n"
                f"- score : 0 (sans intérêt) à 10 (moment historique)\n"
                f"- type_reel : goal/shot/dribble/interception/corner/touche/none\n"
                f"- spectaculaire : visuellement impressionnant\n"
                f"- important : change le cours du match"
            )
        }]

        for i, frame in enumerate(frames):
            content.append({"type": "text",  "text": f"Frame {i+1}/4 :"})
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
            "score":         float(result.get("score", 5.0)),
            "type_reel":     result.get("type_reel", h_type),
            "spectaculaire": result.get("spectaculaire", False),
            "important":     result.get("important", False),
            "description":   result.get("description", ""),
            "titre":         result.get("titre", "")
        }

    except Exception as e:
        print(f"  Highlight scorer error : {e}")
        return None


# ─────────────────────────────────────────
# SCORER TOUS LES HIGHLIGHTS
# ─────────────────────────────────────────
def score_all_highlights(highlights, video_path, sport="football", max_highlights=15):
    """
    Score tous les highlights avec Gemini et les trie par score décroissant.
    Filtre les non-events (touche, corner, none).

    Retourne les highlights enrichis et triés.
    """
    if not GEMINI_AVAILABLE or not highlights:
        return highlights

    print(f"  Scoring Gemini : {len(highlights)} highlights...")

    scored   = []
    filtered = 0

    for i, h in enumerate(highlights):
        result = score_highlight(video_path, h, sport)

        if result is None:
            scored.append(h)
            continue

        # Filtrer les non-events avec score bas
        if result["type_reel"] in ["touche", "corner", "none"] and result["score"] < 4:
            filtered += 1
            continue

        # Enrichir le highlight
        h["score"]         = result["score"]
        h["main_type"]     = result["type_reel"]
        h["spectaculaire"] = result["spectaculaire"]
        h["important"]     = result["important"]
        h["description"]   = result["description"]
        h["titre"]         = result["titre"]

        scored.append(h)

    # Trier : goals/spectaculaires en premier
    scored.sort(key=lambda x: (
        x.get("main_type") == "goal",
        x.get("spectaculaire", False),
        x.get("score", 0)
    ), reverse=True)

    print(f"  Scoring : {len(scored)} gardés | {filtered} filtrés | "
          f"top={scored[0].get('titre','?') if scored else '?'}")

    return scored[:max_highlights]