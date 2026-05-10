# ai/highlight_scorer.py
# -*- coding: utf-8 -*-
"""
Scoring intelligent des highlights par Gemini Vision.
FIX 4 — sur 503 : score fallback conservateur, pas de promotion automatique.
FIX position — goalkeeper_action et defensive_clearance filtrés.
"""

import os
import cv2
import json
import re
import time

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ai.gemini_validator import (
    get_client,
    frame_to_part,
    text_to_part,
    _call_gemini,
    _is_near_goal,
)


# ─────────────────────────────────────────
# EXTRAIRE FRAMES D'UN CLIP
# ─────────────────────────────────────────
def extract_clip_frames(video_path, time_start, time_end, n=4):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []

    time_start = float(time_start or 0)
    time_end   = float(time_end   or 0)
    if time_end <= time_start:
        time_end = time_start + 3.0

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
# SCORE FALLBACK (sans Gemini)
# FIX 4 — conservateur : jamais de promotion automatique à 10
# ─────────────────────────────────────────
FALLBACK_SCORES = {
    "goal":             7.0,
    "shot":             4.0,
    "dribble":          3.0,
    "interception":     3.0,
    "fast_break":       4.5,
    "progressive_run":  2.5,
    "action":           2.0,
    "score":            7.0,
}

def fallback_score(highlight):
    h_type = highlight.get("main_type", "action")
    return FALLBACK_SCORES.get(h_type, 2.0)


# ─────────────────────────────────────────
# SCORER UN HIGHLIGHT
# ─────────────────────────────────────────
def score_highlight(video_path, highlight, sport="football"):
    """
    Score un highlight avec Gemini.
    FIX 4 — utilise _call_gemini() qui gère 503 proprement.
    Retourne None si Gemini indisponible.
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
        client = get_client()
        t      = highlight.get("time_start", 0)
        mins   = int(t // 60)
        secs   = int(t % 60)
        h_type = highlight.get("main_type", "action")

        parts = [text_to_part(
            f"Tu es un expert en {sport}. "
            f"Voici 4 frames d'un clip à {mins:02d}:{secs:02d} "
            f"(type détecté : {h_type}).\n\n"
            f"Évalue ce moment en JSON sans markdown :\n"
            f"{{\n"
            f'  "score": 8.5,\n'
            f'  "type_reel": "goal",\n'
            f'  "spectaculaire": true,\n'
            f'  "important": true,\n'
            f'  "description": "Description courte en français",\n'
            f'  "titre": "Titre accrocheur"\n'
            f"}}\n"
            f"score : 0 (sans intérêt) à 10 (moment historique)\n\n"
            f"ATTENTION — ces situations NE sont PAS des buts, score <= 2 :\n"
            f"- Gardien qui tient ou porte le ballon → type_reel = 'goalkeeper_action'\n"
            f"- Relance à la main ou au pied du gardien → type_reel = 'goalkeeper_action'\n"
            f"- Dégagement de tête d'un défenseur → type_reel = 'defensive_clearance'\n"
            f"- Dégagement du poing du gardien → type_reel = 'goalkeeper_action'\n"
            f"- Passe latérale ou longue balle qui sort en touche sans danger → type_reel = 'touche', score <= 2\n"
            f"- Balle qui sort en touche depuis le milieu du terrain → type_reel = 'touche', score <= 2\n"
            f"- Action sans danger réel hors de la surface adverse → score <= 3"
        )]

        for i, frame in enumerate(frames):
            parts.append(text_to_part(f"Frame {i+1}/4 :"))
            parts.append(frame_to_part(frame))

        # FIX 4 — _call_gemini gère 503 sans accepter silencieusement
        response = _call_gemini(client, parts)

        if response is None:
            return None

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
# FIX 4 + FIX position
# ─────────────────────────────────────────
def score_all_highlights(
    highlights,
    video_path,
    sport          = "football",
    max_highlights = 15,
    frame_w        = 1920,   # FIX position
):
    """
    Score tous les highlights avec Gemini et les trie par score décroissant.
    FIX 4 : score fallback conservateur sur 503.
    FIX position : goalkeeper_action et defensive_clearance filtrés.
    """
    if not GEMINI_AVAILABLE or not highlights:
        for h in highlights:
            if "score" not in h:
                h["score"] = fallback_score(h)
        return sorted(highlights, key=lambda x: -x.get("score", 0))[:max_highlights]

    print(f"  Scoring Gemini : {len(highlights)} highlights...")

    scored                   = []
    filtered                 = 0
    gemini_unavailable_count = 0

    for h in highlights:
        result = score_highlight(video_path, h, sport)

        if result is None:
            # FIX 4 — Gemini indisponible → score fallback conservateur
            gemini_unavailable_count += 1
            h["score"]              = fallback_score(h)
            h["gemini_scored"]      = False
            h["gemini_unavailable"] = True

            # FIX position — même en fallback, on vérifie la position
            # Un goal dont la position x est loin des buts est dégradé
            if h.get("main_type") == "goal":
                # Récupère x depuis les events du highlight si disponible
                events = h.get("events", [])
                xs = [e.get("x", 0) for e in events if e.get("x")]
                if xs:
                    x_mean = sum(xs) / len(xs)
                    if not _is_near_goal(x_mean, frame_w):
                        h["score"] = 1.0  # dégagement → score minimal
                        h["main_type"] = "defensive_clearance"
                        filtered += 1
                        print(f"  FIX position : highlight goal dégradé "
                              f"(x_mean={x_mean:.0f}, loin du but)")
                        continue

            scored.append(h)
            continue

        # FIX position — filtrer goalkeeper_action, defensive_clearance
        if result["type_reel"] in [
            "goalkeeper_action", "defensive_clearance",
            "touche", "corner", "none"
        ] and result["score"] < 5:
            filtered += 1
            continue

        h["score"]         = result["score"]
        # V9.7 — ne jamais promouvoir un shot en goal via le scorer
        original_type = h.get("main_type", "shot")
        scored_type   = result["type_reel"]
        if original_type == "shot" and scored_type == "goal":
            h["main_type"] = "shot"   # on garde le type original
        else:
            h["main_type"] = scored_type
        h["spectaculaire"] = result["spectaculaire"]
        h["important"]     = result["important"]
        h["description"]   = result["description"]
        h["titre"]         = result["titre"]
        h["gemini_scored"] = True

        scored.append(h)

    if gemini_unavailable_count:
        print(f"  ⚠️  {gemini_unavailable_count} highlights scorés en fallback "
              f"(Gemini indisponible)")

    # Tri : goals Gemini-confirmés > goals fallback > spectaculaires > reste
    scored.sort(key=lambda x: (
        x.get("main_type") == "goal" and x.get("gemini_scored", False),
        x.get("main_type") == "goal",
        x.get("spectaculaire", False),
        x.get("score", 0)
    ), reverse=True)

    top = scored[0].get("titre") or scored[0].get("main_type", "?") if scored else "?"
    print(f"  Scoring : {len(scored)} gardés | {filtered} filtrés | top={top}")

    return scored[:max_highlights]