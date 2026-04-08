# ai/highlight_scorer.py
# -*- coding: utf-8 -*-
"""
Scoring intelligent des highlights par Gemini Vision.
FIX 4 — sur 503/indisponibilité : les goals non scorés
         ne sont PAS promus automatiquement.
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
    _call_gemini,        # FIX 4 — utiliser _call_gemini au lieu de generate_content direct
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
# FIX 4 — score conservateur utilisé quand Gemini est indisponible
#          au lieu d'attribuer un score arbitrairement élevé
# ─────────────────────────────────────────
FALLBACK_SCORES = {
    "goal":             7.0,   # But confirmé par le pipeline → score honorable
    "shot":             4.0,   # Tir → score moyen
    "dribble":          3.0,
    "interception":     3.0,
    "fast_break":       4.5,
    "progressive_run":  2.5,
    "action":           2.0,
    "score":            7.0,
}

def fallback_score(highlight):
    """
    Score conservateur basé sur le type, utilisé quand Gemini est indisponible.
    Ne surclasse jamais un goal non confirmé.
    """
    h_type = highlight.get("main_type", "action")
    return FALLBACK_SCORES.get(h_type, 2.0)


# ─────────────────────────────────────────
# SCORER UN HIGHLIGHT
# ─────────────────────────────────────────
def score_highlight(video_path, highlight, sport="football"):
    """
    Score un highlight avec Gemini.
    FIX 4 — utilise _call_gemini() qui gère 503 proprement.
    Retourne None si Gemini indisponible (pas de score fallback élevé).
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
            f"ATTENTION : si le gardien tient le ballon ou effectue une relance, "
            f"ce n'est PAS un but — type_reel = 'goalkeeper_action', score <= 2."
        )]

        for i, frame in enumerate(frames):
            parts.append(text_to_part(f"Frame {i+1}/4 :"))
            parts.append(frame_to_part(frame))

        # FIX 4 — _call_gemini gère 503 sans accepter silencieusement
        response = _call_gemini(client, parts)

        if response is None:
            # 503 ou quota → retourner None, pas de score fallback élevé
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
# FIX 4 — sur 503 : score fallback conservateur au lieu de garder tout
# ─────────────────────────────────────────
def score_all_highlights(highlights, video_path, sport="football", max_highlights=15):
    """
    Score tous les highlights avec Gemini et les trie par score décroissant.
    FIX 4 :
      - Si Gemini retourne None (503) → score fallback conservateur
      - Les goals non confirmés par Gemini restent avec score 7.0 max
        (jamais promus à 10 par défaut)
      - Les goalkeeper_action détectés par Gemini sont filtrés
    """
    if not GEMINI_AVAILABLE or not highlights:
        # Pas de Gemini du tout → fallback sur tous
        for h in highlights:
            if "score" not in h:
                h["score"] = fallback_score(h)
        return sorted(highlights, key=lambda x: -x.get("score", 0))[:max_highlights]

    print(f"  Scoring Gemini : {len(highlights)} highlights...")

    scored   = []
    filtered = 0
    gemini_unavailable_count = 0

    for h in highlights:
        result = score_highlight(video_path, h, sport)

        if result is None:
            # FIX 4 — Gemini indisponible → score fallback conservateur
            gemini_unavailable_count += 1
            h["score"]             = fallback_score(h)
            h["gemini_scored"]     = False
            h["gemini_unavailable"] = True
            scored.append(h)
            continue

        # Filtrer goalkeeper_action et non-events avec score bas
        if result["type_reel"] in ["goalkeeper_action", "touche", "corner", "none"] \
                and result["score"] < 4:
            filtered += 1
            continue

        # Enrichir le highlight
        h["score"]         = result["score"]
        h["main_type"]     = result["type_reel"]
        h["spectaculaire"] = result["spectaculaire"]
        h["important"]     = result["important"]
        h["description"]   = result["description"]
        h["titre"]         = result["titre"]
        h["gemini_scored"] = True

        scored.append(h)

    if gemini_unavailable_count:
        print(f"  ⚠️  {gemini_unavailable_count} highlights scorés en fallback "
              f"(Gemini indisponible)")

    # Trier : goals Gemini-confirmés > goals fallback > spectaculaires > reste
    scored.sort(key=lambda x: (
        x.get("main_type") == "goal" and x.get("gemini_scored", False),
        x.get("main_type") == "goal",
        x.get("spectaculaire", False),
        x.get("score", 0)
    ), reverse=True)

    top = scored[0].get("titre") or scored[0].get("main_type", "?") if scored else "?"
    print(f"  Scoring : {len(scored)} gardés | {filtered} filtrés | top={top}")

    return scored[:max_highlights]