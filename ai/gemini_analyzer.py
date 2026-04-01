# ai/gemini_analyzer.py
# -*- coding: utf-8 -*-
"""
Analyse tactique complète par Gemini Vision.
Remplace l'heuristique de tactical.py par une vraie analyse visuelle.
Multi-sport nativement.
"""

import os
import cv2
import json
import base64
import re

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ai.gemini_validator import get_client, frame_to_part, text_to_part


# ─────────────────────────────────────────
# EXTRAIRE FRAMES REPRÉSENTATIVES
# ─────────────────────────────────────────
def extract_tactical_frames(video_path, fps=25, n=6):
    """
    Extrait n frames réparties sur toute la vidéo
    pour l'analyse tactique globale.
    """
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames       = []

    for i in range(n):
        pos = int(total_frames * (i + 1) / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            frames.append((pos, frame))

    cap.release()
    return frames


# ─────────────────────────────────────────
# ANALYSE TACTIQUE COMPLÈTE
# ─────────────────────────────────────────
def analyze_tactics(video_path, sport="football", fps=25, events=None):
    """
    Analyse tactique complète via Gemini Vision.

    Retourne un dict enrichi par rapport à tactical.py :
    {
        "formation":          "4-3-3",
        "formation_adverse":  "4-4-2",
        "style":              "possession | direct | vertical | pressing",
        "pressing":           true,
        "pressing_zone":      "haut | milieu | bas",
        "transitions":        "rapides | lentes",
        "cotes_dominants":    "gauche | droite | centre",
        "observations":       ["obs1", "obs2", "obs3"],
        "forces":             ["force1", "force2"],
        "faiblesses":         ["faiblesse1", "faiblesse2"],
        "score_estime":       "1-0",
        "gemini_analysed":    true
    }
    """
    if not GEMINI_AVAILABLE:
        return {"gemini_analysed": False}

    try:
        client = get_client()
        frames = extract_tactical_frames(video_path, fps, n=6)

        if not frames:
            return {"gemini_analysed": False}

        # Stats events pour contexte
        context = ""
        if events:
            goals  = sum(1 for e in events if e.get("type") == "goal")
            shots  = sum(1 for e in events if e.get("type") == "shot")
            passes = sum(1 for e in events if e.get("type") == "pass")
            context = (
                f"\nContexte détecté automatiquement : "
                f"{goals} buts, {shots} tirs, {passes} passes."
            )

        parts = [text_to_part(
            f"Tu es un analyste tactique expert en {sport}. "
            f"Voici {len(frames)} frames extraites à intervalles réguliers.{context}\n\n"
            f"Analyse la tactique et réponds en JSON sans markdown :\n"
            f"{{\n"
            f'  "formation": "4-3-3",\n'
            f'  "formation_adverse": "4-4-2",\n'
            f'  "style": "possession",\n'
            f'  "pressing": true,\n'
            f'  "pressing_zone": "haut",\n'
            f'  "transitions": "rapides",\n'
            f'  "cotes_dominants": "gauche",\n'
            f'  "observations": ["obs1", "obs2"],\n'
            f'  "forces": ["force1"],\n'
            f'  "faiblesses": ["faiblesse1"],\n'
            f'  "score_estime": "1-0"\n'
            f"}}"
        )]

        for pos, frame in frames:
            mins = int(pos / fps / 60)
            secs = int((pos / fps) % 60)
            parts.append(text_to_part(f"Frame à {mins:02d}:{secs:02d} :"))
            parts.append(frame_to_part(frame))

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = parts
        )

        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        result["gemini_analysed"] = True

        print(f"  Gemini tactical : {result.get('formation')} | "
              f"{result.get('style')} | pressing={result.get('pressing')}")

        return result

    except Exception as e:
        print(f"  Gemini analyzer error : {e}")
        return {"gemini_analysed": False}