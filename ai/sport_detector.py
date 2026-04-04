# ai/sport_detector.py
# -*- coding: utf-8 -*-

import cv2
import json
import re

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ai.gemini_validator import get_client, frame_to_part, text_to_part, _call_gemini


SUPPORTED_SPORTS = {
    "football":         ["football", "soccer", "foot"],
    "mini-foot":        ["futsal", "mini-foot", "mini foot", "5v5"],
    "basketball":       ["basketball", "basket", "nba"],
    "handball":         ["handball", "hand"],
    "rugby":            ["rugby", "rugby à XV", "rugby à 7"],
    "hockey sur glace": ["hockey sur glace", "ice hockey", "hockey glace"],
    "hockey sur gazon": ["hockey sur gazon", "field hockey"],
    "tennis":           ["tennis"],
    "tennis de table":  ["tennis de table", "ping pong", "ping-pong"],
    "padel":            ["padel", "padel tennis"],
}


def detect_sport(video_path, fallback="football"):
    if not GEMINI_AVAILABLE:
        return fallback

    try:
        # FIX — vérifier le flag quota avant d'appeler Gemini
        from ai.gemini_validator import _quota_exhausted
        if _quota_exhausted:
            print(f"  Sport detector ignoré — quota épuisé → fallback {fallback}")
            return fallback

        client = get_client()
        cap    = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return fallback

        frames = []
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for pos in [int(total * 0.05), int(total * 0.15), int(total * 0.30)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                frames.append(frame)

        cap.release()

        if not frames:
            return fallback

        sports_list = ", ".join(SUPPORTED_SPORTS.keys())

        parts = [text_to_part(
            f"Regarde ces frames et identifie le sport.\n"
            f"Sports possibles : {sports_list}\n\n"
            f"JSON sans markdown :\n"
            f'{{"sport": "football", "confiance": 0.98, "raison": "terrain vert, but visible"}}'
        )]

        for i, frame in enumerate(frames):
            parts.append(text_to_part(f"Frame {i+1} :"))
            parts.append(frame_to_part(frame))

        # FIX — passer par _call_gemini pour gérer quota + retry
        response = _call_gemini(client, parts)
        if response is None:
            return fallback

        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        detected  = result.get("sport", "").lower().strip()
        confiance = float(result.get("confiance", 0))

        for sport_key, aliases in SUPPORTED_SPORTS.items():
            if detected == sport_key or any(a in detected for a in aliases):
                if confiance >= 0.7:
                    print(f"  Sport détecté : {sport_key} "
                          f"(confiance={confiance:.0%}, raison: {result.get('raison','')})")
                    return sport_key

        print(f"  Sport non reconnu : '{detected}' → fallback {fallback}")
        return fallback

    except Exception as e:
        print(f"  Sport detector error : {e} → fallback {fallback}")
        return fallback