# analysis/detect_teams_preview.py
# -*- coding: utf-8 -*-
#
# V5 — Détection couleurs équipes via Gemini Vision
#
# Pipeline :
#   1. Tracker YOLO léger sur 2min → trouver les meilleurs frames
#   2. Sélectionner 5 frames avec beaucoup de joueurs proches
#   3. Gemini Vision analyse chaque frame → couleurs maillot + short
#   4. Vote majoritaire sur les 5 frames → 2 équipes

import os
import cv2
import numpy as np
import base64
import json


# ─────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────
def _frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def _ask_gemini_colors(frame, client):
    """
    Demande à Gemini d'identifier les couleurs des 2 équipes sur une frame.
    Retourne dict avec team_a et team_b ou None si échec.
    """
    img_b64 = _frame_to_base64(frame)

    prompt = """Regarde cette image d'un match de football.
Identifie les 2 équipes distinctes visibles et pour chacune décris :
- la couleur principale du maillot
- la couleur du short

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
{
  "team_a": {
    "jersey": "couleur du maillot (ex: rouge, vert, bordeaux, bleu, blanc, noir, jaune, orange)",
    "short": "couleur du short (ex: noir, blanc, rouge, bleu marine, vert)"
  },
  "team_b": {
    "jersey": "couleur du maillot",
    "short": "couleur du short"
  },
  "confidence": "high/medium/low"
}

Si tu ne vois qu'une seule équipe ou si l'image est floue, réponds :
{"confidence": "low", "team_a": null, "team_b": null}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }
            ]
        )
        text = response.text.strip()
        # Nettoyer les backticks si présents
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"  [PREVIEW] Gemini erreur : {e}")
        return None


def _color_name_to_hex(name):
    """Convertit un nom couleur en hex approximatif pour le dashboard."""
    mapping = {
        "rouge":        "#cc2200",
        "rouge foncé":  "#880000",
        "bordeaux":     "#6b0f2a",
        "vert":         "#2d7a2d",
        "vert foncé":   "#1a4d1a",
        "bleu":         "#1a4dcc",
        "bleu marine":  "#0a1a4d",
        "bleu foncé":   "#0a1a4d",
        "noir":         "#1a1a1a",
        "blanc":        "#f0f0f0",
        "jaune":        "#e6cc00",
        "orange":       "#e67300",
        "violet":       "#6600cc",
        "rose":         "#cc0066",
        "gris":         "#808080",
    }
    if name:
        for key, hex_val in mapping.items():
            if key in name.lower():
                return hex_val
    return "#808080"


def detect_teams_preview(video_path, output_dir="outputs/preview",
                          bootstrap_duration=90.0, sport="football",
                          n_frames=60, analysis_duration=120.0):
    """
    Détecte les couleurs des équipes via Gemini Vision.
    Sélectionne les meilleures frames puis demande à Gemini.
    Fallback sur méthode YOLO+KMeans si Gemini indisponible.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Vérifier disponibilité Gemini ────────────────────────────────────────
    gemini_client = None
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
            print(f"  [PREVIEW] Gemini Vision disponible")
        else:
            print(f"  [PREVIEW] GEMINI_API_KEY manquante → fallback YOLO")
    except Exception as e:
        print(f"  [PREVIEW] Gemini non disponible : {e} → fallback YOLO")

    # ── Ouvrir la vidéo ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Impossible d'ouvrir la vidéo"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  [PREVIEW] {w_orig}x{h_orig} | {total_frames/fps:.0f}s")

    # ── Sélectionner les meilleures frames ───────────────────────────────────
    # Scanner les 3 premières minutes toutes les 10s
    PROCESS_W, PROCESS_H = 960, 540
    scan_limit  = min(int(fps * 180), total_frames)
    best_frames = []   # (score, frame_num, frame)

    try:
        from vision.detector import Detector
        import config
        detector = Detector(sport=sport)

        print(f"  [PREVIEW] Sélection des meilleures frames...")
        for scan_f in range(0, scan_limit, int(fps * 10)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, scan_f)
            ret, frame = cap.read()
            if not ret:
                continue
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            try:
                res = detector.model([small], conf=0.4, verbose=False,
                                      imgsz=int(os.environ.get('YOLO_IMGSZ',
                                               config.YOLO_IMGSZ)))
                # Score = nombre de joueurs grands (proches)
                score = sum(
                    1 for b in res[0].boxes
                    if int(b.cls[0]) == detector.player_cls
                    and float(b.conf[0]) >= 0.4
                    and float(b.xyxy[0][3] - b.xyxy[0][1]) >= PROCESS_H * 0.15
                )
                if score >= 3:
                    best_frames.append((score, scan_f, frame.copy()))
            except Exception:
                continue

        cap.release()

        # Garder les 5 meilleures frames
        best_frames.sort(key=lambda x: -x[0])
        best_frames = best_frames[:5]
        print(f"  [PREVIEW] {len(best_frames)} frames sélectionnées "
              f"(scores: {[s for s,_,_ in best_frames]})")

    except Exception as e:
        cap.release()
        print(f"  [PREVIEW] Sélection frames échouée : {e}")
        # Fallback : prendre 5 frames uniformément réparties
        cap = cv2.VideoCapture(video_path)
        for fid in np.linspace(int(fps*5), min(int(fps*180), total_frames), 5, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if ret:
                best_frames.append((1, int(fid), frame.copy()))
        cap.release()

    if not best_frames:
        return {"success": False, "error": "Aucune frame utilisable"}

    # ── Analyse Gemini ────────────────────────────────────────────────────────
    if gemini_client and best_frames:
        print(f"  [PREVIEW] Analyse Gemini sur {len(best_frames)} frames...")

        results = []
        for score, fid, frame in best_frames:
            result = _ask_gemini_colors(frame, gemini_client)
            if result and result.get("confidence") in ("high", "medium"):
                if result.get("team_a") and result.get("team_b"):
                    results.append(result)
                    print(f"  [PREVIEW] Frame {fid}: "
                          f"A={result['team_a']['jersey']}/{result['team_a']['short']} | "
                          f"B={result['team_b']['jersey']}/{result['team_b']['short']} "
                          f"({result['confidence']})")

        if results:
            # Vote majoritaire sur les couleurs
            from collections import Counter

            def majority_color(key, subkey):
                votes = [r[key][subkey].lower() for r in results
                         if r.get(key) and r[key].get(subkey)]
                if not votes:
                    return "inconnu"
                return Counter(votes).most_common(1)[0][0]

            j_a = majority_color("team_a", "jersey")
            s_a = majority_color("team_a", "short")
            j_b = majority_color("team_b", "jersey")
            s_b = majority_color("team_b", "short")

            name_a = f"{j_a}/{s_a}" if s_a and s_a != j_a else j_a
            name_b = f"{j_b}/{s_b}" if s_b and s_b != j_b else j_b

            print(f"  [PREVIEW] Résultat Gemini: Team A={name_a} | Team B={name_b}")

            # Couleur hex pour le dashboard
            hex_a = _color_name_to_hex(j_a)
            hex_b = _color_name_to_hex(j_b)

            # Preview frame = la meilleure
            preview_0 = preview_1 = None
            try:
                _, _, best_frame = best_frames[0]
                h_f, w_f = best_frame.shape[:2]
                cv2.imwrite(os.path.join(output_dir, "team_0_preview.jpg"), best_frame)
                cv2.imwrite(os.path.join(output_dir, "team_1_preview.jpg"), best_frame)
                preview_0 = os.path.join(output_dir, "team_0_preview.jpg")
                preview_1 = os.path.join(output_dir, "team_1_preview.jpg")
            except Exception:
                pass

            return {
                "success":            True,
                "n_players_analyzed": len(results),
                "method":             "gemini",
                "team_0": {
                    "color_bgr":     _hex_to_bgr(hex_a),
                    "color_name":    name_a,
                    "short_bgr":     _hex_to_bgr(_color_name_to_hex(s_a)),
                    "preview_frame": preview_0,
                },
                "team_1": {
                    "color_bgr":     _hex_to_bgr(hex_b),
                    "color_name":    name_b,
                    "short_bgr":     _hex_to_bgr(_color_name_to_hex(s_b)),
                    "preview_frame": preview_1,
                },
            }

        print(f"  [PREVIEW] Gemini n'a pas pu identifier les équipes → fallback YOLO")

    # ── Fallback : méthode YOLO+LAB ───────────────────────────────────────────
    print(f"  [PREVIEW] Fallback méthode YOLO+LAB...")
    try:
        from analysis.detect_teams_preview_yolo import detect_teams_preview as _yolo_detect
        return _yolo_detect(video_path, output_dir, bootstrap_duration, sport)
    except Exception as e:
        print(f"  [PREVIEW] Fallback échoué : {e}")
        return {"success": False, "error": "Détection impossible"}


def _hex_to_bgr(hex_color):
    """Convertit hex en liste BGR."""
    try:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return [b, g, r]
    except Exception:
        return [128, 128, 128]