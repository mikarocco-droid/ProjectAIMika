# ai/player_tracker_ai.py
# -*- coding: utf-8 -*-
"""
Suivi et analyse d'un joueur spécifique par Gemini Vision.
Intégré au pipeline principal pour le mode "joueur ciblé".
Remplace et améliore scout.py/analyze_player().
"""

import cv2
import json
import re
import os
from collections import Counter

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ai.gemini_validator import get_client, encode_frame


# ─────────────────────────────────────────
# IDENTIFIER LE JOUEUR DANS UNE FRAME
# ─────────────────────────────────────────
def find_player_in_frame(frame, numero=None, couleur=None, position=None):
    """
    Demande à Gemini de trouver un joueur spécifique dans une frame.
    Retourne sa bbox estimée ou None.
    """
    if not GEMINI_AVAILABLE:
        return None

    try:
        client  = get_client()
        h, w    = frame.shape[:2]
        resized = cv2.resize(frame, (640, int(h * 640 / w)))

        desc = []
        if numero:
            desc.append(f"numéro {numero}")
        if couleur:
            desc.append(f"maillot {couleur}")
        if position:
            desc.append(f"joue {position}")

        description = ", ".join(desc) if desc else "joueur principal"

        content = [{
            "type": "text",
            "text": (
                f"Dans cette image de match, trouve le joueur : {description}.\n"
                f"Donne sa position en pourcentage de l'image.\n"
                f"JSON sans markdown :\n"
                f'{{"trouve": true, "x_pct": 0.45, "y_pct": 0.60, '
                f'"w_pct": 0.05, "h_pct": 0.15, "confiance": 0.9, '
                f'"numero_vu": 9}}'
            )
        }, {
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       encode_frame(resized)
            }
        }]

        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = content
        )
        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        if result.get("trouve") and result.get("confiance", 0) >= 0.7:
            return {
                "x_pct":     result["x_pct"],
                "y_pct":     result["y_pct"],
                "w_pct":     result.get("w_pct", 0.05),
                "h_pct":     result.get("h_pct", 0.15),
                "confiance": result["confiance"],
                "numero_vu": result.get("numero_vu")
            }
        return None

    except Exception as e:
        return None


# ─────────────────────────────────────────
# ANALYSER LES ACTIONS D'UN JOUEUR
# ─────────────────────────────────────────
def analyze_player_frames(frames_with_pos, numero=None, couleur=None,
                           position=None, sport="football"):
    """
    Analyse les actions d'un joueur sur un batch de frames.

    frames_with_pos : list de (timestamp_sec, frame_numpy)

    Retourne : list d'observations
    """
    if not GEMINI_AVAILABLE or not frames_with_pos:
        return []

    try:
        client = get_client()

        desc = []
        if numero:
            desc.append(f"#{numero}")
        if couleur:
            desc.append(f"maillot {couleur}")
        if position:
            desc.append(position)
        player_desc = " ".join(desc) if desc else "joueur cible"

        content = [{
            "type": "text",
            "text": (
                f"Analyse les actions du joueur {player_desc} dans ces frames de {sport}.\n"
                f"Pour chaque frame, si le joueur est visible, décris son action.\n"
                f"JSON sans markdown :\n"
                f'{{"observations": [\n'
                f'  {{"frame": 0, "visible": true, "action": "sprint sur le couloir droit", '
                f'"zone": "couloir droit", "avec_ballon": false, "evaluation": 4}},\n'
                f'  {{"frame": 1, "visible": false}}\n'
                f']}}\n\n'
                f"- evaluation : 1 (mauvais) à 5 (excellent)\n"
                f"- zone : surface de réparation / milieu / couloir gauche / couloir droit / "
                f"défense / attaque\n"
                f"- action : description précise en français"
            )
        }]

        for i, (ts, frame) in enumerate(frames_with_pos):
            mins = int(ts // 60)
            secs = int(ts % 60)
            content.append({"type": "text", "text": f"Frame {i} ({mins:02d}:{secs:02d}) :"})
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
        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        obs = result.get("observations", [])
        for o in obs:
            if 0 <= o.get("frame", -1) < len(frames_with_pos):
                o["timestamp"] = frames_with_pos[o["frame"]][0]

        return [o for o in obs if o.get("visible", False)]

    except Exception as e:
        print(f"  Player analyzer error : {e}")
        return []


# ─────────────────────────────────────────
# GÉNÉRER LE RAPPORT JOUEUR
# ─────────────────────────────────────────
def generate_player_report_ai(observations, numero=None, couleur=None,
                               sport="football", duration=600):
    """
    Génère un rapport complet sur un joueur depuis ses observations.
    """
    if not GEMINI_AVAILABLE or not observations:
        return {}

    try:
        client = get_client()

        total_obs    = len(observations)
        avec_ballon  = sum(1 for o in observations if o.get("avec_ballon"))
        evaluations  = [o["evaluation"] for o in observations if o.get("evaluation")]
        note_moy     = round(sum(evaluations) / len(evaluations), 1) if evaluations else 0
        zones        = Counter(o.get("zone", "") for o in observations if o.get("zone"))
        zone_princ   = zones.most_common(1)[0][0] if zones else "inconnue"

        prompt = (
            f"Tu es un scout professionnel de {sport}.\n"
            f"Observations du joueur "
            f"{'#' + str(numero) if numero else ''} "
            f"{'maillot ' + couleur if couleur else ''} "
            f"sur {int(duration/60)} minutes :\n\n"
            f"{json.dumps(observations[:30], ensure_ascii=False, indent=2)}\n\n"
            f"Stats : {total_obs} apparitions, {avec_ballon} touches de balle, "
            f"note moyenne {note_moy}/5, zone principale : {zone_princ}\n\n"
            f"Génère un rapport scout complet en JSON sans markdown :\n"
            f"{{\n"
            f'  "note_globale": 7,\n'
            f'  "resume": "Résumé en 3-4 phrases",\n'
            f'  "points_forts": ["point 1", "point 2", "point 3"],\n'
            f'  "points_faibles": ["point 1", "point 2"],\n'
            f'  "stats": {{\n'
            f'    "touches_balle": {avec_ballon},\n'
            f'    "zone_principale": "{zone_princ}",\n'
            f'    "note_moyenne": {note_moy},\n'
            f'    "apparitions": {total_obs}\n'
            f'  }},\n'
            f'  "moments_cles": [{{"timestamp": 107.0, "description": "...", "evaluation": 5}}],\n'
            f'  "recommandation_coach": "Texte de recommandation",\n'
            f'  "potentiel": "haut | moyen | faible"\n'
            f"}}"
        )

        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = [{"type": "text", "text": prompt}]
        )
        text   = response.text.strip()
        text   = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)

    except Exception as e:
        print(f"  Player report error : {e}")
        return {}


# ─────────────────────────────────────────
# PIPELINE COMPLET ANALYSE JOUEUR
# ─────────────────────────────────────────
def track_and_analyze_player(
    video_path,
    numero          = None,
    couleur         = None,
    position        = None,
    sport           = "football",
    interval_sec    = 10,
    batch_size      = 5
):
    """
    Point d'entrée principal.
    Analyse un joueur ciblé sur toute la vidéo.

    Retourne le rapport complet du joueur.
    """
    if not GEMINI_AVAILABLE:
        print("  Gemini non disponible")
        return {}

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    print(f"  Analyse joueur : "
          f"{'#' + str(numero) if numero else '?'} | "
          f"{couleur or '?'} | {position or '?'}")
    print(f"  Vidéo : {duration/60:.1f} min | intervalle : {interval_sec}s")

    # Extraire les frames à intervalles réguliers
    all_observations = []
    frame_pos        = 0

    while frame_pos < total_frames:
        batch_frames = []

        for _ in range(batch_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            ts    = frame_pos / fps
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            batch_frames.append((ts, frame))
            frame_pos += int(interval_sec * fps)

        if not batch_frames:
            break

        obs = analyze_player_frames(
            batch_frames,
            numero   = numero,
            couleur  = couleur,
            position = position,
            sport    = sport
        )
        all_observations.extend(obs)
        print(f"  [{int(batch_frames[0][0]//60):02d}:{int(batch_frames[0][0]%60):02d}] "
              f"→ {len(obs)} actions détectées")

    cap.release()

    print(f"  Total : {len(all_observations)} observations")

    # Générer le rapport final
    report = generate_player_report_ai(
        observations = all_observations,
        numero       = numero,
        couleur      = couleur,
        sport        = sport,
        duration     = duration
    )

    report["_observations"] = all_observations
    report["_joueur"]       = {
        "numero":   numero,
        "couleur":  couleur,
        "position": position,
        "sport":    sport
    }

    return report