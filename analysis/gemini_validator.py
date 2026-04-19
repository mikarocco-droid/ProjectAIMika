# ai/gemini_validator.py
# -*- coding: utf-8 -*-

import os
import cv2
import json
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
# MODE DEBUG — valeur centralisée dans config.py
# ─────────────────────────────────────────
try:
    from config import DEBUG
except ImportError:
    DEBUG = False

# ─────────────────────────────────────────
# SEUILS DYNAMIQUES (depuis gemini_validation.py)
# ─────────────────────────────────────────
MIN_CONF_BASE = 0.80

def get_dynamic_threshold(event):
    """
    Seuil de confiance Gemini adapté à la confiance de l'event.
    Event très confiant (posthoc score élevé) → seuil plus souple.
    Event peu confiant → seuil plus strict.
    """
    conf = event.get("confidence", 0.5)
    return max(0.75, 0.95 - conf)


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
# FLAGS
# ─────────────────────────────────────────
_quota_exhausted    = False
_gemini_unavailable = False

def _call_gemini(client, parts, max_retries=2):
    global _quota_exhausted, _gemini_unavailable

    if _quota_exhausted:
        return None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model    = "gemini-2.5-flash",
                contents = parts
            )
            _gemini_unavailable = False
            return response

        except Exception as e:
            err_str = str(e)

            if "RESOURCE_EXHAUSTED" in err_str:
                print("  Quota Gemini journalier épuisé — validation désactivée")
                _quota_exhausted = True
                return None

            elif "503" in err_str or "UNAVAILABLE" in err_str:
                _gemini_unavailable = True
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

            elif "429" in err_str and attempt < max_retries - 1:
                m    = re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(m.group(1)) + 2 if m else 30
                wait = min(wait, 30)
                print(f"  Rate limit Gemini — attente {wait}s...")
                time.sleep(wait)
            else:
                raise

    return None


# ─────────────────────────────────────────
# HELPER — position proche d'un but
# ─────────────────────────────────────────
def _is_near_goal(x, frame_w, threshold=0.18):
    if frame_w <= 0:
        return True
    x_pct = x / frame_w
    return x_pct < threshold or x_pct > (1.0 - threshold)


# ─────────────────────────────────────────
# ENCODER FRAME
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
# ─────────────────────────────────────────
def extract_frames_around(video_path, frame_id, fps=25, n_before=2, n_after=2):
    cap    = cv2.VideoCapture(video_path)
    frames = []

    # V9.6 — 5 frames espacées finement autour de l'événement
    # [-15, -5, 0, +5, +15] frames ≈ [-0.6s, -0.2s, 0, +0.2s, +0.6s]
    offsets = sorted(set([-15, -5, 0, 5, 15]))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in offsets:
        target = frame_id + offset
        if target < 0 or target >= total_frames:
            continue
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
# VALIDER UN EVENT — V9.7 MULTI-FRAME
# ─────────────────────────────────────────
def _call_gemini_at_offset(client, video_path, frame_id, fps, event_type, danger, sport):
    """Appel Gemini sur un offset donné. Retourne le résultat parsé ou None."""
    frames_data = extract_frames_around(video_path, frame_id, fps, n_before=2, n_after=2)
    if not frames_data:
        return None

    parts = [text_to_part(
        f"Tu es un expert analyste de {sport}. "
        f"Voici {len(frames_data)} frames dans l'ordre chronologique autour d'un evenement suspect "
        f"(type detecte : {event_type}, danger score : {danger:.1f}/10).\n\n"
        f"CRITERES pour valider un BUT :\n"
        f"- Le ballon franchit ou a franchi la ligne de but\n"
        f"- Le ballon est dans le filet ou vient d'y entrer\n"
        f"- Un joueur celebre ou reagit\n"
        f"- ATTENTION : si le gardien recupere le ballon DANS son but c'est un but\n"
        f"- ATTENTION : but contre son camp = aussi un but\n\n"
        f"CRITERES pour valider un TIR :\n"
        f"- Le ballon est frappe intentionnellement vers le but\n"
        f"- La trajectoire est dirigee vers les cages\n\n"
        f"CE QUI N'EST PAS un but ni un tir :\n"
        f"- Gardien qui tient le ballon DEVANT sa ligne (pas dedans)\n"
        f"- Relance du gardien depuis sa surface\n"
        f"- Degagement defensif loin du but\n"
        f"- Tir qui passe clairement a cote ou tres au-dessus\n\n"
        f"Reponds UNIQUEMENT en JSON valide sans markdown :\n"
        f'{{"type": "goal|shot|corner|touche|goalkeeper_hold|goalkeeper_throw|defensive_clearance|none", '
        f'"confiance": 0.95, '
        f'"equipe": 0, '
        f'"description": "description courte"}}\n\n'
        f"EN CAS DE DOUTE sur un but reponds 'goal' avec confiance 0.65.\n"
        f"Un faux negatif (but rate) est pire qu'un faux positif (faux but)."
    )]

    for _, frame in frames_data:
        parts.append(frame_to_part(frame))

    response = _call_gemini(client, parts)
    if response is None:
        return None

    result = _safe_json_load(response.text)
    if result is None:
        return None

    return {
        "type":        result.get("type", "none"),
        "confiance":   float(result.get("confiance", 0.5)),
        "equipe":      result.get("equipe"),
        "description": result.get("description", "")
    }


def validate_event(video_path, event, fps=25, sport="football"):
    if not GEMINI_AVAILABLE:
        return None

    frame_orig = event.get("frame", 0)
    danger     = event.get("danger", 0)
    event_type = event.get("type", "?")
    source     = event.get("detected_from", event.get("source", "events"))
    tracker_conf = event.get("confidence", 0.5)

    # V9.7 — Offsets multi-frame selon la source
    # goal_posthoc détecte 5-15s AVANT le vrai but visuel
    # → on teste plusieurs offsets en secondes et on vote
    if "posthoc" in str(source):
        offsets_s = [5, 8, 10, 12, 14, 16]  # V9.7 — fenêtre élargie à +16s
        # Couvre les buts détectés jusqu'à 14s avant le moment réel
    else:
        offsets_s = [0, 2, 4]        # events.py plus précis, fenêtre réduite

    if DEBUG:
        print(f"  [PRE-GEMINI] t={event.get('time',0):.2f}s "
              f"source={source} frame={frame_orig} "
              f"offsets={offsets_s}s tracker_conf={tracker_conf:.2f}")

    try:
        client = get_client()
        total_frames = None  # lazy init

        goal_votes = 0
        shot_votes = 0
        best_result = None
        best_conf   = 0.0

        for off_s in offsets_s:
            if _quota_exhausted:
                break

            frame_id = frame_orig + int(off_s * fps)

            # Lazy init total_frames
            if total_frames is None:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            if frame_id >= total_frames:
                continue

            result = _call_gemini_at_offset(
                client, video_path, frame_id, fps, event_type, danger, sport
            )
            time.sleep(0.5)

            if result is None:
                continue

            rtype = result["type"]
            rconf = result["confiance"]

            if DEBUG:
                print(f"    [OFFSET +{off_s}s] type={rtype} conf={rconf:.2f} "
                      f"desc={result.get('description','')[:50]}")

            if rtype == "goal":
                goal_votes += 1
                if rconf > best_conf:
                    best_conf   = rconf
                    best_result = result
                # Early stop — 2 confirmations = suffisant
                if goal_votes >= 2:
                    if DEBUG:
                        print(f"    [EARLY STOP] {goal_votes} votes goal → validation confirmée")
                    break

            elif rtype == "shot":
                shot_votes += 1
                if best_result is None:
                    best_result = result

        # ── Décision finale par vote pondéré ─────────────────────
        score = 0
        if goal_votes >= 2:   score += 3   # confirmation forte
        elif goal_votes == 1: score += 1   # signal faible
        if shot_votes >= 1:   score += 1
        if tracker_conf > 0.9: score += 1

        if DEBUG:
            print(f"  [VOTE] goal={goal_votes} shot={shot_votes} "
                  f"tracker_conf={tracker_conf:.2f} → score={score}")

        if score >= 3:
            # But confirmé — retourner le meilleur résultat goal
            if best_result and best_result["type"] == "goal":
                return best_result
            # Si score=3 via shot+tracker mais pas de goal direct
            # → retourner shot avec boost
            if best_result:
                return best_result

        elif score == 2 and goal_votes >= 1:
            # 1 vote goal sans contexte fort → garder avec conf réduite
            if best_result:
                best_result["confiance"] = min(best_result["confiance"], 0.70)
                return best_result

        # Aucun signal suffisant — confiance basse pour indiquer l'incertitude
        desc = best_result.get("description", "") if best_result else "aucune frame pertinente"
        return {"type": "none", "confiance": 0.3,
                "equipe": None, "description": desc}

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
# ─────────────────────────────────────────
def validate_events_with_gemini(
    events,
    video_path,
    fps        = 25,
    sport      = "football",
    MIN_CONF_GOAL = 0.80,
    MIN_CONF_SHOT = 0.70,
    frame_w    = 1920,
    frame_h    = 1080,
):
    global _gemini_unavailable

    if not GEMINI_AVAILABLE:
        print("  Gemini non disponible — validation ignorée")
        return events

    candidates = [
        e for e in events
        if e.get("type") in ["goal"]
        and e.get("frame", 0) > 0
    ]

    if not candidates:
        return events

    print(f"  Validation Gemini : {len(candidates)} candidats...")
    validated = corrected = removed = 0

    for event in candidates:
        t_pre = event.get("time", 0)
        src   = event.get("detected_from", event.get("source", "?"))
        conf  = event.get("confidence", 0)
        xg    = event.get("xg", 0)
        if DEBUG:
            print(f"  [CANDIDAT] t={int(t_pre//60):02d}:{int(t_pre%60):02d} "
                  f"source={src} "
                  f"tracker_conf={conf:.2f} "
                  f"xg={xg:.3f} "
                  f"frame={event.get('frame',0)}")
        if _quota_exhausted:
            print("  Quota épuisé — tous les buts restants non validés sont rejetés")
            for e in candidates:
                if not e.get("gemini_validated") and e.get("type") == "goal":
                    e["_remove"] = True
                    removed += 1
            break

        result = validate_event(video_path, event, fps, sport)

        if result is None:
            print(f"  Gemini no response : goal t={event.get('time',0):.0f}s gardé")
            event["gemini_validated"] = False
            event["gemini_conf"]      = 0.0
            continue

        validated  += 1
        gemini_type = result["type"]
        confiance   = result["confiance"]
        t_sec = event.get("time", 0)
        tracker_conf = event.get("confidence", 0)
        if DEBUG:
            print(f"  [POST-GEMINI] t={int(t_sec//60):02d}:{int(t_sec%60):02d} "
                  f"gemini={gemini_type} conf={confiance:.2f} "
                  f"tracker_conf={tracker_conf:.2f} "
                  f"threshold={get_dynamic_threshold(event):.2f} "
                  f"desc={result.get('description', '')[:60]}")

        # Seuil dynamique : adapté à la confiance de l'event
        # Un but détecté par goal_posthoc (conf élevée) → seuil plus souple
        threshold = get_dynamic_threshold(event) if event.get("type") == "goal"                     else MIN_CONF_SHOT

        if gemini_type == "goal":
            # 🟢 Gemini confirme → gardé
            if result.get("equipe") is not None:
                event["team"] = result["equipe"]
            if DEBUG:
                print(f"    → BUT CONFIRMÉ (conf={confiance:.2f})")

        elif gemini_type == "shot":
            # Gemini voit un tir → corriger
            event["type"] = "shot"
            corrected += 1
            if DEBUG:
                print(f"    → CORRIGÉ en tir (conf={confiance:.2f})")

        elif gemini_type in ["goalkeeper_hold", "goalkeeper_throw",
                              "defensive_clearance", "corner", "touche"]:
            # 🔴 Signal clair non-but → supprimer si confiant
            if confiance >= 0.85:
                event["_remove"] = True
                removed += 1
                if DEBUG:
                    print(f"    → SUPPRIMÉ ({gemini_type} conf={confiance:.2f})")
            else:
                if DEBUG:
                    print(f"    → GARDÉ malgré {gemini_type} (conf={confiance:.2f} < 0.85)")

        else:
            # 🟡 Zone grise (type=none, vote insuffisant)
            # → décision basée sur signaux physiques
            posthoc_score    = event.get("score", event.get("danger", 0))
            high_conf_phys   = event.get("high_conf_physical", False)
            tracker_conf_val = event.get("confidence", 0)

            if DEBUG:
                print(f"    [DECISION] type={gemini_type} "
                      f"phys={high_conf_phys} "
                      f"score={posthoc_score:.1f} "
                      f"tracker={tracker_conf_val:.2f}")

            if high_conf_phys or posthoc_score >= 8.5:
                if DEBUG:
                    print(f"    → GARDÉ (physique fort : "
                          f"high_conf={high_conf_phys} score={posthoc_score:.1f})")
            elif posthoc_score >= 7.5 and tracker_conf_val > 0.85:
                if DEBUG:
                    print(f"    → GARDÉ (borderline : "
                          f"score={posthoc_score:.1f} tracker={tracker_conf_val:.2f})")
            else:
                event["_remove"] = True
                removed += 1
                if DEBUG:
                    print(f"    → SUPPRIMÉ (physique faible : "
                          f"score={posthoc_score:.1f} tracker={tracker_conf_val:.2f})")

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance

    events = [e for e in events if not e.get("_remove", False)]
    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")
    return events