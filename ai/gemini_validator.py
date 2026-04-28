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
# CACHE FRAMES PyAV — LRU, évite redécodages
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# SEUILS DYNAMIQUES (depuis gemini_validation.py)
# ─────────────────────────────────────────
OFFSETS_POSTHOC = [-20, -10, -5, 0, +5, +10, +15]
OFFSETS_EVENTS  = [-1, 0, 2]

_METRICS = {
    "decode_time": 0.0,
    "gemini_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def get_dynamic_threshold(event):
    """
    Seuil de confiance Gemini adapté à la confiance de l'event.
    Event très confiant (posthoc score élevé) → seuil plus souple.
    Event peu confiant → seuil plus strict.
    """
    conf = event.get("confidence", 0.5)
    return max(0.75, 0.95 - conf)


# ─────────────────────────────────────────
# CACHE FENÊTRES PyAV — évite redécodage des zones temporelles proches
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# CACHE FRAMES PyAV — LRU, max 50 fenêtres
# ─────────────────────────────────────────

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
            t0 = time.time()
            try :
                response = client.models.generate_content(
                    model    = "gemini-2.5-flash",
                    contents = parts
                )
            finally :
                _METRICS["gemini_time"] += time.time() - t0
            
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
# SEEK PRÉCIS — compense les keyframes OpenCV
# ─────────────────────────────────────────
def safe_seek_frame(cap, target_frame, max_jump=30):
    """
    Seek précis vers une frame en avançant depuis le keyframe précédent.
    cap.set(CAP_PROP_POS_FRAMES) est approximatif sur MP4 — cette fonction
    compense en lisant frame par frame depuis max_jump frames avant la cible.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, min(target_frame, total_frames - 1))

    start_frame = max(0, target_frame - max_jump)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current    = start_frame
    last_frame = None

    while current <= target_frame:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        current   += 1

    return last_frame


# ─────────────────────────────────────────
# EXTRAIRE FRAMES AUTOUR D'UN EVENT — PyAV (frame-accurate) + fallback OpenCV
# ─────────────────────────────────────────
def extract_frames_around_opencv(video_path, frame_id, fps=25):
    """Fallback OpenCV avec safe_seek."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    offsets = sorted(set([-15, -5, 0, 5, 15]))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for offset in offsets:
        target = frame_id + offset
        if target < 0 or target >= total_frames:
            continue
        frame = safe_seek_frame(cap, target, max_jump=30)
        if frame is not None:
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            frames.append((offset, frame))
    cap.release()
    return frames


def extract_frames_around(video_path, frame_id, fps=25):
    """
    Version PyAV — frame-accurate.
    Même signature que l'ancienne version OpenCV.
    Retourne : [(offset, frame), ...]
    """
    try:
        import av as _av
    except ImportError:
        return extract_frames_around_opencv(video_path, frame_id, fps)

    offsets = sorted(set([-15, -5, 0, 5, 15]))
    target_time = frame_id / fps
    frames = []

    try:
        container = _get_av_container(video_path)
        if container is None:
            return extract_frames_around_opencv(video_path, frame_id, fps)
        stream = container.streams.video[0]

        for offset in offsets:
            t = target_time + (offset / fps)
            if t < 0:
                continue

            try:
                seek_ts = int(t / float(stream.time_base))

                with _AV_LOCK:
                    container.seek(seek_ts, backward=True, stream=stream)
                    container.flush_buffers()

                best_frame = None
                best_dt = float("inf")

                # decode progressif au lieu de tout charger
                for pkt_frame in container.decode(stream):
                    if pkt_frame.pts is None:
                        continue

                    frame_time = float(pkt_frame.pts * pkt_frame.time_base)
                    dt = abs(frame_time - t)

                    if dt < best_dt:
                        best_dt = dt
                        best_frame = pkt_frame

                    # on est suffisamment proche → stop
                    if best_dt < (1.0 / (fps + 1)):
                        break

                    # on a clairement dépassé la zone utile
                    if frame_time > t + 0.5:
                        break

            except Exception:
                continue
            
            if best_frame is None:
                continue

            img = best_frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            if w > 960:
                img = cv2.resize(img, (960, int(h * 960 / w)))
            frames.append((offset, img))

        # container reste ouvert (cache global)

    except Exception as e:
        print(f"[PyAV] erreur : {e} → fallback OpenCV")
        return extract_frames_around_opencv(video_path, frame_id, fps)

    if not frames:
        return extract_frames_around_opencv(video_path, frame_id, fps)

    return frames

# ─────────────────────────────────────────
# CACHE GLOBAL — fenêtres PyAV (LRU, max 50 entrées)
# ─────────────────────────────────────────
import threading as _threading
_AV_CONTAINERS = {}
_AV_LOCK        = _threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# find_goal_after_shot — Gemini cherche un but dans la fenêtre [shot_t, shot_t+window]
# Remplace goal_posthoc_disappear pour les buts difficiles (caméra face, tracking perdu)
# ─────────────────────────────────────────────────────────────────────────────

def find_goal_after_shot(video_path, shot_time, window=30, fps=25,
                         frame_w=1920, frame_h=1080):
    """
    Analyse la fenêtre [shot_time, shot_time+window] après un tir détecté.
    Envoie des frames espacées de 3s à Gemini avec la question :
    "Est-ce qu'un but a été marqué ? Si oui, à quel timestamp exact ?"

    Returns:
        dict {"is_goal": bool, "timestamp": float, "confidence": float, "desc": str}
        ou None si Gemini indisponible
    """
    global _quota_exhausted, _gemini_unavailable, _gemini_calls, _gemini_time

    if _quota_exhausted or _gemini_unavailable:
        return None

    client = get_client()
    if client is None:
        return None

    # Échantillonner toutes les 3s dans la fenêtre
    sample_times = []
    t = shot_time
    while t <= shot_time + window:
        sample_times.append(round(t, 1))
        t += 3.0

    # Extraire les frames
    parts = []
    valid_times = []
    cap = cv2.VideoCapture(video_path)
    try:
        for st in sample_times:
            frame_id = int(st * fps)
            frame_id = max(0, frame_id)
            frame = safe_seek_frame(cap, frame_id, max_jump=30)
            if frame is not None:
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, int(h * 960 / w)))
                parts.append(frame_to_part(frame))
                valid_times.append(st)
    finally:
        cap.release()

    if not parts:
        return None

    # Prompt Gemini
    times_str = ", ".join(f"{int(t//60):02d}:{int(t%60):02d}" for t in valid_times)
    prompt = f"""You are analyzing a football/soccer match video.
I'm showing you {len(parts)} frames from timestamps: {times_str}
These frames cover {window} seconds after a detected shot on target at {int(shot_time//60):02d}:{int(shot_time%60):02d}.

Your task: determine if a GOAL was scored in this time window.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "is_goal": true or false,
  "timestamp": <exact seconds when ball crosses the line, or null if no goal>,
  "confidence": <0.0 to 1.0>,
  "evidence": "<brief description of what you see: ball in net, goalkeeper reaction, kickoff, etc.>"
}}

Return is_goal=true ONLY if you clearly see:
- ball crossing the goal line OR
- ball inside the net OR
- immediate kickoff after a confirmed scoring sequence

Be conservative. If uncertain, return is_goal=false."""

    parts_with_prompt = [text_to_part(prompt)] + parts

    try:
        t0 = time.time()
        result = _call_gemini(client, parts_with_prompt)
        elapsed = time.time() - t0
        _gemini_calls += 1
        _gemini_time  += elapsed

        if result is None:
            return None

        text = extract_response_text(result)
        parsed = _safe_json_load(text)

        if parsed is None:
            return None

        is_goal    = bool(parsed.get("is_goal", False))
        timestamp  = parsed.get("timestamp")
        confidence = float(parsed.get("confidence", 0.0))
        evidence   = parsed.get("evidence", "")

        # Valider le timestamp dans la fenêtre
        if is_goal and timestamp is not None:
            try:
                timestamp = float(timestamp)
                if not (shot_time - 5 <= timestamp <= shot_time + window + 5):
                    # Timestamp hors fenêtre — utiliser centre de la fenêtre
                    timestamp = shot_time + window / 2
            except (ValueError, TypeError):
                timestamp = shot_time + window / 2
        elif is_goal:
            timestamp = shot_time + window / 2

        log.info(f"  [SHOT→GOAL] shot={shot_time:.1f}s → is_goal={is_goal} "
                 f"t={timestamp} conf={confidence:.2f} | {evidence[:80]}")

        return {
            "is_goal":    is_goal,
            "timestamp":  timestamp,
            "confidence": confidence,
            "desc":       evidence,
        }

    except Exception as e:
        log.warning(f"  [SHOT→GOAL] Erreur : {e}")
        return None

def print_metrics():
    total = _METRICS["cache_hits"] + _METRICS["cache_misses"]
    hit_rate = (_METRICS["cache_hits"] / total) if total else 0
    
    print(
        f"[METRICS] "
        f"decode={_METRICS['decode_time']:.2f}s | "
        f"gemini={_METRICS['gemini_time']:.2f}s | "
        f"cache_hit={hit_rate:.2%} | "
        f"cache_hits={_METRICS['cache_hits']} | "
        f"cache_misses={_METRICS['cache_misses']}"
    )

def _get_av_container(video_path):
    """Cache thread-safe du container PyAV."""
    with _AV_LOCK:
        if video_path not in _AV_CONTAINERS:
            try:
                import av as _av
                _AV_CONTAINERS[video_path] = _av.open(video_path)
            except Exception:
                return None
        return _AV_CONTAINERS[video_path]

def close_all_av_containers():
    """Libère tous les containers PyAV — appeler en fin de pipeline."""
    with _AV_LOCK:
        for c in _AV_CONTAINERS.values():
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass

        _AV_CONTAINERS.clear()

from collections import OrderedDict as _OD

_FRAME_CACHE     = _OD()
_FRAME_CACHE_MAX = 50

def _cache_key(video_path, center_time, window_sec, bucket=0.25):
    """Bucketise le temps pour mutualiser les fenêtres proches."""
    start_b = round((center_time - window_sec) / bucket) * bucket
    end_b   = round((center_time + window_sec) / bucket) * bucket
    return (video_path, start_b, end_b)

def _cache_get(key):
    if key in _FRAME_CACHE:
        _METRICS["cache_hits"] += 1
        _FRAME_CACHE.move_to_end(key)
        return _FRAME_CACHE[key]

    _METRICS["cache_misses"] += 1
    return None

def _cache_put(key, value):
    _FRAME_CACHE[key] = value
    _FRAME_CACHE.move_to_end(key)

    while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
        _FRAME_CACHE.popitem(last=False)


# ─────────────────────────────────────────
# BATCH DECODE PyAV — 1 seek → fenêtre complète
# ─────────────────────────────────────────
def extract_frames_window_pyav(video_path, center_time, window_sec=2.0, fps=25):
    key = _cache_key(video_path, center_time, window_sec)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if _get_av_container(video_path) is None:
        return []

    start_t = max(0.0, center_time - window_sec)
    end_t = center_time + window_sec
    frames = []
    t0 = time.time()

    try:
        container = _get_av_container(video_path)
        if container is None:
            return frames

        stream = container.streams.video[0]
        seek_ts = int(start_t / float(stream.time_base))

        with _AV_LOCK:
            container.seek(seek_ts, backward=True, stream=stream)
            try:
                container.flush_buffers()
            except AttributeError:
                pass  # flush_buffers absent dans certaines versions PyAV

            for pkt_frame in container.decode(stream):
                if pkt_frame.pts is None:
                    continue

                t = float(pkt_frame.pts * pkt_frame.time_base)

                if t < start_t:
                    continue

                if t > end_t:
                    break

                img = pkt_frame.to_ndarray(format="bgr24")

                h, w = img.shape[:2]
                if w > 960:
                    img = cv2.resize(img, (960, int(h * 960 / w)))

                frames.append((t, img))

    except Exception as e:
        print(f"[PyAV batch] erreur : {e}")
        return []

    finally:
        _METRICS["decode_time"] += time.time() - t0

    _cache_put(key, frames)
    return frames
    
    
def pick_closest_frame(frames_window, target_time):
    """Retourne la frame la plus proche du timestamp cible."""
    if not frames_window:
        return None
    best, best_dt = None, float("inf")
    for t, img in frames_window:
        dt = abs(t - target_time)
        if dt < best_dt:
            best_dt, best = dt, img
    return best


# ─────────────────────────────────────────
# HELPER — parse JSON Gemini
# ─────────────────────────────────────────
def _safe_json_load(text):
    if not text:
        return None

    try:
        clean = re.sub(r"```json|```", "", text).strip()

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)

        return json.loads(clean)

    except Exception:
        return None

def select_best_frames(frames_data, max_frames=3):
    scored = []
    prev = None

    for off, frame in frames_data:
        if prev is None:
            score = 1.0
        else:
            diff = cv2.absdiff(prev, frame)
            score = np.mean(diff)

        scored.append((score, off, frame))
        prev = frame

    scored.sort(reverse=True, key=lambda x: x[0])
    return [(off, f) for _, off, f in scored[:max_frames]]

def extract_response_text(response):
    if hasattr(response, "text") and response.text:
        return response.text

    try:
        return "".join(
            part.text
            for cand in response.candidates
            for part in cand.content.parts
            if hasattr(part, "text")
        )
    except Exception:
        return None

# ─────────────────────────────────────────
# VALIDER UN EVENT — V9.7 MULTI-FRAME
# ─────────────────────────────────────────
def _call_gemini_at_offset(client, video_path, frame_id, fps, event_type, danger, sport):
    """Appel Gemini sur un offset donné. Retourne le résultat parsé ou None."""
    center_time = frame_id / fps

    frames_window = extract_frames_window_pyav(
        video_path=video_path,
        center_time=center_time,
        window_sec=1.0,
        fps=fps
    )

    targets = [-0.6, -0.2, 0.0, 0.2, 0.6]

    frames_data = []
    for off in targets:
        frame = pick_closest_frame(
            frames_window,
            center_time + off
        )
        if frame is not None:
            frames_data.append((off, frame))
            
    frames_data = select_best_frames(frames_data, max_frames=3)
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
        f"En cas de doute raisonnable sur un but :\n"
        f"- si le ballon semble avoir franchi la ligne → goal avec confiance 0.75\n"
        f"- sinon → none\n"
        f"IMPORTANT : Ne reponds 'shot' QUE si tu vois CLAIREMENT un tir vers le but.\n"
        f"Si ce n'est pas evident → reponds 'none'. Un faux tir est pire qu'un tir manque."
    )]

    for _, frame in frames_data:
        parts.append(frame_to_part(frame))

    response = _call_gemini(client, parts)
    if response is None:
        return None

    text = extract_response_text(response)
    result = _safe_json_load(text)
    if result is None:
        return None

    return {
        "type":        result.get("type", "none"),
        "confiance":   float(result.get("confiance", 0.5)),
        "equipe":      result.get("equipe"),
        "description": result.get("description", "")
    }
    
def compute_signal_score(rtype, conf, off_s, source):
    # poids temporel (plus proche = plus fiable)
    time_weight = 1.0 / (1.0 + abs(off_s) / 5.0)

    # posthoc = plus bruité
    if "posthoc" in str(source):
        time_weight *= 0.7

    # poids par type
    if rtype == "goal":
        type_weight = 1.0
    elif rtype == "shot":
        type_weight = 0.6

    elif rtype in [
        "goalkeeper_hold",
        "goalkeeper_throw",
        "defensive_clearance",
        "corner",
        "touche"
    ]:
        type_weight = -0.4

    else:
        # none = neutre
        type_weight = 0.0

    return conf * time_weight * type_weight

def validate_event(video_path, event, fps=25, sport="football", frame_w=None):
    if not GEMINI_AVAILABLE:
        return None

    frame_orig   = event.get("frame", 0)
    danger       = event.get("danger", 0)
    event_type   = event.get("type", "?")
    source       = event.get("detected_from", event.get("source", "events"))
    tracker_conf = event.get("confidence", 0.5)
    near_goal = event.get("near_goal", False)
    ball_end_x = event.get("ball_end_x")

    if frame_w is None:
        frame_w = event.get("frame_w", 1920)

    # 🔥 SKIP GEMINI (safe)
    if (
        tracker_conf > 0.97
        and event.get("high_conf_physical")
        and event.get("near_goal", False)
    ):
        return {
            "type": "goal",
            "confiance": tracker_conf,
            "_goal_votes": 1,
            "_shot_votes": 0,
            "description": "high confidence tracker skip"
        }

    # Offsets triés (plus proches d'abord)
    offsets_s = OFFSETS_POSTHOC if "posthoc" in str(source) else OFFSETS_EVENTS
    offsets_s = sorted(offsets_s, key=lambda x: abs(x))

    _t_event = event.get('time', 0)
    print(f"  [PRE-GEMINI] t={_t_event:.2f}s "
          f"source={source} offsets={offsets_s} conf={tracker_conf:.2f}")
    # Log zone cible pour debug 02:14 et 09:44
    if 100 <= _t_event <= 160 or 570 <= _t_event <= 600:
        print(f"  [DEBUG ZONE] Candidat dans zone cible — analyse détaillée activée")

    try:
        client = get_client()

        goal_score = 0.0
        shot_score = 0.0
        neg_score  = 0.0

        goal_votes = 0
        shot_votes = 0

        best_result      = None
        best_conf        = 0.0
        best_goal_offset = None   # offset où Gemini a vu un goal
        best_shot_offset = None   # offset où Gemini a vu le meilleur shot
        best_shot_conf   = 0.0
        checked_core     = False
        
        total_frames = get_video_frame_count(video_path)
        
        for off_s in offsets_s:
            if _quota_exhausted:
                break

            frame_id = frame_orig + int(off_s * fps)
            t_analyzed = frame_id / fps

            # Log détaillé pour zones cibles (02:14 et 09:44)
            t_event = frame_orig / fps
            if 100 <= t_event <= 160 or 570 <= t_event <= 600:
                print(f"    [GEMINI FRAME] event_t={t_event:.1f}s off={off_s:+d}s → frame={frame_id} t={t_analyzed:.1f}s")

            if frame_id < 0 or frame_id >= total_frames:
                continue

            result = _call_gemini_at_offset(
                client, video_path, frame_id, fps, event_type, danger, sport
            )

            if off_s != offsets_s[-1]:
                time.sleep(0.1)

            if result is None:
                continue

            rtype = result["type"]
            rconf = result["confiance"]

            signal = compute_signal_score(rtype, rconf, off_s, source)

            print(f"    [OFFSET {off_s:+}s] {rtype} conf={rconf:.2f} signal={signal:.3f}")

            if rtype == "goal":
                goal_votes += 1
                goal_score += signal

                if rconf > best_conf:
                    best_conf        = rconf
                    best_result      = result
                    best_goal_offset = off_s

            elif rtype == "shot":
                shot_votes += 1

                shot_signal = signal

                # tir loin du but → probablement dégagement / relance
                if not near_goal:
                    shot_signal *= 0.4

                # ballon finit au milieu du terrain → souvent faux tir
                if ball_end_x is not None:
                    x_pct = ball_end_x / frame_w

                    if 0.2 < x_pct < 0.8:
                        shot_signal *= 0.5

                shot_score += shot_signal

                print(
                    f"      shot_adjusted={shot_signal:.3f} "
                    f"(near_goal={near_goal}, ball_end_x={ball_end_x})"
                )

                
                if rconf > best_shot_conf:
                    best_shot_conf   = rconf
                    best_shot_offset = off_s

            else:
                neg_score += abs(signal)

            # 🔥 EARLY STOP INTELLIGENT (score-based only)
            if goal_score > 1.2:
                print("    [EARLY STOP] goal score suffisant")
                break

            if "posthoc" in str(source):
                if off_s >= 0:
                    checked_core = True
            else:
                if off_s == 0:
                    checked_core = True

            if checked_core and neg_score > 1.5:
                break

        # ─────────────────────────────
        # 🔥 REFINE (déclenché intelligemment)
        # ─────────────────────────────
        need_refine = (
            goal_votes == 0
            and (
                shot_score > 0.8
                or (shot_score > 0.55 and near_goal)
            )
            and best_shot_offset is not None
        )

        if need_refine and not _quota_exhausted:
            print(f"    [REFINE] autour offset={best_shot_offset}s")

            base_time = (frame_orig / fps) + best_shot_offset

            frames_window = extract_frames_window_pyav(
                video_path,
                center_time=base_time,
                window_sec=1.5,
                fps=fps,
            )

            if frames_window:
                refine_offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]

                refine_frames = []
                for roff in refine_offsets:
                    frame = pick_closest_frame(frames_window, base_time + roff)
                    if frame is not None:
                        refine_frames.append((roff, frame))

                if refine_frames:
                    parts = [text_to_part(
                        f"Analyse {len(refine_frames)} frames autour d'un evenement. "
                        "But = ballon dans filet ou franchit ligne. "
                        "Shot = tir vers but. Sinon none.\n"
                        '{"type":"goal|shot|none","confiance":0.9,"description":"court"}'
                    )]

                    for i, (roff, frm) in enumerate(refine_frames):
                        parts.append(text_to_part(f"Frame {i} t+{roff:+.1f}s"))
                        parts.append(frame_to_part(frm))

                    response = _call_gemini(client, parts)
                    
                    if response:
                        text = extract_response_text(response)
                        r2 = _safe_json_load(text)
                        if r2:
                            rtype = r2.get("type", "none")
                            rconf = float(r2.get("confiance", 0.5))

                            signal = compute_signal_score(rtype, rconf, 0, source)

                            print(f"    [REFINE RESULT] {rtype} conf={rconf:.2f}")

                            if rtype == "goal":
                                goal_votes += 1
                                goal_score += signal

                                if rconf > best_conf:
                                    best_conf = rconf
                                    best_result = {
                                        "type": "goal",
                                        "confiance": rconf,
                                        "description": r2.get("description", "")
                                    }

        # ─────────────────────────────
        # 🎯 DECISION FINALE
        # ─────────────────────────────
        final_score = goal_score - neg_score

        print(f"  [FINAL SCORE] goal={goal_score:.2f} neg={neg_score:.2f} total={final_score:.2f}")
        print(
            f"[DEBUG FINAL] goal_votes={goal_votes} "
            f"shot_votes={shot_votes} "
        )
        if final_score > 1.5:
            return {
                "type": "goal",
                "confiance": best_conf if best_conf > 0 else 0.7,
                "description": best_result.get("description", "") if best_result else "",
                "_goal_votes": goal_votes,
                "_shot_votes": shot_votes,
            }

        if goal_votes >= 1 and goal_score >= 0.5:
            return {
                "type": "goal",
                "confiance": min(best_conf, 0.7),
                "description": best_result.get("description", "") if best_result else "",
                "_goal_votes": goal_votes,
                "_shot_votes": shot_votes,
            }

        return {
            "type": "none",
            "confiance": 0.3,
            "description": best_result.get("description", "") if best_result else "",
            "_goal_votes": goal_votes,
            "_shot_votes": shot_votes,
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

        text = extract_response_text(response)
        if not text:
            return {}

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
# ------------------------------------------
# RECUPERER RESOLUTION VIDEO
# ------------------------------------------
def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w <= 0:
        w = 1920
    if h <= 0:
        h = 1080

    return w, h

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return max(total, 1)

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
):
    video_w, video_h = get_video_dimensions(video_path)
    
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

    # Pré-lier chaque but au tir le plus proche (avant Gemini)
    # → xG disponible même si Gemini crashe
    shots_all = [e for e in events if e.get("type") == "shot"]
    for cand in candidates:
        if float(cand.get("xg", 0) or 0) > 0:
            continue
        t_c = cand.get("time", 0)
        best_s, best_dt = None, 999
        for s in shots_all:
            dt = abs(s.get("time", 0) - t_c)
            if dt < best_dt and dt <= 8.0:
                best_dt, best_s = dt, s
        if best_s:
            cand["xg"]           = best_s.get("xg", 0.10)
            cand["linked_shot"]  = best_s.get("time", 0)

    print(f"  Validation Gemini : {len(candidates)} candidats...")
    validated = corrected = removed = 0

    for event in candidates:
        t_pre = event.get("time", 0)
        src   = event.get("detected_from", event.get("source", "?"))
        conf  = event.get("confidence", 0)
        xg    = event.get("xg", 0)
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

        result = validate_event(video_path, event, fps, sport,frame_w=video_w)

        if result is None:
            if event.get("confidence", 0) > 0.95:
                print("Gemini indisponible → conservé car tracker très confiant")
            else:
                event["_remove"] = True
                removed += 1
            continue

        validated  += 1
        gemini_type = result["type"]
        confiance = result["confiance"]
        goal_votes = result.get("_goal_votes", 0)
        shot_votes = result.get("_shot_votes", 0)
        t_sec = event.get("time", 0)
        tracker_conf = event.get("confidence", 0)
        print(f"  [POST-GEMINI] t={int(t_sec//60):02d}:{int(t_sec%60):02d} "
              f"gemini={gemini_type} conf={confiance:.2f} "
              f"tracker_conf={tracker_conf:.2f} "
              f"threshold={get_dynamic_threshold(event):.2f} "
              f"desc={result.get('description', '')[:60]}")

        # Seuil dynamique : adapté à la confiance de l'event
        # Un but détecté par goal_posthoc (conf élevée) → seuil plus souple
        threshold = get_dynamic_threshold(event) if event.get("type") == "goal"                     else MIN_CONF_SHOT

        if gemini_type == "goal" and (confiance >= threshold or goal_votes >= 1):
            # 🟢 Gemini confirme → gardé
            if result.get("equipe") is not None:
                event["team"] = result["equipe"]
            print(f"    → BUT CONFIRMÉ (conf={confiance:.2f})")

        elif gemini_type == "shot" and goal_votes == 0:
            # Gemini voit un tir → corriger
            event["type"] = "shot"
            corrected += 1
            print(f"    → CORRIGÉ en tir (conf={confiance:.2f})")

        elif gemini_type in ["goalkeeper_hold", "goalkeeper_throw",
                              "defensive_clearance", "corner", "touche"]:
            # 🔴 Signal clair non-but → supprimer si confiant
            if confiance >= 0.85:
                event["_remove"] = True
                removed += 1
                print(f"    → SUPPRIMÉ ({gemini_type} conf={confiance:.2f})")
            else:
                print(f"    → GARDÉ malgré {gemini_type} (conf={confiance:.2f} < 0.85)")

        else:
            # 🟡 Zone grise (type=none, vote insuffisant)
            # → décision basée sur signaux physiques
            posthoc_score    = event.get("score", event.get("danger", 0))
            high_conf_phys   = event.get("high_conf_physical", False)
            tracker_conf_val = event.get("confidence", 0)

            print(f"    [DECISION] type={gemini_type} "
                  f"phys={high_conf_phys} "
                  f"score={posthoc_score:.1f} "
                  f"tracker={tracker_conf_val:.2f}")

            # Zone grise — après refine, on exige goal_votes >= 1
            # Si Gemini n'a pas vu de but même avec le zoom fin → supprimé
            has_recent_shot = event.get("shot_linked", False)
            has_rebound     = event.get("rebound", False)
            gemini_saw_goal = result.get("_goal_votes", 0) >= 1

            if gemini_saw_goal:
                print("    → GARDÉ (Gemini a vu au moins 1 goal)")

            elif high_conf_phys and tracker_conf_val > 0.95:
                print("    → GARDÉ (signal physique très fort)")

            elif has_recent_shot and has_rebound and tracker_conf_val > 0.90:
                # Tir récent + rebond filet + tracker confiant → garder
                print(f"    → GARDÉ (shot_linked + rebound + tracker={tracker_conf_val:.2f})")

            else:
                event["_remove"] = True
                removed += 1
                print("    → SUPPRIMÉ")

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance

    events = [e for e in events if not e.get("_remove", False)]
    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")

    # Libérer les containers PyAV en fin de validation
    close_all_av_containers()

    return events