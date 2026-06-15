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
OFFSETS_POSTHOC = [0, -5, 5, 20]  # +20 ajouté : voir le kickoff après le but (signal fort)
OFFSETS_EVENTS = [-1, 0, 2, 10]

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
rebound_sig         = 0.15   # bonus signal pour candidats avec rebond filet

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
                         frame_w=1920, frame_h=1080,
                         confirmed_goal_times=None,
                         kickoff_offset=0):
    """
    Analyse la fenêtre [shot_time, shot_time+window] après un tir détecté.
    Envoie des frames espacées à Gemini avec la question :
    "Est-ce qu'un but a été marqué ? Si oui, à quel timestamp exact ?"

    Returns:
        dict {"is_goal": bool, "timestamp": float, "confidence": float, "desc": str}
        ou None si Gemini indisponible
    """
    global _quota_exhausted, _gemini_unavailable

    if _quota_exhausted or _gemini_unavailable:
        return None

    client = get_client()
    if client is None:
        return None

    # Stratégie d'échantillonnage optimisée — 8 frames max pour réduire le temps Gemini
    # Frame 1 : t+0s  — état au moment du tir
    # Frame 2 : t+2s  — ballon dans le filet ou gardien qui plonge
    # Frame 3 : t+4s  — réaction immédiate gardien / célébration
    # Frame 4 : t+7s  — célébration ou reprise du jeu
    # Frame 5 : t+12s — si célébration, elle est visible ici
    # Frame 6 : t+18s — remise en jeu commence
    # Frame 7 : t+25s — kickoff au centre ou corner/touche visible
    # Frame 8 : t+window — confirmation finale
    offsets = [0, 2, 4, 7, 12, 18, 25, window]
    sample_times = []
    seen = set()
    for off in offsets:
        t = round(shot_time + off, 1)
        if t not in seen and t <= shot_time + window:
            sample_times.append(t)
            seen.add(t)

    # Extraire les frames en une seule passe séquentielle
    # (évite les seeks répétés qui peuvent sauter des frames sur fichier Drive/réseau)
    parts = []
    valid_times = []
    cap = cv2.VideoCapture(video_path)
    try:
        if not sample_times:
            pass
        else:
            # Seek unique au début de la fenêtre
            first_frame_id = max(0, int(sample_times[0] * fps) - 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
            current_frame_id = first_frame_id

            # Construire un set des frame_ids cibles
            targets = {int(st * fps): st for st in sample_times}
            target_ids = sorted(targets.keys())
            target_idx = 0

            while target_idx < len(target_ids):
                target_id = target_ids[target_idx]

                # Avancer jusqu'à la frame cible
                while current_frame_id < target_id:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    current_frame_id += 1

                # Lire la frame cible
                ret, frame = cap.read()
                if ret and frame is not None:
                    current_frame_id += 1
                    h, w = frame.shape[:2]
                    if w > 960:
                        frame = cv2.resize(frame, (960, int(h * 960 / w)))
                    parts.append(frame_to_part(frame))
                    valid_times.append(targets[target_id])
                else:
                    break

                target_idx += 1
    finally:
        cap.release()

    if not parts:
        return None

    # ── EARLY STOP — prompt minimal sur frames 0-5s ─────────────────────
    # À 0-5s après le tir : chercher UNIQUEMENT le ballon dans le filet
    # Pas de kickoff possible à ce stade → prompt simplifié et rapide
    # Seuil 0.90 minimum, au moins 2 frames pour éviter faux positif
    EARLY_STOP_MAX_OFFSET = 5.0
    EARLY_STOP_MIN_CONF   = 0.90

    early_parts = [p for p, t in zip(parts, valid_times)
                   if t - shot_time <= EARLY_STOP_MAX_OFFSET]
    early_times = [t for t in valid_times
                   if t - shot_time <= EARLY_STOP_MAX_OFFSET]

    if len(early_parts) >= 2:
        _et_str = ", ".join(f"{int(t//60):02d}:{int(t%60):02d}" for t in early_times)
        _early_prompt = f"""Football match analysis.
{len(early_parts)} frames from {_et_str}, taken 0-5 seconds after a shot on goal at {int(shot_time//60):02d}:{int(shot_time%60):02d}.

Question: Is the ball clearly INSIDE the goal (behind the goal line, inside the net)?

Rules — read carefully:
- YES: ball is fully past the goal line, inside the net, net is VISIBLY BULGING/DEFORMED by the ball (net pushed inward)
- YES: goalkeeper is retrieving the ball from INSIDE the net (body inside goal area)
- NO: ball is near the net or in front of it but net is flat/undisturbed → answer NO
- NO: ball is beside the post or outside the goal frame → answer NO
- NO: goalkeeper holding/catching the ball in his hands or arms (even if inside the goal area)
- NO: ball is BEHIND the goal (outside the net, on the other side of the goal frame) → this is a corner kick or goal kick, NOT a goal
- NO: ball visible behind the goal structure from outside → corner kick situation
- NO: goalkeeper picking up the ball while standing upright
- NO: ball in front of the goal or on the goal line
- NO: goalkeeper holding/catching ball in front of the goal
- NO: ball near the post but not inside
- NO: ball anywhere outside the net
- NO: goal kick — goalkeeper or defender kicking a stationary ball outward from the 6-yard box (ball moving AWAY from net toward midfield) → answer NO immediately
- NO: goalkeeper distributing the ball (punting or throwing it outward from inside the goal) → answer NO immediately
- NO: out-of-match content — children playing casually (small bodies, no team jerseys), empty pitch, informal game, or post-match activity → answer NO immediately

Return ONLY valid JSON:
{{"is_goal": true or false, "timestamp": <seconds or null>, "confidence": <0.0-1.0>, "evidence": "<describe exactly: ball position relative to net/line>"}}
confidence=0.95 only if ball is unmistakably inside the net.
Default to is_goal=false if any doubt."""
        try:
            _resp = client.models.generate_content(
                model    = "gemini-2.5-flash",
                contents = [_early_prompt] + list(early_parts),
            )
            _data = _safe_json_load(_resp.text.strip())
            if (_data and _data.get("is_goal")
                    and _data.get("confidence", 0) >= EARLY_STOP_MIN_CONF):
                _ts = _data.get("timestamp")
                print(f"  [SHOT→GOAL EARLY] shot={shot_time:.1f}s → BUT t={_ts} "
                      f"conf={_data['confidence']:.2f} | {_data.get('evidence','')[:80]}")
                # Valider que le timestamp est dans la fenêtre réelle
                _ts_validated = None
                if _ts is not None:
                    try:
                        _ts_f = float(_ts)
                        if shot_time - 5 <= _ts_f <= shot_time + window + 5:
                            _ts_validated = _ts_f
                        else:
                            # Timestamp hors fenêtre (ex: Gemini renvoie mm:ss au lieu de secondes)
                            _ts_validated = shot_time + 2
                    except (ValueError, TypeError):
                        _ts_validated = shot_time + 2
                else:
                    _ts_validated = shot_time + 2
                return {
                    "is_goal":    True,
                    "timestamp":  _ts_validated,
                    "confidence": _data["confidence"],
                    "desc":       _data.get("evidence", ""),
                    "goal_votes": 2,   # early stop = ballon dans filet évident = score >= 5
                    "goal_score": 8,
                }
        except Exception:
            pass  # early stop échoue → continue avec analyse complète

        # Prompt Gemini — signaux post-but + strict sur kickoff vs remise en touche
    times_str = ", ".join(f"{int(t//60):02d}:{int(t%60):02d}" for t in valid_times)
    prompt = f"""You are analyzing a football/soccer match video.
I'm showing you {len(parts)} frames from timestamps: {times_str}
These frames cover {window} seconds after a detected shot on target at {int(shot_time//60):02d}:{int(shot_time%60):02d}.

Focus ONLY on ball position relative to the goal line. Ignore all other context unless it directly confirms a goal.

Your task: determine if a GOAL was scored in this time window.
This is an amateur/semi-professional football match filmed from the sideline.
The goalkeeper wears a different colored jersey from both outfield teams.
The goals have white posts and white or colored nets at each end of the pitch.

IMPORTANT TEMPORAL RULE:
Interpret these frames as a continuous sequence of ONE event, not independent snapshots.
If the ball is inside the net in any frame, the goal happened slightly BEFORE that frame.
If a celebration is visible, the goal happened several seconds before.
Do NOT treat each frame as isolated — the story builds across frames.

A goal is confirmed ONLY by ONE of these TWO evidences:

EVIDENCE A — BALL IN NET (most reliable):
- Ball is physically visible INSIDE the net (behind the goal line, between the posts)
- OR goalkeeper is crouching/diving to retrieve ball FROM INSIDE the net
- The net must be visibly deformed or ball clearly behind the line
- The goalkeeper's position INSIDE the goal area confirms the ball went in
- The goalkeeper must be CROUCHING or DIVING to pick up the ball — NOT simply standing in goal waiting
- FAST GOALS: in amateur football the ball can enter and exit the net in under 1 second (net deforms briefly then returns to normal). If the net shows a brief deformation in frame 1 or 2 even if the ball is no longer visible inside → this counts as ball-in-net evidence (+3).
- If the net appears BRIEFLY DEFORMED in an early frame but flat in subsequent frames, this is consistent with a fast goal where the ball rebounded out — do NOT discount this as a negative signal.
- CRITICAL DISTINCTION — PENALTY BOX vs GOAL NET: The penalty area is flat WHITE LINES painted on the GROUND (a rectangle on the grass). The goal net is a 3D structure BEHIND the goal line, elevated, attached to posts. Do NOT confuse flat ground lines with a net. If what you see is a rectangle of white lines on the grass with players standing inside it → this is the penalty box, NOT a goal net. Score 0 for ball-in-net if the "net" is actually the penalty area markings.

AMBIGUITY RULE (very important):
If the ball position is unclear, partially hidden, or you are not 100% certain it crossed the line:
→ ALWAYS return is_goal=false
→ NEVER guess a goal from partial visibility
→ When in doubt = no goal

EVIDENCE B — CENTER KICKOFF (very specific restart):
A center kickoff is valid ONLY if ALL of these conditions are met simultaneously:
1. Ball is exactly at the CENTER SPOT (geometric middle of the pitch)
2. BOTH teams are clearly on OPPOSITE halves — visible midfield line symmetry
3. ALL players are STATIC or walking very slowly — nobody running
4. NO cluster of players near the ball — they are spread across the whole field
5. The scene looks like a PHOTO — frozen, organized, symmetric
6. You can clearly see the halfway line separating the two teams

If ANY of these conditions is missing → it is NOT a center kickoff → is_goal=false

This is COMPLETELY DIFFERENT from:
- Free kick: players clustered in one zone, not symmetric across field
- Throw-in: players near sideline, not at center
- Normal play: players running in various directions
- Defensive repositioning: players walking back but NOT at center spot
- Players spread but still moving = NOT a kickoff
- Goalkeeper standing upright in goal = normal positioning, NOT retrieving ball from net
- Ball near goal but outside net = NOT a goal, even if goalkeeper is nearby

DO NOT interpret as a goal:
- Throw-in: player on the SIDELINE throwing or preparing to throw the ball
- Players clustered or fighting near the SIDELINE = throw-in situation, NOT goal
- Players grouped on ONE side of the field = NOT a kickoff (kickoff = spread across whole field)
- Free kick: players standing around ball anywhere on the pitch
- Players running or fighting for the ball = normal play, NOT celebration
- Players raising one arm = could be calling for the ball, NOT necessarily celebrating
- Corner kick: player near the corner flag
- Any restart near the sideline or touchline = NOT a goal kickoff
- If the ball or players are near the sideline = almost certainly a throw-in, NOT a goal
- Defensive free kick in the middle of the pitch: one player about to kick stationary ball, others spread = NOT a kickoff
- Players walking back to positions after a foul = NOT a celebration, NOT a goal
- GOALKEEPER HOLDING BALL: goalkeeper catching, holding, or securing the ball in hands/arms = NOT a goal, even inside goal area or crouching
- GOALKEEPER STANDING WITH BALL: goalkeeper upright or slightly bent holding ball = NOT a goal
- DEFENSIVE CLEARANCE: ball kicked or headed away from goal by a defender = NOT a goal
- CROSS OR CENTER: ball played from wing into penalty area but cleared by defender = NOT a goal
- BALL HIT POST: ball bouncing off post or crossbar without clearly entering the net = NOT a goal
- BALL BEHIND THE GOAL (OUTSIDE): ball visible behind the goal structure from outside the net — this means corner kick or goal kick, NOT a goal. The ball must be INSIDE the net between the posts, not behind the goal frame from the exterior.
- DEFENSIVE FREE KICK NEAR GOAL: player about to kick a stationary ball near the penalty area, others standing around = NOT a goal, NOT a kickoff
- GOAL KICK (coup de pied de but): goalkeeper or defender placing ball on/near the 6-yard box line and kicking it outward (away from goal, toward midfield) = NOT a goal. Key signals: ball starts inside or very near the goal area, no attacking players nearby, ball is kicked AWAY from the net toward the field. This is a restart after the ball went out, NOT a goal. Score -5 immediately.
- GOALKEEPER PUNTING / DISTRIBUTION: goalkeeper holding ball inside goal area then kicking/throwing it outward = NOT a goal. The ball is moving AWAY from the net.
- BALL CAUGHT BY GOALKEEPER THEN CLEARED: goalkeeper catches ball and punts/throws it = NOT a goal, even if ball was near goal line
- LONG BALL INTO GOALKEEPER: ball played toward goal that goalkeeper catches or holds comfortably = NOT a goal (no danger)
- REFEREE ON PITCH NEAR PLAYERS: if the referee (black or yellow kit) is visibly active on the pitch standing near a group of players → this indicates a stoppage (injury, foul, incident), NOT a goal. Players grouping around the referee = stoppage, not celebration.
- INJURY STOPPAGE: players gathering around a player on the ground, referee nearby, ball out of play near the touchline or penalty area = injury stoppage, NOT a goal. Score -5 immediately.
- PENALTY BOX LINES MISREAD AS NET: if the "deformation" or "rectangle" visible is flat on the ground with white lines → it is the penalty box painted on grass, NOT the goal net. Do not award +3 for this.
- OUT-OF-MATCH CONTENT: if the scene shows children playing casually (small bodies, no team jerseys, informal play), spectators on the pitch, an empty pitch with no organized game, or a goal kick/free kick situation from a completely different informal context → this is NOT a football match goal. Score -10 immediately, is_goal=false.
  Signs of out-of-match content: very small players (children), no visible team colors/jerseys, casual clothing, no referee visible anywhere, completely empty pitch, or the context clearly resembles post-match/pre-match informal activity.

For celebrations to count as evidence they must be UNAMBIGUOUS:
- Multiple players from SAME team running toward each other with arms wide open
- Players jumping on each other, clearly hugging in joy
- NOT: players jogging, disputing a ball, or one player raising a hand

SCORING SYSTEM — assign a score based on ALL frames combined:

POSITIVE signals (accumulate across frames):
+5 : center kickoff clearly visible (ball at center spot, both teams on opposite halves, static formation)
+4 : unambiguous multi-player celebration (multiple players running toward each other, arms wide, hugging)
+3 : ball unmistakably INSIDE the net — net is visibly BULGING/DEFORMED by the ball, AND ball is clearly behind the goal line between the posts. NOT just near the net.
+3 : net clearly deformed/bulging in an early frame even if ball is no longer visible inside (fast goal, ball rebounded out)
+3 : attacking players walking/jogging back toward center circle after action
+2 : players from scoring team showing clear joy reactions (arms up, jumping, turning to teammates) even if full group celebration not yet formed
+1 : ball near goal line but position unclear

NEGATIVE signals (subtract immediately):
-10: out-of-match content detected (children playing, empty pitch, informal game, no team jerseys) → is_goal=false, override ALL positives, stop scoring
-5 : goalkeeper holding/catching/securing ball in hands or arms → is_goal=false, override ALL positives
-5 : referee visible and active on pitch near players (injury stoppage, foul stoppage) → is_goal=false, override ALL positives
-3 : players gathered around a player lying on the ground (injury) = NOT a celebration
-4 : ball clearly kicked/headed away from goal (defensive clearance)
-4 : corner kick or throw-in visible immediately after
-4 : ball visible BEHIND the goal from outside (behind the goal frame/net exterior) → corner kick or goal kick situation
-5 : goal kick detected (goalkeeper/defender kicking stationary ball outward from 6-yard box, ball moving away from net) → is_goal=false
-3 : goalkeeper standing upright with ball (not diving, not retrieving from inside net)
-3 : defensive free kick near goal area (stationary ball, players standing around)
-1 : ball visible beside/around the post or outside the frame of the goal (not between the posts) — weak signal, overridden by ball-in-net evidence
-2 : ball visible outside the net after the action

CRITICAL RULE — avoid false positives on amateur pitches:
The net is always visible in the background. A ball NEAR the net or in front of it is NOT a goal.
Only award +3 if the net is clearly DEFORMED (pushed inward) by the ball, proving it crossed the line.
If the net appears flat/undisturbed and the ball is near it → score 0 for that signal, not +3.

DECISION:
- total_score >= 5 → is_goal=true ONLY IF at least one physical signal is also present.
  Physical signals = net deformation (+3), ball unmistakably in net (+3), celebration (+4 or +2), players walking back to center (+3).
  CRITICAL EXCEPTION: if the ONLY positive signal is center kickoff (+5) with NO physical signal → is_goal=false.
  Reason: a kickoff can follow any stoppage (foul, corner, free kick), not only goals.
  Example: kickoff +5, ball outside net -2 = score 3 → is_goal=false (kickoff alone, negative physical signal)
  Example: kickoff +5, net deformation +3 = score 8 → is_goal=true (physical signal present)
- total_score == 4 → is_goal=true if celebration (+4) OR net deformation (+3) is present
- total_score 3 → is_goal=true ONLY if net deformation clearly confirmed (+3 signal present)
- total_score <= 2 → is_goal=false
- goalkeeper holding ball detected (-5) → is_goal=false immediately, no exception

Confidence mapping:
- score >= 8 → confidence=0.95
- score 6-7 → confidence=0.90
- score 5 → confidence=0.85
- score 4 with celebration or net deformation → confidence=0.80
- score 3 with net deformation → confidence=0.80
- score <= 2 → is_goal=false, confidence=0.0

Return ONLY valid JSON, no markdown:
{{
  "is_goal": true or false,
  "timestamp": <seconds when ball crossed line, or null>,
  "confidence": <0.0 to 1.0>,
  "goal_score": <integer: your computed total score>,
  "evidence": "<list each signal detected with its score — e.g. 'center kickoff +5, celebration +4 = 9' or 'goalkeeper holding ball -5 = -5, is_goal=false'>"
}}

DEFAULT TO is_goal=false if total_score <= 2 or goalkeeper holding ball detected."""

    parts_with_prompt = [text_to_part(prompt)] + parts

    try:
        t0 = time.time()
        result = _call_gemini(client, parts_with_prompt)
        elapsed = time.time() - t0
        _METRICS["cache_misses"] += 1
        _METRICS["gemini_time"]  += elapsed

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
        goal_score = int(parsed.get("goal_score", 0))

        # FIX kickoff fantôme — si le signal est UNIQUEMENT un kickoff (+5)
        # SANS aucun signal physique (ballon dans filet, gardien qui récupère)
        # et qu'un but déjà confirmé existe dans les 60s précédentes → rejeter
        # PATCH v3 :
        # - "walking back" SEUL sans filet/gardien = kickoff (pas physique)
        # - walking back + kickoff + "+3" ensemble = signal valide
        # - fenêtre 200s → 60s
        if is_goal and confirmed_goal_times:
            _evidence_lower = evidence.lower()
            _has_net_signal = any(kw in _evidence_lower for kw in [
                "inside the net", "inside net", "net deform",
                "goalkeeper", "retrieving", "ball unmistakably",
                "ball clearly inside", "bulg",
            ])
            _has_walking_back = any(kw in _evidence_lower for kw in [
                "walking back", "walking/jogging back", "jogging back",
                "toward center circle", "back toward center",
            ])
            _has_physical_signal = (
                _has_net_signal
                or (_has_walking_back and "+5" in evidence and "+3" in evidence)
            )
            _kickoff_only = (
                "+5" in evidence
                and not _has_physical_signal
            )
            if _kickoff_only:
                _recent_goal = any(
                    0 < shot_time - gt < 60
                    for gt in confirmed_goal_times
                )
                if _recent_goal:
                    print(f"  [SHOT→GOAL] ❌ Kickoff fantôme rejeté t={int(shot_time//60):02d}:{int(shot_time%60):02d} "
                          f"— kickoff sans signal physique, but confirmé dans 60s (score={goal_score})")
                    return {
                        "is_goal":    False,
                        "timestamp":  None,
                        "confidence": 0.0,
                        "desc":       f"kickoff fantôme rejeté (but confirmé dans 60s)",
                        "goal_votes": 0,
                        "goal_score": 0,
                    }

        # Convertir goal_score en goal_votes pour la logique pipeline
        # Règle stricte v9.8 :
        # Si Gemini voit lui-même le ballon hors du filet (signaux -1 ou -2 présents),
        # le kickoff seul ne peut pas compenser — exiger score >= 9 pour 2 votes
        # Sinon (preuves directes solides) : score >= 6 suffit
        _has_ball_outside = ("outside the net" in evidence.lower()
                             or "beside" in evidence.lower()
                             or "hors du filet" in evidence.lower()
                             or "outside goal" in evidence.lower())
        if _has_ball_outside:
            goal_votes = 2 if goal_score >= 10 else (1 if goal_score >= 7 else 0)
        else:
            goal_votes = 2 if goal_score >= 6 else (1 if goal_score >= 4 else 0)

        # Valider le timestamp dans la fenêtre
        if is_goal and timestamp is not None:
            try:
                timestamp = float(timestamp)
                # PATCH v3 : si Gemini retourne le timestamp du kickoff (visible à +20s)
                # plutôt que celui du but, corriger vers shot_time+2
                # Critère : timestamp proche de la fin de la fenêtre = kickoff vu, pas le but
                # On garde le timestamp Gemini seulement s'il est dans les 10 premières secondes
                # de la fenêtre (tir → but immédiat) ; au-delà c'est probablement le kickoff
                if timestamp > shot_time + min(10, window * 0.4):
                    # Timestamp trop tardif = kickoff visible → recentrer sur le tir
                    timestamp = shot_time + 2
                    print(f"    [TIMESTAMP FIX] kickoff retourné → recentré sur shot+2s")
                if not (shot_time - 5 <= timestamp <= shot_time + window + 5):
                    # Timestamp hors fenêtre — utiliser centre de la fenêtre
                    timestamp = shot_time + window / 2
            except (ValueError, TypeError):
                timestamp = shot_time + window / 2
        elif is_goal:
            timestamp = shot_time + window / 2

        print(f"  [SHOT→GOAL] shot={shot_time:.1f}s → is_goal={is_goal} "
              f"t={timestamp} conf={confidence:.2f} score={goal_score}\n"
              f"  EVIDENCE FULL: {evidence}")

        print(f"  [SHOT→GOAL SCORE] goal_score={goal_score} → goal_votes={goal_votes}")
        return {
            "is_goal":    is_goal,
            "timestamp":  timestamp,
            "confidence": confidence,
            "desc":       evidence,
            "goal_votes": goal_votes,
            "goal_score": goal_score,
        }

    except Exception as e:
        print(f"  [SHOT→GOAL] Erreur : {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# find_goal_after_shot_v2 — Architecture Observation → Décision
#
# Principe fondamental :
#   Gemini observe uniquement — il ne conclut JAMAIS "is_goal"
#   Le code Python prend la décision finale
#
# Différences vs V1 :
#   - Prompt ne contient pas les mots "goal", "is_goal", "did a goal"
#   - Gemini retourne un JSON d'observations pures
#   - Le scoring est entièrement dans le code Python
#   - Pas de seuil fixe au départ — logs pour calibration
# ─────────────────────────────────────────────────────────────────────────────

def _score_observation_v2(obs: dict) -> tuple[float, list[str]]:
    """
    Calcule le score but à partir des observations Gemini.
    Gemini n'a jamais vu cette logique — il observe, Python décide.

    Retourne (score, [raisons])
    """
    score = 0.0
    reasons = []

    ball_loc = obs.get("ball_location", "unknown")
    kickoff  = obs.get("center_kickoff_visible", False)
    to_center = obs.get("players_moving_to_center", False)
    celebration = obs.get("celebration_visible", False)
    gk_inside = obs.get("goalkeeper_inside_goal", False)
    gk_visible = obs.get("goalkeeper_visible", False)
    referee = obs.get("referee_visible", False)

    # ── Signaux négatifs — override immédiat ──────────────────────────────
    if referee:
        score -= 1.0
        reasons.append("referee_visible -1")

    # ── Signaux positifs ──────────────────────────────────────────────────

    # Ballon dans le filet — signal physique direct
    if ball_loc == "inside_net":
        score += 3.0
        reasons.append("ball_inside_net +3")
    elif ball_loc == "goal_area":
        score += 0.5
        reasons.append("ball_goal_area +0.5")

    # Gardien à l'intérieur du but (récupération) — SEULEMENT si ballon en zone
    # Sans signal ballon, le gardien dans le but c'est normal
    if gk_inside and ball_loc in ("inside_net", "goal_area"):
        score += 1.5
        reasons.append("goalkeeper_inside_goal +1.5")

    # Coup d'envoi — signal de contexte faible (réduit depuis +5 en V1)
    if kickoff:
        score += 1.0
        reasons.append("center_kickoff +1")

    # Joueurs qui reviennent au centre — corroborant
    if to_center:
        score += 1.0
        reasons.append("players_moving_to_center +1")

    # Célébration — signal faible, ambigu, metadata uniquement
    if celebration:
        score += 0.5
        reasons.append("celebration +0.5")

    return score, reasons


def find_goal_after_shot_v2(video_path, shot_time, window=30, fps=25,
                             frame_w=1920, frame_h=1080,
                             confirmed_goal_times=None,
                             kickoff_offset=0):
    """
    V2 — Architecture Observation → Décision.

    Gemini retourne uniquement des observations visuelles structurées.
    Aucun champ "is_goal" dans le prompt ou la réponse Gemini.
    Le code Python calcule le score et prend la décision.

    Returns:
        dict compatible V1 :
        {"is_goal": bool, "timestamp": float, "confidence": float,
         "desc": str, "goal_votes": int, "goal_score": float}
        ou None si Gemini indisponible
    """
    global _quota_exhausted, _gemini_unavailable

    if _quota_exhausted or _gemini_unavailable:
        return None

    client = get_client()
    if client is None:
        return None

    # ── Extraction frames (identique V1) ──────────────────────────────────
    offsets = [0, 2, 4, 7, 12, 18, 25, window]
    sample_times = []
    seen = set()
    for off in offsets:
        t = round(shot_time + off, 1)
        if t not in seen and t <= shot_time + window:
            sample_times.append(t)
            seen.add(t)

    parts = []
    valid_times = []
    cap = cv2.VideoCapture(video_path)
    try:
        if sample_times:
            first_frame_id = max(0, int(sample_times[0] * fps) - 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_id)
            current_frame_id = first_frame_id
            targets = {int(st * fps): st for st in sample_times}
            target_ids = sorted(targets.keys())
            target_idx = 0
            while target_idx < len(target_ids):
                target_id = target_ids[target_idx]
                while current_frame_id < target_id:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    current_frame_id += 1
                ret, frame = cap.read()
                if ret and frame is not None:
                    current_frame_id += 1
                    h, w = frame.shape[:2]
                    if w > 960:
                        frame = cv2.resize(frame, (960, int(h * 960 / w)))
                    parts.append(frame_to_part(frame))
                    valid_times.append(targets[target_id])
                else:
                    break
                target_idx += 1
    finally:
        cap.release()

    if not parts:
        return None

    # ── Prompt d'observation pure — Gemini ne sait pas qu'on cherche un but ──
    times_str = ", ".join(f"{int(t//60):02d}:{int(t%60):02d}" for t in valid_times)
    shot_str  = f"{int(shot_time//60):02d}:{int(shot_time%60):02d}"

    prompt_v2 = f"""You are analyzing a football/soccer match.
I am showing you {len(parts)} frames from timestamps: {times_str}
These frames cover {window} seconds of play starting at {shot_str}.

Your task: describe ONLY what you can directly observe in these frames.
Do NOT interpret, conclude, or infer what happened. Only report visible facts.

For EACH of the following fields, report what you observe across ALL frames combined:

ball_visible: true if the ball is visible in at least one frame, false otherwise.

ball_location: where is the ball located in the frames? Choose the MOST specific location observed:
  - "inside_net"   : ball is physically INSIDE the net, clearly behind the goal line between the posts, net is visibly pushed inward
  - "goal_area"    : ball is within the 6-yard box area in front of goal (NOT inside net)
  - "penalty_area" : ball is inside the penalty box but outside the 6-yard box
  - "midfield"     : ball is in the middle third of the pitch
  - "out_of_play"  : ball is out of bounds (touchline, corner, etc.)
  - "unknown"      : ball not visible or location unclear

  IMPORTANT: "inside_net" means the ball has physically crossed the goal line and is inside the net structure.
  Do NOT use "inside_net" if the ball is near the net but outside it, or if the net appears flat/undisturbed.
  Do NOT confuse the painted penalty box lines on the ground with the 3D goal net structure.

goal_visible: true if the goal structure (posts + net) is visible in at least one frame.

goalkeeper_visible: true if the goalkeeper (different colored jersey) is visible.

goalkeeper_inside_goal: true ONLY if the goalkeeper is physically inside the goal structure (between the posts, behind the goal line), crouching or retrieving something. NOT simply standing in front of the goal.

center_kickoff_visible: true ONLY if you can clearly see:
  - ball positioned at the exact center spot of the pitch
  - players from both teams positioned on opposite halves
  - static, organized formation visible
  All three conditions must be simultaneously visible. If any condition is missing, return false.

players_moving_to_center: true if you observe players walking or jogging toward the center circle of the pitch.

celebration_visible: true ONLY if you observe multiple players from the same team clearly embracing, jumping on each other, or running toward each other with arms open in obvious joy. NOT: one player raising a hand, players jogging, players disputing.

referee_visible: true if you observe the referee (typically in black, yellow, or fluorescent kit, different from both teams) visibly active on the pitch.

confidence: your overall confidence in the accuracy of these observations (0.0 to 1.0).

Return ONLY valid JSON, no markdown, no explanation:
{{
  "ball_visible": true or false,
  "ball_location": "inside_net|goal_area|penalty_area|midfield|out_of_play|unknown",
  "goal_visible": true or false,
  "goalkeeper_visible": true or false,
  "goalkeeper_inside_goal": true or false,
  "center_kickoff_visible": true or false,
  "players_moving_to_center": true or false,
  "celebration_visible": true or false,
  "referee_visible": true or false,
  "confidence": 0.0
}}"""

    try:
        t0 = time.time()
        result = _call_gemini(client, [text_to_part(prompt_v2)] + parts)
        elapsed = time.time() - t0
        _METRICS["gemini_time"] += elapsed

        if result is None:
            return None

        text   = extract_response_text(result)
        parsed = _safe_json_load(text)

        if parsed is None:
            return None

        # ── Scoring Python ────────────────────────────────────────────────
        score, reasons = _score_observation_v2(parsed)
        reasons_str = ", ".join(reasons) if reasons else "no signals"

        # ── Log pour calibration (pas de décision binaire encore) ─────────
        print(f"  [V2 OBS] shot={shot_time:.1f}s | score={score:.1f} | {reasons_str}")
        print(f"  [V2 OBS] ball={parsed.get('ball_location','?')} "
              f"kickoff={parsed.get('center_kickoff_visible','?')} "
              f"to_center={parsed.get('players_moving_to_center','?')} "
              f"celebration={parsed.get('celebration_visible','?')} "
              f"gk_inside={parsed.get('goalkeeper_inside_goal','?')} "
              f"referee={parsed.get('referee_visible','?')}")

        # ── Décision — seuil provisoire pour tests, ajuster après calibration ──
        # Score >= 4.0 : inside_net (+3) + au moins un corroborant
        # Score >= 2.0 : kickoff + to_center uniquement → is_goal=True provisoire
        # À calibrer sur vidéo 1 + vidéo 2 avant de fixer définitivement
        THRESHOLD = 2.0
        is_goal_v2 = (score >= THRESHOLD)

        # Confiance basée sur score
        if score >= 4.5:
            conf_v2 = 0.90
        elif score >= 3.0:
            conf_v2 = 0.80
        elif score >= 2.0:
            conf_v2 = 0.70
        else:
            conf_v2 = 0.0

        # Timestamp estimé
        timestamp_v2 = shot_time + 2 if is_goal_v2 else None

        # goal_votes compatible pipeline V1
        goal_votes_v2 = 2 if score >= 4.5 else (1 if score >= 2.0 else 0)

        print(f"  [V2 DECISION] score={score:.1f} → is_goal={is_goal_v2} "
              f"conf={conf_v2:.2f} votes={goal_votes_v2}")

        return {
            "is_goal":    is_goal_v2,
            "timestamp":  timestamp_v2,
            "confidence": conf_v2,
            "desc":       reasons_str,
            "goal_votes": goal_votes_v2,
            "goal_score": score,
        }

    except Exception as e:
        print(f"  [V2] Erreur : {e}")
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

    start_t = max(0.0, center_time - window_sec)
    end_t   = center_time + window_sec
    frames  = []
    t0      = time.time()

    # Tentative PyAV
    pyav_ok = False
    if _get_av_container(video_path) is not None:
        try:
            container = _get_av_container(video_path)
            stream    = container.streams.video[0]
            seek_ts   = int(start_t / float(stream.time_base))
            with _AV_LOCK:
                container.seek(seek_ts, backward=True, stream=stream)
                try:
                    container.flush_buffers()
                except AttributeError:
                    pass
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
            pyav_ok = len(frames) > 0
        except Exception as e:
            print(f"[PyAV batch] erreur : {e} — fallback OpenCV")
            frames = []

    # Fallback OpenCV si PyAV a echoue ou indisponible
    if not pyav_ok:
        try:
            cap   = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_offsets = [-window_sec, -window_sec * 0.5, 0.0, window_sec * 0.5, window_sec]
            for off in sample_offsets:
                t_target  = center_time + off
                frame_idx = max(0, min(int(t_target * fps), total - 1))
                frame     = safe_seek_frame(cap, frame_idx, max_jump=30)
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > 960:
                        frame = cv2.resize(frame, (960, int(h * 960 / w)))
                    frames.append((t_target, frame))
            cap.release()
        except Exception as e2:
            print(f"[OpenCV fallback] erreur : {e2}")

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
    # PATCH v3 : terminal_goal → offsets élargis pour voir le filet avant (-3s)
    # et le kickoff après (+5s, +20s) — les offsets courts [-1,0,+2] ratent souvent le signal
    if "terminal_goal" in str(source):
        offsets_s = [-3, 0, 5, 20]
    elif "posthoc" in str(source):
        offsets_s = OFFSETS_POSTHOC
    else:
        offsets_s = OFFSETS_EVENTS
    offsets_s = sorted(offsets_s, key=lambda x: abs(x))

    _t_event = event.get('time', 0)
    # Note : offsets enrichis {1, 2} retirés — coût +2 appels Gemini sans gain prouvé

    print(f"  [PRE-GEMINI] t={_t_event:.2f}s "
          f"source={source} offsets={offsets_s} conf={tracker_conf:.2f}")

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

                # Mémoriser les timestamps où Gemini confirme un tir dangereux
                # → utilisé pour marquer les events tirs comme occasions à montrer
                if rconf >= 0.90:
                    if "_shot_offsets_seen" not in event:
                        event["_shot_offsets_seen"] = []
                    event["_shot_offsets_seen"].append(t_analyzed)

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

            # PATCH v3 : terminal_goal → ne pas couper sur neg_score
            # les offsets +5s/+20s peuvent voir des situations normales avant le kickoff
            if checked_core and neg_score > 1.5 and "terminal_goal" not in str(source):
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
        # Si Gemini a vu un but (goal_votes >= 1), ne pas laisser neg_score l'annuler
        # Une touche/corner avant un but est normal (corner → but, remise → but)
        final_score_no_neg = goal_score if goal_votes >= 1 else final_score

        print(f"  [FINAL SCORE] goal={goal_score:.2f} neg={neg_score:.2f} total={final_score:.2f} (no_neg={final_score_no_neg:.2f})")
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

        # Override supprimé v9.8 : goal_votes >= 1 seul ne suffit pas à ignorer les pénalités
        # Un score négatif = signal fort (gardien tient le ballon, touche, etc.)
        # Exiger goal_votes >= 2 ET final_score > 0
        if goal_votes >= 2 and final_score > 0:
            print(f"  [GOAL 2VOTES+POS] goal_votes={goal_votes} final_score={final_score:.2f} → but confirmé")
            return {
                "type": "goal",
                "confiance": min(best_conf, 0.70),
                "description": best_result.get("description", "") if best_result else "",
                "_goal_votes": goal_votes,
                "_shot_votes": shot_votes,
            }

        # Cas rebond : tir contré puis but dans la même fenêtre
        # Contrainte temporelle : goal APRÈS shot, dans les 6s
        # ex: tir contré à offset 0s → but à offset +5s = rebond → valider
        if (goal_votes >= 1 and shot_votes >= 1 and goal_score >= 0.25
                and best_goal_offset is not None and best_shot_offset is not None
                and best_goal_offset > best_shot_offset
                and (best_goal_offset - best_shot_offset) <= 6):
            print(f"  [REBOND] shot@{best_shot_offset:+d}s → goal@{best_goal_offset:+d}s "
                  f"(Δ={best_goal_offset - best_shot_offset}s) → BUT confirmé")
            return {
                "type": "goal",
                "confiance": min(best_conf, 0.65),
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

def _visual_match_score(v1, v2):
    """
    Score de similarité entre 2 descriptions visuelles (0.0 → 1.0).
    Comparaison par champ texte — local au match, pas de ML.
    Pondération : boots et socks sont les plus discriminants.
    """
    if not v1 or not v2:
        return 0.0

    WEIGHTS = {
        "boots":       0.30,   # très discriminant (couleur unique par joueur)
        "socks":       0.25,   # très discriminant (position + couleur)
        "hair":        0.20,   # discriminant si coupe distinctive
        "skin":        0.10,   # utile mais moins unique
        "body":        0.08,   # morphologie — variable selon angle caméra
        "role":        0.07,   # utile si rôle stable dans le match
    }

    total_weight = 0.0
    total_score  = 0.0

    for field, weight in WEIGHTS.items():
        a = (v1.get(field) or "").lower().strip()
        b = (v2.get(field) or "").lower().strip()

        if not a or not b:
            continue  # champ absent → ne pénalise pas

        # Similarité mot-clé : cherche des mots communs
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            continue

        common  = words_a & words_b
        union   = words_a | words_b
        jaccard = len(common) / len(union)

        # Bonus si correspondance exacte sur un mot fort
        # (ex: "rouge" dans les 2, ou "rasé" dans les 2)
        strong_keywords = {
            "rouge", "bleu", "noir", "blanc", "vert", "jaune", "orange",
            "rasé", "long", "court", "bouclé", "tressé", "chauve",
            "haut", "bas", "mi-mollet",
            "grand", "petit", "svelte", "trapu",
            "gauche", "droit", "ailier", "attaquant", "milieu",
        }
        bonus = 0.1 if (common & strong_keywords) else 0.0

        field_score = min(1.0, jaccard + bonus)
        total_score  += field_score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(total_score / total_weight, 3)


def read_highlight_visuals(video_path, highlight_events, fps=25, highlight_clips=None):
    """
    Pour chaque highlight (tir, action dangereuse, but), demande à Gemini
    de décrire visuellement le joueur principal.

    Enrichit chaque event avec :
      event["scorer_visual"]  — description visuelle structurée
      event["player_jersey"]  — numéro si lisible (ne remplace pas si déjà connu)

    Retourne un pool de référence visuelle :
      { jersey_num: [visual_dict, ...] }  — tous les visuels par numéro connu
    """
    if not GEMINI_AVAILABLE or not highlight_events:
        return {}

    visual_pool = {}   # { jersey_num: [visual_dict, ...] }

    try:
        client = get_client()
    except Exception:
        return {}

    for ev in highlight_events:
        ev_time  = float(ev.get("time") or ev.get("time_start") or 0)
        frame_ev = int(ev.get("frame", ev_time * fps))
        ev_type  = (ev.get("main_type") or ev.get("type") or "action").lower()
        pid      = str(ev.get("player", ""))
        t_mm     = f"{int(ev_time//60):02d}:{int(ev_time%60):02d}"

        # Skip si déjà une description visuelle
        if ev.get("scorer_visual"):
            # Alimenter quand même le pool si jersey connu
            _j = ev.get("player_jersey") or ev.get("player_jersey")
            if _j and ev.get("scorer_visual"):
                visual_pool.setdefault(int(_j), []).append(ev["scorer_visual"])
            continue

        try:
            frames = []
            clip_path = None

            if highlight_clips:
                best_match = min(
                    highlight_clips.items(),
                    key=lambda x: abs(float(x[0]) - ev_time),
                    default=(None, None)
                )
                if best_match[1] and abs(float(best_match[0]) - ev_time) < 60:
                    clip_path = best_match[1]

            if clip_path and os.path.exists(clip_path):
                clip_offset = 12.0 if ev_type == "shot" else 25.0
                cap = cv2.VideoCapture(clip_path)
                clip_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                for offset_s in [-3, -2, -1, 0]:
                    fid = max(0, int((clip_offset + offset_s) * clip_fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, fr = cap.read()
                    if ret:
                        frames.append(fr)
                cap.release()
                source_label = "clip HD"
            else:
                cap = cv2.VideoCapture(video_path)
                for offset_s in [-3, -2, -1, 0]:
                    fid = max(0, frame_ev + int(offset_s * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, fr = cap.read()
                    if ret:
                        frames.append(fr)
                cap.release()
                source_label = "source"

            if not frames:
                continue

            parts = [text_to_part(
                f"Action de type '{ev_type}' à {t_mm}. "
                f"Voici {len(frames)} frames prises dans les 3 secondes AVANT l'action.\n"
                f"Réponds UNIQUEMENT en JSON sans markdown :\n"
                f'{{\n'
                f'  "jersey": <numero entier ou null>,\n'
                f'  "visual": {{\n'
                f'    "hair": "<coupe et couleur, ex: cheveux noirs rasés, chauve>",\n'
                f'    "boots": "<couleur et marque si visible, ex: chaussures rouges Nike>",\n'
                f'    "socks": "<position et couleur, ex: chaussettes blanches basses>",\n'
                f'    "body": "<morphologie, ex: grand svelte, petit trapu>",\n'
                f'    "sleeves": "<manches longues ou courtes>",\n'
                f'    "skin": "<teinte peau, ex: peau claire, peau foncée>",\n'
                f'    "role": "<rôle apparent, ex: ailier gauche, attaquant axial>",\n'
                f'    "accessories": "<bandeaux, gants, genouillères ou null>",\n'
                f'    "confidence": "<high | medium | low>"\n'
                f'  }}\n'
                f'}}\n'
                f"- jersey : numéro de maillot du joueur principal de l'action (null si illisible)\n"
                f"- visual : décris le joueur principal de l'action\n"
                f"- ne devine pas un numéro au hasard"
            )]
            for fr in frames:
                if source_label == "clip HD":
                    parts.append(frame_to_part(fr))
                else:
                    parts.append(frame_to_part(cv2.resize(fr, (960, 540))))

            response = _call_gemini(client, parts)
            if response is None:
                continue

            text = extract_response_text(response)
            if not text:
                continue

            text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)

            jersey = data.get("jersey")
            visual = data.get("visual") or {}

            if visual and isinstance(visual, dict):
                ev["scorer_visual"] = {
                    "hair":        visual.get("hair"),
                    "boots":       visual.get("boots"),
                    "socks":       visual.get("socks"),
                    "body":        visual.get("body"),
                    "sleeves":     visual.get("sleeves"),
                    "skin":        visual.get("skin"),
                    "role":        visual.get("role"),
                    "accessories": visual.get("accessories"),
                    "confidence":  visual.get("confidence", "low"),
                    "_source":     source_label,
                    "_event_time": t_mm,
                    "_event_type": ev_type,
                }

            if jersey is not None and 1 <= int(jersey) <= 99:
                jersey_num = int(jersey)
                if not ev.get("player_jersey"):
                    ev["player_jersey"] = jersey_num
                # Alimenter le pool de référence visuelle
                if visual:
                    visual_pool.setdefault(jersey_num, []).append(ev["scorer_visual"])
                print(f"  [HIGHLIGHT VIS] {t_mm} ({ev_type}) → #{jersey_num}")
            else:
                print(f"  [HIGHLIGHT VIS] {t_mm} ({ev_type}) → numéro illisible — visual stocké")

        except Exception as e:
            print(f"  [HIGHLIGHT VIS] erreur {t_mm} : {e}")
            continue

    print(f"  [HIGHLIGHT VIS] Pool local : {len(visual_pool)} joueurs référencés "
          f"({sum(len(v) for v in visual_pool.values())} descriptions)")
    return visual_pool


def read_goal_scorers(video_path, goal_events, fps=25, highlight_clips=None, visual_pool=None):
    """
    Pour chaque but confirmé, demande à Gemini le numéro du buteur et du passeur.
    
    Si highlight_clips est fourni (dict {event_time: clip_path}),
    utilise les clips Full HD (1920×1080) pour une meilleure lisibilité des numéros.
    Sinon fallback sur la source vidéo.
    
    highlight_clips : dict {goal_time_float: clip_path_str}
    """
    if not GEMINI_AVAILABLE or not goal_events:
        return {}

    jersey_map = {}

    try:
        client = get_client()
    except Exception:
        return {}

    for ev in goal_events:
        goal_t  = float(ev.get("time", 0))
        frame_g = int(ev.get("frame", goal_t * fps))
        pid     = str(ev.get("player", ""))

        # Déjà identifié
        if pid and pid in jersey_map:
            continue

        try:
            frames = []
            t_mm = f"{int(goal_t//60):02d}:{int(goal_t%60):02d}"

            # Priorité : clip highlight Full HD si disponible
            clip_path = None
            if highlight_clips:
                # Trouver le clip le plus proche temporellement
                best_match = min(
                    highlight_clips.items(),
                    key=lambda x: abs(float(x[0]) - goal_t),
                    default=(None, None)
                )
                if best_match[1] and abs(float(best_match[0]) - goal_t) < 60:
                    clip_path = best_match[1]

            if clip_path and os.path.exists(clip_path):
                clip_offset = 25.0
                cap = cv2.VideoCapture(clip_path)
                clip_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                # Offsets recentrés autour de l'impact du tir
                # Évite les frames trop en amont où un autre joueur tient le ballon
                for offset_s in [-2.0, -1.5, -1.0, -0.5, -0.2, 0, 0.2]:
                    fid = max(0, int((clip_offset + offset_s) * clip_fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, fr = cap.read()
                    if ret:
                        frames.append(fr)
                cap.release()
                source_label = "clip HD"
            else:
                cap = cv2.VideoCapture(video_path)
                for offset_s in [-2.0, -1.5, -1.0, -0.5, -0.2, 0, 0.2]:
                    fid = max(0, frame_g + int(offset_s * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, fr = cap.read()
                    if ret:
                        frames.append(fr)
                cap.release()
                source_label = "source"

            if not frames:
                continue

            parts = [text_to_part(
                f"But marqué à {t_mm}. Voici {len(frames)} frames centrées autour du moment du tir "
                f"(depuis le {source_label}, pleine résolution).\n"
                f"PROCESSUS D'IDENTIFICATION EN 3 ÉTAPES — dans cet ordre :\n"
                f"\n"
                f"ÉTAPE 1 — IDENTIFIER LE TIREUR PAR SON GESTE\n"
                f"  Cherche le joueur dont la jambe est en EXTENSION ACTIVE vers le ballon.\n"
                f"  Posture de frappe : jambe tendue, corps orienté vers le but, pied au contact du ballon.\n"
                f"  Ignore les joueurs qui courent vers le tireur, sautent, ou lèvent les bras.\n"
                f"\n"
                f"ÉTAPE 2 — IDENTIFIER LA COULEUR DE SON MAILLOT\n"
                f"  Note la couleur principale du maillot du tireur identifié à l'étape 1.\n"
                f"  IMPORTANT : la couleur peut différer de l'équipe majoritaire dans le plan.\n"
                f"  (ex: un attaquant en bleu entouré de défenseurs rouges — le buteur est le bleu).\n"
                f"  Des adversaires plus proches de la caméra ne sont PAS le buteur.\n"
                f"\n"
                f"ÉTAPE 3 — LIRE LE NUMÉRO SUR CE JOUEUR UNIQUEMENT\n"
                f"  Lis le numéro sur le dos ou la poitrine du maillot du tireur identifié.\n"
                f"  Ne lis pas les numéros des autres joueurs.\n"
                f"  Le gardien porte un maillot de couleur différente (gardien ≠ buteur).\n"
                f"\n"
                f"Réponds UNIQUEMENT en JSON sans markdown, format exact :\n"
                f'{{\n'
                f'  "buteur": <numero entier ou null>,\n'
                f'  "buteur_couleur": "<couleur principale du maillot du tireur, ex: bleu, rouge, blanc, noir>",\n'
                f'  "buteur_confiance": "<high | medium | low — confiance sur l\'identification>",\n'
                f'  "passeur": <numero entier ou null>,\n'
                f'  "visual": {{\n'
                f'    "hair": "<coupe et couleur, ex: cheveux noirs rasés, chauve>",\n'
                f'    "boots": "<couleur, ex: chaussures rouges Nike, chaussures noires>",\n'
                f'    "socks": "<position et couleur, ex: chaussettes blanches basses>",\n'
                f'    "body": "<morphologie, ex: grand svelte, petit trapu>",\n'
                f'    "sleeves": "<manches longues ou courtes>",\n'
                f'    "skin": "<teinte peau, ex: peau claire, peau foncée>",\n'
                f'    "role": "<rôle apparent, ex: ailier gauche, attaquant axial>",\n'
                f'    "accessories": "<bandeaux, gants, genouillères ou null>",\n'
                f'    "confidence": "<high | medium | low>"\n'
                f'  }}\n'
                f'}}\n'
                f"Règles finales :\n"
                f"- buteur null si numéro illisible — ne devine pas un numéro au hasard\n"
                f"- buteur_couleur toujours renseigné si un tireur est identifiable\n"
                f"- passeur : numéro du joueur qui fait la dernière passe décisive (null si inconnu)\n"
                f"- pour visual : décris le buteur identifié\n"
                f"- si aucun joueur identifiable, mets null pour tous les champs"
            )]
            for fr in frames:
                # Clip HD : envoyer à taille native (1920×1080 pour lire les numéros)
                # Source : réduire mais garder lisible (960×540 au lieu de 640×360)
                if source_label == "clip HD":
                    parts.append(frame_to_part(fr))  # pleine résolution
                else:
                    parts.append(frame_to_part(cv2.resize(fr, (960, 540))))

            response = _call_gemini(client, parts)
            if response is None:
                continue

            text = extract_response_text(response)
            if not text:
                continue

            text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)

            buteur          = data.get("buteur")
            passeur         = data.get("passeur")
            buteur_couleur  = (data.get("buteur_couleur") or "").strip().lower()
            buteur_confiance = data.get("buteur_confiance", "low")
            visual          = data.get("visual") or {}

            # Stocker la signature visuelle + couleur équipe
            if visual and isinstance(visual, dict):
                ev["scorer_visual"] = {
                    "hair":        visual.get("hair"),
                    "boots":       visual.get("boots"),
                    "socks":       visual.get("socks"),
                    "body":        visual.get("body"),
                    "sleeves":     visual.get("sleeves"),
                    "skin":        visual.get("skin"),
                    "role":        visual.get("role"),
                    "accessories": visual.get("accessories"),
                    "confidence":  visual.get("confidence", "low"),
                    "_source":     source_label,
                    "_goal_time":  t_mm,
                }
                _vis_conf = visual.get("confidence", "?")
                _vis_log  = (
                    f"hair={visual.get('hair','?')} | "
                    f"boots={visual.get('boots','?')} | "
                    f"socks={visual.get('socks','?')} | "
                    f"role={visual.get('role','?')} | "
                    f"conf={_vis_conf}"
                )
                print(f"  [VISUAL] {t_mm} → {_vis_log}")

            # Stocker la couleur maillot du buteur — utilisée pour validation croisée
            if buteur_couleur:
                ev["scorer_team_color"]      = buteur_couleur
                ev["scorer_id_confidence"]   = buteur_confiance
                print(f"  [COULEUR BUTEUR] {t_mm} → maillot={buteur_couleur} confiance_id={buteur_confiance}")

            # ── VALIDATION CROISÉE COULEUR ─────────────────────────────────────
            # Compare la couleur Gemini avec la couleur de l'équipe du tir.
            # Si l'event a un team associé et une couleur connue → validation possible.
            # Si discordance forte → numéro peu fiable → forcer fallback visuel.
            _known_colors = ev.get("_team_colors", {})   # dict {team_id: "bleu"|"rouge"|...}
            _color_valid  = True
            if buteur_couleur and _known_colors:
                # Récupérer la couleur de l'équipe qui a tiré (pas toutes les équipes)
                _shooter_team = ev.get("team")
                if _shooter_team is not None and _shooter_team in _known_colors:
                    _expected_color = _known_colors[_shooter_team]
                    # Correspondance souple : "bleu foncé" contient "bleu"
                    _color_match = (
                        buteur_couleur in _expected_color
                        or _expected_color in buteur_couleur
                    )
                    if not _color_match:
                        print(f"  [COULEUR CROSS] {t_mm} → Gemini='{buteur_couleur}' "
                              f"≠ équipe tireur='{_expected_color}' (team={_shooter_team}) "
                              f"→ numéro invalidé, fallback visuel")
                        _color_valid = False
                        buteur = None   # invalider → fallback visuel prend le relais
                    else:
                        print(f"  [COULEUR CROSS] {t_mm} → '{buteur_couleur}' "
                              f"✓ correspond équipe {_shooter_team}='{_expected_color}'")

            if buteur is not None and _color_valid and 1 <= int(buteur) <= 99:
                key = pid if pid else f"goal_{t_mm}"
                jersey_map[key] = int(buteur)
                ev["player_jersey"] = int(buteur)
                print(f"  [JERSEY GOAL] {t_mm} → buteur=#{buteur} ({buteur_couleur})"
                      f"{f' passeur=#{passeur}' if passeur else ''}")
            else:
                print(f"  [JERSEY GOAL] {t_mm} → numéro illisible ({buteur_couleur}) — description visuelle stockée")

            if passeur is not None and 1 <= int(passeur) <= 99:
                ev["assist_jersey"] = int(passeur)

        except Exception as e:
            print(f"  [JERSEY GOAL] erreur {t_mm} : {e}")
            continue

    # ── FALLBACK VISUEL LOCAL AU MATCH ──────────────────────────────────────
    # Pool de référence : buts avec jersey confirmé + tous les highlights
    _ref_pool = []  # [(jersey_num, scorer_visual_dict)]

    # 1. Depuis les buts eux-mêmes (jersey confirmé par OCR dans ce run)
    for ev in goal_events:
        _j = ev.get("player_jersey")
        _v = ev.get("scorer_visual")
        if _j and _v and ev.get("player_jersey_source") != "visual_match":
            _ref_pool.append((_j, _v))

    # 2. Depuis le pool externe des highlights (tirs + actions dangereuses)
    if visual_pool:
        for jersey_num, visuals in visual_pool.items():
            for _v in visuals:
                _ref_pool.append((jersey_num, _v))

    if _ref_pool:
        print(f"  [VISUAL MATCH] Pool disponible : {len(_ref_pool)} références "
              f"({len(set(j for j,_ in _ref_pool))} joueurs distincts)")
        for ev in goal_events:
            _ocr_jersey  = ev.get("player_jersey")
            _ocr_source  = ev.get("player_jersey_source", "ocr")
            _ocr_conf    = ev.get("scorer_id_confidence", "low")
            if _ocr_source == "visual_match":
                continue  # déjà traité

            _v = ev.get("scorer_visual")
            if not _v:
                continue

            best_jersey, best_score = None, 0
            for ref_jersey, ref_visual in _ref_pool:
                sc = _visual_match_score(_v, ref_visual)
                if sc > best_score:
                    best_score = sc
                    best_jersey = ref_jersey

            _t_mm = _v.get("_goal_time", "??:??")
            if best_jersey and best_score >= 0.5:
                # Seuil selon confiance OCR — low/medium peuvent être écrasés
                _threshold = 0.50 if not _ocr_jersey else (
                    0.60 if _ocr_conf == "low" else
                    0.75 if _ocr_conf == "medium" else
                    0.85  # high — seuil strict
                )
                if best_score >= _threshold:
                    _prev = f"#{_ocr_jersey}" if _ocr_jersey else "inconnu"
                    ev["player_jersey"]        = best_jersey
                    ev["player_jersey_source"] = "visual_match"
                    ev["player_jersey_conf"]   = round(best_score, 2)
                    pid = str(ev.get("player", ""))
                    key = pid if pid else f"goal_{_t_mm}"
                    jersey_map[key] = best_jersey
                    print(f"  [VISUAL MATCH] {_t_mm} → #{best_jersey} "
                          f"(score={best_score:.2f}, remplace {_prev} conf={_ocr_conf})")
                else:
                    print(f"  [VISUAL MATCH] {_t_mm} → #{best_jersey} score={best_score:.2f} "
                          f"< seuil {_threshold:.2f} (OCR=#{_ocr_jersey} conf={_ocr_conf}) → OCR conservé")
            else:
                print(f"  [VISUAL MATCH] {_t_mm} → aucune correspondance "
                      f"(meilleur={best_score:.2f}) → buteur=inconnu")
    else:
        print("  [VISUAL MATCH] Pool vide — pas de fallback possible")

    return jersey_map

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
            gemini_goal_votes = result.get("_goal_votes", 0)

            _desc_lower = result.get("description", "").lower()
            _has_celebration = any(w in _desc_lower for w in
                ["célébr", "celebr", "joie", "jump", "saute", "fête", "fete",
                 "bras lev", "arms", "franchit la ligne", "dans le filet"])

            if gemini_goal_votes >= 2:
                print(f"    → GARDÉ (Gemini a vu {gemini_goal_votes} goal)")

            elif gemini_goal_votes >= 1 and _has_celebration and confiance >= 0.85:
                print(f"    → GARDÉ (1 vote goal conf={confiance:.2f} >= 0.85 + célébration)")

            elif high_conf_phys and tracker_conf_val > 0.95 and gemini_goal_votes >= 1:
                print("    → GARDÉ (signal physique très fort + 1 vote Gemini)")

            else:
                # Gemini n a pas confirmé → supprimé
                event["_remove"] = True
                removed += 1
                print("    → SUPPRIMÉ (Gemini n a pas confirmé)")

        event["gemini_validated"] = True
        event["gemini_type"]      = gemini_type
        event["gemini_conf"]      = confiance
        # Stocker la description complète pour le commentary
        _full_desc = result.get("description", "")
        if _full_desc:
            event["description"] = _full_desc

    events = [e for e in events if not e.get("_remove", False)]
    print(f"  Gemini : {validated} validés | {corrected} corrigés | {removed} supprimés")

    # ── Marquer les tirs confirmés par Gemini comme occasions dangereuses ──
    # Pendant la validation, certains offsets retournent "shot conf >= 0.90"
    # → marquer les events tirs proches comme gemini_shot_confirmed=True
    # → utilisé par _shot_qualifies pour les inclure dans les highlights
    _shot_timestamps_confirmed = set()
    for e in events:
        for _ts in e.get("_shot_offsets_seen", []):
            _shot_timestamps_confirmed.add(_ts)

    if _shot_timestamps_confirmed:
        for e in events:
            if e.get("type") == "shot":
                t_s = e.get("time", 0)
                if any(abs(t_s - _ts) <= 8.0 for _ts in _shot_timestamps_confirmed):
                    e["gemini_shot_confirmed"] = True

    # Libérer les containers PyAV en fin de validation
    close_all_av_containers()

    return events