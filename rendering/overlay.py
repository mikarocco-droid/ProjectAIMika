# rendering/overlay.py
# -*- coding: utf-8 -*-
#
# Overlay vidéo professionnel pour les highlights.
#
# Fonctionnalités :
#   - Scoreboard permanent en haut de chaque clip
#   - Animation "⚽ BUT !" pour les clips de buts
#   - Timestamp de l'action
#   - Noms d'équipes + score en temps réel
#
# Usage :
#   from rendering.overlay import render_highlights_with_overlay
#   render_highlights_with_overlay(highlights, events, teams, output_dir)

import os
import cv2
import numpy as np
import subprocess
from datetime import timedelta


# ─────────────────────────────────────────
# HELPERS COULEUR
# ─────────────────────────────────────────
def bgr_to_name(bgr):
    """Nom couleur depuis BGR."""
    if not bgr:
        return "?"
    try:
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        pixel   = np.uint8([[[b, g, r]]])
        hsv     = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h = int(hsv[0]) * 2
        s = int(hsv[1] / 255 * 100)
        v = int(hsv[2] / 255 * 100)
    except Exception:
        return "?"
    if s < 15: return "blanc" if v > 70 else ("gris" if v > 30 else "noir")
    if v < 20: return "noir"
    if 0 <= h < 20 or 340 <= h <= 360: return "bordeaux" if v < 50 else "rouge"
    if 20 <= h < 35:   return "orange"
    if 35 <= h < 75:   return "jaune"
    if 75 <= h < 155:  return "vert"
    if 155 <= h < 185: return "cyan"
    if 185 <= h < 265: return "bleu marine" if v < 45 else "bleu"
    if 265 <= h < 295: return "violet"
    if 295 <= h < 340: return "rose"
    return "?"


def hex_to_bgr(hex_color):
    """#RRGGBB → (B, G, R)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def team_color_for_display(teams_data, team_id):
    """
    Retourne la couleur BGR d'une équipe pour l'affichage.
    Fallback sur des couleurs par défaut si non disponible.
    """
    defaults = {0: (80, 120, 220), 1: (60, 60, 200)}
    if not teams_data:
        return defaults.get(team_id, (150, 150, 150))
    td = teams_data.get(team_id) or teams_data.get(str(team_id)) or {}
    bgr = td.get("color_bgr")
    if bgr:
        return tuple(int(x) for x in bgr)
    return defaults.get(team_id, (150, 150, 150))


def team_name_display(teams_data, team_id):
    """Retourne le nom d'équipe pour l'affichage."""
    if not teams_data:
        return f"Équipe {team_id + 1}"
    td = teams_data.get(team_id) or teams_data.get(str(team_id)) or {}
    name = td.get("name")
    if name:
        return str(name)
    color = bgr_to_name(td.get("color_bgr"))
    return f"Équipe {color.title()}" if color and color != "?" else f"Équipe {team_id + 1}"


def fmt_time(seconds):
    """Formate un temps en MM:SS."""
    seconds = max(0, float(seconds or 0))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ─────────────────────────────────────────
# DESSIN SCOREBOARD
# ─────────────────────────────────────────
def draw_scoreboard(frame, team_home_name, team_away_name,
                    score_home, score_away,
                    action_time, is_goal=False,
                    team_home_bgr=None, team_away_bgr=None):
    """
    Dessine un scoreboard professionnel en haut de la frame.

    Layout :
    ┌──────────────────────────────────────────────────────────────┐
    │ [couleur] RSC STAVELOT B     1 - 0     FC LIÈGE [couleur]  │
    │                                                    02:22    │
    └──────────────────────────────────────────────────────────────┘
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Fond scoreboard ──────────────────────────────────────────
    bar_h   = 52
    padding = 8

    # Fond semi-transparent noir
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 20), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # ── Ligne de séparation colorée ───────────────────────────────
    cv2.rectangle(frame, (0, bar_h), (w, bar_h + 3),
                  (0, 180, 220), -1)   # cyan accent

    # ── Couleurs équipes (bandes latérales) ───────────────────────
    band_w = 6
    c_home = team_home_bgr or (80, 120, 220)
    c_away = team_away_bgr or (60, 60, 200)
    cv2.rectangle(frame, (0, 0), (band_w, bar_h), c_home, -1)
    cv2.rectangle(frame, (w - band_w, 0), (w, bar_h), c_away, -1)

    # ── Noms équipes ──────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_bold  = cv2.FONT_HERSHEY_DUPLEX
    text_color = (240, 240, 240)
    gray_color = (160, 160, 170)

    # Équipe domicile (gauche)
    home_text = team_home_name.upper()[:20]
    cv2.putText(frame, home_text, (band_w + 12, 22),
                font_bold, 0.55, text_color, 1, cv2.LINE_AA)

    # Équipe visiteur (droite — aligné à droite)
    away_text = team_away_name.upper()[:20]
    (aw, _), _ = cv2.getTextSize(away_text, font_bold, 0.55, 1)
    cv2.putText(frame, away_text, (w - band_w - aw - 12, 22),
                font_bold, 0.55, text_color, 1, cv2.LINE_AA)

    # ── Score central ─────────────────────────────────────────────
    score_str = f"{score_home}  -  {score_away}"
    (sw, _), _ = cv2.getTextSize(score_str, font_bold, 0.85, 2)
    score_x = (w - sw) // 2

    # Fond score
    cv2.rectangle(frame,
                  (score_x - 14, 4),
                  (score_x + sw + 14, bar_h - 4),
                  (30, 30, 40), -1)

    # Score texte
    score_color = (80, 220, 80) if is_goal else (255, 255, 255)
    cv2.putText(frame, score_str, (score_x, 32),
                font_bold, 0.85, score_color, 2, cv2.LINE_AA)

    # ── Timestamp ─────────────────────────────────────────────────
    time_str = fmt_time(action_time)
    (tw, _), _ = cv2.getTextSize(time_str, font, 0.45, 1)
    cv2.putText(frame, time_str, ((w - tw) // 2, bar_h - 8),
                font, 0.45, gray_color, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────
# ANIMATION BUT
# ─────────────────────────────────────────
def draw_goal_animation(frame, scorer_name, team_name,
                        score_home, score_away,
                        progress=1.0):
    """
    Dessine l'animation "⚽ BUT !" en overlay.

    progress : 0.0→1.0 — animation d'entrée (slide + fade)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Calcul position avec animation slide ─────────────────────
    panel_h = 80
    panel_w = min(480, w - 40)
    panel_x = (w - panel_w) // 2

    # Slide depuis le bas avec ease-out
    ease     = 1 - (1 - progress) ** 3
    panel_y  = int(h * 0.65 - panel_h // 2 * ease)
    alpha    = min(1.0, progress * 2)

    if panel_y + panel_h > h or panel_y < 0:
        return frame

    # ── Fond panel ────────────────────────────────────────────────
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (10, 10, 15), -1)

    # Bordure accent
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + 4),
                  (0, 200, 80), -1)   # vert but

    # ── Texte "⚽ BUT !" ──────────────────────────────────────────
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_sm   = cv2.FONT_HERSHEY_SIMPLEX

    goal_text = "BUT !"
    (gw, _), _ = cv2.getTextSize(goal_text, font, 1.1, 2)
    cv2.putText(overlay, goal_text,
                (panel_x + (panel_w - gw) // 2, panel_y + 38),
                font, 1.1, (80, 255, 80), 2, cv2.LINE_AA)

    # ── Buteur ────────────────────────────────────────────────────
    if scorer_name and scorer_name != "?":
        scorer_text = f"{scorer_name}  ({team_name})"
        (sctw, _), _ = cv2.getTextSize(scorer_text, font_sm, 0.55, 1)
        cv2.putText(overlay, scorer_text,
                    (panel_x + (panel_w - sctw) // 2, panel_y + 62),
                    font_sm, 0.55, (200, 200, 210), 1, cv2.LINE_AA)

    # ── Score mis en avant ────────────────────────────────────────
    score_str  = f"{score_home} - {score_away}"
    (ssw, _), _ = cv2.getTextSize(score_str, font, 0.9, 2)
    # Affiché à droite du panel
    cv2.putText(overlay, score_str,
                (panel_x + panel_w - ssw - 16, panel_y + 38),
                font, 0.9, (255, 255, 100), 2, cv2.LINE_AA)

    # ── Fusion avec alpha ─────────────────────────────────────────
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ─────────────────────────────────────────
# CALCUL SCORE EN TEMPS RÉEL
# ─────────────────────────────────────────
def compute_score_at_time(goals, t, teams_data=None):
    """
    Calcule le score domicile/visiteur à un instant t.

    goals : liste d'events de type "goal"
    t     : timestamp en secondes
    """
    score = {0: 0, 1: 0}
    for g in goals:
        if float(g.get("time", 0) or 0) <= t:
            team = g.get("team")
            if team in (0, 1):
                score[team] += 1
            elif str(team) in ("0", "1"):
                score[int(team)] += 1
    return score.get(0, 0), score.get(1, 0)


# ─────────────────────────────────────────
# RENDU D'UN CLIP AVEC OVERLAY
# ─────────────────────────────────────────
def render_clip_with_overlay(input_path, output_path,
                              team_home_name, team_away_name,
                              score_home, score_away,
                              action_time, is_goal=False,
                              scorer_name=None,
                              team_home_bgr=None, team_away_bgr=None,
                              goal_anim_duration=3.0):
    """
    Ajoute le scoreboard + animation but sur un clip vidéo.

    Utilise OpenCV frame par frame puis ffmpeg pour ré-encoder.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = output_path + "_tmp.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    frame_idx         = 0
    goal_anim_frames  = int(goal_anim_duration * fps) if is_goal else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_current = action_time - (frame_idx / fps)

        # ── Scoreboard permanent ──────────────────────────────────
        frame = draw_scoreboard(
            frame,
            team_home_name  = team_home_name,
            team_away_name  = team_away_name,
            score_home      = score_home,
            score_away      = score_away,
            action_time     = action_time,
            is_goal         = is_goal,
            team_home_bgr   = team_home_bgr,
            team_away_bgr   = team_away_bgr,
        )

        # ── Animation but (sur les N premières secondes du clip) ──
        if is_goal and frame_idx < goal_anim_frames:
            progress = frame_idx / max(goal_anim_frames * 0.4, 1)
            progress = min(1.0, progress)
            frame = draw_goal_animation(
                frame,
                scorer_name = scorer_name or "?",
                team_name   = team_home_name if score_home > score_away else team_away_name,
                score_home  = score_home,
                score_away  = score_away,
                progress    = progress,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ── Ré-encoder avec ffmpeg pour compatibilité + audio ─────────
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if result.returncode == 0 and os.path.exists(output_path):
        return output_path

    # Fallback sans audio si ffmpeg échoue
    if os.path.exists(tmp_path):
        os.rename(tmp_path, output_path)
        return output_path
    return None


# ─────────────────────────────────────────
# RENDU COMPLET DE TOUS LES HIGHLIGHTS
# ─────────────────────────────────────────
def render_highlights_with_overlay(highlights, events, teams_data,
                                    output_dir, sport="football"):
    """
    Ajoute scoreboard + animations sur tous les highlights.

    Args:
        highlights  : liste de highlights (depuis pipeline)
        events      : liste d'events (pour calculer le score)
        teams_data  : dict {team_id: {name, color_bgr, ...}}
        output_dir  : dossier de sortie
        sport       : sport (pour adapter l'overlay)

    Returns:
        Liste des highlights enrichis avec "file_overlay"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extraire les buts pour calcul score en temps réel
    goals = [e for e in events if e.get("type") in ("goal", "score")]

    # Noms et couleurs équipes
    home_name  = team_name_display(teams_data, 0)
    away_name  = team_name_display(teams_data, 1)
    home_bgr   = team_color_for_display(teams_data, 0)
    away_bgr   = team_color_for_display(teams_data, 1)

    enriched = []

    for i, h in enumerate(highlights):
        clip_path = h.get("file")
        if not clip_path or not os.path.exists(str(clip_path)):
            enriched.append(h)
            continue

        h_type      = h.get("main_type", "shot")
        action_time = float(h.get("time_start", 0) or 0)
        is_goal     = h_type in ("goal", "score")

        # Score au moment de l'action
        score_h, score_a = compute_score_at_time(goals, action_time)
        # Si c'est un but, on affiche le score APRÈS le but
        if is_goal:
            team_scorer = h.get("team")
            if team_scorer == 0 or str(team_scorer) == "0":
                score_h += 1
            elif team_scorer == 1 or str(team_scorer) == "1":
                score_a += 1

        # Nom du buteur
        scorer_name = None
        if is_goal:
            pid    = h.get("player")
            jersey = str(pid).replace("P", "#") if pid else "?"
            scorer_name = jersey

        # Chemin de sortie
        base     = os.path.splitext(os.path.basename(clip_path))[0]
        out_path = os.path.join(output_dir, f"{base}_overlay.mp4")

        print(f"  [OVERLAY] Clip {i+1}/{len(highlights)} "
              f"{'⚽' if is_goal else '🎯'} "
              f"{score_h}-{score_a} t={fmt_time(action_time)}")

        rendered = render_clip_with_overlay(
            input_path      = clip_path,
            output_path     = out_path,
            team_home_name  = home_name,
            team_away_name  = away_name,
            score_home      = score_h,
            score_away      = score_a,
            action_time     = action_time,
            is_goal         = is_goal,
            scorer_name     = scorer_name,
            team_home_bgr   = home_bgr,
            team_away_bgr   = away_bgr,
        )

        h_new = dict(h)
        h_new["file_overlay"] = rendered or clip_path
        enriched.append(h_new)

    return enriched


# ─────────────────────────────────────────
# REEL FINAL AVEC OVERLAY
# ─────────────────────────────────────────
def create_overlay_reel(highlights_with_overlay, output_path,
                         team_home, team_away, final_score):
    """
    Concatène les clips avec overlay en un reel final.
    Ajoute un écran titre au début et un écran score final.

    Args:
        highlights_with_overlay : liste de highlights avec "file_overlay"
        output_path             : chemin du reel final
        team_home               : nom équipe domicile
        team_away               : nom équipe visiteur
        final_score             : (score_home, score_away)
    """
    clips = [
        h.get("file_overlay") or h.get("file")
        for h in highlights_with_overlay
        if (h.get("file_overlay") or h.get("file"))
        and os.path.exists(h.get("file_overlay") or h.get("file", ""))
    ]

    if not clips:
        return None

    # Créer liste de concat pour ffmpeg
    list_file = output_path + "_list.txt"
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    try:
        os.remove(list_file)
    except Exception:
        pass

    return output_path if result.returncode == 0 else None


# ─────────────────────────────────────────
# COMPATIBILITÉ — Classes attendues par main.py
# ─────────────────────────────────────────
class TeamColorDetector:
    """
    Détecteur de couleurs d'équipes pour l'overlay vidéo.
    Maintient le score en temps réel et les noms d'équipes.
    Accepte tous les kwargs pour compatibilité avec main.py.
    """
    def __init__(self, teams_data=None, sample_frames=60, **kwargs):
        self.teams_data   = teams_data or {}
        self.sample_frames = sample_frames
        self.score        = {0: 0, 1: 0}
        self._goals_seen  = set()
        self._samples     = []      # frames collectées pour calibration
        self._calibrated  = False
        self._centroids   = None

    def add_frame(self, frame):
        """Collecte une frame pour la calibration des couleurs équipes."""
        if self._calibrated or frame is None:
            return
        self._samples.append(frame)
        if len(self._samples) >= self.sample_frames:
            self._calibrate()

    def _calibrate(self):
        """KMeans sur les frames collectées pour trouver les 2 couleurs équipes."""
        try:
            import numpy as np
            all_colors = []
            for frame in self._samples:
                h, w = frame.shape[:2]
                # Zone centrale du terrain (évite tribunes)
                crop = frame[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
                # Sample aléatoire de pixels
                pixels = crop.reshape(-1, 3).astype(np.float32)
                idx = np.random.choice(len(pixels), min(500, len(pixels)), replace=False)
                all_colors.extend(pixels[idx])

            if len(all_colors) < 20:
                return

            samples = np.array(all_colors, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centroids = cv2.kmeans(
                samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
            )
            self._centroids  = centroids
            self._calibrated = True
            c0 = tuple(int(x) for x in centroids[0])
            c1 = tuple(int(x) for x in centroids[1])
            print(f"  [TeamColorDetector] calibré : team0={c0} team1={c1}")
        except Exception as e:
            print(f"  [TeamColorDetector] calibration ignorée : {e}")

    def _dominant_color(self, patch):
        """
        Retourne la couleur BGR dominante d'un crop joueur.
        Utilisé par main.py dans assign_teams_by_color().
        """
        if patch is None or patch.size == 0:
            return None
        try:
            h = patch.shape[0]
            torse = patch[int(h * 0.15):int(h * 0.45), :]
            if torse.size == 0:
                return None
            hsv  = cv2.cvtColor(torse, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > 60
            if mask.sum() >= 10:
                color = torse[mask].mean(axis=0)
            else:
                color = torse.mean(axis=(0, 1))
            return tuple(int(x) for x in color)
        except Exception:
            return None

    def get_team(self, color):
        """Retourne 0 ou 1 selon la couleur BGR la plus proche."""
        if not self._calibrated or self._centroids is None:
            return None
        try:
            import numpy as np
            c = np.array(color, dtype=np.float32)
            d0 = np.linalg.norm(c - self._centroids[0])
            d1 = np.linalg.norm(c - self._centroids[1])
            return 0 if d0 < d1 else 1
        except Exception:
            return None

    def update(self, frame, tracked):
        """
        Appelé par main.py à chaque frame.
        Collecte des samples de couleur et assigne les équipes aux joueurs trackés.
        tracked : liste de dicts avec 'bbox' et éventuellement 'team'
        """
        if frame is None:
            return

        # Collecter la frame pour calibration
        self.add_frame(frame)

        if not self._calibrated:
            return

        # Assigner une équipe à chaque joueur tracké
        try:
            import numpy as np
            h_f, w_f = frame.shape[:2]
            for p in (tracked or []):
                if p.get("team") is not None:
                    continue
                bbox = p.get("bbox") or p.get("box")
                if not bbox:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_f, x2); y2 = min(h_f, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ch = crop.shape[0]
                torse = crop[int(ch*0.15):int(ch*0.45), :]
                if torse.size == 0:
                    continue
                # Filtre saturation
                hsv = cv2.cvtColor(torse, cv2.COLOR_BGR2HSV)
                mask = hsv[:, :, 1] > 60
                if mask.sum() >= 10:
                    color = torse[mask].mean(axis=0).astype(np.float32)
                else:
                    color = torse.mean(axis=(0, 1)).astype(np.float32)
                p["team"] = self.get_team(color)
        except Exception:
            pass

    def update_score(self, events, current_time):
        """Met à jour le score depuis les events jusqu'à current_time."""
        for e in events:
            eid = id(e)
            if eid in self._goals_seen:
                continue
            if e.get("type") in ("goal", "score"):
                t = float(e.get("time", 0) or 0)
                if t <= current_time:
                    team = e.get("team")
                    if team in (0, 1):
                        self.score[team] += 1
                    elif str(team) in ("0", "1"):
                        self.score[int(team)] += 1
                    self._goals_seen.add(eid)
        return self.score[0], self.score[1]

    def get_team_name(self, team_id):
        return team_name_display(self.teams_data, team_id)

    def get_team_color(self, team_id):
        return team_color_for_display(self.teams_data, team_id)


class Overlay:
    """
    Classe principale d'overlay vidéo.
    Encapsule scoreboard + animation but pour un clip.
    Compatibilité main.py : Overlay(fps=fps) + overlay.render(frame, players, ball, events, frame_id)
    """
    def __init__(self, teams_data=None, sport="football", fps=25, **kwargs):
        self.teams_data  = teams_data or {}
        self.sport       = sport
        self.fps         = fps
        self.detector    = TeamColorDetector(teams_data)
        self._score      = {0: 0, 1: 0}
        self._goals_seen = set()

    def render(self, frame, players, ball, events, frame_id):
        """
        Appelé par main.py pour annoter chaque frame en temps réel.
        Dessine : bboxes joueurs, scoreboard, animation but si nécessaire.
        """
        if frame is None:
            return frame

        # Mettre à jour le score depuis les events de cette frame
        for e in (events or []):
            eid = id(e)
            if eid not in self._goals_seen and e.get("type") in ("goal", "score"):
                team = e.get("team")
                if team in (0, 1):
                    self._score[team] += 1
                elif str(team) in ("0", "1"):
                    self._score[int(team)] += 1
                self._goals_seen.add(eid)

        is_goal     = any(e.get("type") in ("goal","score") for e in (events or []))
        action_time = float(frame_id) / max(self.fps, 1)

        # Dessiner les bboxes joueurs
        try:
            for p in (players or []):
                bbox = p.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                team  = p.get("team")
                color = self.detector.get_team_color(team) if team is not None else (150, 150, 150)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Numéro maillot si dispo
                jersey = p.get("jersey") or p.get("jersey_number")
                if jersey:
                    cv2.putText(frame, f"#{jersey}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception:
            pass

        # Dessiner la balle
        try:
            if ball and ball.get("center"):
                cx, cy = int(ball["center"][0]), int(ball["center"][1])
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
        except Exception:
            pass

        # Scoreboard en haut
        try:
            frame = draw_scoreboard(
                frame,
                team_home_name = self.detector.get_team_name(0),
                team_away_name = self.detector.get_team_name(1),
                score_home     = self._score.get(0, 0),
                score_away     = self._score.get(1, 0),
                action_time    = action_time,
                is_goal        = is_goal,
                team_home_bgr  = self.detector.get_team_color(0),
                team_away_bgr  = self.detector.get_team_color(1),
            )
        except Exception:
            pass

        return frame

    def draw_frame(self, frame, action_time, score_home=0, score_away=0,
                   is_goal=False, scorer_name=None, goal_progress=1.0):
        """Applique scoreboard + animation but sur une frame."""
        home_name = self.detector.get_team_name(0)
        away_name = self.detector.get_team_name(1)
        home_bgr  = self.detector.get_team_color(0)
        away_bgr  = self.detector.get_team_color(1)

        frame = draw_scoreboard(
            frame,
            team_home_name = home_name,
            team_away_name = away_name,
            score_home     = score_home,
            score_away     = score_away,
            action_time    = action_time,
            is_goal        = is_goal,
            team_home_bgr  = home_bgr,
            team_away_bgr  = away_bgr,
        )

        if is_goal and goal_progress < 1.0:
            frame = draw_goal_animation(
                frame,
                scorer_name = scorer_name or "?",
                team_name   = home_name if score_home >= score_away else away_name,
                score_home  = score_home,
                score_away  = score_away,
                progress    = goal_progress,
            )
        return frame

    def render_clip(self, input_path, output_path, action_time,
                    events=None, is_goal=False, scorer_name=None):
        """Render un clip avec overlay complet."""
        score_h, score_a = 0, 0
        if events:
            score_h, score_a = compute_score_at_time(events, action_time)
            if is_goal:
                # Chercher l'équipe qui a marqué
                for e in events:
                    if (e.get("type") in ("goal","score")
                            and abs(float(e.get("time",0) or 0) - action_time) < 5):
                        team = e.get("team")
                        if team in (0, "0"): score_h += 1
                        elif team in (1, "1"): score_a += 1
                        break

        return render_clip_with_overlay(
            input_path     = input_path,
            output_path    = output_path,
            team_home_name = self.detector.get_team_name(0),
            team_away_name = self.detector.get_team_name(1),
            score_home     = score_h,
            score_away     = score_a,
            action_time    = action_time,
            is_goal        = is_goal,
            scorer_name    = scorer_name,
            team_home_bgr  = self.detector.get_team_color(0),
            team_away_bgr  = self.detector.get_team_color(1),
        )