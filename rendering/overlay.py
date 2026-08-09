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

    # Garde-fous ajoutés suite au bug diagnostiqué sur Andrimont :
    # la calibration se faisait sur les 60 premières frames ANALYSÉES du
    # clip fourni, sans vérifier leur contenu — sur un clip découpé depuis
    # le tout début de la vidéo (terrain vide / échauffement dispersé), ces
    # frames ne contiennent pas assez de vrais joueurs en configuration de
    # match, produisant des centroïdes KMeans sur du bruit (ex: deux teintes
    # bleu-vert quasi identiques, écart ~14, alors que les vraies couleurs
    # étaient jaune/rouge).
    MIN_PLAYERS_PER_FRAME_FOR_CALIBRATION = 10   # frame comptée seulement si assez de joueurs présents
    MIN_CENTROID_DISTANCE = 40.0                  # distance BGR mini entre les 2 couleurs trouvées
    UNCLASSIFIED_DISTANCE_FACTOR = 2.5            # cf. get_team() : au-delà de ce multiple de la
                                                    # dispersion intra-équipe mesurée, une couleur
                                                    # n'est classée dans AUCUNE équipe (probable
                                                    # arbitre/autre) plutôt que forcée dans la plus
                                                    # proche des deux. Valeur de départ raisonnable,
                                                    # PAS ENCORE validée visuellement sur des cas réels
                                                    # — à ajuster si besoin après vérification.

    def __init__(self, teams_data=None, sample_frames=60, **kwargs):
        self.teams_data    = teams_data or {}
        self.sample_frames = sample_frames
        self.score         = {0: 0, 1: 0}
        self._goals_seen   = set()
        self._jersey_colors = []   # couleurs de torses collectées (maillots)
        self._n_frames     = 0     # frames "valides" vues pour calibration
        self._calibrated   = False
        self._centroids    = None
        self._calibration_attempts = 0   # nb de tentatives de calibration rejetées (diagnostic)
        self._total_frames_seen = 0      # compteur total (soupape de sécurité anti-blocage)
        self._max_classification_dist = None  # cf. get_team() : seuil au-delà duquel une couleur
                                                # n'est classée dans aucune équipe

    def add_frame(self, frame):
        """Obsolète — la calibration se fait via update() sur les torses joueurs."""
        pass

    def _extract_jersey_color(self, frame, bbox):
        """Extrait la couleur dominante du torse d'un joueur (zone maillot).

        FIX important : x1/x2/y1/y2 doivent être bornés à [0, largeur/hauteur]
        AVANT le slicing — sinon une valeur négative (trace fantôme hors
        cadre) déclenche le comportement d'indexation négative de NumPy
        (compte depuis la fin du tableau), produisant un crop qui couvre
        presque toute l'image au lieu d'un tout petit bout de rien.
        Diagnostiqué sur Andrimont : bbox=(-83,483,-56,558) → crop de
        1865px de large au lieu des 27px attendus."""
        try:
            import numpy as np
            h_f, w_f = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w_f))
            x2 = max(0, min(x2, w_f))
            y1 = max(0, min(y1, h_f))
            y2 = max(0, min(y2, h_f))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            ch = crop.shape[0]
            # Zone torse : 15%-45% de la hauteur du joueur
            torse = crop[int(ch * 0.15):int(ch * 0.45), :]
            if torse.size == 0:
                return None
            # Filtre taille minimale : un crop trop petit donne une couleur
            # non fiable (quelques pixels de bord/pelouse), à écarter plutôt
            # qu'à faire semblant qu'elle est exploitable.
            if torse.shape[0] < 6 or torse.shape[1] < 10:
                return None
            hsv  = cv2.cvtColor(torse, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1] > 60   # pixels saturés = couleur maillot
            if mask.sum() >= 10:
                color = torse[mask].mean(axis=0).astype(np.float32)
            else:
                color = torse.mean(axis=(0, 1)).astype(np.float32)
            return color
        except Exception:
            return None

    def _calibrate(self):
        """KMeans sur les couleurs de torses collectées — 2 centroides = 2 équipes.
        Rejette la calibration si les deux couleurs trouvées sont trop
        proches (signe que l'échantillon ne contenait pas 2 vraies équipes
        distinctes, ex: terrain vide/quasi-vide) — continue alors à
        collecter plutôt que de verrouiller un résultat probablement faux."""
        try:
            import numpy as np
            if len(self._jersey_colors) < 20:
                return
            samples  = np.array(self._jersey_colors, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centroids = cv2.kmeans(
                samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
            )

            centroid_distance = float(np.linalg.norm(centroids[0] - centroids[1]))
            if centroid_distance < self.MIN_CENTROID_DISTANCE:
                self._calibration_attempts += 1
                print(f"  [TeamColorDetector] calibration rejetée "
                      f"(couleurs trop proches, distance={centroid_distance:.1f} "
                      f"< {self.MIN_CENTROID_DISTANCE}) — poursuite de la collecte "
                      f"(tentative {self._calibration_attempts})")
                # On ne verrouille pas : on continue à accumuler des échantillons.
                # Repousse le prochain essai de sample_frames supplémentaires.
                self._n_frames = 0
                # Garde-fou anti-boucle infinie : au-delà de 5 tentatives,
                # on accepte quand même le résultat pour ne pas bloquer le pipeline.
                if self._calibration_attempts < 5:
                    return

            self._centroids  = centroids
            self._calibrated = True

            # Dispersion intra-équipe mesurée (pas devinée) : à quel point les
            # couleurs d'une même équipe varient naturellement (éclairage,
            # angle...). Sert de référence pour get_team() : une couleur trop
            # loin des DEUX centroïdes (au-delà de UNCLASSIFIED_DISTANCE_FACTOR
            # fois cette dispersion) ne sera classée dans aucune équipe,
            # plutôt que forcée dans la plus proche des deux — nécessaire pour
            # qu'un arbitre à la couleur nettement différente soit repérable
            # comme tel (cf. diagnostic Andrimont : get_team() forçait TOUJOURS
            # 0 ou 1, rendant la détection de l'arbitre structurellement
            # impossible dans detect_c_capitaines.py).
            labels_flat = labels.flatten()
            dist_intra = []
            for k in (0, 1):
                pts = samples[labels_flat == k]
                if len(pts) > 0:
                    dist_intra.extend(np.linalg.norm(pts - centroids[k], axis=1).tolist())
            self._max_classification_dist = (
                float(np.mean(dist_intra)) * self.UNCLASSIFIED_DISTANCE_FACTOR
                if dist_intra else None
            )

            c0 = tuple(int(x) for x in centroids[0])
            c1 = tuple(int(x) for x in centroids[1])
            print(f"  [TeamColorDetector] calibré sur maillots : team0={c0} team1={c1} "
                  f"(distance={centroid_distance:.1f}, {len(self._jersey_colors)} échantillons, "
                  f"seuil_non_classe={self._max_classification_dist:.1f})")
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
        """Retourne 0 ou 1 selon la couleur BGR la plus proche — ou None si
        la couleur est trop éloignée des DEUX équipes (probable arbitre ou
        autre), plutôt que de forcer systématiquement un choix binaire.

        FIX important : l'ancienne version retournait toujours 0 ou 1,
        jamais None pour une couleur valide — rendant structurellement
        impossible la détection d'un arbitre à couleur distincte (diagnostic
        Andrimont : le champ 'hors équipe' ne contenait alors que des
        échecs d'extraction, jamais de vraies couleurs différentes)."""
        if not self._calibrated or self._centroids is None:
            return None
        try:
            import numpy as np
            c = np.array(color, dtype=np.float32)
            d0 = np.linalg.norm(c - self._centroids[0])
            d1 = np.linalg.norm(c - self._centroids[1])
            min_d = min(d0, d1)
            if self._max_classification_dist is not None and min_d > self._max_classification_dist:
                return None  # couleur trop loin des 2 équipes -> non classée
            return 0 if d0 < d1 else 1
        except Exception:
            return None

    def update(self, frame, tracked):
        """
        Appelé par main.py à chaque frame.
        Phase 1 (frames 0→sample_frames) : collecte les couleurs de torses
        pour calibrer les 2 centroides équipes.
        Phase 2 (après calibration) : assigne team=0 ou team=1 à chaque joueur.
        """
        if frame is None:
            return

        # ── Phase 1 : collecte pour calibration ──────────────────────
        if not self._calibrated:
            self._total_frames_seen = getattr(self, "_total_frames_seen", 0) + 1
            # Ne garder que les joueurs RÉELLEMENT détectés cette frame
            # (pas une position purement prédite par Kalman après perte de
            # la détection — diagnostiqué sur Andrimont : bbox "confirmées"
            # mais dérivées de plusieurs dizaines de pixels à côté du vrai
            # joueur pendant le rassemblement, faussant la couleur captée).
            joueurs_frais = [p for p in (tracked or []) if p.get("time_since_update", 0) == 0]
            n_joueurs_frame = len(joueurs_frais)
            # Ne compte cette frame pour la calibration que si assez de
            # joueurs sont présents (signe qu'on est bien en configuration
            # de match, pas sur un terrain vide/quasi-vide en début de clip).
            if n_joueurs_frame < self.MIN_PLAYERS_PER_FRAME_FOR_CALIBRATION:
                # Soupape de sécurité : si on n'atteint jamais assez de
                # joueurs simultanés (match avec occlusions fréquentes,
                # cadrage serré...), ne pas bloquer indéfiniment la
                # calibration — on assouplit après un long moment plutôt
                # que de ne jamais assigner aucune équipe.
                if self._total_frames_seen >= 10 * self.sample_frames:
                    for p in joueurs_frais:
                        bbox = p.get("bbox") or p.get("box")
                        if not bbox:
                            continue
                        color = self._extract_jersey_color(frame, bbox)
                        if color is not None:
                            self._jersey_colors.append(color)
                    self._n_frames += 1
                    if self._n_frames >= self.sample_frames:
                        self._calibrate()
                return
            self._n_frames += 1
            for p in joueurs_frais:
                bbox = p.get("bbox") or p.get("box")
                if not bbox:
                    continue
                color = self._extract_jersey_color(frame, bbox)
                if color is not None:
                    self._jersey_colors.append(color)

            if self._n_frames >= self.sample_frames:
                self._calibrate()
            return

        # ── Phase 2 : assignation équipe ─────────────────────────────
        # Seuls les joueurs RÉELLEMENT détectés cette frame reçoivent une
        # couleur/équipe. Une position purement prédite (Kalman) sans
        # confirmation reste team=None plutôt que de deviner sur une bbox
        # qui a pu dériver loin du vrai joueur.
        try:
            for p in (tracked or []):
                if p.get("team") is not None:
                    continue
                if p.get("time_since_update", 0) != 0:
                    continue
                bbox = p.get("bbox") or p.get("box")
                if not bbox:
                    continue
                color = self._extract_jersey_color(frame, bbox)
                if color is not None:
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