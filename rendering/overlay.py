# rendering/overlay.py

import cv2
import numpy as np
from collections import defaultdict
import config


# ─────────────────────────────────────────
# COULEURS FALLBACK (si détection échoue)
# ─────────────────────────────────────────
FALLBACK_COLORS = {
    0:    (0,   200, 255),   # équipe A → jaune
    1:    (255,  80,  80),   # équipe B → rouge
    None: (180, 180, 180),   # inconnu  → gris
}

TEXT_COLOR   = (255, 255, 255)
SHADOW_COLOR = (0,   0,   0)


# ─────────────────────────────────────────
# DÉTECTION AUTOMATIQUE COULEURS MAILLOTS
# ─────────────────────────────────────────
class TeamColorDetector:
    """
    Détecte automatiquement la couleur dominante
    du maillot de chaque équipe sur les premières frames.
    """

    def __init__(self, sample_frames=60):
        """
        sample_frames : nombre de frames analysées
                        avant de figer les couleurs
        """
        self.sample_frames  = sample_frames
        self.frame_count    = 0
        self.locked         = False

        # team_id → liste de couleurs BGR moyennes
        self._samples = defaultdict(list)

        # Couleurs finales détectées
        self.team_colors = dict(FALLBACK_COLORS)

    # ─────────────────────────────────────────
    # EXTRACTION COULEUR DOMINANTE D'UN CROP
    # ─────────────────────────────────────────
    def _dominant_color(self, patch):
        """
        Retourne la couleur BGR dominante d'un crop joueur.
        On analyse uniquement le tiers supérieur (zone maillot).
        """
        h, w = patch.shape[:2]
        if h < 20 or w < 10:
            return None

        # Zone maillot : entre 20% et 60% de la hauteur
        top    = int(h * 0.20)
        bottom = int(h * 0.60)
        roi    = patch[top:bottom, :]

        if roi.size == 0:
            return None

        # Redimensionner pour accélérer le calcul
        roi = cv2.resize(roi, (20, 20))

        # Convertir en HSV pour exclure le fond vert (terrain)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Masque anti-fond vert (terrain)
        mask_green = cv2.inRange(hsv,
            np.array([35,  40,  40]),
            np.array([85, 255, 255])
        )
        mask = cv2.bitwise_not(mask_green)

        pixels = roi[mask > 0]

        if len(pixels) < 10:
            return None

        # Couleur moyenne des pixels non-verts
        mean_color = pixels.mean(axis=0).astype(int)
        return tuple(mean_color.tolist())

    # ─────────────────────────────────────────
    # MISE À JOUR À CHAQUE FRAME
    # ─────────────────────────────────────────
    def update(self, frame, players):
        """
        Collecte les échantillons de couleur sur les
        premières frames puis fige les couleurs détectées.
        """
        if self.locked:
            return

        self.frame_count += 1

        for p in players:
            team = p.get("team")
            if team is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            h_f, w_f = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_f, x2); y2 = min(h_f, y2)

            patch = frame[y1:y2, x1:x2]
            color = self._dominant_color(patch)

            if color is not None:
                self._samples[team].append(color)

        # Après sample_frames frames → figer les couleurs
        if self.frame_count >= self.sample_frames:
            self._lock_colors()

    # ─────────────────────────────────────────
    # CALCUL COULEURS FINALES
    # ─────────────────────────────────────────
    def _lock_colors(self):
        """
        Calcule la couleur moyenne par équipe
        et la convertit en BGR pour OpenCV.
        """
        for team, samples in self._samples.items():
            if len(samples) < 5:
                continue

            arr = np.array(samples, dtype=np.float32)

            # Moyenne robuste : exclure les 10% extrêmes
            for ch in range(3):
                col = arr[:, ch]
                low, high = np.percentile(col, [10, 90])
                arr = arr[(arr[:, ch] >= low) & (arr[:, ch] <= high)]

            if len(arr) == 0:
                continue

            mean = arr.mean(axis=0).astype(int)
            self.team_colors[team] = (int(mean[0]), int(mean[1]), int(mean[2]))

        self.locked = True

        print("🎨 Couleurs équipes détectées :")
        for team, color in self.team_colors.items():
            if team is not None:
                print(f"   Équipe {team} → BGR{color}")

    # ─────────────────────────────────────────
    # ACCESSEUR
    # ─────────────────────────────────────────
    def get_color(self, team):
        return self.team_colors.get(team, FALLBACK_COLORS[None])

    def reset(self):
        """Réinitialise entre deux vidéos."""
        self.__init__(self.sample_frames)


# ─────────────────────────────────────────
# UTILITAIRES DESSIN
# ─────────────────────────────────────────
def draw_text_with_shadow(frame, text, pos, font_scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1),
                font, font_scale, SHADOW_COLOR, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                font, font_scale, TEXT_COLOR,   thickness,   cv2.LINE_AA)


def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=6, thickness=2):
    pts = [
        (x1+radius, y1), (x2-radius, y1),
        (x2, y1+radius), (x2, y2-radius),
        (x2-radius, y2), (x1+radius, y2),
        (x1, y2-radius), (x1, y1+radius),
    ]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, thickness)

    cv2.ellipse(frame, (x1+radius, y1+radius), (radius, radius), 180,  0, 90, color, thickness)
    cv2.ellipse(frame, (x2-radius, y1+radius), (radius, radius), 270,  0, 90, color, thickness)
    cv2.ellipse(frame, (x2-radius, y2-radius), (radius, radius),   0,  0, 90, color, thickness)
    cv2.ellipse(frame, (x1+radius, y2-radius), (radius, radius),  90,  0, 90, color, thickness)


# ─────────────────────────────────────────
# OVERLAY JOUEURS
# ─────────────────────────────────────────
def draw_players(frame, players, color_detector, jersey_map=None):
    for p in players:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        team   = p.get("team")
        tid    = p.get("id", "?")
        jersey = p.get("jersey") or (jersey_map or {}).get(tid)
        color  = color_detector.get_color(team)

        draw_rounded_rect(frame, x1, y1, x2, y2, color, thickness=2)

        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 4, color, -1)

        label   = f"#{jersey}" if jersey else f"ID{tid}"
        label_y = max(y1 - 6, 12)
        draw_text_with_shadow(frame, label, (x1, label_y))

    return frame


# ─────────────────────────────────────────
# OVERLAY BALLON
# ─────────────────────────────────────────
def draw_ball(frame, ball):
    if ball is None:
        return frame

    cx, cy       = [int(v) for v in ball["center"]]
    interpolated = ball.get("interpolated", False)
    color        = (100, 255, 100) if interpolated else (0, 255, 0)
    thickness    = 1               if interpolated else 2

    cv2.circle(frame, (cx, cy), 12, color, thickness)
    cv2.circle(frame, (cx, cy),  3, color, -1)

    if interpolated:
        draw_text_with_shadow(frame, "~", (cx+14, cy+5), font_scale=0.4)

    return frame


# ─────────────────────────────────────────
# OVERLAY EVENTS (flash)
# ─────────────────────────────────────────
EVENT_LABELS = {
    "goal":         "BUT !",
    "score":        "BUT !",
    "shot":         "TIR",
    "interception": "INTERCEPTION",
    "dribble":      "DRIBBLE",
    "long_pass":    "PASSE LONGUE",
    "pass":         "PASSE",
}

def draw_event_flash(frame, events, frame_id, flash_duration=30):
    key_events = [
        e for e in events
        if e.get("type") in EVENT_LABELS
        and abs(e.get("frame", -999) - frame_id) < flash_duration
    ]
    if not key_events:
        return frame

    e     = max(key_events, key=lambda x: x.get("frame", 0))
    label = EVENT_LABELS.get(e["type"], e["type"].upper())

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = 35

    cv2.putText(frame, label, (tx+1, ty+1), font, font_scale, SHADOW_COLOR, thickness+2, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,   ty),   font, font_scale, TEXT_COLOR,   thickness,   cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────
# OVERLAY SCOREBOARD
# ─────────────────────────────────────────
def draw_scoreboard(frame, frame_id, fps, score=None):
    h, w    = frame.shape[:2]
    seconds = int(frame_id / fps) if fps > 0 else 0
    chrono  = f"{seconds//60:02d}:{seconds%60:02d}"
    text    = f"  {score.get('A',0)} - {score.get('B',0)}  |  {chrono}" \
              if score else chrono

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    margin = 8
    rx1 = w - tw - margin*2 - 10
    ry1 = 8
    rx2 = w - 10
    ry2 = 8 + th + margin*2

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    tx = rx1 + margin
    ty = ry2 - margin
    cv2.putText(frame, text, (tx+1, ty+1), font, font_scale, SHADOW_COLOR, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,   ty),   font, font_scale, TEXT_COLOR,   thickness,   cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────
# LÉGENDE COULEURS ÉQUIPES
# ─────────────────────────────────────────
def draw_legend(frame, color_detector):
    """
    Affiche en bas à gauche un petit carré
    de la couleur détectée pour chaque équipe.
    """
    h, w = frame.shape[:2]
    x, y = 10, h - 50

    for team in [0, 1]:
        color = color_detector.get_color(team)
        label = f"Equipe {team + 1}"

        cv2.rectangle(frame, (x, y), (x+20, y+16), color, -1)
        draw_text_with_shadow(frame, label, (x+26, y+13), font_scale=0.45)
        y += 22

    return frame


# ─────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────
class Overlay:

    def __init__(self, fps=None, show_scoreboard=True,
                 show_events=True, show_legend=True,
                 sample_frames=60):

        self.fps             = fps or config.FPS
        self.show_scoreboard = show_scoreboard
        self.show_events     = show_events
        self.show_legend     = show_legend
        self.score           = {"A": 0, "B": 0}

        # Détection automatique couleurs
        self.color_detector  = TeamColorDetector(sample_frames=sample_frames)

    def update_score(self, events):
        for e in events:
            if e.get("type") in ["goal", "score"]:
                team = e.get("team")
                if team == 0:   self.score["A"] += 1
                elif team == 1: self.score["B"] += 1

    def render(self, frame, players, ball, events, frame_id, jersey_map=None):
        """
        Applique tous les overlays sur une frame.
        La détection de couleur se fait automatiquement
        sur les premières frames puis se fige.
        """

        # Apprentissage couleurs sur les N premières frames
        self.color_detector.update(frame, players)

        self.update_score(events)

        frame = draw_players(frame, players, self.color_detector, jersey_map)
        frame = draw_ball(frame, ball)

        if self.show_events:
            frame = draw_event_flash(frame, events, frame_id)

        if self.show_scoreboard:
            frame = draw_scoreboard(frame, frame_id, self.fps, self.score)

        if self.show_legend:
            frame = draw_legend(frame, self.color_detector)

        return frame

    def reset(self):
        """Réinitialise entre deux vidéos."""
        self.color_detector.reset()
        self.score = {"A": 0, "B": 0}