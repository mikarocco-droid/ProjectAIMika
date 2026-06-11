"""
GeometryExtractor — Sprint BC
==============================
Extrait les structures géométriques du terrain depuis une frame vidéo.

Principe : accumulation temporelle légère.
- Pas d'homographie complète (Sprint D).
- Objectif : fournir des coordonnées normalisées fiables pour
  goal_left, goal_right, penalty_area_left, penalty_area_right.

Sortie par frame :
  GeometryObservation {
    goal_left_x    : float | None   # x normalisé [0,1] du poteau gauche
    goal_right_x   : float | None   # x normalisé [0,1] du poteau droit
    penalty_left_x : float | None
    penalty_right_x: float | None
    sideline_y_top : float | None   # y normalisé de la ligne haute
    sideline_y_bot : float | None
    confidence     : float          # [0, 1]
    method         : str            # "hough" | "contour" | "fallback"
  }
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Paramètres ─────────────────────────────────────────────────────────────────

# Zone d'intérêt — élargie pour capturer les poteaux de but
# Sur low_side_zoom, les poteaux peuvent commencer dès y=0.15
ROI_Y_MIN = 0.10   # élargi de 0.20 → 0.10
ROI_Y_MAX = 0.97

# Détection lignes Hough — seuils souples pour vidéos compressées/low_side_zoom
HOUGH_THRESHOLD   = 30    # réduit de 60 → 30 pour détecter lignes peu contrastées
HOUGH_MIN_LENGTH  = 40    # réduit de 80 → 40 pour courtes lignes partiellement visibles
HOUGH_MAX_GAP     = 30    # augmenté pour tolérer les interruptions

# Largeur but — élargie pour low_side_zoom (caméra très proche du but gauche)
MIN_GOAL_WIDTH_PCT  = 0.02   # 2% = ~38px
MAX_GOAL_WIDTH_PCT  = 0.18   # 18% = ~346px (but très proche caméra)

# Largeur minimale surface de réparation
MIN_PENALTY_WIDTH_PCT = 0.08

# Score minimum pour publier une observation
MIN_CONFIDENCE = 0.30


# ── Structures de données ───────────────────────────────────────────────────────

@dataclass
class GeometryObservation:
    """Observation géométrique pour une frame."""
    goal_left_x:     Optional[float] = None
    goal_right_x:    Optional[float] = None
    penalty_left_x:  Optional[float] = None
    penalty_right_x: Optional[float] = None
    sideline_y_top:  Optional[float] = None
    sideline_y_bot:  Optional[float] = None
    confidence:      float = 0.0
    method:          str   = "none"
    frame_idx:       int   = 0

    def has_goal(self) -> bool:
        return self.goal_left_x is not None or self.goal_right_x is not None

    def has_penalty(self) -> bool:
        return self.penalty_left_x is not None and self.penalty_right_x is not None

    def to_dict(self) -> dict:
        return {
            "goal_left_x":     self.goal_left_x,
            "goal_right_x":    self.goal_right_x,
            "penalty_left_x":  self.penalty_left_x,
            "penalty_right_x": self.penalty_right_x,
            "sideline_y_top":  self.sideline_y_top,
            "sideline_y_bot":  self.sideline_y_bot,
            "confidence":      self.confidence,
            "method":          self.method,
            "frame_idx":       self.frame_idx,
        }


# ── Extracteur principal ────────────────────────────────────────────────────────

class GeometryExtractor:
    """
    Extrait les structures géométriques terrain depuis une frame BGR.

    Usage :
        extractor = GeometryExtractor(frame_w=1920, frame_h=1080)
        obs = extractor.extract(frame, frame_idx=500)
    """

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080, fps: float = 25.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._fps    = fps
        self._n_calls = 0
        self._n_success = 0

    def extract(self, frame: np.ndarray, frame_idx: int = 0,
                debug: bool = False) -> GeometryObservation:
        """
        Extrait les structures géométriques d'une frame.
        Retourne une GeometryObservation avec confidence ≥ 0.
        debug=True : logs détaillés pour BC.2 diagnostic.
        """
        self._n_calls += 1
        h, w = frame.shape[:2]

        obs = GeometryObservation(frame_idx=frame_idx)

        # 1. Détecter les lignes blanches du terrain
        lines, debug_info = self._detect_white_lines_debug(frame, w, h)

        if debug:
            t_s = frame_idx / max(1, getattr(self, '_fps', 25.0))
            print(f"  [GEOM_DEBUG] frame={frame_idx} t={t_s:.1f}s "
                  f"grass={debug_info['grass_ratio']:.2f} "
                  f"white_px={debug_info['white_pixels']} "
                  f"hough_lines={debug_info['hough_lines']} "
                  f"fallback={debug_info['used_fallback']}")

        if lines is None or len(lines) == 0:
            obs.method = "no_lines"
            if debug:
                print(f"  [GEOM_DEBUG]   → reject: no_lines")
            return obs

        # 2. Classifier les lignes : verticales (poteaux) vs horizontales (lignes)
        v_lines, h_lines = self._classify_lines(lines, w, h)

        # 3. Chercher les structures but
        goal_obs = self._find_goal_structures(v_lines, h_lines, w, h)
        if goal_obs:
            obs.goal_left_x     = goal_obs.get("left_x")
            obs.goal_right_x    = goal_obs.get("right_x")
            obs.confidence      += goal_obs.get("confidence", 0) * 0.5
            obs.method           = "hough_goal"

        # 4. Chercher la surface de réparation
        penalty_obs = self._find_penalty_area(v_lines, h_lines, w, h)
        if penalty_obs:
            obs.penalty_left_x  = penalty_obs.get("left_x")
            obs.penalty_right_x = penalty_obs.get("right_x")
            obs.confidence      += penalty_obs.get("confidence", 0) * 0.3

        # 5. Chercher les lignes de touche (sidelines)
        sideline_obs = self._find_sidelines(h_lines, w, h)
        if sideline_obs:
            obs.sideline_y_top  = sideline_obs.get("y_top")
            obs.sideline_y_bot  = sideline_obs.get("y_bot")
            obs.confidence      += sideline_obs.get("confidence", 0) * 0.2

        obs.confidence = min(1.0, obs.confidence)

        if obs.confidence >= MIN_CONFIDENCE:
            self._n_success += 1

        return obs

    # ── Détection lignes blanches ─────────────────────────────────────────────

    def _detect_white_lines(self, frame, w, h):
        """Compatibilité — appelle la version debug avec debug_info ignoré."""
        lines, _ = self._detect_white_lines_debug(frame, w, h)
        return lines

    def _detect_white_lines_debug(self, frame, w, h):
        """Détecte les lignes blanches + retourne des métriques de diagnostic."""
        debug_info = {
            "grass_ratio": 0.0,
            "white_pixels": 0,
            "hough_lines": 0,
            "used_fallback": False,
        }

        y0 = int(h * ROI_Y_MIN)
        y1 = int(h * ROI_Y_MAX)
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv,
                                  np.array([25, 25, 25]),
                                  np.array([95, 255, 255]))

        roi_pixels = roi.shape[0] * roi.shape[1]
        debug_info["grass_ratio"] = float(np.count_nonzero(green_mask)) / max(1, roi_pixels)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        debug_info["white_pixels"] = int(np.count_nonzero(white_mask))

        green_dilated = cv2.dilate(green_mask, np.ones((5, 5), np.uint8), iterations=2)
        line_mask = cv2.bitwise_and(white_mask, green_dilated)

        edges = cv2.Canny(line_mask, 30, 120)
        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi/180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LENGTH,
            maxLineGap=HOUGH_MAX_GAP,
        )
        debug_info["hough_lines"] = len(lines) if lines is not None else 0

        # Fallback : image grise directe si peu de lignes
        if lines is None or len(lines) < 3:
            edges_fb = cv2.Canny(gray, 40, 130)
            lines_fb = cv2.HoughLinesP(
                edges_fb,
                rho=1, theta=np.pi/180,
                threshold=max(15, HOUGH_THRESHOLD // 2),
                minLineLength=max(20, HOUGH_MIN_LENGTH // 2),
                maxLineGap=HOUGH_MAX_GAP * 2,
            )
            if lines_fb is not None:
                lines = lines_fb
                debug_info["used_fallback"] = True
                debug_info["hough_lines"] = len(lines_fb)

        if lines is None:
            return None, debug_info

        offset_y = y0
        adjusted = []
        for ln in lines:
            x1, y1_, x2, y2_ = ln[0]
            adjusted.append([x1, y1_ + offset_y, x2, y2_ + offset_y])
        return adjusted, debug_info

    # ── Classification lignes ─────────────────────────────────────────────────

    def _classify_lines(self, lines, w, h):
        """Sépare les lignes verticales (poteaux) des horizontales (lignes)."""
        v_lines = []  # angle > 60° par rapport à horizontal
        h_lines = []  # angle < 30°

        for ln in lines:
            x1, y1, x2, y2 = ln
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)
            if length < 20:
                continue
            angle = np.degrees(np.arctan2(dy, max(dx, 1)))

            if angle > 60:
                v_lines.append({
                    "x_mean": (x1 + x2) / 2 / w,  # normalisé [0,1]
                    "y_top":  min(y1, y2) / h,
                    "y_bot":  max(y1, y2) / h,
                    "length": length,
                    "raw": (x1, y1, x2, y2),
                })
            elif angle < 30:
                h_lines.append({
                    "y_mean": (y1 + y2) / 2 / h,
                    "x_left":  min(x1, x2) / w,
                    "x_right": max(x1, x2) / w,
                    "length": length,
                    "raw": (x1, y1, x2, y2),
                })

        return v_lines, h_lines

    # ── Structures but ────────────────────────────────────────────────────────

    def _find_goal_structures(self, v_lines, h_lines, w, h) -> Optional[dict]:
        """
        Cherche une paire de poteaux verticaux formant un but.
        Retourne {"left_x", "right_x", "confidence"} ou None.
        """
        if len(v_lines) < 2:
            return None

        best = None
        best_conf = 0.0

        # Trier par x
        v_sorted = sorted(v_lines, key=lambda l: l["x_mean"])

        for i in range(len(v_sorted)):
            for j in range(i + 1, len(v_sorted)):
                left  = v_sorted[i]
                right = v_sorted[j]
                width = right["x_mean"] - left["x_mean"]

                if not (MIN_GOAL_WIDTH_PCT <= width <= MAX_GOAL_WIDTH_PCT):
                    continue

                # Les deux poteaux doivent avoir des y similaires (même hauteur)
                y_overlap = min(left["y_bot"], right["y_bot"]) - max(left["y_top"], right["y_top"])
                if y_overlap < 0.05:
                    continue

                # Chercher une barre transversale horizontale entre eux
                crossbar = self._find_crossbar(h_lines, left["x_mean"], right["x_mean"])

                conf = 0.5
                if crossbar:
                    conf += 0.3
                conf += min(0.2, (left["length"] + right["length"]) / 400)

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "left_x":    left["x_mean"],
                        "right_x":   right["x_mean"],
                        "confidence": conf,
                    }

        return best

    def _find_crossbar(self, h_lines, x_left, x_right) -> bool:
        """Vérifie s'il existe une ligne horizontale entre x_left et x_right."""
        for hl in h_lines:
            if hl["x_left"] <= x_left + 0.02 and hl["x_right"] >= x_right - 0.02:
                return True
        return False

    # ── Surface de réparation ─────────────────────────────────────────────────

    def _find_penalty_area(self, v_lines, h_lines, w, h) -> Optional[dict]:
        """
        Cherche la surface de réparation comme un rectangle de largeur typique.
        """
        if len(v_lines) < 2:
            return None

        v_sorted = sorted(v_lines, key=lambda l: l["x_mean"])

        for i in range(len(v_sorted)):
            for j in range(i + 1, len(v_sorted)):
                left  = v_sorted[i]
                right = v_sorted[j]
                width = right["x_mean"] - left["x_mean"]

                if not (MIN_PENALTY_WIDTH_PCT <= width <= 0.35):
                    continue

                # La surface doit être plus large qu'un but
                if width <= MAX_GOAL_WIDTH_PCT:
                    continue

                conf = 0.4
                conf += min(0.3, (left["length"] + right["length"]) / 300)

                return {
                    "left_x":    left["x_mean"],
                    "right_x":   right["x_mean"],
                    "confidence": conf,
                }
        return None

    # ── Lignes de touche ──────────────────────────────────────────────────────

    def _find_sidelines(self, h_lines, w, h) -> Optional[dict]:
        """
        Cherche les lignes de touche horizontales (haut et bas du terrain).
        """
        if not h_lines:
            return None

        # Lignes longues (> 40% de la largeur) probablement des lignes de touche
        long_h = [l for l in h_lines if l["length"] / w > 0.40]

        if not long_h:
            return None

        y_values = sorted(l["y_mean"] for l in long_h)

        if len(y_values) == 1:
            y = y_values[0]
            # Si dans le bas de l'image → sideline bas
            if y > 0.60:
                return {"y_top": None, "y_bot": y, "confidence": 0.4}
            else:
                return {"y_top": y, "y_bot": None, "confidence": 0.4}

        return {
            "y_top":      y_values[0],
            "y_bot":      y_values[-1],
            "confidence": min(0.8, 0.3 + len(long_h) * 0.1),
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_calls":   self._n_calls,
            "n_success": self._n_success,
            "rate":      self._n_success / max(1, self._n_calls),
        }