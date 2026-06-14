"""
GeometryExtractor — Sprint BC.3A
==================================
Extrait les structures géométriques du terrain depuis une frame vidéo.

BC.3A — Refactor sémantique (2026-06-14)
-----------------------------------------
Problème BC.2 : goal_left_x / goal_right_x prétendaient identifier des
poteaux alors que la caméra latérale voit des *lignes verticales* dont la
nature réelle est inconnue.

  Vidéo 1 : obs.goal_left_x = 0.052  → probablement ligne de but
  Vidéo 2 : obs.goal_left_x = 0.221  → probablement ligne de surface
  → même variable, deux sémantiques différentes.

Nouveau modèle : GeometryExtractor n'interprète plus, il observe.
  vertical_anchor_x    : position x normalisée de la structure verticale
  vertical_anchor_conf : confiance dans la détection [0,1]
  vertical_anchor_type : "goal_line" | "penalty_line" | "unknown"
                          (hint seulement — FieldAnchorModel tranche)

Les anciens champs goal_left_x / goal_right_x sont dépréciés → None.
L'interprétation (H1=goal_line vs H2=penalty_line) est déléguée à
FieldAnchorModel (BC.3C).

Sortie par frame :
  GeometryObservation {
    # Nouveaux champs BC.3A
    vertical_anchor_x    : float | None   # x normalisé [0,1]
    vertical_anchor_conf : float          # [0,1]
    vertical_anchor_type : str            # "goal_line"|"penalty_line"|"unknown"

    # Champs existants conservés
    penalty_left_x  : float | None
    penalty_right_x : float | None
    sideline_y_top  : float | None
    sideline_y_bot  : float | None
    confidence      : float
    method          : str

    # DÉPRÉCIÉS — toujours None depuis BC.3A
    goal_left_x     : None   # était ambigu, remplacé par vertical_anchor_x
    goal_right_x    : None
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
    """
    Observation géométrique pour une frame — BC.3A.

    Philosophie : observer sans interpréter.
    L'extracteur rapporte ce qu'il voit (une structure verticale à x=0.054).
    FieldAnchorModel décide ce que c'est (ligne de but ? ligne de surface ?).
    """
    # ── Nouveaux champs BC.3A ──────────────────────────────────────────────────
    # La structure verticale la plus saillante détectée dans la frame.
    # Ne présume PAS de sa nature — c'est FieldAnchorModel qui tranche.
    vertical_anchor_x:    Optional[float] = None   # x normalisé [0,1]
    vertical_anchor_conf: float           = 0.0    # confiance détection [0,1]
    vertical_anchor_type: str             = "unknown"
    # "unknown"      : GeometryExtractor ne sait pas
    # "goal_line"    : hint basé sur position < 10% (très probablement but)
    # "penalty_line" : hint basé sur position 10-25% + largeur cohérente
    # Ces hints sont des suggestions, pas des certitudes.

    # ── Champs existants conservés ────────────────────────────────────────────
    penalty_left_x:  Optional[float] = None
    penalty_right_x: Optional[float] = None
    sideline_y_top:  Optional[float] = None
    sideline_y_bot:  Optional[float] = None
    confidence:      float = 0.0
    method:          str   = "none"
    frame_idx:       int   = 0

    # ── Champs DÉPRÉCIÉS BC.3A ────────────────────────────────────────────────
    # Conservés pour compatibilité descendante (code existant ne plante pas).
    # Toujours None depuis BC.3A — ne plus lire ces champs.
    goal_left_x:  Optional[float] = None   # DÉPRÉCIÉ → lire vertical_anchor_x
    goal_right_x: Optional[float] = None   # DÉPRÉCIÉ

    def has_vertical_anchor(self) -> bool:
        """True si une structure verticale a été détectée."""
        return self.vertical_anchor_x is not None

    def has_penalty(self) -> bool:
        return self.penalty_left_x is not None and self.penalty_right_x is not None

    # Compatibilité descendante — code qui appelle has_goal() ne plante pas.
    def has_goal(self) -> bool:
        """DÉPRÉCIÉ BC.3A — utiliser has_vertical_anchor()."""
        return self.has_vertical_anchor()

    def to_dict(self) -> dict:
        return {
            # Nouveaux champs BC.3A
            "vertical_anchor_x":    self.vertical_anchor_x,
            "vertical_anchor_conf": self.vertical_anchor_conf,
            "vertical_anchor_type": self.vertical_anchor_type,
            # Champs conservés
            "penalty_left_x":  self.penalty_left_x,
            "penalty_right_x": self.penalty_right_x,
            "sideline_y_top":  self.sideline_y_top,
            "sideline_y_bot":  self.sideline_y_bot,
            "confidence":      self.confidence,
            "method":          self.method,
            "frame_idx":       self.frame_idx,
            # Dépréciés — None
            "goal_left_x":     None,   # DÉPRÉCIÉ
            "goal_right_x":    None,   # DÉPRÉCIÉ
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
        self._n_calls   = 0
        self._n_success = 0
        # BC.4 — accumulation observations pour alimentation directe FieldAnchorModel.
        # Le pipeline lit _observations au lieu de reconstruire des fakes depuis
        # _field_model._history (BC.2 legacy).
        # Seules les obs avec vertical_anchor_x valide sont stockées.
        self._observations: list = []   # list[GeometryObservation]

    def extract(self, frame: np.ndarray, frame_idx: int = 0,
                debug: bool = False) -> GeometryObservation:
        """
        Extrait les structures géométriques d'une frame.
        Retourne une GeometryObservation avec confidence ≥ 0.

        BC.3A : peuple vertical_anchor_x / _conf / _type au lieu de
        goal_left_x / goal_right_x.
        debug=True : logs enrichis BC.3B.
        """
        self._n_calls += 1
        h, w = frame.shape[:2]

        obs = GeometryObservation(frame_idx=frame_idx)

        # 1. Détecter les lignes blanches du terrain
        lines, debug_info = self._detect_white_lines_debug(frame, w, h)

        if debug:
            t_s = frame_idx / max(1, getattr(self, '_fps', 25.0))
            print(f"  [GEOM] frame={frame_idx} t={t_s:.1f}s "
                  f"grass={debug_info['grass_ratio']:.2f} "
                  f"white_px={debug_info['white_pixels']} "
                  f"hough_lines={debug_info['hough_lines']} "
                  f"fallback={debug_info['used_fallback']}")

        if lines is None or len(lines) == 0:
            obs.method = "no_lines"
            if debug:
                print(f"  [GEOM]   → reject: no_lines")
            return obs

        # 2. Classifier les lignes : verticales vs horizontales
        v_lines, h_lines = self._classify_lines(lines, w, h)

        if debug:
            n_ignored = len(lines) - len(v_lines) - len(h_lines)
            print(f"  [GEOM]   → vertical={len(v_lines)} horizontal={len(h_lines)} "
                  f"ignored={n_ignored}")
            if v_lines:
                v_by_x = sorted(v_lines, key=lambda l: l['x_mean'])[:5]
                xvals = [f"{int(l['x_mean']*w)}px(L={l['length']:.0f})" for l in v_by_x]
                print(f"  [GEOM]   → top v_lines x: {' '.join(xvals)}")
            elif lines and len(lines) > 5:
                angles = []
                for ln in lines:
                    x1, y1, x2, y2 = ln
                    dx, dy = abs(x2-x1), abs(y2-y1)
                    length = (dx**2 + dy**2) ** 0.5
                    if length >= 0.10 * h:
                        angles.append((round(np.degrees(np.arctan2(dy, max(dx, 1))), 1), int(length)))
                if angles:
                    angles.sort(reverse=True)
                    top = ' '.join(f"{a}°({l}px)" for a, l in angles[:6])
                    print(f"  [GEOM]   → long_lines angles: {top}")
                    near_v = [a for a, _ in angles if a > 40]
                    if near_v:
                        print(f"  [GEOM]   → angles 40-60°: {near_v[:8]} ← poteaux probables")

        # 3. BC.3A — chercher la structure verticale principale (anchor)
        anchor_result = self._find_vertical_anchors(v_lines, h_lines, w, h, debug=debug)
        if anchor_result:
            obs.vertical_anchor_x    = anchor_result["anchor_x"]
            obs.vertical_anchor_conf = anchor_result["confidence"]
            obs.vertical_anchor_type = anchor_result["anchor_type"]
            obs.confidence          += anchor_result["confidence"] * 0.5
            obs.method               = "hough_anchor"

            if debug:
                print(f"  [GEOM]   → ANCHOR: x={obs.vertical_anchor_x:.3f} "
                      f"({int(obs.vertical_anchor_x*w)}px) "
                      f"type={obs.vertical_anchor_type} "
                      f"conf={obs.vertical_anchor_conf:.2f}")
        elif debug:
            print(f"  [GEOM]   → no_anchor "
                  f"(besoin ≥1 v_line longue dans [{MIN_GOAL_WIDTH_PCT:.2f}, {MAX_GOAL_WIDTH_PCT:.2f}])")

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

        # BC.4 — accumuler les obs avec anchor valide pour FieldAnchorModel
        if obs.vertical_anchor_x is not None:
            self._observations.append(obs)

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

    def _classify_lines(self, lines, w, h,
                         min_vertical_length: float = 0.10):
        """
        Sépare les lignes verticales (poteaux) des horizontales (lignes).

        min_vertical_length : longueur minimale pour une ligne verticale,
            exprimée en fraction de la hauteur de l'image.
            Filtre les courtes lignes de bruit (joueurs, publicités).
            Valeur par défaut : 10% de la hauteur = ~108px sur 1080px.
        """
        v_lines = []  # angle > 60° par rapport à horizontal
        h_lines = []  # angle < 30°
        min_v_px = min_vertical_length * h

        for ln in lines:
            x1, y1, x2, y2 = ln
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)
            if length < 20:
                continue
            angle = np.degrees(np.arctan2(dy, max(dx, 1)))

            if angle > 50:   # CAS A: élargi de 60° → 50° pour caméras inclinées
                if length >= min_v_px:   # filtrer les verticales courtes
                    v_lines.append({
                        "x_mean": (x1 + x2) / 2 / w,
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

    # ── BC.3A : Anchor vertical principal ───────────────────────────────────────

    def _find_vertical_anchors(self, v_lines, h_lines, w, h,
                               debug: bool = False) -> Optional[dict]:
        """
        BC.3A — Trouve la structure verticale la plus saillante sans l'interpréter.

        Retourne :
          {
            "anchor_x":    float,   # x normalisé de la structure principale
            "confidence":  float,   # [0,1]
            "anchor_type": str,     # hint : "goal_line"|"penalty_line"|"unknown"
          }
        ou None si aucune structure verticale fiable trouvée.

        Note sur anchor_type :
          Ce n'est qu'un hint heuristique pour aider FieldAnchorModel.
          La vraie décision d'interprétation appartient à FieldAnchorModel (BC.3C).
          - "goal_line"    : x < 0.10 → très probablement ligne de but
          - "penalty_line" : 0.10 ≤ x ≤ 0.28 → probablement ligne de surface
          - "unknown"      : hors plage, ou signal ambiguë
        """
        if not v_lines:
            if debug:
                print(f"  [GEOM]   → no_anchor: 0 v_lines")
            return None

        # Sélectionner la ligne verticale la plus longue dans la zone gauche
        # (le but est toujours du côté gauche sur cette caméra)
        # Zone élargie : x < 0.35 pour capturer aussi la ligne de surface
        candidates = [l for l in v_lines if l["x_mean"] < 0.40]

        if not candidates:
            # Fallback : prendre toutes les v_lines, prendre la plus à gauche
            candidates = v_lines

        # Trier par longueur décroissante + position gauche (penalise les structures lointaines)
        # Score composite : length_norm × (1 - x_mean) pour favoriser les structures à gauche
        def anchor_score(l):
            length_norm = min(1.0, l["length"] / (0.4 * h))   # normalisé sur 40% hauteur
            position_bonus = max(0.0, 1.0 - l["x_mean"] / 0.40)
            return length_norm * 0.7 + position_bonus * 0.3

        best = max(candidates, key=anchor_score)
        anchor_x = best["x_mean"]

        # Confiance : basée sur longueur de la ligne
        length_norm = min(1.0, best["length"] / (0.3 * h))
        # Bonus si une barre transversale existe à ce x (indice de but réel)
        has_bar = self._has_horizontal_near(h_lines, anchor_x, tolerance=0.05)
        conf = length_norm * 0.6 + (0.25 if has_bar else 0.0)
        conf = min(1.0, conf)

        # Hint sémantique (ne pas sur-interpréter)
        if anchor_x < 0.10:
            anchor_type = "goal_line"
        elif 0.10 <= anchor_x <= 0.28:
            anchor_type = "penalty_line"
        else:
            anchor_type = "unknown"

        if debug:
            print(f"  [GEOM]   → anchor_best: x={anchor_x:.3f}({int(anchor_x*w)}px) "
                  f"length={best['length']:.0f}px "
                  f"has_bar={has_bar} score={anchor_score(best):.2f} "
                  f"→ type_hint={anchor_type}")
            # Afficher aussi les autres candidats pour comparaison
            others = sorted(candidates, key=anchor_score, reverse=True)[1:4]
            for o in others:
                print(f"  [GEOM]     alt: x={o['x_mean']:.3f}({int(o['x_mean']*w)}px) "
                      f"length={o['length']:.0f}px score={anchor_score(o):.2f}")

        return {
            "anchor_x":    anchor_x,
            "confidence":  conf,
            "anchor_type": anchor_type,
        }

    def _has_horizontal_near(self, h_lines, anchor_x: float,
                             tolerance: float = 0.05) -> bool:
        """Vérifie si une ligne horizontale passe près de anchor_x (indice de barre)."""
        for hl in h_lines:
            if hl["x_left"] <= anchor_x + tolerance and hl["x_right"] >= anchor_x - tolerance:
                return True
        return False

    # ── DÉPRÉCIÉ BC.3A ────────────────────────────────────────────────────────

    def _find_goal_structures(self, v_lines, h_lines, w, h,
                              debug: bool = False) -> Optional[dict]:
        """
        DÉPRÉCIÉ BC.3A — utiliser _find_vertical_anchors().

        Conservé pour compatibilité mais n'est plus appelé par extract().
        Cherchait une paire de poteaux (goal_left_x / goal_right_x) —
        approche abandonnée car sémantiquement incorrecte sur caméra latérale.
        """
        if len(v_lines) < 2:
            if debug:
                print(f"  [GEOM]   → [DEPRECATED] no_goal: <2 v_lines ({len(v_lines)})")
            return None

        best = None
        best_conf = 0.0
        reject_width = 0
        reject_overlap = 0

        v_sorted = sorted(v_lines, key=lambda l: l["x_mean"])

        for i in range(len(v_sorted)):
            for j in range(i + 1, len(v_sorted)):
                left  = v_sorted[i]
                right = v_sorted[j]
                width = right["x_mean"] - left["x_mean"]

                if not (MIN_GOAL_WIDTH_PCT <= width <= MAX_GOAL_WIDTH_PCT):
                    reject_width += 1
                    continue

                y_overlap = min(left["y_bot"], right["y_bot"]) - max(left["y_top"], right["y_top"])
                if y_overlap < 0.05:
                    reject_overlap += 1
                    continue

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