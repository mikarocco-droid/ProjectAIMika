"""
FieldModel — Sprint BC
========================
Accumule les observations géométriques frame par frame
et produit un état terrain stable via EMA temporelle.

Principe :
  - Chaque observation est pondérée par sa confidence
  - EMA avec alpha=0.15 → lissage fort, résistant aux fausses détections
  - Un état est publié seulement quand confidence cumulée ≥ MIN_CONFIDENCE

Sortie :
  field_state = {
      "goal_left_x":     0.04,    # x normalisé [0,1]
      "goal_right_x":    0.15,
      "penalty_left_x":  0.00,
      "penalty_right_x": 0.22,
      "sideline_y_top":  0.28,
      "sideline_y_bot":  0.92,
      "confidence":      0.82,
      "n_observations":  47,
  }
"""

import json
import time
from typing import Optional
from dataclasses import dataclass, field

from geometry.geometry_extractor import GeometryObservation


# ── Paramètres ─────────────────────────────────────────────────────────────────

EMA_ALPHA         = 0.15   # lissage fort : 0.15 * nouveau + 0.85 * historique
MIN_OBS_TO_PUBLISH = 5     # observations minimum avant de publier un état
MIN_CONFIDENCE    = 0.40   # confiance minimale pour publier


# ── Accumulateur EMA pour une valeur scalaire ───────────────────────────────────

class _EMAValue:
    """EMA pondérée par la confidence."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha   = alpha
        self.value   = None
        self.conf    = 0.0
        self.n_obs   = 0

    def update(self, new_value: Optional[float], weight: float = 1.0):
        if new_value is None:
            return
        if self.value is None:
            self.value = new_value
            self.conf  = weight
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
            self.conf  = min(1.0, self.alpha * weight + (1 - self.alpha) * self.conf)
        self.n_obs += 1

    def get(self) -> Optional[float]:
        return round(self.value, 4) if self.value is not None else None


# ── Modèle terrain ─────────────────────────────────────────────────────────────

class FieldModel:
    """
    Accumule les observations et maintient un modèle terrain cohérent.

    Usage :
        model = FieldModel()
        for frame in frames:
            obs = extractor.extract(frame)
            model.update(obs)

        state = model.get_state()
        # → {"goal_left_x": 0.04, ..., "confidence": 0.78}
    """

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._goal_left     = _EMAValue(alpha)
        self._goal_right    = _EMAValue(alpha)
        self._penalty_left  = _EMAValue(alpha)
        self._penalty_right = _EMAValue(alpha)
        self._sideline_top  = _EMAValue(alpha)
        self._sideline_bot  = _EMAValue(alpha)
        self._conf          = _EMAValue(alpha)
        self._n_obs         = 0
        self._n_good        = 0
        self._last_update   = 0.0

        # Historique léger pour diagnostics
        self._history: list[dict] = []
        self._max_history = 200

    def update(self, obs: GeometryObservation):
        """Intègre une observation dans le modèle."""
        self._n_obs += 1
        w = obs.confidence

        if w < 0.10:
            return  # observation trop faible, ignorer

        self._n_good += 1
        self._goal_left.update(obs.goal_left_x,     w)
        self._goal_right.update(obs.goal_right_x,   w)
        self._penalty_left.update(obs.penalty_left_x,  w)
        self._penalty_right.update(obs.penalty_right_x, w)
        self._sideline_top.update(obs.sideline_y_top,   w)
        self._sideline_bot.update(obs.sideline_y_bot,   w)
        self._conf.update(w, 1.0)
        self._last_update = time.time()

        if len(self._history) < self._max_history:
            self._history.append({
                "frame": obs.frame_idx,
                "conf": round(w, 3),
                "goal_left": obs.goal_left_x,
                "goal_right": obs.goal_right_x,
            })

    def get_state(self) -> dict:
        """
        Retourne l'état terrain actuel.
        Retourne None si pas assez d'observations.
        """
        if self._n_good < MIN_OBS_TO_PUBLISH:
            return {
                "confidence": 0.0,
                "n_observations": self._n_obs,
                "status": "insufficient_data",
            }

        conf = self._conf.get() or 0.0
        if conf < MIN_CONFIDENCE:
            return {
                "confidence": conf,
                "n_observations": self._n_obs,
                "status": "low_confidence",
            }

        return {
            "goal_left_x":     self._goal_left.get(),
            "goal_right_x":    self._goal_right.get(),
            "penalty_left_x":  self._penalty_left.get(),
            "penalty_right_x": self._penalty_right.get(),
            "sideline_y_top":  self._sideline_top.get(),
            "sideline_y_bot":  self._sideline_bot.get(),
            "confidence":      round(conf, 3),
            "n_observations":  self._n_obs,
            "n_good":          self._n_good,
            "status":          "ok",
        }

    def is_ready(self) -> bool:
        """True si le modèle a suffisamment d'observations fiables."""
        state = self.get_state()
        return state.get("status") == "ok"

    def ball_in_goal(self, bx: float, by: float,
                     side: str = "left") -> Optional[bool]:
        """
        Détermine si le ballon est à l'intérieur du but.
        Retourne None si la géométrie n'est pas connue.

        bx, by : position normalisée [0,1] du ballon
        side   : "left" | "right"
        """
        state = self.get_state()
        if state.get("status") != "ok":
            return None

        if side == "left":
            gl = state.get("goal_left_x")
            gr = state.get("goal_right_x")
            if gl is None or gr is None:
                return None
            # Le ballon est dans le but gauche si bx est entre gl et gr
            # et suffisamment bas (dans la zone but)
            in_x = gl - 0.01 <= bx <= gr + 0.02
            return in_x

        if side == "right":
            gl = state.get("goal_left_x")
            gr = state.get("goal_right_x")
            if gl is None or gr is None:
                return None
            in_x = gr - 0.02 <= bx <= 1.0
            return in_x

        return None

    def distance_to_goal(self, bx: float, side: str = "left") -> Optional[float]:
        """
        Estime la distance normalisée du ballon au centre du but.
        Retourne None si la géométrie n'est pas connue.
        """
        state = self.get_state()
        if state.get("status") != "ok":
            return None

        if side == "left":
            gl = state.get("goal_left_x")
            gr = state.get("goal_right_x")
            if gl is None or gr is None:
                return None
            goal_center = (gl + gr) / 2
            return abs(bx - goal_center)

        return None

    def reset(self):
        """Remet à zéro le modèle (nouveau match)."""
        self.__init__(self.alpha)

    def summary(self):
        """Affiche l'état courant."""
        state = self.get_state()
        print(f"  [FIELD_MODEL] ── État terrain ─────────────────────────")
        print(f"  [FIELD_MODEL]   Status         : {state.get('status')}")
        print(f"  [FIELD_MODEL]   Confiance      : {state.get('confidence', 0):.2f}")
        print(f"  [FIELD_MODEL]   Observations   : {state.get('n_good', 0)}/{state.get('n_observations', 0)}")

        if state.get("status") == "ok":
            gl = state.get("goal_left_x")
            gr = state.get("goal_right_x")
            pl = state.get("penalty_left_x")
            pr = state.get("penalty_right_x")
            st = state.get("sideline_y_top")
            sb = state.get("sideline_y_bot")

            if gl is not None:
                width = (gr or 0) - gl
                print(f"  [FIELD_MODEL]   But gauche     : x=[{gl:.3f}, {gr:.3f}] largeur={width:.3f}")
            if pl is not None:
                print(f"  [FIELD_MODEL]   Surface gauche : x=[{pl:.3f}, {pr:.3f}]")
            if st is not None:
                print(f"  [FIELD_MODEL]   Lignes touche  : y=[{st:.3f}, {sb:.3f}]")
        print(f"  [FIELD_MODEL] ──────────────────────────────────────────")
