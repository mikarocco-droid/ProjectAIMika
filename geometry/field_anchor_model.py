"""
FieldAnchorModel — Sprint BC.3C (inférence géométrique par hypothèses)
=======================================================================

BC.3C remplace le calcul mécanique de _recompute() par un moteur
d'hypothèses FIFA + scoring de cohérence.

Philosophie BC.3C :
  GeometryExtractor observe une structure verticale à anchor_x=0.054.
  Il ne sait pas ce que c'est.
  FieldAnchorModel teste 3 hypothèses et garde la plus cohérente.

  H1 : anchor = ligne de but
       → goal_line_x = anchor_x
       → penalty_line_x = anchor_x + 16.5m (si px_per_m connu)

  H2 : anchor = ligne de surface
       → penalty_line_x = anchor_x
       → goal_line_x = anchor_x - 16.5m
       → valide même si goal_line_x < 0 (but hors champ = caméra décalée)

  H3 : anchor = poteau proche du but
       → géométriquement identique à H1 (poteau ≈ ligne de but sur caméra lointaine)

  Score de cohérence :
    s_camera  : accord avec camera_profile (prior p01-p99 du ballon)
    s_fifa    : lignes dans le champ [0,1]
    s_stable  : signal stable (σ/mean faible)

  Quand le score de l'hypothèse gagnante dépasse SCORE_READY_THRESHOLD,
  le modèle est Ready=True et les coordonnées terrain sont exploitables.

Sources d'information (par ordre de fiabilité) :
  1. camera_profile → goal_left_pct, sideline_y_top   (conf≈0.60)
  2. GeometryExtractor → vertical_anchor_x (BC.3A), sideline_y_bot
  3. Constantes FIFA → goal_width=7.32m, penalty_depth=16.5m

Usage :
  anchor = FieldAnchorModel()
  anchor.set_camera_prior(goal_left_pct=0.101, sideline_top_pct=0.28)
  anchor.update_from_observation(obs)   # obs.vertical_anchor_x depuis BC.3A

  if anchor.is_ready():
      anchor.ball_in_goal(bx=0.04)           → True / False
      anchor.distance_to_goal_m(bx=0.20)    → float (mètres)
      anchor.get_hypothesis()                → HypothesisResult
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ── Constantes terrain FIFA ────────────────────────────────────────────────────
GOAL_WIDTH_M    = 7.32    # m
PENALTY_DEPTH_M = 16.5    # m  (ligne de but → ligne de surface)
PENALTY_WIDTH_M = 40.32   # m  (largeur totale de la surface)

# ── Paramètres du moteur d'hypothèses ─────────────────────────────────────────
SCORE_READY_THRESHOLD  = 0.40   # score minimum pour déclarer Ready=True
CAMERA_PRIOR_CONF      = 0.60   # confiance par défaut dans camera_profile
CAMERA_TOLERANCE       = 0.15   # ±15% = plage réaliste d'erreur camera_prior
MIN_OBS_FOR_INFERENCE  = 5      # observations minimum avant de trancher

# px_per_m de bootstrap quand on n'a qu'une seule ancre
# Valeur typique pour une caméra à ~30m du but : 1920px / 105m ≈ 18px/m
# Sera affinée dès qu'on a deux ancres cohérentes
BOOTSTRAP_PX_PER_M     = 18.0


# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class HypothesisResult:
    """Résultat de l'inférence d'hypothèse."""
    name:             str              # "H1_goal_line" | "H2_penalty_line" | "H3_near_post" | "none"
    score:            float            # [0,1]
    goal_line_x:      Optional[float]  # x normalisé ligne de but
    penalty_line_x:   Optional[float]  # x normalisé ligne de surface
    px_per_m:         Optional[float]  # pixels par mètre
    score_camera:     float = 0.0      # contribution cohérence camera_prior
    score_fifa:       float = 0.0      # contribution cohérence FIFA
    score_stable:     float = 0.0      # contribution stabilité signal

    def is_valid(self) -> bool:
        return self.goal_line_x is not None and self.score > 0


# ── Modèle principal ───────────────────────────────────────────────────────────

class FieldAnchorModel:
    """
    Modèle terrain par inférence géométrique.

    Accumule les observations GeometryExtractor (vertical_anchor_x),
    teste 3 hypothèses FIFA et expose les coordonnées terrain dès que
    le score de cohérence dépasse SCORE_READY_THRESHOLD.
    """

    def __init__(self):
        # Prior depuis camera_profile
        self._camera_goal_left:   Optional[float] = None
        self._camera_prior_conf:  float           = CAMERA_PRIOR_CONF
        self.sideline_y_top:      Optional[float] = None

        # Accumulation observations (vertical_anchor_x depuis BC.3A)
        self._anchor_obs:         list[float]     = []   # toutes les valeurs brutes
        self._anchor_conf_sum:    float           = 0.0  # somme des confiances
        self._sideline_bot_sum:   float           = 0.0
        self._sideline_bot_n:     int             = 0

        # Résolution frame
        self.frame_width_px:  int = 1920
        self.frame_height_px: int = 1080

        # Résultat courant de l'inférence
        self._best_hypothesis: Optional[HypothesisResult] = None
        self._n_obs:           int = 0

        # Exposés publiquement pour compatibilité avec le pipeline existant
        self.goal_prior_left:  Optional[float] = None
        self.goal_prior_right: Optional[float] = None
        self.px_per_m:         Optional[float] = None

    # ── Alimentation ──────────────────────────────────────────────────────────

    def set_frame_size(self, width: int, height: int):
        self.frame_width_px  = width
        self.frame_height_px = height
        self._run_inference()

    def set_camera_prior(self,
                         goal_left_pct:    float,
                         sideline_top_pct: Optional[float] = None,
                         prior_conf:       float = CAMERA_PRIOR_CONF):
        """
        Initialise le prior depuis camera_profile.

        goal_left_pct : est_goal_left / frame_w  (ex: 193/1920 ≈ 0.101)
        prior_conf    : 0.60 = mesure indirecte (p01-p99 du ballon, pas
                        détection directe du poteau).
        """
        self._camera_goal_left  = goal_left_pct
        self._camera_prior_conf = prior_conf
        if sideline_top_pct is not None:
            self.sideline_y_top = sideline_top_pct
        # Le prior seul initialise goal_prior_left pour compatibilité
        self.goal_prior_left = goal_left_pct
        self._run_inference()

    def update_from_observation(self, obs) -> bool:
        """
        Intègre une observation GeometryExtractor.

        Priorité de lecture (BC.3A) :
          1. obs.vertical_anchor_x + obs.vertical_anchor_conf
          2. obs.penalty_line_x (BC.2 legacy)
          3. obs.goal_left_x (déprécié BC.2)
        """
        updated = False

        # Source BC.3A
        anchor_x    = getattr(obs, 'vertical_anchor_x', None)
        anchor_conf = getattr(obs, 'vertical_anchor_conf', None)
        anchor_type = getattr(obs, 'vertical_anchor_type', 'unknown')

        # Correction BC.4 — filtre ancres instables :
        # type="unknown" + x > 0.30 → milieu de terrain probable (pas le but ni la surface)
        # Ces obs gonflaient σ à 0.178 (supérieur à la moyenne = signal inutilisable)
        if anchor_x is not None and anchor_type == "unknown" and anchor_x > 0.30:
            if len(self._anchor_obs) < 5 or len(self._anchor_obs) % 20 == 0:
                # Logger seulement périodiquement pour éviter le spam
                print(f"  [ANCHOR] rejected type=unknown x={anchor_x:.3f}"
                      f" conf={anchor_conf:.2f}" if anchor_conf else
                      f"  [ANCHOR] rejected type=unknown x={anchor_x:.3f}")
            anchor_x = None  # ne pas accumuler

        # Fallback BC.2 legacy
        if anchor_x is None and getattr(obs, 'vertical_anchor_x', None) is None:
            anchor_x    = getattr(obs, 'penalty_line_x', None) or obs.goal_left_x
            anchor_conf = obs.confidence

        if anchor_x is not None and (anchor_conf or 0) >= 0.30:
            self._anchor_obs.append(anchor_x)
            self._anchor_conf_sum += anchor_conf
            updated = True

        if getattr(obs, 'sideline_y_bot', None) is not None:
            self._sideline_bot_sum += obs.sideline_y_bot
            self._sideline_bot_n   += 1
            updated = True

        if updated:
            self._n_obs += 1
            self._run_inference()

        return updated

    # ── Moteur d'hypothèses BC.3C ──────────────────────────────────────────────

    def _run_inference(self):
        """
        Teste H1 / H2 / H3 et retient l'hypothèse avec le meilleur score.
        Met à jour goal_prior_left, px_per_m, goal_prior_right, _best_hypothesis.
        """
        if not self._anchor_obs:
            return

        anchor_mean = sum(self._anchor_obs) / len(self._anchor_obs)
        anchor_std  = _stddev(self._anchor_obs)
        n_obs       = len(self._anchor_obs)

        # px_per_m courant : depuis sidelines si dispo, sinon bootstrap
        px_per_m = self._estimate_px_per_m()

        candidates = [
            self._score_h1(anchor_mean, anchor_std, n_obs, px_per_m),
            self._score_h2(anchor_mean, anchor_std, n_obs, px_per_m),
            self._score_h3(anchor_mean, anchor_std, n_obs, px_per_m),
        ]

        best = max(candidates, key=lambda h: h.score)
        self._best_hypothesis = best

        # Propager vers attributs publics pour compatibilité pipeline
        if best.is_valid() and best.score >= SCORE_READY_THRESHOLD:
            self.goal_prior_left  = best.goal_line_x
            self.px_per_m         = best.px_per_m
            if best.px_per_m:
                gw_pct = (GOAL_WIDTH_M * best.px_per_m) / self.frame_width_px
                self.goal_prior_right = best.goal_line_x + gw_pct

    def _score_h1(self, anchor_mean: float, anchor_std: float,
                  n_obs: int, px_per_m: float) -> HypothesisResult:
        """
        H1 : anchor = ligne de but (goal_line).
        Cohérent si anchor_mean ≈ camera_prior.
        """
        goal_x    = anchor_mean
        penalty_x = goal_x + (PENALTY_DEPTH_M * px_per_m / self.frame_width_px)

        s_camera = self._score_camera(goal_x)
        s_fifa   = self._score_fifa(goal_x, penalty_x)
        s_stable = self._score_stability(anchor_std, anchor_mean, n_obs)

        score = 0.50 * s_camera + 0.30 * s_fifa + 0.20 * s_stable

        return HypothesisResult(
            name           = "H1_goal_line",
            score          = score,
            goal_line_x    = goal_x,
            penalty_line_x = penalty_x,
            px_per_m       = px_per_m,
            score_camera   = s_camera,
            score_fifa     = s_fifa,
            score_stable   = s_stable,
        )

    def _score_h2(self, anchor_mean: float, anchor_std: float,
                  n_obs: int, px_per_m: float) -> HypothesisResult:
        """
        H2 : anchor = ligne de surface (penalty_line).
        goal_line = anchor - 16.5m.
        Valide même si goal_line < 0 (but hors champ gauche = caméra décalée).
        """
        penalty_x = anchor_mean
        goal_x    = penalty_x - (PENALTY_DEPTH_M * px_per_m / self.frame_width_px)

        s_camera = self._score_camera(goal_x)
        s_fifa   = self._score_fifa(goal_x, penalty_x)
        s_stable = self._score_stability(anchor_std, anchor_mean, n_obs)

        score = 0.50 * s_camera + 0.30 * s_fifa + 0.20 * s_stable

        return HypothesisResult(
            name           = "H2_penalty_line",
            score          = score,
            goal_line_x    = goal_x,
            penalty_line_x = penalty_x,
            px_per_m       = px_per_m,
            score_camera   = s_camera,
            score_fifa     = s_fifa,
            score_stable   = s_stable,
        )

    def _score_h3(self, anchor_mean: float, anchor_std: float,
                  n_obs: int, px_per_m: float) -> HypothesisResult:
        """
        H3 : anchor = poteau proche (near_post).
        Géométriquement : poteau ≈ ligne de but sur caméra lointaine.
        Sur caméra proche (low_side_zoom), le poteau est légèrement en retrait.
        H3 est H1 avec un décalage empirique de +0.5m.
        Si H3 > H1, ça indique que la caméra est proche du but.
        """
        post_offset_pct = (0.5 * px_per_m) / self.frame_width_px
        goal_x    = anchor_mean - post_offset_pct   # poteau est à droite de la ligne
        penalty_x = goal_x + (PENALTY_DEPTH_M * px_per_m / self.frame_width_px)

        s_camera = self._score_camera(goal_x)
        s_fifa   = self._score_fifa(goal_x, penalty_x)
        s_stable = self._score_stability(anchor_std, anchor_mean, n_obs)

        # H3 légèrement pénalisé vs H1 : on préfère la lecture directe sauf preuve contraire
        score = (0.50 * s_camera + 0.30 * s_fifa + 0.20 * s_stable) * 0.90

        return HypothesisResult(
            name           = "H3_near_post",
            score          = score,
            goal_line_x    = goal_x,
            penalty_line_x = penalty_x,
            px_per_m       = px_per_m,
            score_camera   = s_camera,
            score_fifa     = s_fifa,
            score_stable   = s_stable,
        )

    # ── Fonctions de score ─────────────────────────────────────────────────────

    def _score_camera(self, goal_line_x: float) -> float:
        """
        Cohérence entre goal_line_x dérivée et camera_prior.
        Score = 1.0 si accord parfait, 0.0 si écart ≥ CAMERA_TOLERANCE.
        Si pas de prior : 0.5 (neutre, ne pénalise pas).
        """
        if self._camera_goal_left is None:
            return 0.50
        diff = abs(goal_line_x - self._camera_goal_left)
        raw  = max(0.0, 1.0 - diff / CAMERA_TOLERANCE)
        # Pondérer par la confiance dans le prior
        return self._camera_prior_conf * raw + (1 - self._camera_prior_conf) * 0.50

    def _score_fifa(self, goal_line_x: float, penalty_line_x: float) -> float:
        """
        Cohérence avec les contraintes physiques FIFA.
          - goal_line > -0.10  (but peut être légèrement hors champ)
          - penalty_line > goal_line  (toujours vrai par construction)
          - penalty_line < 0.50  (surface pas à plus de 50% de l'image)
          - goal_line < 0.30  (but pas trop loin à droite sur caméra gauche)
        """
        score = 1.0
        # But trop hors champ gauche (> 10% hors frame)
        if goal_line_x < -0.10:
            score -= 0.50
        # Surface trop loin à droite
        if penalty_line_x > 0.50:
            score -= 0.30
        # But trop loin à droite (caméra ne vise pas le bon but)
        if goal_line_x > 0.30:
            score -= 0.40
        # Contrainte d'ordre (toujours vérifiée par construction, mais on s'en assure)
        if penalty_line_x <= goal_line_x:
            score -= 0.80
        return max(0.0, score)

    def _score_stability(self, std: float, mean: float, n_obs: int) -> float:
        """
        Stabilité du signal observé.
        σ/mean faible = signal stable = score élevé.
        Bonus si beaucoup d'observations.
        """
        if mean == 0 or n_obs == 0:
            return 0.0
        cv = std / abs(mean)   # coefficient de variation
        s_cv = max(0.0, 1.0 - cv / 0.20)   # 0.20 = 20% de variation = score=0
        # Bonus progressif d'observations (plafonné à MIN_OBS_FOR_INFERENCE×3)
        s_n  = min(1.0, n_obs / (MIN_OBS_FOR_INFERENCE * 3))
        return 0.70 * s_cv + 0.30 * s_n

    # ── Estimation px_per_m ────────────────────────────────────────────────────

    def _estimate_px_per_m(self) -> float:
        """
        Estime px_per_m depuis les sources disponibles.

        Priorité :
          1. Deux ancres cohérentes (camera_prior + mean_obs) : mesure directe
          2. Lignes de touche (sideline_top + sideline_bot + FIFA width)
          3. Bootstrap (valeur typique pour stade amateur)
        """
        # Source 1 : camera_prior + obs si les deux existent et sont cohérentes
        if self._camera_goal_left is not None and len(self._anchor_obs) >= MIN_OBS_FOR_INFERENCE:
            mean_obs = sum(self._anchor_obs) / len(self._anchor_obs)
            # Si obs ≈ camera_prior → H1 : anchor = goal_line → px_per_m indéterminé par ça seul
            # Si obs > camera_prior + 5% → probablement H2 → depth = obs - camera_prior
            depth_pct = mean_obs - self._camera_goal_left
            if depth_pct > 0.05:
                # La caméra voit la surface et le prior est la ligne de but
                depth_px = depth_pct * self.frame_width_px
                return depth_px / PENALTY_DEPTH_M

        # Source 2 : sidelines
        sbot = self.sideline_y_bot
        stop = self.sideline_y_top
        if sbot is not None and stop is not None and sbot > stop:
            pitch_height_pct = sbot - stop
            pitch_height_px  = pitch_height_pct * self.frame_height_px
            # FIFA : largeur terrain 68m. Hauteur vue = approximation perspective.
            # Sur caméra latérale à mi-terrain, ~40m visible en hauteur.
            # Heuristique : utiliser 40m comme référence Y
            return pitch_height_px / 40.0

        # Source 3 : bootstrap
        return BOOTSTRAP_PX_PER_M

    # ── Propriétés calculées ───────────────────────────────────────────────────

    @property
    def penalty_line_obs(self) -> Optional[float]:
        """Moyenne des observations vertical_anchor_x accumulées."""
        if not self._anchor_obs:
            return None
        return sum(self._anchor_obs) / len(self._anchor_obs)

    @property
    def sideline_y_bot(self) -> Optional[float]:
        if self._sideline_bot_n == 0:
            return None
        return self._sideline_bot_sum / self._sideline_bot_n

    def is_ready(self) -> bool:
        """
        True si le modèle a une hypothèse avec score ≥ SCORE_READY_THRESHOLD
        et suffisamment d'observations.
        """
        if self._best_hypothesis is None:
            return False
        if len(self._anchor_obs) < MIN_OBS_FOR_INFERENCE:
            return False
        return self._best_hypothesis.score >= SCORE_READY_THRESHOLD

    def get_hypothesis(self) -> Optional[HypothesisResult]:
        """Retourne l'hypothèse courante (gagnante), ou None."""
        return self._best_hypothesis

    # ── Requêtes géométriques ──────────────────────────────────────────────────

    def ball_in_goal(self, bx: float) -> Optional[bool]:
        """
        True si le ballon est dans le but (à gauche de la ligne de but).
        None si modèle non prêt.
        """
        if self.goal_prior_left is None or not self.is_ready():
            # Fallback : utiliser le prior camera seul si disponible
            if self._camera_goal_left is not None:
                return bx < self._camera_goal_left
            return None
        return bx < self.goal_prior_left

    def ball_in_penalty(self, bx: float) -> Optional[bool]:
        """True si le ballon est dans la surface. None si modèle non prêt."""
        h = self._best_hypothesis
        if h is None or h.penalty_line_x is None:
            return None
        return bx < h.penalty_line_x

    def distance_to_goal_m(self, bx: float) -> Optional[float]:
        """
        Distance en mètres à la ligne de but.
        Positif = devant le but. Négatif = dans le filet.
        None si px_per_m indisponible.
        """
        if self.goal_prior_left is None or self.px_per_m is None:
            return None
        dist_px = (bx - self.goal_prior_left) * self.frame_width_px
        return dist_px / self.px_per_m

    def distance_to_penalty_m(self, bx: float) -> Optional[float]:
        """Distance en mètres à la ligne de surface."""
        h = self._best_hypothesis
        if h is None or h.penalty_line_x is None or h.px_per_m is None:
            return None
        dist_px = (bx - h.penalty_line_x) * self.frame_width_px
        return dist_px / h.px_per_m

    # ── Résumé ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        h = self._best_hypothesis
        return {
            "ready":              self.is_ready(),
            "n_observations":     len(self._anchor_obs),
            "anchor_mean":        round(sum(self._anchor_obs)/len(self._anchor_obs), 4)
                                  if self._anchor_obs else None,
            "anchor_std":         round(_stddev(self._anchor_obs), 4)
                                  if len(self._anchor_obs) > 1 else None,
            "camera_prior":       self._camera_goal_left,
            "best_hypothesis":    h.name if h else "none",
            "hypothesis_score":   round(h.score, 3) if h else None,
            "goal_line_x":        round(h.goal_line_x, 4) if h and h.goal_line_x else None,
            "penalty_line_x":     round(h.penalty_line_x, 4) if h and h.penalty_line_x else None,
            "px_per_m":           round(h.px_per_m, 2) if h and h.px_per_m else None,
            "sideline_y_top":     self.sideline_y_top,
            "sideline_y_bot":     self.sideline_y_bot,
        }

    def summary(self):
        s   = self.get_state()
        h   = self._best_hypothesis
        obs = self._anchor_obs
        print("  [ANCHOR] ── FieldAnchorModel BC.3C ──────────────────────")
        print(f"  [ANCHOR]   Ready            : {s['ready']}")
        print(f"  [ANCHOR]   n_observations   : {s['n_observations']}")
        if obs:
            print(f"  [ANCHOR]   anchor_mean/std  : {s['anchor_mean']} / {s['anchor_std']}")
        print(f"  [ANCHOR]   camera_prior     : {s['camera_prior']}")
        if h:
            print(f"  [ANCHOR]   best_hypothesis  : {h.name}  score={h.score:.3f}")
            print(f"  [ANCHOR]     s_camera={h.score_camera:.2f}  "
                  f"s_fifa={h.score_fifa:.2f}  s_stable={h.score_stable:.2f}")
            print(f"  [ANCHOR]   goal_line_x      : {s['goal_line_x']}")
            print(f"  [ANCHOR]   penalty_line_x   : {s['penalty_line_x']}")
            print(f"  [ANCHOR]   px_per_m         : {s['px_per_m']}")
        print(f"  [ANCHOR]   sideline_y       : top={s['sideline_y_top']} bot={s['sideline_y_bot']}")
        print("  [ANCHOR] ──────────────────────────────────────────────────")


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _stddev(values: list[float]) -> float:
    """Écart-type population."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))