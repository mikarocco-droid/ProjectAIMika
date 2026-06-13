"""
FieldAnchorModel — Sprint BC.2 (pivot architecture)
=====================================================
Remplace le FieldModel basé sur la détection de poteaux.

Philosophie :
  Les poteaux ne sont pas détectables par Hough sur les caméras latérales
  compressées (angles 1-36°, jamais >50°).

  Ce modèle reconstruit le terrain à partir de structures réellement
  visibles : ligne de surface, lignes de touche, + prior du camera_profile.

Sources d'information (par ordre de fiabilité) :
  1. camera_profile  → est_goal_left, sideline_y_top   (haute fiabilité)
  2. GeometryExtractor → penalty_line_obs, sideline_bot  (fiabilité moyenne)
  3. Constantes FIFA  → goal_width=7.32m, penalty_depth=16.5m

Usage dans le pipeline :
  anchor = FieldAnchorModel()
  anchor.set_camera_prior(
      goal_left_pct=0.101,      # depuis camera_profile
      sideline_top_pct=0.28,    # depuis calibration
  )
  anchor.update_from_observation(obs)  # depuis GeometryExtractor

  # Requêtes
  anchor.ball_in_goal(bx=0.04)         → True
  anchor.ball_in_penalty(bx=0.15)      → True
  anchor.distance_to_goal_m(bx=0.20)  → 1.43 m
"""

from typing import Optional


# ── Constantes terrain standard ────────────────────────────────────────────────
GOAL_WIDTH_M     = 7.32    # m
PENALTY_DEPTH_M  = 16.5    # m  (ligne de but → ligne de surface)
PENALTY_WIDTH_M  = 40.32   # m  (largeur totale de la surface)


class FieldAnchorModel:
    """
    Modèle terrain basé sur des ancres observables plutôt que sur les poteaux.

    Les ancres sont :
      goal_prior_left  : position ligne de but gauche (camera_profile)
      penalty_line_obs : position ligne de surface (GeometryExtractor)
      px_per_m         : échelle pixels/mètre (déduite des deux ancres)
      sideline_y_top   : ligne de touche haute (calibration)
      sideline_y_bot   : ligne de touche basse (GeometryExtractor)
    """

    def __init__(self):
        # Ancres depuis camera_profile (prior haute fiabilité)
        self.goal_prior_left:  Optional[float] = None   # x normalisé [0,1]
        self.sideline_y_top:   Optional[float] = None   # y normalisé [0,1]

        # Ancres depuis GeometryExtractor (accumulées par EMA)
        self._penalty_obs_sum:  float = 0.0
        self._penalty_obs_n:    int   = 0
        self._sideline_bot_sum: float = 0.0
        self._sideline_bot_n:   int   = 0

        # Dérivés
        self.px_per_m:          Optional[float] = None
        self.goal_prior_right:  Optional[float] = None

        self._n_obs = 0
        self._goal_prior_left_camera: Optional[float] = None
        self._prior_conf: float = 0.60
        self.frame_width_px: int = 1920   # à surcharger via set_frame_size()
        self.frame_height_px: int = 1080

    # ── Alimentation depuis le pipeline ────────────────────────────────────────

    def set_frame_size(self, width: int, height: int):
        """Injecter la résolution réelle — évite de hardcoder 1920."""
        self.frame_width_px  = width
        self.frame_height_px = height
        self._recompute()

    def set_camera_prior(self,
                         goal_left_pct:    float,
                         sideline_top_pct: Optional[float] = None,
                         prior_conf:       float = 0.60):
        """
        Initialise les ancres depuis le camera_profile.

        goal_left_pct    : est_goal_left / frame_w   (ex: 193/1920 = 0.101)
        sideline_top_pct : y_min de la zone terrain    (ex: 0.28)
        prior_conf       : confiance dans ce prior [0,1].
            0.60 = camera_profile est approximatif (p01-p99 du ballon, pas
                   une mesure directe). Permet une fusion pondérée ultérieure.
        """
        self._goal_prior_left_camera = goal_left_pct
        self._prior_conf             = prior_conf
        self.goal_prior_left         = goal_left_pct   # valeur courante
        if sideline_top_pct is not None:
            self.sideline_y_top = sideline_top_pct
        self._recompute()

    def update_from_observation(self, obs) -> bool:
        """
        Intègre une observation GeometryExtractor.
        Retourne True si l'observation a enrichi le modèle.
        """
        updated = False

        # penalty_line_x : la structure détectée par GeometryExtractor sur cette
        # caméra n'est pas un poteau mais la ligne de surface de réparation.
        # obs.goal_left_x est le nom hérité — on lit penalty_line_x en priorité.
        penalty_x = getattr(obs, 'penalty_line_x', None) or obs.goal_left_x
        if penalty_x is not None and obs.confidence >= 0.30:
            self._penalty_obs_sum += penalty_x
            self._penalty_obs_n   += 1
            updated = True

        if obs.sideline_y_bot is not None:
            self._sideline_bot_sum += obs.sideline_y_bot
            self._sideline_bot_n   += 1
            updated = True

        if updated:
            self._n_obs += 1
            self._recompute()

        return updated

    # ── Recalcul des dérivés ───────────────────────────────────────────────────

    def _recompute(self):
        """Recalcule px_per_m et goal_prior_right depuis les ancres disponibles."""
        penalty_obs = self.penalty_line_obs

        if self.goal_prior_left is not None and penalty_obs is not None:
            depth_pct = penalty_obs - self.goal_prior_left
            if depth_pct > 0.02:  # au moins 2% d'écart pour être fiable
                depth_px = depth_pct * self.frame_width_px
                self.px_per_m = depth_px / PENALTY_DEPTH_M

                # Fusion pondérée goal_left : camera_prior × prior_conf
                # + contrainte géométrique × (1 - prior_conf)
                # La contrainte : penalty_obs - 16.5m × px_per_m
                camera_val  = getattr(self, '_goal_prior_left_camera', self.goal_prior_left)
                prior_conf  = getattr(self, '_prior_conf', 0.60)
                obs_derived = penalty_obs - (PENALTY_DEPTH_M * self.px_per_m / self.frame_width_px)
                self.goal_prior_left = (prior_conf * camera_val
                                       + (1 - prior_conf) * obs_derived)

                goal_width_pct = (GOAL_WIDTH_M * self.px_per_m) / self.frame_width_px
                self.goal_prior_right = self.goal_prior_left + goal_width_pct

    # ── Propriétés calculées ───────────────────────────────────────────────────

    @property
    def penalty_line_obs(self) -> Optional[float]:
        """Moyenne des observations de la ligne de surface."""
        if self._penalty_obs_n == 0:
            return None
        return self._penalty_obs_sum / self._penalty_obs_n

    @property
    def sideline_y_bot(self) -> Optional[float]:
        if self._sideline_bot_n == 0:
            return None
        return self._sideline_bot_sum / self._sideline_bot_n

    def is_ready(self) -> bool:
        """True si le modèle a suffisamment d'ancres pour raisonner."""
        return (self.goal_prior_left is not None
                and self.penalty_line_obs is not None
                and self.px_per_m is not None)

    # ── Requêtes géométriques ──────────────────────────────────────────────────

    def ball_in_goal(self, bx: float) -> Optional[bool]:
        """
        True si le ballon est à gauche de la ligne de but (dans le filet).
        None si le modèle n'est pas initialisé.
        """
        if self.goal_prior_left is None:
            return None
        return bx < self.goal_prior_left

    def ball_in_penalty(self, bx: float) -> Optional[bool]:
        """
        True si le ballon est dans la surface de réparation.
        None si le modèle n'est pas initialisé.
        """
        if self.penalty_line_obs is None:
            return None
        return bx < self.penalty_line_obs

    def distance_to_goal_m(self, bx: float) -> Optional[float]:
        """
        Distance en mètres du ballon à la ligne de but.
        Positif = devant le but, négatif = dans le filet.
        None si px_per_m non disponible.
        """
        if self.goal_prior_left is None or self.px_per_m is None:
            return None
        dist_px = (bx - self.goal_prior_left) * self.frame_width_px
        return dist_px / self.px_per_m

    def distance_to_penalty_m(self, bx: float) -> Optional[float]:
        """Distance en mètres du ballon à la ligne de surface."""
        if self.penalty_line_obs is None or self.px_per_m is None:
            return None
        dist_px = (bx - self.penalty_line_obs) * self.frame_width_px
        return dist_px / self.px_per_m

    # ── Résumé ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "ready":            self.is_ready(),
            "goal_prior_left":  self.goal_prior_left,
            "goal_prior_right": self.goal_prior_right,
            "penalty_line_obs": self.penalty_line_obs,
            "sideline_y_top":   self.sideline_y_top,
            "sideline_y_bot":   self.sideline_y_bot,
            "px_per_m":         round(self.px_per_m, 2) if self.px_per_m else None,
            "n_observations":   self._n_obs,
        }

    def summary(self):
        s = self.get_state()
        print("  [ANCHOR] ── FieldAnchorModel ────────────────────────")
        print(f"  [ANCHOR]   Ready          : {s['ready']}")
        print(f"  [ANCHOR]   goal_prior     : left={s['goal_prior_left']} right={s['goal_prior_right']}")
        print(f"  [ANCHOR]   penalty_line   : {s['penalty_line_obs']} (obs={self._penalty_obs_n})")
        print(f"  [ANCHOR]   sideline_y     : top={s['sideline_y_top']} bot={s['sideline_y_bot']}")
        print(f"  [ANCHOR]   px_per_m       : {s['px_per_m']}")
        print(f"  [ANCHOR]   n_observations : {s['n_observations']}")
        print("  [ANCHOR] ──────────────────────────────────────────────")
