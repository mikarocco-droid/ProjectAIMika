"""
FieldHomography — Sprint D (placeholder)
==========================================
Ce module sera implémenté dans Sprint D uniquement si
la reconstruction géométrique de Sprint BC atteint ses limites.

Sprint BC fournit des coordonnées normalisées [0,1] dans l'image.
Sprint D transformera ces coordonnées en mètres réels sur le terrain.

Terrain standard football :
  105m × 68m
  Surface de réparation : 40.32m × 16.5m
  But : 7.32m de large × 2.44m de haut
  Distance du point de penalty : 11m

Pré-requis pour Sprint D :
  - 4+ points de correspondance image ↔ terrain (sprints BC)
  - Confiance du modèle terrain ≥ 0.70
  - Méthode recommandée : cv2.findHomography avec RANSAC

Placeholder intentionnel — ne pas implémenter avant que Sprint BC
ne démontre ses limites.
"""


class FieldHomography:
    """
    Placeholder Sprint D.
    Transforme les coordonnées image [0,1] en mètres réels.
    """

    FIELD_LENGTH = 105.0  # mètres
    FIELD_WIDTH  = 68.0
    GOAL_WIDTH   = 7.32
    GOAL_HEIGHT  = 2.44
    PENALTY_DIST = 11.0

    def __init__(self):
        self._H = None  # matrice homographie 3x3
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def image_to_field(self, x_norm: float, y_norm: float):
        """
        Convertit des coordonnées image normalisées en coordonnées terrain (mètres).
        Retourne None si homographie non calculée.
        """
        if not self._ready:
            return None
        # Sprint D : implémenter cv2.perspectiveTransform ici
        return None

    def field_to_image(self, x_m: float, y_m: float):
        """
        Convertit des coordonnées terrain (mètres) en coordonnées image normalisées.
        """
        if not self._ready:
            return None
        return None

    def distance_to_goal_meters(self, bx: float, by: float) -> float:
        """
        Calcule la distance réelle en mètres du ballon au centre du but.
        Retourne -1 si homographie non disponible.
        """
        if not self._ready:
            return -1.0
        pos = self.image_to_field(bx, by)
        if pos is None:
            return -1.0
        # Sprint D : calculer distance euclidienne au centre du but
        return -1.0
