# vision/ocr.py

import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract non installé — OCR désactivé")


# ─────────────────────────────────────────
# PRÉTRAITEMENT IMAGE
# ─────────────────────────────────────────
def preprocess_patch(patch):
    """
    Prépare le crop d'un joueur pour l'OCR.
    On isole la zone maillot (tiers central de la bbox).
    """
    h, w = patch.shape[:2]

    # Garder uniquement le tiers central vertical (zone numéro)
    top    = h // 4
    bottom = 3 * h // 4
    patch  = patch[top:bottom, :]

    # Agrandir pour faciliter l'OCR
    patch = cv2.resize(patch, (w * 3, (bottom - top) * 3),
                       interpolation=cv2.INTER_CUBIC)

    # Niveaux de gris
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Contraste adaptatif
    gray = cv2.equalizeHist(gray)

    # Seuillage binaire + inversé (chiffres noirs sur blanc)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh


# ─────────────────────────────────────────
# LECTURE NUMÉRO SUR UN CROP
# ─────────────────────────────────────────
def read_number_from_patch(patch):
    """
    Tente de lire un numéro de maillot depuis un crop joueur.
    Retourne un entier ou None si non détecté.
    """
    if not TESSERACT_AVAILABLE:
        return None

    processed = preprocess_patch(patch)

    # Config Tesseract : chiffres uniquement, mode bloc
    config = (
        "--psm 8 "           # bloc de texte unique
        "--oem 3 "           # moteur LSTM
        "-c tessedit_char_whitelist=0123456789"
    )

    try:
        text = pytesseract.image_to_string(processed, config=config)
        text = text.strip()

        if text.isdigit() and 1 <= int(text) <= 99:
            return int(text)

    except Exception:
        pass

    return None


# ─────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────
class OCRReader:

    def __init__(self, min_confidence=0.6):
        self.min_confidence = min_confidence
        # Cache : track_id → numéro détecté + score confiance
        self._cache = {}

    # ─────────────────────────────────────────
    # LECTURE SUR UN JOUEUR TRACKÉ
    # ─────────────────────────────────────────
    def read_jersey(self, frame, player):
        """
        Lit le numéro de maillot d'un joueur tracké.

        player : dict avec "id", "bbox"
        frame  : image BGR complète

        Retourne le numéro (int) ou None.
        """
        track_id = player.get("id")

        # Si déjà détecté avec confiance → retourner depuis cache
        if track_id in self._cache:
            cached = self._cache[track_id]
            if cached["confidence"] >= self.min_confidence:
                return cached["number"]

        x1, y1, x2, y2 = [int(v) for v in player["bbox"]]

        # Sécurité bords image
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        patch  = frame[y1:y2, x1:x2]
        number = read_number_from_patch(patch)

        if number is not None:
            # Mise à jour cache avec score de confiance
            entry = self._cache.get(track_id, {"number": None, "confidence": 0, "votes": {}})
            votes = entry.get("votes", {})
            votes[number] = votes.get(number, 0) + 1

            # Numéro le plus voté = le plus fiable
            best        = max(votes, key=votes.get)
            total_votes = sum(votes.values())
            confidence  = votes[best] / total_votes

            self._cache[track_id] = {
                "number":     best,
                "confidence": confidence,
                "votes":      votes
            }

            return best

        # Retourner ce qu'on a en cache même si confiance faible
        if track_id in self._cache:
            return self._cache[track_id]["number"]

        return None

    # ─────────────────────────────────────────
    # LECTURE SUR TOUS LES JOUEURS D'UNE FRAME
    # ─────────────────────────────────────────
    def read_all(self, frame, players):
        """
        Lit les numéros de tous les joueurs trackés.

        Retourne la liste players enrichie avec "jersey" :
        [{"id": 1, "bbox": [...], "jersey": 9}, ...]
        """
        for p in players:
            p["jersey"] = self.read_jersey(frame, p)

        return players

    # ─────────────────────────────────────────
    # RÉSUMÉ DES NUMÉROS DÉTECTÉS
    # ─────────────────────────────────────────
    def get_jersey_map(self):
        """
        Retourne un dict track_id → numéro maillot
        pour tous les joueurs détectés avec confiance suffisante.
        """
        return {
            tid: data["number"]
            for tid, data in self._cache.items()
            if data["confidence"] >= self.min_confidence
            and data["number"] is not None
        }

    def reset(self):
        """Vide le cache — à appeler entre deux vidéos."""
        self._cache = {}