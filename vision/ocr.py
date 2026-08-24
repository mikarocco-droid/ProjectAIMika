# vision/ocr.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("pytesseract non installe — OCR desactive")


# ─────────────────────────────────────────
# PRÉTRAITEMENT IMAGE
# ─────────────────────────────────────────
def preprocess_patch(patch):
    h, w = patch.shape[:2]

    if h < 20 or w < 10:
        return None

    # Zone maillot : entre 20% et 60% de la hauteur
    top    = int(h * 0.20)
    bottom = int(h * 0.60)
    roi    = patch[top:bottom, :]

    if roi.size == 0:
        return None

    # Agrandir
    roi = cv2.resize(roi, (w * 3, (bottom - top) * 3),
                     interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


# ─────────────────────────────────────────
# LECTURE NUMÉRO
# ─────────────────────────────────────────
def read_number_from_patch(patch):
    if not TESSERACT_AVAILABLE:
        return None

    processed = preprocess_patch(patch)
    if processed is None:
        return None

    cfg = (
        "--psm 8 "  # V5.1 : teste 6/7/8/10/11/13 sur image propre -
                    # seul psm=8 (et 10/13) lit correctement "47" en
                    # entier ; psm=7 (tente initialement) ne lit que "7"
                    # - regression evitee par test reel avant deploiement
        "--oem 3 "
        "-c tessedit_char_whitelist=0123456789"
    )

    try:
        text = pytesseract.image_to_string(processed, config=cfg)
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

    def __init__(self, min_confidence=0.6, ocr_every_n_frames=30):
        self.min_confidence    = min_confidence
        self.ocr_every_n_frames = ocr_every_n_frames  # ← 1 fois par seconde
        self._cache            = {}
        self._frame_counter    = 0

    # ─────────────────────────────────────────
    # LECTURE SUR UN JOUEUR
    # ─────────────────────────────────────────
    def read_jersey(self, frame, player):
        track_id = player.get("id")

        # Retourner depuis cache si confiance suffisante
        if track_id in self._cache:
            cached = self._cache[track_id]
            if cached["confidence"] >= self.min_confidence:
                return cached["number"]

        if not TESSERACT_AVAILABLE:
            return None

        x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
        h_frame, w_frame = frame.shape[:2]

        # FIX V5.1 : padding horizontal de 20% de la largeur de la bbox,
        # de chaque cote, AVANT le clip aux bords de la frame. Sans ca,
        # un numero a deux chiffres proche du bord de la bbox (joueur
        # legerement de biais, bbox mal centree sur le torse) se fait
        # couper d'un cote - Tesseract en mode "un seul mot" (psm 8) lit
        # alors correctement le SEUL chiffre restant, avec une confiance
        # elevee et repetee (erreur systematique, pas du bruit aleatoire,
        # donc non filtree par le systeme de vote). Confirme visuellement
        # sur Raeren : #17 et #44 clairement lisibles a l'ecran, mais lus
        # une seule fois chacun sur tout le match dans jersey_map_brut
        # (94% des lectures totales sont des chiffres simples 1-9, alors
        # qu'aucun numero a deux chiffres n'apparait plus d'une fois).
        pad = int((x2 - x1) * 0.20)
        x1 -= pad
        x2 += pad

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h_frame, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        patch  = frame[y1:y2, x1:x2]
        number = read_number_from_patch(patch)

        if number is not None:
            entry = self._cache.get(track_id, {
                "number": None, "confidence": 0, "votes": {}
            })
            votes = entry.get("votes", {})
            votes[number] = votes.get(number, 0) + 1

            best       = max(votes, key=votes.get)
            confidence = votes[best] / sum(votes.values())

            self._cache[track_id] = {
                "number":     best,
                "confidence": confidence,
                "votes":      votes
            }
            return best

        if track_id in self._cache:
            return self._cache[track_id]["number"]

        return None

    # ─────────────────────────────────────────
    # LECTURE SUR TOUS LES JOUEURS
    # ─────────────────────────────────────────
    def read_all(self, frame, players, frame_id=None):
        """
        Lit les numéros de maillot.
        OCR actif seulement 1 frame sur ocr_every_n_frames.
        Les autres frames retournent le cache.
        """
        self._frame_counter += 1

        # Toujours retourner le jersey depuis le cache
        for p in players:
            tid = p.get("id")
            if tid in self._cache:
                p["jersey"] = self._cache[tid].get("number")
            else:
                p["jersey"] = None

        # OCR seulement 1 frame sur N
        if not TESSERACT_AVAILABLE:
            return players

        if self._frame_counter % self.ocr_every_n_frames != 0:
            return players

        # Frame OCR — lire les numéros
        for p in players:
            jersey = self.read_jersey(frame, p)
            p["jersey"] = jersey

        return players

    # ─────────────────────────────────────────
    # JERSEY MAP
    # ─────────────────────────────────────────
    def get_jersey_map(self):
        return {
            tid: data["number"]
            for tid, data in self._cache.items()
            if data["confidence"] >= self.min_confidence
            and data["number"] is not None
        }

    def reset(self):
        self._cache         = {}
        self._frame_counter = 0