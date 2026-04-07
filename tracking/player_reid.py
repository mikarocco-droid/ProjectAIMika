# tracking/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class PlayerReID:
    """
    ReID hybride : position + couleur maillot + embedding histogramme
    + memory intelligente avec TTL (Time To Live)

    TTL system :
    - joueur actif    : vu dans les 5 dernières secondes (125 frames @25fps)
    - joueur en veille: pas vu depuis 5-30s → récupérable si revient
    - joueur oublié   : pas vu depuis >30s → supprimé de la mémoire
    """

    # TTL en frames (@25fps)
    TTL_ACTIVE  = 125   # 5s  — joueur considéré actif
    TTL_SLEEP   = 750   # 30s — joueur en veille, récupérable
    # Au-delà de TTL_SLEEP → supprimé

    def __init__(self, max_distance=80, fps=25):
        self.max_distance = max_distance
        self.fps          = fps
        self.memory       = {}    # pid -> {center, color, embedding, last_seen, active}
        self.next_id      = 0
        self.frame_count  = 0

    # ─────────────────────────────
    # COULEUR DOMINANTE MAILLOT
    # ─────────────────────────────
    def _extract_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0])

        h = crop.shape[0]
        # Torse uniquement (évite pelouse + shorts)
        crop = crop[int(h * 0.2):int(h * 0.6), :]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0])

        return crop.mean(axis=(0, 1)).astype(float)

    def _color_distance(self, c1, c2):
        return np.linalg.norm(c1 - c2)

    # ─────────────────────────────
    # EMBEDDING HISTOGRAMME
    # ─────────────────────────────
    def _extract_embedding(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512)

        crop = cv2.resize(crop, (32, 64))
        hist = cv2.calcHist(
            [crop], [0, 1, 2], None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )
        return cv2.normalize(hist, hist).flatten()

    def _embedding_distance(self, e1, e2):
        return np.linalg.norm(e1 - e2)

    # ─────────────────────────────
    # SCORE GLOBAL
    # ─────────────────────────────
    def _compute_score(self, det, mem):
        pos_d = np.linalg.norm(
            np.array(det["center"]) - np.array(mem["center"])
        )
        col_d = self._color_distance(det["color"], mem["color"])
        emb_d = self._embedding_distance(det["embedding"], mem["embedding"])

        # Pénalité si joueur en veille (pas vu depuis longtemps)
        frames_absent = self.frame_count - mem["last_seen"]
        sleep_penalty = 1.0 + (frames_absent / self.TTL_SLEEP) * 0.5

        return (pos_d * 0.5 + col_d * 0.3 + emb_d * 0.2) * sleep_penalty

    # ─────────────────────────────
    # MEMORY — nettoyage TTL
    # ─────────────────────────────
    def _cleanup_memory(self):
        """
        Supprime les joueurs non vus depuis plus de TTL_SLEEP frames.
        Appelé à chaque frame pour garder la mémoire propre.
        """
        to_delete = [
            pid for pid, mem in self.memory.items()
            if self.frame_count - mem["last_seen"] > self.TTL_SLEEP
        ]
        for pid in to_delete:
            del self.memory[pid]

    # ─────────────────────────────
    # ASSIGNATION ID
    # ─────────────────────────────
    def _assign_id(self, det):
        best_id    = None
        best_score = float("inf")

        for pid, mem in self.memory.items():
            score = self._compute_score(det, mem)
            if score < best_score and score < self.max_distance:
                best_score = score
                best_id    = pid

        if best_id is not None:
            # Mettre à jour la mémoire du joueur
            self.memory[best_id].update({
                "center":    det["center"],
                "color":     det["color"],
                "embedding": det["embedding"],
                "last_seen": self.frame_count,
                "active":    True
            })
            return best_id

        # Nouveau joueur
        pid              = self.next_id
        self.memory[pid] = {
            **det,
            "last_seen": self.frame_count,
            "active":    True
        }
        self.next_id += 1
        return pid

    # ─────────────────────────────
    # STATS (debug)
    # ─────────────────────────────
    def stats(self):
        active  = sum(
            1 for m in self.memory.values()
            if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE
        )
        sleeping = sum(
            1 for m in self.memory.values()
            if self.TTL_ACTIVE < self.frame_count - m["last_seen"] <= self.TTL_SLEEP
        )
        return {
            "total_ids": self.next_id,
            "in_memory": len(self.memory),
            "active":    active,
            "sleeping":  sleeping
        }

    # ─────────────────────────────
    # PROCESS FRAME
    # ─────────────────────────────
    def process(self, frame, detections):
        """
        Appelé depuis vision/tracker.py après ByteTrack.
        detections : liste de dicts avec "id", "bbox", "center", "conf"
        Retourne la même liste avec "id" stabilisé par ReID.
        """
        self.frame_count += 1

        # Nettoyer les joueurs oubliés (>30s sans apparition)
        self._cleanup_memory()

        results = []

        for det in detections:
            bbox   = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            enriched = {
                "center":    center,
                "color":     self._extract_color(frame, bbox),
                "embedding": self._extract_embedding(frame, bbox),
            }

            reid_id = self._assign_id(enriched)

            results.append({
                **det,
                "player_id":  reid_id,
                "tracker_id": det.get("id"),
                "id":         reid_id,
                "center":     list(center)
            })

        return results

    def reset(self):
        self.memory      = {}
        self.next_id     = 0
        self.frame_count = 0