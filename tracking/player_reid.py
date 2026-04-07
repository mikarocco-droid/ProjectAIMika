# tracking/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class PlayerReID:
    """
    ReID hybride calibré :
    - position + couleur maillot + embedding histogramme
    - contrainte spatiale (SPATIAL_MAX_DIST appris par MatchLearner)
    - contrainte inter-équipes (jamais fusion A/B)
    - TTL adaptatif (seuil plus strict pour joueurs en veille)
    - MAX_PLAYERS = 25
    """

    TTL_ACTIVE  = 125    # 5s  @25fps
    TTL_SLEEP   = 500    # 20s @25fps
    MAX_PLAYERS = 25

    # Seuils de matching (ajustés dynamiquement via set_spatial_max_dist)
    SPATIAL_MAX_DIST = 200.0
    THRESHOLD_ACTIVE = 90.0
    THRESHOLD_SLEEP  = 120.0

    def __init__(self, fps=25):
        self.fps         = fps
        self.memory      = {}
        self.next_id     = 0
        self.frame_count = 0

    def set_spatial_max_dist(self, dist):
        """Appelé depuis vision/tracker.py avec la valeur apprise."""
        self.SPATIAL_MAX_DIST = max(100.0, min(400.0, float(dist)))

    # ─────────────────────────────────────────
    # COULEUR MAILLOT (torse)
    # ─────────────────────────────────────────
    def _extract_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0])
        h    = crop.shape[0]
        crop = crop[int(h * 0.2):int(h * 0.6), :]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0])
        return crop.mean(axis=(0, 1)).astype(float)

    # ─────────────────────────────────────────
    # ÉQUIPE depuis couleur maillot
    # ─────────────────────────────────────────
    def _infer_team(self, color):
        r, g, b = color[2], color[1], color[0]  # BGR
        if r > 150 and g < 120 and b < 120:
            return 1   # rouge
        if b > 150 and g > 150 and r < 100:
            return 0   # cyan/bleu
        return None

    # ─────────────────────────────────────────
    # EMBEDDING HISTOGRAMME
    # ─────────────────────────────────────────
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
            [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        return cv2.normalize(hist, hist).flatten()

    # ─────────────────────────────────────────
    # CONTRAINTE SPATIALE
    # ─────────────────────────────────────────
    def _spatial_gate(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2)) < self.SPATIAL_MAX_DIST

    # ─────────────────────────────────────────
    # SCORE GLOBAL
    # ─────────────────────────────────────────
    def _compute_score(self, det, mem):
        pos_d = np.linalg.norm(np.array(det["center"]) - np.array(mem["center"]))
        col_d = np.linalg.norm(det["color"]     - mem["color"])
        emb_d = np.linalg.norm(det["embedding"] - mem["embedding"])

        frames_absent = self.frame_count - mem["last_seen"]
        penalty       = 1.0 + (frames_absent / self.TTL_SLEEP) * 0.5

        return (pos_d * 0.5 + col_d * 0.3 + emb_d * 0.2) * penalty

    # ─────────────────────────────────────────
    # CLEANUP TTL
    # ─────────────────────────────────────────
    def _cleanup_memory(self):
        to_del = [
            pid for pid, mem in self.memory.items()
            if self.frame_count - mem["last_seen"] > self.TTL_SLEEP
        ]
        for pid in to_del:
            del self.memory[pid]

    # ─────────────────────────────────────────
    # ASSIGNATION ID
    # ─────────────────────────────────────────
    def _assign_id(self, det):
        active_count = sum(
            1 for m in self.memory.values()
            if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE
        )

        best_id    = None
        best_score = float("inf")

        for pid, mem in self.memory.items():
            # Contrainte spatiale
            if not self._spatial_gate(det["center"], mem["center"]):
                continue

            # Contrainte inter-équipes
            dt = det.get("team")
            mt = mem.get("team")
            if dt is not None and mt is not None and dt != mt:
                continue

            # Seuil adaptatif
            frames_absent = self.frame_count - mem["last_seen"]
            threshold = (self.THRESHOLD_ACTIVE
                         if frames_absent <= self.TTL_ACTIVE
                         else self.THRESHOLD_SLEEP)

            score = self._compute_score(det, mem)
            if score < best_score and score < threshold:
                best_score = score
                best_id    = pid

        if best_id is not None:
            self.memory[best_id].update({
                "center":    det["center"],
                "color":     det["color"],
                "embedding": det["embedding"],
                "team":      det.get("team", self.memory[best_id].get("team")),
                "last_seen": self.frame_count,
            })
            return best_id

        # Nouveau joueur — MAX_PLAYERS
        if active_count >= self.MAX_PLAYERS:
            if self.memory:
                return min(
                    self.memory.keys(),
                    key=lambda p: np.linalg.norm(
                        np.array(det["center"]) - np.array(self.memory[p]["center"])
                    )
                )
            return 0

        pid = self.next_id
        self.memory[pid] = {**det, "last_seen": self.frame_count}
        self.next_id += 1
        return pid

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────
    def stats(self):
        active   = sum(1 for m in self.memory.values()
                       if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE)
        sleeping = len(self.memory) - active
        return {
            "total_ids":       self.next_id,
            "in_memory":       len(self.memory),
            "active":          active,
            "sleeping":        sleeping,
            "spatial_max_dist": self.SPATIAL_MAX_DIST,
        }

    # ─────────────────────────────────────────
    # PROCESS FRAME
    # ─────────────────────────────────────────
    def process(self, frame, detections):
        self.frame_count += 1
        self._cleanup_memory()

        results = []
        for det in detections:
            bbox   = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            color  = self._extract_color(frame, bbox)
            team   = self._infer_team(color)

            enriched = {
                "center":    center,
                "color":     color,
                "embedding": self._extract_embedding(frame, bbox),
                "team":      team,
            }

            reid_id = self._assign_id(enriched)

            results.append({
                **det,
                "id":         reid_id,
                "player_id":  reid_id,
                "tracker_id": det.get("id"),
                "center":     list(center),
                "team":       team,
            })

        return results

    def reset(self):
        self.memory      = {}
        self.next_id     = 0
        self.frame_count = 0