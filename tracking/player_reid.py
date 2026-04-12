# tracking/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import defaultdict


class PlayerReID:
    """
    ReID hybride calibré :
    - position + couleur maillot + embedding histogramme
    - contrainte spatiale (SPATIAL_MAX_DIST appris par MatchLearner)
    - contrainte inter-équipes (jamais fusion A/B)
    - TTL adaptatif (seuil plus strict pour joueurs en veille)
    - MAX_PLAYERS = 25
    - Calibration dynamique des couleurs équipes (KMeans)
    - jersey_map intégré : stabilise les IDs via numéro maillot
    """

    TTL_ACTIVE  = 125    # 5s  @25fps
    TTL_SLEEP   = 500    # 20s @25fps
    MAX_PLAYERS = 25

    SPATIAL_MAX_DIST = 200.0
    THRESHOLD_ACTIVE = 90.0
    THRESHOLD_SLEEP  = 120.0

    CALIB_FRAMES     = 50
    CALIB_MIN_SAMPLE = 8

    def __init__(self, fps=25):
        self.fps         = fps
        self.memory      = {}
        self.next_id     = 0
        self.frame_count = 0

        # Calibration équipes
        self._team_colors_calibrated = False
        self._team_color_samples     = []
        self._team_centroids         = None

        # jersey_map intégré : {reid_id → jersey_number}
        # Permet de stabiliser les IDs via les numéros détectés
        self._jersey_map             = {}
        # Mapping inverse : jersey_number → reid_id canonique
        self._jersey_to_canonical    = {}

    def set_spatial_max_dist(self, dist):
        self.SPATIAL_MAX_DIST = max(100.0, min(400.0, float(dist)))

    def update_jersey_map(self, jersey_map):
        """
        Intègre le jersey_map externe (depuis Gemini OCR) dans le ReID.
        Permet de lier les IDs numériques aux numéros de maillots.
        """
        for pid, jersey in jersey_map.items():
            pid_str    = str(pid)
            jersey_str = str(jersey) if jersey is not None else None
            if jersey_str:
                self._jersey_map[pid_str] = jersey_str
                # Premier ID avec ce jersey = canonique
                if jersey_str not in self._jersey_to_canonical:
                    self._jersey_to_canonical[jersey_str] = pid_str

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
    # CALIBRATION ÉQUIPES (KMeans dynamique)
    # ─────────────────────────────────────────
    def _calibrate_teams(self):
        if len(self._team_color_samples) < self.CALIB_MIN_SAMPLE:
            return

        samples  = np.array(self._team_color_samples, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centroids = cv2.kmeans(
            samples, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )

        self._team_centroids         = centroids
        self._team_colors_calibrated = True

        c0 = centroids[0].astype(int)
        c1 = centroids[1].astype(int)
        print(f"  ReID équipes calibrées : "
              f"équipe0=BGR({c0[0]},{c0[1]},{c0[2]}) "
              f"équipe1=BGR({c1[0]},{c1[1]},{c1[2]}) "
              f"({len(samples)} samples)")

    def _infer_team(self, color):
        c = color.astype(np.float32)

        if not self._team_colors_calibrated:
            if np.any(c > 10):
                self._team_color_samples.append(c)
            if (len(self._team_color_samples) >= self.CALIB_MIN_SAMPLE
                    and self.frame_count >= self.CALIB_FRAMES):
                self._calibrate_teams()
            return None

        d0 = np.linalg.norm(c - self._team_centroids[0])
        d1 = np.linalg.norm(c - self._team_centroids[1])

        if abs(d0 - d1) < 15.0:
            return None

        return 0 if d0 < d1 else 1

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
    # Amélioration : si deux IDs ont le même jersey → retourner le canonique
    # ─────────────────────────────────────────
    def _assign_id(self, det):
        active_count = sum(
            1 for m in self.memory.values()
            if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE
        )

        best_id    = None
        best_score = float("inf")

        for pid, mem in self.memory.items():
            if not self._spatial_gate(det["center"], mem["center"]):
                continue

            dt = det.get("team")
            mt = mem.get("team")
            if dt is not None and mt is not None and dt != mt:
                continue

            frames_absent = self.frame_count - mem["last_seen"]
            threshold = (self.THRESHOLD_ACTIVE
                         if frames_absent <= self.TTL_ACTIVE
                         else self.THRESHOLD_SLEEP)

            score = self._compute_score(det, mem)
            if score < best_score and score < threshold:
                best_score = score
                best_id    = pid

        if best_id is not None:
            # Mise à jour mémoire avec lissage exponentiel sur la couleur
            alpha = 0.3
            mem   = self.memory[best_id]
            mem["center"]    = det["center"]
            mem["color"]     = alpha * det["color"] + (1 - alpha) * mem["color"]
            mem["embedding"] = alpha * det["embedding"] + (1 - alpha) * mem["embedding"]
            mem["team"]      = det.get("team") or mem.get("team")
            mem["last_seen"] = self.frame_count
            return best_id

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

            existing_team = det.get("team")
            team = existing_team if existing_team is not None else self._infer_team(color)

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

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────
    def stats(self):
        active   = sum(1 for m in self.memory.values()
                       if self.frame_count - m["last_seen"] <= self.TTL_ACTIVE)
        sleeping = len(self.memory) - active
        return {
            "total_ids":        self.next_id,
            "in_memory":        len(self.memory),
            "active":           active,
            "sleeping":         sleeping,
            "spatial_max_dist": self.SPATIAL_MAX_DIST,
            "teams_calibrated": self._team_colors_calibrated,
        }

    # ─────────────────────────────────────────
    # TEAM DISTRIBUTION
    # ─────────────────────────────────────────
    def get_team_distribution(self):
        dist = defaultdict(int)
        for mem in self.memory.values():
            t = mem.get("team")
            if t is not None:
                dist[t] += 1
        return dict(dist)

    def reset(self):
        self.memory                  = {}
        self.next_id                 = 0
        self.frame_count             = 0
        self._team_colors_calibrated = False
        self._team_color_samples     = []
        self._team_centroids         = None
        self._jersey_map             = {}
        self._jersey_to_canonical    = {}