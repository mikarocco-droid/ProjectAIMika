# vision/tracker.py
# -*- coding: utf-8 -*-

import numpy as np

try:
    from boxmot import ByteTrack
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("boxmot non installe — fallback DeepSort")

import config
from tracking.player_reid import PlayerReID


class Tracker:
    def __init__(self):
        # ── Tracker principal ──────────────────
        if BOXMOT_AVAILABLE:
            self.tracker = ByteTrack(
                track_thresh = 0.35,
                track_buffer = 150,
                match_thresh = 0.85,
                frame_rate   = config.FPS
            )
            self.mode = "bytetrack"
            print("  Tracker : ByteTrack (buffer=150)")
        else:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(max_age=90)
            self.mode    = "deepsort"
            print("  Tracker : DeepSort (fallback)")

        # ── Re-ID — stabilise les IDs entre les frames ──
        # max_distance=50px : si un joueur réapparaît à moins de 50px
        # de sa dernière position connue, il garde le même ID
        self.reid = PlayerReID(max_distance=50)

    def update(self, players, frame):
        if not players:
            return []

        if self.mode == "bytetrack":
            results = self._update_bytetrack(players, frame)
        else:
            results = self._update_deepsort(players, frame)

        # ── Appliquer Re-ID sur les résultats du tracker ──
        # Réduit les doublons d'IDs (398 maillots → ~22 joueurs)
        reid_results = self.reid.process_tracks(results)
        return reid_results

    def _update_bytetrack(self, players, frame):
        dets = np.array([
            [p["bbox"][0], p["bbox"][1],
             p["bbox"][2], p["bbox"][3],
             p["conf"], 0]
            for p in players
        ], dtype=np.float32)

        tracks  = self.tracker.update(dets, frame)
        results = []

        for t in tracks:
            x1, y1, x2, y2, track_id = (
                t[0], t[1], t[2], t[3], int(t[4])
            )
            results.append({
                "id":     track_id,
                "bbox":   [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "conf":   float(t[5]) if len(t) > 5 else 1.0
            })

        return results

    def _update_deepsort(self, players, frame):
        detections = []
        for p in players:
            x1, y1, x2, y2 = p["bbox"]
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], p["conf"], "player")
            )

        tracks  = self.tracker.update_tracks(detections, frame=frame)
        results = []

        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            results.append({
                "id":     t.track_id,
                "bbox":   [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "conf":   1.0
            })

        return results

    def reset(self):
        self.__init__()