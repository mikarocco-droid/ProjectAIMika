# vision/tracker.py
# -*- coding: utf-8 -*-
# DeepSort mobilenet FP16 GPU — meilleur compromis vitesse/qualité sur T4
# ByteTrack natif testé et abandonné : Kalman+Hungarian Python pur > GPU batch

import numpy as np
import config
from analysis.player_reid import PlayerReID


class Tracker:
    def __init__(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        import torch as _torch
        _has_gpu = _torch.cuda.is_available()
        self.tracker = DeepSort(
            max_age      = 90,
            embedder     = "mobilenet",
            half         = _has_gpu,
            bgr          = True,
            embedder_gpu = _has_gpu,
        )
        self.mode = "deepsort"
        print(f"  Tracker : DeepSort (mobilenet {'FP16 GPU' if _has_gpu else 'FP32 CPU'})")

        self.reid = PlayerReID(fps=config.FPS)
        self.reid.set_spatial_max_dist(80)

    def update(self, players, frame):
        if not players:
            return []

        results = self._update_deepsort(players, frame)
        results = self.reid.process(frame, results)

        for r in results:
            if "player_id" in r:
                r["tracker_id"] = r.get("id", r["player_id"])
                r["id"]         = r["player_id"]

        return results

    def _update_deepsort(self, players, frame):
        detections = [
            ([p["bbox"][0], p["bbox"][1],
              p["bbox"][2] - p["bbox"][0],
              p["bbox"][3] - p["bbox"][1]], p["conf"], "player")
            for p in players
        ]
        tracks  = self.tracker.update_tracks(detections, frame=frame)
        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            results.append({
                "id":     t.track_id,
                "bbox":   [x1, y1, x2, y2],
                "center": [(x1+x2)/2, (y1+y2)/2],
                "conf":   1.0
            })
        return results

    def reset(self):
        self.__init__()