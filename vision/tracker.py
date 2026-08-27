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
                "conf":   1.0,
                # FIX : expose la fraîcheur de la piste. time_since_update=0
                # signifie une VRAIE détection YOLO cette frame ; >0 signifie
                # une position purement prédite par Kalman (le joueur n'a pas
                # été redétecté depuis N frames — occlusion, rassemblement
                # dense...). Diagnostiqué sur Andrimont : des bbox "confirmées"
                # mais dérivées de plusieurs dizaines de pixels à côté du
                # vrai joueur, faussant l'extraction de couleur de maillot.
                "time_since_update": t.time_since_update,
            })

        # V5.2 FIX : deduplique par tracker_id AU SEIN de la meme frame.
        # Demontre sur Goe (65% des frames) puis Raeren (~90% des frames,
        # CPU et GPU) : la sortie brute de update_tracks() peut contenir
        # DEUX entrees distinctes pour le MEME track_id, a des positions
        # differentes et des time_since_update differents (donc deux
        # objets Track distincts partageant un ID, pas une reference
        # dupliquee). Cause exacte (deep_sort_realtime lui-meme, ou nos
        # detections en entree) non elucidee - voir V4_CLUSTERING_ET_
        # AUTOPSIE_RF.md §13.1. Ce correctif ne resout pas la cause, il
        # neutralise le symptome le plus dommageable (classification
        # equipe/evenements dupliques en aval) : on garde la version la
        # PLUS FRAICHE (time_since_update le plus bas) de chaque doublon,
        # celle la plus susceptible de refleter une vraie detection YOLO
        # de cette frame plutot qu'une prediction Kalman perimee.
        # V5.2 DIAGNOSTIC - avant de continuer a deviner pourquoi le taux de
        # doublons ne baisse pas malgre le fix (mesure sur Andrimont CPU :
        # 88.3% apres fix, quasi identique a avant) - affiche explicitement
        # ce qui se passe A CE POINT PRECIS, une fois par 200 frames pour
        # ne pas noyer le log. A retirer une fois la cause comprise.
        if not hasattr(self, "_diag_frame_count"):
            self._diag_frame_count = 0
        self._diag_frame_count += 1
        _diag_actif = (self._diag_frame_count % 200 == 1)

        ids_bruts = [r["id"] for r in results]
        from collections import Counter as _Counter_diag
        _doublons_bruts = {k: v for k, v in _Counter_diag(ids_bruts).items() if v > 1}
        if _diag_actif and _doublons_bruts:
            print(f"    [DIAG DEDUP frame#{self._diag_frame_count}] AVANT dedup : "
                  f"{len(results)} entrees, doublons bruts={_doublons_bruts}")

        par_id = {}
        ids_vus = set()
        n_doublons = 0
        for r in results:
            tid = r["id"]
            if tid in ids_vus:
                n_doublons += 1
                if r["time_since_update"] < par_id[tid]["time_since_update"]:
                    par_id[tid] = r
            else:
                ids_vus.add(tid)
                par_id[tid] = r
        results = list(par_id.values())

        if _diag_actif and _doublons_bruts:
            ids_apres = [r["id"] for r in results]
            _doublons_apres = {k: v for k, v in _Counter_diag(ids_apres).items() if v > 1}
            print(f"    [DIAG DEDUP frame#{self._diag_frame_count}] APRES dedup : "
                  f"{len(results)} entrees, n_doublons_comptes={n_doublons}, "
                  f"doublons_restants={_doublons_apres}")

        return results

    def reset(self):
        self.__init__()