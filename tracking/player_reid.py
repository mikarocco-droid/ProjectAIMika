# tracking/player_reid.py
# -*- coding: utf-8 -*-

import numpy as np


class PlayerReID:
    """
    Stabilise les IDs joueurs entre les frames.
    Quand ByteTrack perd un joueur et lui réassigne un nouvel ID,
    PlayerReID détecte que c'est le même joueur (même position)
    et lui redonne son ID d'origine.

    Résultat : 398 IDs uniques → ~22 joueurs stables
    """

    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.memory       = {}   # reid_id -> dernière bbox_center
        self.id_map       = {}   # tracker_id -> reid_id
        self.next_id      = 1

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _get_or_create_reid_id(self, tracker_id, bbox_center):
        # Si on a déjà un mapping pour ce tracker_id, on le réutilise
        if tracker_id in self.id_map:
            reid_id = self.id_map[tracker_id]
            self.memory[reid_id] = bbox_center
            return reid_id

        # Chercher un joueur proche en mémoire
        for reid_id, prev_center in self.memory.items():
            if self._distance(bbox_center, prev_center) < self.max_distance:
                # Même joueur — on réutilise son reid_id
                self.id_map[tracker_id] = reid_id
                self.memory[reid_id]    = bbox_center
                return reid_id

        # Nouveau joueur
        reid_id                 = self.next_id
        self.next_id           += 1
        self.memory[reid_id]    = bbox_center
        self.id_map[tracker_id] = reid_id
        return reid_id

    def process_tracks(self, tracks):
        """
        tracks : liste de dicts avec "id", "bbox", "center", "conf"
        Retourne la même liste avec "id" stabilisé par Re-ID.
        """
        results = []

        for t in tracks:
            bbox   = t.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            reid_id = self._get_or_create_reid_id(t["id"], center)

            results.append({
                **t,
                "id":          reid_id,
                "tracker_id":  t["id"],   # garde l'ID original pour debug
                "center":      [center[0], center[1]]
            })

        return results

    def reset(self):
        self.memory  = {}
        self.id_map  = {}
        self.next_id = 1